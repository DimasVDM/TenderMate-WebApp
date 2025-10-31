# function_app.py — TenderMate HTTP endpoint (chat + retrieval) met Azure AI Foundry gpt-4o
# - JSON + multipart (PDF/DOCX)
# - Hybride search (BM25 + vector) met semantic config
# - Compacte context caps zodat er ruimte blijft voor lange quickscans
# - gpt-4o als default (minder strikt dan gpt-5)

import azure.functions as func
import logging
import os
import json
import io
import re
import zipfile
import traceback
import binascii
from typing import List, Dict, Any

# ---- Bestandslezers ----
from pypdf import PdfReader
import docx
from docx.opc.exceptions import PackageNotFoundError

# ---- Multipart decoder ----
from requests_toolbelt.multipart.decoder import MultipartDecoder

# ---- Azure Search ----
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery, QueryType

# ---- Azure OpenAI (chat + embeddings) ----
from openai import AzureOpenAI

# =============================================================================
# ENV / Config
# =============================================================================

# Azure AI Search
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
AZURE_SEARCH_API_KEY  = os.environ.get("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_INDEX    = os.environ.get("AZURE_SEARCH_INDEX", "sharepoint-vectorizer-direct-v2")

TEXT_VECTOR_FIELD = "text_vector"
SEMANTIC_CONFIG   = os.environ.get("SEMANTIC_CONFIG", "sharepoint-vectorizer-semantic-configuration")
TOP_K             = int(os.environ.get("TOP_K", "5"))

# Azure AI Foundry (OpenAI)
AOAI_ENDPOINT         = os.environ.get("AOAI_ENDPOINT", "").rstrip("/")
AOAI_API_KEY          = os.environ.get("AOAI_API_KEY", "")
# >>> We zetten de default hier op gpt-4o <<<
AOAI_CHAT_DEPLOYMENT  = os.environ.get("AOAI_CHAT_DEPLOYMENT", "gpt-4o")
AOAI_EMBED_DEPLOYMENT = os.environ.get("AOAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

# Output-budgets (completion tokens) – genoeg voor jouw lange quickscan
AOAI_MAX_OUT_QUICKSCAN = int(os.environ.get("AOAI_MAX_OUT_QUICKSCAN", "2400"))
AOAI_MAX_OUT_DRAFT     = int(os.environ.get("AOAI_MAX_OUT_DRAFT", "2000"))
AOAI_MAX_OUT_REVIEW    = int(os.environ.get("AOAI_MAX_OUT_REVIEW", "1600"))
AOAI_MAX_OUT_INFO      = int(os.environ.get("AOAI_MAX_OUT_INFO", "800"))

# Document/context caps (karakters) – iets krapper zodat het model nog kan schrijven
DOC_CAP_QUICK_REVIEW = int(os.environ.get("DOC_CAP_QUICK_REVIEW", "5500"))
DOC_CAP_DRAFT        = int(os.environ.get("DOC_CAP_DRAFT", "8000"))
CTX_PER_DOC_CHARS    = int(os.environ.get("CTX_PER_DOC_CHARS", "500"))
CTX_MAX_DOCS         = int(os.environ.get("CTX_MAX_DOCS", "3"))
CTX_MAX_TOTAL_CHARS  = int(os.environ.get("CTX_MAX_TOTAL_CHARS", "3000"))
MAX_CONTEXT_DOCS     = int(os.environ.get("MAX_CONTEXT_DOCS", "10"))

# CORS
ALLOWED_ORIGINS = set([
    "https://nice-bay-0e1280e03.2.azurestaticapps.net",
])
ALLOWED_METHODS = "GET,POST,OPTIONS"
ALLOWED_HEADERS = "Content-Type,Authorization,x-functions-key"
ALLOW_CREDENTIALS = "true"

# =============================================================================

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

_search_client: SearchClient = None
_aoai_client: AzureOpenAI    = None


def _cors_headers(req: func.HttpRequest) -> Dict[str, str]:
    origin = (req.headers.get("Origin") or "").rstrip("/")
    if origin in ALLOWED_ORIGINS:
        return {
            "Access-Control-Allow-Origin": origin,
            "Vary": "Origin",
            "Access-Control-Allow-Credentials": ALLOW_CREDENTIALS,
            "Access-Control-Allow-Methods": ALLOWED_METHODS,
            "Access-Control-Allow-Headers": ALLOWED_HEADERS,
        }
    return {}


def get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:
        if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
            raise RuntimeError("AZURE_SEARCH_ENDPOINT/API_KEY ontbreken.")
        _search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
        )
    return _search_client


def get_aoai_client() -> AzureOpenAI:
    global _aoai_client
    if _aoai_client is None:
        if not AOAI_ENDPOINT or not AOAI_API_KEY:
            raise RuntimeError("AOAI_ENDPOINT/API_KEY ontbreken.")
        _aoai_client = AzureOpenAI(
            api_key=AOAI_API_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint=AOAI_ENDPOINT,
        )
    return _aoai_client

# =============================================================================
# Utilities
# =============================================================================

def _clip(txt: str, max_chars: int) -> str:
    if not txt:
        return ""
    return txt if len(txt) <= max_chars else txt[:max_chars]


def _hex_prefix(b: bytes, n: int = 16) -> str:
    return binascii.hexlify(b[:n]).decode().upper()


def _is_probably_docx(file_bytes: bytes) -> bool:
    return zipfile.is_zipfile(io.BytesIO(file_bytes))


def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def _read_docx_text(file_bytes: bytes) -> str:
    if not _is_probably_docx(file_bytes):
        sig = _hex_prefix(file_bytes, 8)
        raise zipfile.BadZipFile(f"Not a DOCX/ZIP (magic={sig})")
    d = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in d.paragraphs])


def embed_text(text: str) -> List[float]:
    client = get_aoai_client()
    resp = client.embeddings.create(
        model=AOAI_EMBED_DEPLOYMENT,
        input=text
    )
    return resp.data[0].embedding


def _dedupe_key(hit: Dict[str, Any]) -> str:
    pid = (hit.get("parent_id") or "").strip()
    src = (hit.get("source") or "").strip()
    head = (hit.get("content") or "")[:80].strip()
    return f"{pid}|{src}|{head}"


def _get_field(r, name, default=""):
    try:
        v = getattr(r, name, None)
        if v is not None:
            return v
        if hasattr(r, "document") and isinstance(r.document, dict):
            v = r.document.get(name)
            if v is not None:
                return v
        try:
            v = r.get(name, None)
            if v is not None:
                return v
        except Exception:
            pass
    except Exception:
        pass
    return default


def _join_docs_for_context(docs, per_doc_chars=CTX_PER_DOC_CHARS,
                           max_docs=CTX_MAX_DOCS, max_context_chars=CTX_MAX_TOTAL_CHARS) -> str:
    blocks, total = [], 0
    for d in docs[:max_docs]:
        src = d.get("source") or ""
        pid = d.get("parent_id") or ""
        label = src if not pid else f"{src} (parent: {pid})"
        content = _clip((d.get("content", "") or "").strip(), per_doc_chars)
        if not content:
            continue
        block = f"Source: {label}\nContent:\n{content}"
        if total + len(block) > max_context_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def format_history_for_prompt_flow(conversation: List[Dict[str, str]]) -> List[Dict[str, Dict[str, str]]]:
    chat_history = []
    for i in range(0, len(conversation), 2):
        if (
            i + 1 < len(conversation)
            and isinstance(conversation[i], dict)
            and isinstance(conversation[i + 1], dict)
            and conversation[i].get("role") == "user"
            and conversation[i + 1].get("role") == "bot"
        ):
            chat_history.append(
                {
                    "inputs":  {"chat_input":  conversation[i].get("content", "")},
                    "outputs": {"chat_output": conversation[i + 1].get("content", "")},
                }
            )
    return chat_history

# =============================================================================
# Search
# =============================================================================

def _expand_queries(user_q: str, mode: str, doc_hint: str = "") -> List[str]:
    base = (user_q or "").strip()
    mode = (mode or "").upper()
    hint = (doc_hint or "")[:300]

    txt = base.lower()
    tenderish = any(k in txt for k in [
        "aanbested", "gunnings", "pve", "programma van eisen", "arbit", "gibit", "arvodi",
        "uea", "offerte", "score", "quickscan", "referentie", "proof of concept"
    ])
    if not tenderish:
        return [base] if base else []

    keywords_common = [
        "(prijs OR prijsplafond OR 'geraamde waarde' OR kosten)",
        "(contractduur OR looptijd OR verlenging OR startdatum OR einddatum)",
        "(weging OR beoordelingscriteria OR paginabudget OR demo OR PoC)",
        "(referentie OR geschiktheid OR KO OR uitsluitingsgronden)",
        "(GIBIT OR ARBIT OR ARVODI OR 'Verwerkersovereenkomst' OR AVG OR DPIA)",
        "('TLS en HTTP response headers' OR TLS OR HSTS OR CSP)",
        "(Common Ground OR ZGW OR STUF OR ZKN OR OData OR CSV OR 'Power BI')",
        "('EN 301 549' OR WCAG OR toegankelijkheid OR B1 OR voorleesfunctie)",
        "('Concern Informatie Architectuur' OR exitstrategie OR SLA OR support)",
        "('Verklaring geen Russische betrokkenheid' OR 'Wachtkamerovereenkomst' OR Conceptovereenkomst)"
    ]
    mode_bonus = []
    if mode == "QUICKSCAN":
        mode_bonus = ["(go/no-go OR kwalificatie OR showstoppers OR risico’s)"]
    elif mode == "REVIEW":
        mode_bonus = ["(KPI OR SMART OR dekkingsmatrix OR 'boven verwachting')"]
    else:
        mode_bonus = ["(structuur OR hoofdstukindeling OR 'paginabudget verdeling')"]

    out = []
    if base:
        out.append(base)
        out.append(f"{base} {' '.join(keywords_common + mode_bonus)}")
        out.append(f"{base} prijsplafond OR 'geraamde waarde' OR contractduur OR 2026 OR 2027 OR 2028")
        if hint:
            out.append(f"{base} {hint}")

    seen, uniq = set(), []
    for q in out:
        q = q.strip()
        if q and q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq


def hybrid_search_multi(queries: List[str], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    sc = get_search_client()
    all_hits: Dict[str, Dict[str, Any]] = {}

    for q in queries:
        vec = embed_text(q)
        vq = VectorizedQuery(vector=vec, k_nearest_neighbors=top_k, fields=TEXT_VECTOR_FIELD)
        results = sc.search(
            search_text=q,
            vector_queries=[vq],
            top=top_k,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name=SEMANTIC_CONFIG,
        )
        for r in results:
            hit = {
                "content":   _get_field(r, "chunk", ""),
                "source":    _get_field(r, "title", ""),
                "parent_id": _get_field(r, "parent_id", ""),
            }
            if not hit["content"]:
                continue
            k = _dedupe_key(hit)
            if k not in all_hits:
                all_hits[k] = hit

    return list(all_hits.values())[: (top_k * 2)]

# =============================================================================
# Prompt rendering
# =============================================================================

def detect_mode(explicit_mode: str, chat_input: str) -> str:
    m = (explicit_mode or "").strip().upper()
    if m in ("QUICKSCAN", "REVIEW", "DRAFT", "INFO"):
        return m
    txt = (chat_input or "").lower()
    tenderish = any(k in txt for k in [
        "aanbested", "quickscan", "gunnings", "pve", "programma van eisen", "arbit", "gibit",
        "arvodi", "uea", "offerte", "score", "referentie", "paginabudget", "kpi"
    ])
    smalltalk = (len(txt.split()) <= 6) and (not tenderish) and any(
        g in txt for g in ["hoi", "hallo", "hey", "wat kan", "kun jij", "help", "wie ben"]
    )
    if smalltalk:
        return "INFO"
    if "quickscan" in txt or "go/no-go" in txt or "go no go" in txt:
        return "QUICKSCAN"
    if "review" in txt or "beoordeel" in txt or "verbeter" in txt:
        return "REVIEW"
    return "DRAFT" if tenderish else "INFO"


def render_system_and_user(mode: str, document_text: str, context_text: str,
                           chat_history_pf: List[Dict[str, Any]], chat_input: str):
    hist_lines = []
    for item in chat_history_pf:
        hist_lines.append("user:\n" + (item["inputs"].get("chat_input") or ""))
        hist_lines.append("assistant:\n" + (item["outputs"].get("chat_output") or ""))
    history_block = "\n".join(hist_lines)

    common_suffix = f"""
context: {context_text}

chat history:
{history_block}

user:
{chat_input}
""".strip()

    if mode == "QUICKSCAN":
        system = (
            "Je bent TenderMate, de AI-tenderanalist en strategisch bid-adviseur van IT-Workz.\n"
            "STRICT_FACT_MODE: gebruik uitsluitend 'document_text' en 'context'. "
            "Ontbreekt info → schrijf exact: “Niet gespecificeerd in stukken.” "
            "Schrijf in helder Nederlands, met de koppen hieronder."
        )
        user = f"""
document_text: {document_text or ''}

Maak een **quickscan (kwalificatie / go-no-go)** op basis van CONTEXT en aangeleverd document.
Volg deze koppen exact:

## Samenvatting in één oogopslag
- **Advies**: 🟢 / 🟡 / 🔴 — met 1 zin waarom.
- **Belangrijkste 3 showstoppers/risico’s** (met mitigatie-hint).
- **Belangrijkste 3 scoringskansen**.

## Feitenblad (tabel)
| Veld | Waarde |
|---|---|
| **Naam aanbesteding / opdrachtgever** | ... |
| **Soort aanbesteding/procedure** | ... |
| **Sector/organisatie-type** | ... |
| **Aantal locaties/scholen** | ... |
| **Aantal medewerkers** | ... |
| **Aantal leerlingen/studenten** | ... |
| **# vragenronden** | ... |
| **Vragen stellen – kanaal/portaal** | ... |
| **Inkoopadviseur / contact** | ... |
| **Contractduur (basis + opties)** | ... |
| **Contractwaarde / richtprijs / prijsplafond** | ... |

## Aanleiding en doel
## Essentie van de uitvraag (scope)
## KO-criteria en voorwaarden (checklist)
## Architectuur & Standaarden
## Programma van Eisen/Wensen (samengevat per cluster)
## Analyse van Vereiste Disciplines
## Weging, paginabudget en planning
## Budget en concurrentiepositie
## Showstoppers / risico’s (met mitigatie of verhelderingsvraag)
## Concurrentie-analyse
## Referenties (advies)
## In te dienen bewijsstukken & status
## Standaard- en verdiepingsvragen
## Actiechecklist (intern)
## Go/No-Go indicatie (conclusie)
""".strip() + "\n\n" + common_suffix

    elif mode == "REVIEW":
        system = (
            "Je bent TenderMate, AI-kwaliteitsauditor voor aanbestedingen. "
            "Beoordeel t.o.v. gunningskader en CONTEXT. Ontbrekend = “Niet gespecificeerd in stukken.”"
        )
        user = f"""
document_text: {document_text or ''}

Beoordeel en versterk de tekst t.o.v. gunningscriterium.
Volg deze koppen exact:
## Korte overall beoordeling (in 5 bullets)
## Dekking & structuur t.o.v. gunningskader (incl. dekkingsmatrix)
## KO/voorwaarden & juridische punten
## Architectuur & standaarden
## KPI-audit (SMART)
## Stijl & tone of voice
## Paginabudget & visuele opbouw
## “Boven verwachting” (onderscheidend)
## Herschrijfsuggesties (voorbeeld)
## Referenties & bewijs
## Actiechecklist (intern)
## Eindoordeel (kort)
""".strip() + "\n\n" + common_suffix

    elif mode == "INFO":
        system = "Je bent TenderMate, een AI-assistent voor aanbestedingen. Antwoord kort en praktisch."
        user = f"""
Geef in bullets:
- Wat jij kunt (QUICKSCAN / DRAFT / REVIEW)
- Hoe je CONTEXT gebruikt
- Hoe ik een document aanlever
- Hoe ik betere antwoorden krijg
""".strip() + "\n\n" + common_suffix

    else:  # DRAFT
        system = (
            "Je bent TenderMate, AI-tekstarchitect voor aanbestedingen. STRICT_FACT_MODE: alleen document_text/context."
        )
        user = f"""
document_text: {document_text or ''}

Genereer een scoregerichte concepttekst met:
## Executive summary (scorefocus in 5 bullets)
## Begrips- & beoordelingskader
## Voorstelstructuur conform criterium
## Uitwerking per subcriterium
## Planning & mijlpalen
## Rollen & RACI
## KPI-overzicht
## Paginabudget & opmaak
## Referenties & bewijs
## Verhelderingsvragen
""".strip() + "\n\n" + common_suffix

    return system, user

# =============================================================================
# Chat call — gpt-4o friendly
# =============================================================================

def call_chat(system_text: str, user_text: str, mode_hint: str = "") -> str:
    client = get_aoai_client()
    model_name = AOAI_CHAT_DEPLOYMENT.lower()

    # outputbudget
    mh = (mode_hint or "").upper()
    if mh == "QUICKSCAN":
        max_out = AOAI_MAX_OUT_QUICKSCAN
    elif mh == "DRAFT":
        max_out = AOAI_MAX_OUT_DRAFT
    elif mh == "REVIEW":
        max_out = AOAI_MAX_OUT_REVIEW
    else:
        max_out = AOAI_MAX_OUT_INFO

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]

    # 1) probeer stream + response_format
    try:
        stream = client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=messages,
            stream=True,
            # gpt-4o: max_tokens ok
            max_tokens=max_out,
            temperature=0.7,
            response_format={"type": "text"},
        )
        chunks: List[str] = []
        for event in stream:
            try:
                delta = event.choices[0].delta
            except Exception:
                continue
            c = getattr(delta, "content", None)
            if isinstance(c, str) and c:
                chunks.append(c)
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") in ("text", "output_text"):
                        t = p.get("text")
                        if isinstance(t, str) and t:
                            chunks.append(t)
        txt = "".join(chunks).strip()
        if txt:
            return txt
    except Exception as e:
        logging.warning(f"stream with response_format failed, fallback to non-stream. Error: {e}")

    # 2) non-stream, met response_format
    try:
        resp = client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=messages,
            max_tokens=max_out,
            temperature=0.7,
            response_format={"type": "text"},
        )
        parts = []
        for ch in resp.choices:
            c = getattr(ch.message, "content", None)
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") in ("text", "output_text"):
                        t = p.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
        final_txt = "\n".join(parts).strip()
        if final_txt:
            return final_txt
    except Exception as e:
        logging.warning(f"non-stream with response_format failed, try plain. Error: {e}")

    # 3) allerlaatste fallback: plain call
    try:
        resp = client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=messages,
            max_tokens=max_out,
            temperature=0.7,
        )
        parts = []
        for ch in resp.choices:
            c = getattr(ch.message, "content", None)
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
        final_txt = "\n".join(parts).strip()
        if final_txt:
            return final_txt
    except Exception as e:
        logging.exception(f"plain chat call failed: {e}")
        return f"AI-dienst error: {e}"

    return "Er kwam geen leesbare tekst terug van het AI-model. Probeer het nogmaals of verlaag de omvang."

# =============================================================================
# HTTP Function
# =============================================================================

@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # CORS preflight
        if req.method == "OPTIONS":
            return func.HttpResponse(status_code=204, headers=_cors_headers(req))

        if req.method == "GET":
            return func.HttpResponse("OK - TenderMate TalkToTenderBot - gpt-4o", status_code=200,
                                     mimetype="text/plain", headers=_cors_headers(req))

        # intake logging
        try:
            ct = (req.headers.get("Content-Type") or "").lower()
            raw_len = len(req.get_body() or b"")
            logging.info(f"req: ct={ct}, raw_len={raw_len}")
        except Exception:
            pass

        question = ""
        conversation = []
        document_text = ""
        mode = ""
        content_type = (req.headers.get("Content-Type") or "").lower()

        # multipart
        if "multipart/form-data" in content_type:
            body_bytes = req.get_body() or b""
            try:
                dec = MultipartDecoder(body_bytes, content_type)
            except Exception:
                return func.HttpResponse("Ongeldig multipart-verzoek.", status_code=400,
                                         mimetype="text/plain", headers=_cors_headers(req))

            text_fields = {}
            file_part = None
            file_mt = ""
            file_name = ""

            for p in dec.parts:
                cd = p.headers.get(b'Content-Disposition', b'').decode(errors="ignore")
                ctype = p.headers.get(b'Content-Type', b'').decode(errors="ignore").lower()
                if 'filename=' in cd:
                    file_part = p
                    file_mt = ctype or "application/octet-stream"
                    m = re.search(r'filename\*?="?([^";]+)"?', cd, flags=re.IGNORECASE)
                    file_name = (m.group(1) if m else "").strip()
                else:
                    m = re.search(r'name="([^"]+)"', cd or "", flags=re.IGNORECASE)
                    if m:
                        text_fields[m.group(1)] = p.text

            question = (text_fields.get("question") or "").strip()
            mode = (text_fields.get("mode") or "").strip()
            try:
                conversation = json.loads(text_fields.get("conversation") or "[]")
            except Exception:
                conversation = []

            if file_part is not None:
                file_bytes = file_part.content or b""
                ext = (os.path.splitext(file_name or "")[1] or "").lower()
                try:
                    if ("pdf" in file_mt) or (ext == ".pdf"):
                        document_text = _read_pdf_text(file_bytes)
                    elif ("wordprocessingml" in file_mt) or (ext == ".docx") or (file_mt == "application/octet-stream" and ext == ".docx"):
                        document_text = _read_docx_text(file_bytes)
                    else:
                        return func.HttpResponse("Bestandstype niet ondersteund (upload PDF of DOCX).",
                                                 status_code=415, mimetype="text/plain", headers=_cors_headers(req))
                except zipfile.BadZipFile:
                    sig = _hex_prefix(file_bytes)
                    return func.HttpResponse(
                        f"Kon DOCX niet lezen (magic={sig}). Sla opnieuw op als .docx.",
                        status_code=422, mimetype="text/plain", headers=_cors_headers(req))
                except PackageNotFoundError:
                    return func.HttpResponse("Kon DOCX niet openen. Sla opnieuw op als .docx.",
                                             status_code=422, mimetype="text/plain", headers=_cors_headers(req))
                except Exception as e:
                    return func.HttpResponse(f"Fout bij lezen van document: {repr(e)}",
                                             status_code=422, mimetype="text/plain", headers=_cors_headers(req))

        # json
        elif "application/json" in content_type:
            try:
                body = req.get_json()
            except ValueError:
                raw = req.get_body()
                try:
                    body = json.loads(raw.decode("utf-8", errors="ignore"))
                except Exception:
                    body = {}
            question      = (body.get("question") or body.get("chat_input") or "").strip()
            conversation  = body.get("conversation", []) or []
            document_text = body.get("document_text", "") or ""
            mode          = (body.get("mode") or "").strip()
        else:
            return func.HttpResponse(
                f"Content-Type '{content_type}' niet ondersteund.",
                status_code=415, mimetype="text/plain", headers=_cors_headers(req)
            )

        # mode bepalen
        mode_final = detect_mode(mode, question)

        # retrieval
        if mode_final == "INFO":
            docs = []
            context_text = ""
        else:
            queries = _expand_queries(question, mode_final, document_text)
            try:
                docs = hybrid_search_multi(queries, top_k=TOP_K)
            except Exception as e:
                logging.exception(f"Search error: {e}")
                docs = []
            context_text = _join_docs_for_context(
                docs,
                per_doc_chars=CTX_PER_DOC_CHARS,
                max_docs=CTX_MAX_DOCS,
                max_context_chars=CTX_MAX_TOTAL_CHARS,
            )

        # doc trimmen
        if mode_final in ("QUICKSCAN", "REVIEW"):
            document_text = _clip(document_text, DOC_CAP_QUICK_REVIEW)
        elif mode_final == "DRAFT":
            document_text = _clip(document_text, DOC_CAP_DRAFT)
        else:
            document_text = _clip(document_text, 6000)

        # conversation kort
        if isinstance(conversation, list) and len(conversation) > 8:
            conversation = conversation[-8:]
        chat_history_pf = format_history_for_prompt_flow(conversation)

        # prompt renderen
        system_text, user_text = render_system_and_user(
            mode=mode_final,
            document_text=document_text,
            context_text=context_text,
            chat_history_pf=chat_history_pf,
            chat_input=question or "Analyseer het bijgevoegde document."
        )

        # call model
        try:
            chat_output = call_chat(system_text, user_text, mode_hint=mode_final)
        except Exception as e:
            logging.exception(f"AI call error: {e}")
            return func.HttpResponse(f"AI-dienst error: {repr(e)}",
                                     status_code=502, mimetype="text/plain", headers=_cors_headers(req))

        return func.HttpResponse(
            body=json.dumps({"chat_output": chat_output}),
            status_code=200, mimetype="application/json", headers=_cors_headers(req)
        )

    except Exception as e:
        logging.error("UNHANDLED", exc_info=True)
        body_text = f"Unhandled server error: {repr(e)}\n{traceback.format_exc()}"
        return func.HttpResponse(body_text, status_code=500,
                                 mimetype="text/plain", headers=_cors_headers(req))
