# function_app.py — TenderMate HTTP endpoint (chat + retrieval, GPT-5 via Azure AI Foundry)
# - Multipart (PDF/DOCX) en JSON
# - Hybride search (BM25 + vector) met semantic rerank
# - GPT-5 safe: temperature=1.0 + max_completion_tokens + STREAMING (vangt non-text af)
# - Non-stream fallback
# - Iets conservatievere contextcaps om window issues te voorkomen

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

AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
AZURE_SEARCH_API_KEY  = os.environ.get("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_INDEX    = os.environ.get("AZURE_SEARCH_INDEX", "sharepoint-vectorizer-direct-v2")

TEXT_VECTOR_FIELD = "text_vector"
SEMANTIC_CONFIG   = os.environ.get("SEMANTIC_CONFIG", "sharepoint-vectorizer-semantic-configuration")
TOP_K             = int(os.environ.get("TOP_K", "5"))

AOAI_ENDPOINT         = os.environ.get("AOAI_ENDPOINT", "").rstrip("/")
AOAI_API_KEY          = os.environ.get("AOAI_API_KEY", "")
AOAI_CHAT_DEPLOYMENT  = os.environ.get("AOAI_CHAT_DEPLOYMENT", "gpt-5")
AOAI_EMBED_DEPLOYMENT = os.environ.get("AOAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

# Output-budgets (completion tokens)
AOAI_MAX_OUT_QUICKSCAN = int(os.environ.get("AOAI_MAX_OUT_QUICKSCAN", "2800"))
AOAI_MAX_OUT_DRAFT     = int(os.environ.get("AOAI_MAX_OUT_DRAFT", "2200"))
AOAI_MAX_OUT_REVIEW    = int(os.environ.get("AOAI_MAX_OUT_REVIEW", "1800"))
AOAI_MAX_OUT_INFO      = int(os.environ.get("AOAI_MAX_OUT_INFO", "900"))

# Document/context caps (karakters)
# Document/context caps (karakters) – iets krapper zodat gpt-5 ruimte houdt voor output
DOC_CAP_QUICK_REVIEW = int(os.environ.get("DOC_CAP_QUICK_REVIEW", "5500"))
DOC_CAP_DRAFT        = int(os.environ.get("DOC_CAP_DRAFT", "8000"))
CTX_PER_DOC_CHARS    = int(os.environ.get("CTX_PER_DOC_CHARS", "500"))
CTX_MAX_DOCS         = int(os.environ.get("CTX_MAX_DOCS", "3"))
CTX_MAX_TOTAL_CHARS  = int(os.environ.get("CTX_MAX_TOTAL_CHARS", "3000"))


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
        # Let op: deze api_version werkt met GPT-5 chat/completions
        _aoai_client = AzureOpenAI(
            api_key=AOAI_API_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint=AOAI_ENDPOINT
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


def _get_field(r, name, default=""):
    """Robuust veld uitlezen uit Azure Search result (obj/document/dict)."""
    try:
        # 1) property op object
        v = getattr(r, name, None)
        if v is not None:
            return v
        # 2) via onderliggende document dict
        if hasattr(r, "document") and isinstance(r.document, dict):
            v = r.document.get(name)
            if v is not None:
                return v
        # 3) als r dict-achtig is
        try:
            v = r.get(name, None)
            if v is not None:
                return v
        except Exception:
            pass
        return default
    except Exception:
        return default

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

    # unique
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
                "parent_id": _get_field(r, "parent_id", "")
            }
            if not hit["content"]:
                continue
            k = _dedupe_key(hit)
            if k not in all_hits:
                all_hits[k] = hit

    return list(all_hits.values())[: (top_k * 2)]

# =============================================================================
# Prompt rendering (met jouw volledige instructies, niets verwijderd)
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
            "Je bent TenderMate, de elite AI-tenderanalist en strategisch bid-adviseur van IT-Workz.\n"
            "Werk in STRICT_FACT_MODE:\n"
            "- Gebruik alleen informatie uit 'document_text' en 'context'.\n"
            "- Als iets ontbreekt: schrijf exact “Niet gespecificeerd in stukken.”\n"
            "- Geen webzoeken, geen aannames, geen externe feiten.\n\n"
            "Stijl:\n"
            "- Schrijf in helder Nederlands, compact maar volledig.\n"
            "- Gebruik duidelijke Markdown-koppen (##) en subtitels (###).\n"
            "- Gebruik ✅/❌ en 🟢/🔴 waar gevraagd.\n"
            "- Plaats tabellen waar dat de leesbaarheid verhoogt."
        )
        user = f"""
document_text: {document_text or ''}

Maak een strategische **quickscan (kwalificatie / go-no-go)** op basis van **CONTEXT** (IT-Workz kennisbank, eerdere aanbestedingen, referenties) en het **aangeleverde document**. 
Volg **deze koppen exact in deze volgorde** en houd je aan STRICT_FACT_MODE.

## Samenvatting in één oogopslag
- **Advies**: 🟢 Voorwaardelijke Go / 🟡 Twijfel / 🔴 No-Go — in één zin waarom.
- **Belangrijkste 3 showstoppers/risico’s** (kort, met mitigatie-hint).
- **Belangrijkste 3 scoringskansen** (kort).

## Feitenblad (tabel)
| Veld | Waarde |
|---|---|
| **Naam aanbesteding / opdrachtgever** | [waarde of Niet gespecificeerd in stukken] |
| **Soort aanbesteding/procedure** | [...] |
| **Sector/organisatie-type** | (bv. Gemeente / Onderwijssoort) |
| **Aantal locaties/scholen** | [...] |
| **Aantal medewerkers** | [...] |
| **Aantal leerlingen/studenten** | [...] |
| **# vragenronden** | [...] |
| **Vragen stellen – kanaal/portaal** | [...] |
| **Inkoopadviseur / contact** | [...] |
| **Contractduur (basis + opties)** | [...] |
| **Contractwaarde / richtprijs / prijsplafond** | [...] |

## Aanleiding en doel
Vat beknopt het “waarom” en de doelstellingen samen. Citeer 1–3 sleutelzinnen indien nuttig.

## Essentie van de uitvraag (scope)
- Kernomvang & doelgroep (specificeer sector/onderwijssoort).
- Koppel expliciet aan CONTEXT (Producten- en Diensten Catalogus, referenties).

## KO-criteria en voorwaarden (checklist)
- **Uitsluitingsgronden / geschiktheidseisen**: ✅/❌ per onderdeel.
- **Normen/certificeringen** (ISO/ISAE/NEN/ENSIA, AVG/DPIA): 🟢/🔴.
- **Juridische kaders**: GIBIT/ARBIT/ARVODI, AVG/DPA, DPIA, securitybeleid (TLS/HTTP headers).
- **Match met IT-Workz Catalogus**: matches/gaten t.o.v. CONTEXT.
- **Conclusie haalbaarheid (harde eisen)**: 1 alinea.

## Architectuur & Standaarden
- Common Ground / ZGW-/STUF(-ZKN) / OData — status.
- Microsoft-integraties (Azure AD/SSO/MFA, Graph, SharePoint/Teams, Power BI).
- Toegankelijkheid: **WCAG/EN 301 549**; expliciet “Niet gespecificeerd in stukken” indien onbekend.

## Programma van Eisen/Wensen (samengevat per cluster)
- **Projectorganisatie & implementatie**
- **Techniek/architectuur & integraties**
- **Privacy/AVG & security (TLS/headers)**
- **Beheer/Support/SLA**
- **Adoptie/Training**
- **Planning/Migratie**

## Analyse van Vereiste Disciplines
[ ] Lijst kernrollen o.b.v. CONTEXT.  
[ ] Voorlopig bid-team o.b.v. match met CONTEXT.

## Weging, paginabudget en planning
- **Weging**: percentages indien aanwezig.
- **Paginabudget**: limiet + verdelingsvoorstel.
- **Planning & deadlines**: alle mijlpalen (vragenronde, indiening, demo/PoC, gunning, start).

## Budget en concurrentiepositie
- **Budgetsignalen** (plafond/raming/prijsmechaniek).
- **Concurrenten (indicatief)** en **positie IT-Workz** o.b.v. CONTEXT.

## Showstoppers / risico’s (met mitigatie of verhelderingsvraag)
- Lijst met concrete eisen/risico’s + korte mitigatie/te-stellen vraag.

## Concurrentie-analyse
- (Indien via CONTEXT) huidige dienstverlener/bekende concurrenten; sterke/zwakke punten.

## Referenties (advies)
- 2–3 **typen** referenties (branche/omvang/complexiteit) + “waarom dit scoort”.

## In te dienen bewijsstukken & **status**
- **UEA, KvK, ISO/ISAE, verzekeringen, DPA/Verwerkers, prijzenblad, referentieverklaring**.
- Status: **🟢 Geldig / 🔴 Verlopen/ontbreekt / Niet gespecificeerd in stukken**.

## Standaard- en verdiepingsvragen
- Standaardvragen (scope/eisen/voorwaarden/planning).
- Verdiepingsvragen (strategisch).

## Actiechecklist (intern)
- (Taak; **Eigenaar**; **Datum**) — 6–10 concrete acties.

## Go/No-Go indicatie (conclusie)
- **🟢/🟡/🔴** + 4–6 onderbouwende bullets.
""".strip() + "\n\n" + common_suffix

    elif mode == "REVIEW":
        system = (
            "Je bent TenderMate, AI-kwaliteitsauditor en scoringscoach voor aanbestedingen van IT-Workz.\n"
            "Werk in STRICT_FACT_MODE:\n"
            "- Beoordeel uitsluitend t.o.v. 'document_text', gunningskader en CONTEXT.\n"
            "- Geen aannames; ontbrekende info = “Niet gespecificeerd in stukken.”\n"
            "Stijl: concreet en actiegericht; tabellen en voorbeeldherschrijvingen toegestaan."
        )
        user = f"""
document_text: {document_text or ''}

OPDRACHT
Beoordeel en **versterk** de tekst t.o.v. gunningscriterium en beoordelingskader. 
Houd je aan STRICT_FACT_MODE en volg **deze koppen exact**:

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
        system = "Je bent TenderMate, een AI-assistent voor aanbestedingen. Antwoord kort, duidelijk en praktisch."
        user = f"""
Geef in bullets:
- Wat jij kunt (QUICKSCAN / DRAFT / REVIEW) en wanneer welke modus.
- Hoe je CONTEXT gebruikt (SharePoint → Azure AI Search → relevante fragmenten).
- Hoe ik documenten kan aanleveren (PDF/DOCX upload).
- Tips om betere antwoorden te krijgen.
""".strip() + "\n\n" + common_suffix

    else:  # DRAFT
        system = (
            "Je bent TenderMate, AI-tekstarchitect voor aanbestedingen van IT-Workz.\n"
            "STRICT_FACT_MODE: alleen 'document_text' en 'context'. Ontbrekend = “Niet gespecificeerd in stukken.”"
        )
        user = f"""
document_text: {document_text or ''}

OPDRACHT
Genereer een **scoregericht concept** conform gunningscriterium. 
Volg **deze koppen exact** (fact-only):

## Executive summary (scorefocus in 5 bullets)
## Begrips- & beoordelingskader (facts)
## Voorstelstructuur conform criterium
## Uitwerking per subcriterium (aanpak, architectuur, KPI’s, risico’s, bewijs)
## Planning & mijlpalen
## Rollen & RACI (compact)
## KPI-overzicht (tabel)
## Paginabudget & opmaak
## Referenties & bewijs (gericht)
## Verhelderingsvragen
## Bronnen/quotes (compact)
""".strip() + "\n\n" + common_suffix

    return system, user

# =============================================================================
# Chat call (GPT-5: streaming + max_completion_tokens + text-only)
# =============================================================================

def call_chat(system_text: str, user_text: str, mode_hint: str = "") -> str:
    """
    GPT-5-safe call voor Azure:
    - geen streaming (want jouw deployment stuurt dan niks)
    - geen 'modalities'
    - eerst met response_format={"type":"text"}, bij 400 meteen zonder
    """
    client = get_aoai_client()
    is_gpt5 = AOAI_CHAT_DEPLOYMENT.lower().startswith("gpt-5")

    # 1) bepaal outputbudget per modus
    mode = (mode_hint or "").upper()
    if mode == "QUICKSCAN":
        max_out = AOAI_MAX_OUT_QUICKSCAN
    elif mode == "DRAFT":
        max_out = AOAI_MAX_OUT_DRAFT
    elif mode == "REVIEW":
        max_out = AOAI_MAX_OUT_REVIEW
    else:
        max_out = AOAI_MAX_OUT_INFO

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]

    # ====== helper om tekst uit completion te vissen ======
    def _extract_text(resp) -> str:
        parts = []
        try:
            for choice in resp.choices:
                msg = getattr(choice, "message", None)
                if not msg:
                    continue
                c = getattr(msg, "content", None)
                # vorm 1: plain string
                if isinstance(c, str) and c.strip():
                    parts.append(c.strip())
                    continue
                # vorm 2: list-of-parts (gpt-5)
                if isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") in ("text", "output_text"):
                            t = p.get("text")
                            if isinstance(t, str) and t.strip():
                                parts.append(t.strip())
        except Exception as e:
            logging.exception(f"extract_text error: {e}")
        return "\n".join(parts).strip()

    # ====== GPT-5 pad: non-stream ======
    if is_gpt5:
        base_kwargs = {
            "model": AOAI_CHAT_DEPLOYMENT,
            "messages": messages,
            "tool_choice": "none",
            "max_completion_tokens": max_out,
            "temperature": 1.0,
        }

        # ---- poging 1: mét response_format ----
        try_kwargs = dict(base_kwargs)
        try_kwargs["response_format"] = {"type": "text"}
        try:
            resp = client.chat.completions.create(**try_kwargs)
            text = _extract_text(resp)
            if text:
                return text
            logging.error(f"GPT-5 gaf lege text terug (met response_format). Raw: {resp}")
        except Exception as e:
            # check of het zo'n 400 unknown_parameter is
            msg = str(e)
            logging.warning(f"GPT-5 eerste call faalde, probeer zonder response_format. Error: {msg}")
            # we vallen hieronder automatisch terug

        # ---- poging 2: ZONDER response_format ----
        try:
            resp = client.chat.completions.create(**base_kwargs)
            text = _extract_text(resp)
            if text:
                return text
            logging.error(f"GPT-5 gaf lege text terug (zonder response_format). Raw: {resp}")
        except Exception as e:
            logging.exception(f"GPT-5 call (zonder response_format) error: {e}")
            return f"AI-dienst error: {e}"

        return "Er kwam geen leesbare tekst terug van het AI-model. Probeer het nogmaals of verlaag de omvang (minder context of kortere vraag)."

    # ====== NIET-gpt-5 pad (mag streamen) ======
    def _collect_text_from_delta(delta) -> str:
        out = []
        c = getattr(delta, "content", None)
        if isinstance(c, str) and c:
            out.append(c)
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("text", "output_text"):
                    t = p.get("text")
                    if isinstance(t, str) and t:
                        out.append(t)
        return "".join(out)

    # stream eerst
    try:
        stream = client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=messages,
            stream=True,
            max_tokens=max_out,
            temperature=0.7,
            tool_choice="none",
            response_format={"type": "text"},
        )
        chunks: List[str] = []
        for event in stream:
            try:
                delta = event.choices[0].delta
            except Exception:
                continue
            piece = _collect_text_from_delta(delta)
            if piece:
                chunks.append(piece)
        txt = "".join(chunks).strip()
        if txt:
            return txt
    except Exception as e:
        logging.exception(f"stream (niet-gpt5) error: {e}")

    # non-stream fallback
    try:
        resp = client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=messages,
            max_tokens=max_out,
            temperature=0.7,
            tool_choice="none",
            response_format={"type": "text"},
        )
        text = _extract_text(resp)
        if text:
            return text
    except Exception as e:
        logging.exception(f"non-stream (niet-gpt5) error: {e}")

    return "Er kwam geen leesbare tekst terug van het AI-model. Probeer het nogmaals of verlaag de omvang (minder context of kortere vraag)."


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
            return func.HttpResponse("OK - TenderMate TalkToTenderBot - vA.51", status_code=200,
                                     mimetype="text/plain", headers=_cors_headers(req))

        # Basic intake logging
        try:
            ct = (req.headers.get("Content-Type") or "").lower()
            cl = req.headers.get("Content-Length") or "?"
            raw_len = len(req.get_body() or b"")
            logging.info(f"req.method={req.method}, ct={ct}, len={cl}, raw_len={raw_len}")
        except Exception:
            pass

        question = ""
        conversation = []
        document_text = ""
        mode = ""
        content_type = (req.headers.get("Content-Type") or "").lower()

        # ---------------- Multipart (PDF/DOCX upload) ----------------
        if "multipart/form-data" in content_type:
            body_bytes = req.get_body() or b""
            try:
                dec = MultipartDecoder(body_bytes, content_type)
            except Exception:
                return func.HttpResponse("Ongeldig multipart-verzoek (decode fout).",
                                         status_code=400, mimetype="text/plain", headers=_cors_headers(req))

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
                        return func.HttpResponse("Bestandstype niet ondersteund. Upload een PDF of DOCX.",
                                                 status_code=415, mimetype="text/plain", headers=_cors_headers(req))
                except zipfile.BadZipFile:
                    sig = _hex_prefix(file_bytes)
                    return func.HttpResponse(
                        f"Kon DOCX niet lezen: bestand lijkt geen geldig DOCX/ZIP (magic={sig}). "
                        "Controleer IRM/wachtwoord-beveiliging of sla opnieuw op als .docx.",
                        status_code=422, mimetype="text/plain", headers=_cors_headers(req))
                except PackageNotFoundError:
                    return func.HttpResponse("Kon DOCX niet openen (geen geldig Office-package). Sla opnieuw op als .docx en probeer opnieuw.",
                                             status_code=422, mimetype="text/plain", headers=_cors_headers(req))
                except Exception as e:
                    return func.HttpResponse(f"Fout bij lezen van document: {repr(e)}",
                                             status_code=422, mimetype="text/plain", headers=_cors_headers(req))

        # ---------------- JSON ----------------
        elif "application/json" in content_type:
            body = {}
            try:
                body = req.get_json()
            except ValueError:
                raw = req.get_body()
                if raw:
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
                f"Content-Type '{content_type}' niet ondersteund (verwacht JSON of multipart/form-data).",
                status_code=415, mimetype="text/plain", headers=_cors_headers(req)
            )

        # ---- Mode bepalen ----
        mode_final = detect_mode(mode, question)

        # ---- Retrieval (geen context voor INFO) ----
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
                max_context_chars=CTX_MAX_TOTAL_CHARS
            )

        # ---- Document trimming per mode ----
        if mode_final in ("QUICKSCAN", "REVIEW"):
            document_text = _clip(document_text, DOC_CAP_QUICK_REVIEW)
        elif mode_final == "DRAFT":
            document_text = _clip(document_text, DOC_CAP_DRAFT)
        else:
            document_text = _clip(document_text, 6000)

        # Kort chatverleden
        if isinstance(conversation, list) and len(conversation) > 8:
            conversation = conversation[-8:]

        chat_history_pf = format_history_for_prompt_flow(conversation)

        # Log promptlengtes voor diagnose
        try:
            logging.info(f"lens: system={len(mode_final)}, doc={len(document_text)}, ctx={len(context_text)}, hist={len(json.dumps(chat_history_pf))}, q={len(question)}")
        except Exception:
            pass

        # ---- Render prompts ----
        system_text, user_text = render_system_and_user(
            mode=mode_final,
            document_text=document_text,
            context_text=context_text,
            chat_history_pf=chat_history_pf,
            chat_input=question or "Analyseer het bijgevoegde document."
        )

        # ---- Call AOAI ----
        try:
            chat_output = call_chat(system_text, user_text, mode_hint=mode_final)
        except Exception as e:
            logging.exception(f"AOAI chat error: {e}")
            return func.HttpResponse(f"AI-dienst error: {repr(e)}",
                                     status_code=502, mimetype="text/plain", headers=_cors_headers(req))

        return func.HttpResponse(
            body=json.dumps({"chat_output": chat_output}),
            status_code=200, mimetype="application/json", headers=_cors_headers(req)
        )

    except Exception as e:
        logging.error("UNHANDLED", exc_info=True)
        body_text = f"Unhandled server error: {repr(e)}\n{traceback.format_exc()}"
        return func.HttpResponse(body_text, status_code=500, mimetype="text/plain", headers=_cors_headers(req))
