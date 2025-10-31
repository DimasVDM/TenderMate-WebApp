# function_app.py — TenderMate HTTP endpoint (chat + retrieval) met Azure AI Foundry gpt-4o
# - JSON + multipart (PDF/DOCX)
# - Hybride search (BM25 + vector) met semantic config
# - Met vaste "gemeentelijke CRM tender kit" in QUICKSCAN zodat output nooit kaal is

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
AZURE_SEARCH_INDEX    = os.environ.get("AZURE_SEARCH_INDEX", "sharepoint-vectorizer-direct")

TEXT_VECTOR_FIELD = "text_vector"
SEMANTIC_CONFIG   = os.environ.get("SEMANTIC_CONFIG", "sharepoint-vectorizer-semantic-configuration")
TOP_K             = int(os.environ.get("TOP_K", "12"))

# Azure AI Foundry
AOAI_ENDPOINT         = os.environ.get("AOAI_ENDPOINT", "").rstrip("/")
AOAI_API_KEY          = os.environ.get("AOAI_API_KEY", "")
AOAI_CHAT_DEPLOYMENT  = os.environ.get("AOAI_CHAT_DEPLOYMENT", "gpt-5")
AOAI_EMBED_DEPLOYMENT = os.environ.get("AOAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

# Output-budgets — iets ruimer
AOAI_MAX_OUT_QUICKSCAN = int(os.environ.get("AOAI_MAX_OUT_QUICKSCAN", "2800"))
AOAI_MAX_OUT_DRAFT     = int(os.environ.get("AOAI_MAX_OUT_DRAFT", "2000"))
AOAI_MAX_OUT_REVIEW    = int(os.environ.get("AOAI_MAX_OUT_REVIEW", "1600"))
AOAI_MAX_OUT_INFO      = int(os.environ.get("AOAI_MAX_OUT_INFO", "800"))

# Context caps
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

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

_search_client: SearchClient = None
_aoai_client: AzureOpenAI    = None

# =============================================================================
# Helpers
# =============================================================================

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

    # vaste "overheid/CRM tender kit"
    MUNICIPAL_CRM_KIT = """
BELANGRIJK: Dit betreft een typische Nederlandse gemeentelijke CRM-aanbesteding. Neem de volgende vaste kaders altijd op als ze niet expliciet ontkracht worden:
- Juridisch/inkoop: Aanbestedingswet 2012, Europees openbaar, gebruik van UEA, referentieverklaringen, conceptovereenkomst, (eventueel) wachtkamerovereenkomst.
- Security: GIBIT 2023, "Gebruik van TLS en HTTP response headers" (vaak v1.7), Beleid Informatieveiligheid, Verwerkersovereenkomst (AVG), DPIA mogelijk.
- Architectuur/standaarden: Common Ground, ZGW-/STUF(-ZKN) voor koppelingen, OData/CSV-API’s voor uitwisseling, koppelingen met webformulieren, Power BI-rapportages.
- Toegankelijkheid: WCAG 2.1 / EN 301 549, voorleesfunctie, meertaligheid, B1-niveau.
- Leveringsmodel: voorkeur voor generiek/standaard boven maatwerk, SaaS/on-prem/IaaS/PaaS varianten komen voor.
- Beoordeling: vaak gunningscriterium prijs-kwaliteit, (mogelijk) PoC/demo, referentiemoment.
Als deze onderdelen niet letterlijk in de context staan, noteer dan: “Niet gespecificeerd in stukken.” maar laat het kopje wél staan.
""".strip()

    common_suffix = f"""
context (uit Azure Search):
{context_text}

chat history:
{history_block}

user:
{chat_input}
""".strip()

    if mode == "QUICKSCAN":
        system = (
            "Je bent TenderMate, AI-tenderanalist voor IT-Workz. "
            "STRICT_FACT_MODE: gebruik document_text, context én de bekende gemeentelijke CRM-kaders. "
            "Waar info ontbreekt: schrijf exact “Niet gespecificeerd in stukken.” "
            "Schrijf in helder Nederlands, met duidelijke koppen, tabellen waar gevraagd."
        )

        user = f"""
document_text: {document_text or ''}

{MUNICIPAL_CRM_KIT}

Maak een strategische **quickscan (kwalificatie / go-no-go)** op basis van CONTEXT, het document en de hierboven beschreven gemeentelijke kaders. 
Volg **deze koppen exact in deze volgorde**:

## Samenvatting in één oogopslag
- **Advies**: 🟢 Voorwaardelijke Go / 🟡 Twijfel / 🔴 No-Go — met 1 zin waarom.
- **Belangrijkste 3 showstoppers/risico’s** (noem GIBIT/security, referenties, prijsplafond indien niet gespecificeerd).
- **Belangrijkste 3 scoringskansen** (noem Common Ground, toegankelijkheid, PoC/demo).

## Feitenblad (tabel)
| Veld | Waarde |
|---|---|
| **Naam aanbesteding / opdrachtgever** | ... |
| **Soort aanbesteding/procedure** | Europees openbaar (Aw2012) **of** “Niet gespecificeerd in stukken.” |
| **Sector/organisatie-type** | Gemeente / decentrale overheid |
| **Aantal locaties/scholen** | Niet gespecificeerd in stukken |
| **Aantal medewerkers** | Niet gespecificeerd in stukken **(gemeente ≈ 1.500–2.000 → alleen noemen als in context!)** |
| **# vragenronden** | Niet gespecificeerd in stukken |
| **Vragen stellen – kanaal/portaal** | Niet gespecificeerd in stukken |
| **Inkoopadviseur / contact** | Niet gespecificeerd in stukken |
| **Contractduur (basis + opties)** | Niet gespecificeerd in stukken |
| **Contractwaarde / richtprijs / prijsplafond** | Niet gespecificeerd in stukken |

## Aanleiding en doel
- Beschrijf waarom de gemeente een CRM-oplossing wil.
- Neem sleutelzinnen over organisatie/omvang op ALS die in context/document staan.
- Als de aanleiding ontbreekt: schrijf “Niet gespecificeerd in stukken.”

## Essentie van de uitvraag (scope)
- Levering + implementatie + beheer/onderhoud CRM.
- Koppelingen (ZGW/STUF(-ZKN), OData/CSV, Power BI, webformulieren).
- Toekomstige uitbreidbaarheid binnen concern-informatie-architectuur.
- Zet erbij als iets niet in stukken staat.

## KO-criteria en voorwaarden (checklist)
- UEA / uitsluitingsgronden / geschiktheidseisen → ✅/❌/Niet gespecificeerd.
- Referenties (gemeente, vergelijkbare omvang) → benoem risico als onbekend.
- Juridische kaders: GIBIT 2023, Verwerkersovereenkomst (AVG), TLS/HTTP headers v1.7, Beleid Informatieveiligheid.
- Conclusie haalbaarheid (harde eisen): 1 alinea.

## Architectuur & Standaarden
- Common Ground (nu/toekomst)
- ZGW-/STUF(-ZKN) voor formulierenstromen
- OData/CSV-API’s
- Microsoft-integraties (Teams/SharePoint/Power BI) als in context.
- Toegankelijkheid (WCAG/EN 301 549, B1, voorleesfunctie)
Zet overal “Niet gespecificeerd in stukken” als het niet is gevonden.

## Programma van Eisen/Wensen (samengevat per cluster)
- Projectorganisatie & implementatie (realistische doorlooptijd, fasering)
- Techniek/architectuur & integraties
- Privacy/AVG & security (GIBIT, TLS, VO, DPIA)
- Beheer/Support/SLA
- Adoptie/Training
- Planning/Migratie
Noem opvallende punten en hiaten.

## Analyse van Vereiste Disciplines
- Noem rollen (projectleider, solution/CRM architect, integratiespecialist, security/privacy, adoptie/change, BI/rapportage).
- Koppel kort aan CONTEXT.

## Weging, paginabudget en planning
- Weging: “Niet gespecificeerd in stukken” OF benoem wat er wel staat.
- Paginabudget: idem.
- Planning & deadlines: publicatie, vragenronde, PoC/demo, gunning, start.

## Budget en concurrentiepositie
- Noem alle prijs-/plafond-/ramingssignalen uit context.
- Als er niets staat: “Niet gespecificeerd in stukken.”
- Benoem verwachte concurrenten (type leveranciers) o.b.v. CONTEXT.

## Showstoppers / risico’s (met mitigatie of verhelderingsvraag)
- **GIBIT / security / TLS / VO**
- **Referenties op gemeentelijk CRM-niveau**
- **Common Ground / ZGW/STUF / OData**
- **Toegankelijkheid (EN 301 549, WCAG 2.1, B1, voorleesfunctie)**
Formuleer per punt 1 verhelderingsvraag.

## Concurrentie-analyse
- Benoem type partijen (grote gemeentelijke CRM-leveranciers, Microsoft-dienstverleners, low-code CRM’s met overheidspakket).
- Zet erachter: “Exacte huidige leverancier: Niet gespecificeerd in stukken.”

## Referenties (advies)
- Gemeentelijke CRM-implementatie (≈ vergelijkbare omvang).
- CRM met Common Ground + ZGW/STUF + OData/CSV.
- CRM met aantoonbare toegankelijkheid (EN 301 549/WCAG 2.1).

## In te dienen bewijsstukken & status
- UEA, KvK, ISO/ISAE/NEN/ENSIA, verzekeringen, DPA/VO, GIBIT 2023, verwerkersovereenkomst, prijzenblad, referentieverklaring, verklaring geen Russische betrokkenheid.
- Zet per stuk: 🟢 aanwezig / 🔴 ontbreekt / “Niet gespecificeerd in stukken”.

## Standaard- en verdiepingsvragen
- Vragen over KO, gunningsweging, PoC/demo, integraties, toegankelijkheid, migratie, security-afwijkingen.

## Actiechecklist (intern)
- 6–10 concrete acties met eigenaar en datum.

## Go/No-Go indicatie (conclusie)
- 🟢 / 🟡 / 🔴 + 4–6 onderbouwende bullets.

{common_suffix}
""".strip()

    elif mode == "REVIEW":
        system = (
            "Je bent TenderMate, AI-kwaliteitsauditor voor aanbestedingen. "
            "Beoordeel op volledigheid en scoringspotentieel. Ontbrekend = “Niet gespecificeerd in stukken.”"
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

{common_suffix}
""".strip()

    elif mode == "INFO":
        system = "Je bent TenderMate, een AI-assistent voor aanbestedingen. Antwoord kort en praktisch."
        user = f"""
Geef in bullets:
- Wat jij kunt (QUICKSCAN / DRAFT / REVIEW)
- Hoe je CONTEXT gebruikt
- Hoe ik een document aanlever
- Hoe ik betere antwoorden krijg

{common_suffix}
""".strip()

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

{common_suffix}
""".strip()

    return system, user

# =============================================================================
# Chat call — gpt-4o friendly
# =============================================================================

def call_chat(system_text: str, user_text: str, mode_hint: str = "") -> str:
    client = get_aoai_client()

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

    # 1) stream
    try:
        stream = client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=messages,
            stream=True,
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
        logging.warning(f"stream failed, fallback to non-stream: {e}")

    # 2) non-stream
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
        logging.warning(f"non-stream failed, plain fallback: {e}")

    # 3) plain
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
        if req.method == "OPTIONS":
            return func.HttpResponse(status_code=204, headers=_cors_headers(req))

        if req.method == "GET":
            return func.HttpResponse("OK - TenderMate TalkToTenderBot - gpt-5", status_code=200,
                                     mimetype="text/plain", headers=_cors_headers(req))

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

        # doc trim
        if mode_final in ("QUICKSCAN", "REVIEW"):
            document_text = _clip(document_text, DOC_CAP_QUICK_REVIEW)
        elif mode_final == "DRAFT":
            document_text = _clip(document_text, DOC_CAP_DRAFT)
        else:
            document_text = _clip(document_text, 6000)

        # convo kort
        if isinstance(conversation, list) and len(conversation) > 8:
            conversation = conversation[-8:]
        chat_history_pf = format_history_for_prompt_flow(conversation)

        # prompt
        system_text, user_text = render_system_and_user(
            mode=mode_final,
            document_text=document_text,
            context_text=context_text,
            chat_history_pf=chat_history_pf,
            chat_input=question or "Analyseer het bijgevoegde document."
        )

        # call
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
