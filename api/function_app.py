# function_app.py — Serverless PF-vervanger (hybride search + AOAI chat)
# - Ondersteunt PDF/DOCX uploads
# - PF-achtige retrieval: multi-query expansie + hybride search + semantic rerank
# - Semantic config: sharepoint-vectorizer-semantic-configuration
# - Modes: QUICKSCAN / DRAFT / REVIEW (met fallback op vraag)

import azure.functions as func
import logging
import os
import json
import io
import re
import zipfile
import traceback
import binascii
import time
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

# ------------------ ENV / Config ------------------

AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_INDEX   = os.environ.get("AZURE_SEARCH_INDEX", "sharepoint-vectorizer-direct")

# Vector veld & semantic config zoals in jouw index
TEXT_VECTOR_FIELD = "text_vector"
SEMANTIC_CONFIG   = "sharepoint-vectorizer-semantic-configuration"
TOP_K             = int(os.environ.get("TOP_K", "6"))

# Azure OpenAI
AOAI_ENDPOINT          = os.environ.get("AOAI_ENDPOINT", "").rstrip("/")
AOAI_API_KEY           = os.environ.get("AOAI_API_KEY", "")
AOAI_CHAT_DEPLOYMENT   = os.environ.get("AOAI_CHAT_DEPLOYMENT", "gpt-5")  # pas aan naar jouw deploymentnaam
AOAI_EMBED_DEPLOYMENT  = os.environ.get("AOAI_EMBED_DEPLOYMENT", "text-embedding-3-large")  # 3072-dim

# Contextlimieten
MAX_CONTEXT_DOCS   = int(os.environ.get("MAX_CONTEXT_DOCS", "10"))

# ---------------------------------------------------

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Lazy clients
_search_client = None
_aoai_client   = None

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
            azure_endpoint=AOAI_ENDPOINT
        )
    return _aoai_client

# --- Simple length guards (char-based; good enough without a tokenizer) ---
def _clip(txt: str, max_chars: int) -> str:
    if not txt:
        return ""
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars]

def _dedupe_key(hit: Dict[str, Any]) -> str:
    pid = (hit.get("parent_id") or "").strip()
    src = (hit.get("source") or "").strip()
    head = (hit.get("content") or "")[:80].strip()
    return f"{pid}|{src}|{head}"

def _join_docs_for_context(docs, per_doc_chars=1600, max_docs=10, max_context_chars=16000) -> str:
    blocks = []
    total = 0
    for d in docs[:max_docs]:
        src = d.get("source") or ""
        pid = d.get("parent_id") or ""
        label = src if not pid else f"{src} (parent: {pid})"
        content = _clip((d.get("content","") or "").strip(), per_doc_chars)
        if not content:
            continue
        block = f"Source: {label}\nContent:\n{content}"
        if total + len(block) > max_context_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)

# ---------------- Helpers: conversation & parsing ----------------

def format_history_for_prompt_flow(conversation: List[Dict[str, str]]) -> List[Dict[str, Dict[str, str]]]:
    """
    Conversatie in PF-achtig formaat (zodat we bestaande formatting kunnen hergebruiken).
    """
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

def _hex_prefix(b: bytes, n: int = 16) -> str:
    return binascii.hexlify(b[:n]).decode().upper()

def _is_probably_docx(file_bytes: bytes) -> bool:
    return zipfile.is_zipfile(io.BytesIO(file_bytes))

def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for p in reader.pages:
        chunks.append(p.extract_text() or "")
    return "\n".join(chunks)

def _read_docx_text(file_bytes: bytes) -> str:
    if not _is_probably_docx(file_bytes):
        sig = _hex_prefix(file_bytes, 8)
        raise zipfile.BadZipFile(f"Not a DOCX/ZIP (magic={sig})")
    d = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in d.paragraphs])

# ---------------- Embeddings & Search ----------------

def embed_text(text: str) -> List[float]:
    client = get_aoai_client()
    resp = client.embeddings.create(
        model=AOAI_EMBED_DEPLOYMENT,
        input=text
    )
    return resp.data[0].embedding

def _expand_queries(user_q: str, mode: str, doc_hint: str = "") -> List[str]:
    """
    PF-achtige query-expansie: voeg high-yield termen toe zodat bedragen, looptijden,
    KO’s en standaarden sneller gevonden worden.
    """
    base = (user_q or "").strip()
    mode = (mode or "").upper()
    hint = (doc_hint or "")[:300]

    keywords_common = [
        "(prijs OR prijsplafond OR budget OR 'geraamde waarde' OR kosten)",
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
    else:  # DRAFT
        mode_bonus = ["(structuur OR hoofdstukindeling OR 'paginabudget verdeling')"]

    queries = [base] if base else []
    core = " ".join(keywords_common + mode_bonus)
    queries.append(f"{base} {core}".strip())
    queries.append(f"{base} prijsplafond OR 'geraamde waarde' OR contractduur OR 2026 OR 2027 OR 2028")
    if hint:
        queries.append(f"{base} {hint}")

    seen, out = set(), []
    for q in queries:
        qn = (q or "").strip()
        if qn and qn not in seen:
            out.append(qn)
            seen.add(qn)
    return out

def hybrid_search_multi(queries: List[str], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Voert hybride search uit voor meerdere query-varianten en merge't de resultaten.
    Emuleert PF's 'modify_query_with_history' + 'lookup(top_k)' gedrag.
    """
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
                "content":   r.get("chunk", "") or "",
                "source":    r.get("title", "") or "",
                "parent_id": r.get("parent_id", "") or ""
            }
            if not hit["content"]:
                continue
            k = _dedupe_key(hit)
            if k not in all_hits:
                all_hits[k] = hit

    merged = list(all_hits.values())
    # cap op ~2*top_k zodat context niet explodeert
    return merged[: (top_k * 2)]

def build_context(docs: List[Dict[str, Any]]) -> str:
    """
    (Niet meer gebruikt in de handler; behouden voor compat.)
    """
    blocks = []
    for d in docs:
        src = d.get("source") or ""
        pid = d.get("parent_id") or ""
        label = src if not pid else f"{src} (parent: {pid})"
        blocks.append(f"Content:\n{d.get('content','')}\nSource: {label}")
    return "\n\n".join(blocks)

# ---------------- Prompt rendering ----------------

def detect_mode(explicit_mode: str, chat_input: str) -> str:
    """
    Neem expliciete mode over; anders heuristisch uit vraag.
    """
    m = (explicit_mode or "").strip().upper()
    if m in ("QUICKSCAN", "REVIEW", "DRAFT"):
        return m
    txt = (chat_input or "").lower()
    if "quickscan" in txt or "go/no-go" in txt or "go no go" in txt:
        return "QUICKSCAN"
    if "review" in txt or "beoordeel" in txt or "verbeter" in txt:
        return "REVIEW"
    return "DRAFT"

def render_system_and_user(mode: str, document_text: str, context_text: str, chat_history_pf: List[Dict[str, Any]], chat_input: str):
    """
    Render system+user rollen voor Chat API op basis van gevraagde mode.
    """
    # Chat history in plain tekst (zoals je PF deed)
    hist_lines = []
    for item in chat_history_pf:
        hist_lines.append("user:\n" + (item["inputs"].get("chat_input") or ""))
        hist_lines.append("assistant:\n" + (item["outputs"].get("chat_output") or ""))
    history_block = "\n".join(hist_lines)

    # Gemeenschappelijk blok
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
            "- Je mag korte letterlijk geciteerde fragmenten opnemen met aanhalingstekens.\n"
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
- **Uitsluitingsgronden / geschiktheidseisen**: ✅/❌ per onderdeel (referentiesector/omvang/complexiteit; technische/beroepsbekwaamheid).
- **Normen/certificeringen** (ISO/ISAE/NEN/ENSIA, AVG/DPIA): 🟢 Aanwezig / 🔴 Ontbreekt.
- **Juridische kaders**: benoem **GIBIT/ARBIT/ARVODI**, AVG/Verwerkersovereenkomst, DPIA, securitybeleid (TLS/HTTP headers).
- **Match met IT-Workz Catalogus**: Benoem matches/gaten t.o.v. CONTEXT.
- **Conclusie haalbaarheid (harde eisen)**: 1 alinea.

## Architectuur & Standaarden (overheidsspecifiek waar relevant)
- **Common Ground / ZGW-/STUF(-ZKN) / OData**: status & relevantie.
- **Integraties Microsoft-stapel**: Azure AD (SSO/MFA), Graph API, SharePoint/Teams, Power BI.
- **Toegankelijkheid**: **WCAG/EN 301 549**; noem expliciet als dit niet gespecificeerd is.

## Programma van Eisen/Wensen (samengevat per cluster)
- **Projectorganisatie & implementatie**
- **Techniek/architectuur & integraties**
- **Privacy/AVG & security (TLS/headers)**
- **Beheer/Support/SLA**
- **Adoptie/Training**
- **Planning/Migratie**
Noteer opvallende punten en hiaten. Gebruik bullets.

## Analyse van Vereiste Disciplines
[ ] Lijst de belangrijkste uitvoeringsrollen o.b.v. CONTEXT (alleen relevante intern bekende rollen).
[ ] Stel een voorlopig bid-team samen o.b.v. match met CONTEXT.

## Weging, paginabudget en planning
- **Weging**: noteer percentages indien aanwezig; anders “Niet gespecificeerd in stukken”.
- **Paginabudget**: noteer limiet. **Voorstel verdeling** per onderdeel (o.b.v. weging).
- **Planning & deadlines**: alle mijlpalen (vragenronde, indiening, demo/PoC, gunning, start).

## Budget en concurrentiepositie
- **Budgetsignalen** (plafond/raming/prijsmechaniek).
- **Concurrenten (indicatief)** en **positie IT-Workz** o.b.v. CONTEXT (sterktes/zwaktes kort).

## Showstoppers / risico’s (met mitigatie of verhelderingsvraag)
- Korte lijst met concrete eisen/risico’s + korte mitigatie/te-stellen vraag.

## Concurrentie-analyse
- Benoem (indien mogelijk via CONTEXT) huidige dienstverlener/bekende concurrenten; sterke/zwakke punten.

## Referenties (advies)
- Adviseer 2–3 **typen** referenties (branche/omvang/complexiteit) en **waarom** deze scoren. Koppel aan CONTEXT.

## In te dienen bewijsstukken & **status**
- **UEA, KvK, ISO/ISAE, verzekeringen, DPA/Verwerkersovereenkomst, prijzenblad, referentieverklaring**.
- Zet per stuk: **🟢 Geldig / 🔴 Verlopen/ontbreekt / Niet gespecificeerd in stukken**.
- Voeg vervaldata toe als die in de CONTEXT staan.

## Standaard- en verdiepingsvragen
- **Standaardvragen**: verduidelijk scope/eisen/voorwaarden/planning.
- **Verdiepingsvragen**: strategische “vraag achter de vraag”.

## Actiechecklist (intern)
- (Taak; **Eigenaar**; **Datum**) — 6–10 concrete acties die het bidproces starten.

## Go/No-Go indicatie (conclusie)
- **🟢/🟡/🔴** + 4–6 onderbouwende bullets (KO, referenties, normen, concurrentie, fit met Catalogus).
""".strip() + "\n\n" + common_suffix

    elif mode == "REVIEW":
        system = (
            "Je bent TenderMate, AI-kwaliteitsauditor en scoringscoach voor aanbestedingen van IT-Workz.\n"
            "Werk in STRICT_FACT_MODE:\n"
            "- Beoordeel uitsluitend t.o.v. het aangeleverde 'document_text', het gunningskader en de CONTEXT.\n"
            "- Geen aannames; ontbrekende info = “Niet gespecificeerd in stukken.”\n\n"
            "Stijl:\n"
            "- Wees concreet, actiegericht, en scoregericht.\n"
            "- Gebruik checklijsten, tabellen en voorbeeldherschrijvingen.\n"
            "- Lever geen generieke adviezen: maak ze toetsbaar (SMART, dekking, bewijs)."
        )

        user = f"""
document_text: {document_text or ''}

OPDRACHT
Beoordeel en **versterk** de tekst t.o.v. gunningscriterium en beoordelingskader. 
Verbeter op **volledigheid, relevantie, bewijswaarde en scoringspotentieel**. 
Houd je aan STRICT_FACT_MODE en volg **deze koppen exact**:

## Korte overall beoordeling (in 5 bullets)
- Sterkste punten, grootste gaten, verwachte score-impact.

## Dekking & structuur t.o.v. gunningskader
- **Dekkingsmatrix (tabel)** – criteriumonderdeel × (Gedekt: ✅/⚠/❌) + 1 regel toelichting per cel.
- Wel/geen aansluiting op gevraagde indeling en beoordelingsaspecten.

## KO/voorwaarden & juridische punten
- KO-eisen, uitsluitingsgronden, geschiktheid, normen (ISO/ISAE/NEN/ENSIA), AVG/DPIA, **GIBIT/ARBIT/ARVODI**, security (TLS/HTTP headers).  
- Conclusie haalbaarheid + expliciete hiaten.

## Architectuur & standaarden (waar relevant)
- Common Ground / ZGW-/STUF(-ZKN) / OData.
- Microsoft-integraties (Azure AD, Graph, SharePoint/Teams, Power BI).
- Toegankelijkheid **WCAG/EN 301 549**.  
Noteer wat ontbreekt en wat explicieter moet.

## KPI-audit (SMART)
- Tabel met huidige KPI-zinnen → **verbeterde SMART-variant** + meetmethode + eigenaar.
- Voeg 2–4 **score-verhogende KPI’s** toe die passen bij het criterium.

## Stijl & tone of voice (doelgroep)
- Past de toon bij onderwijs/gemeente? Jargon/leesbaarheid, actief taalgebruik, benefits/impact.

## Paginabudget & visuele opbouw
- Is de tekst binnen limiet? Waar inkorten/uitbreiden.  
- DTP-tips: tabellen/figuren die score verhogen (bijv. RACI, roadmap, KPI-dashboard).

## “Boven verwachting” (onderscheidend)
- 5–8 concrete voorstellen (bv. PoC/demo-script, adoptie-interventies, security-assurance, datagedreven monitoring).

## Herschrijfsuggesties (voorbeeld)
- **Voorbeeldblok(ken)**: geef 1–2 kernparagrafen opnieuw geschreven in **score-proof** stijl (max. ~300 woorden elk).

## Referenties & bewijs (gericht)
- Adviseer 2–3 **typen** referenties + kort “waarom dit scoort”; koppel aan CONTEXT.

## Actiechecklist (intern)
- 8–12 concrete acties met **Eigenaar** en **Datum** (KPI’s aanscherpen, bewijs verzamelen, structuur corrigeren, visuals maken, legal/security checks).

## Eindoordeel (kort)
- Verwachte score en 3 belangrijkste verbeteracties met grootste impact.
""".strip() + "\n\n" + common_suffix

    else:  # DRAFT
        system = (
            "Je bent TenderMate, AI-tekstarchitect voor aanbestedingen van IT-Workz.\n"
            "Werk in STRICT_FACT_MODE:\n"
            "- Gebruik alleen informatie uit 'document_text' en 'context'.\n"
            "- Als iets ontbreekt: schrijf exact “Niet gespecificeerd in stukken.”\n"
            "- Geen webzoeken, geen aannames.\n\n"
            "Stijl:\n"
            "- Schrijf in helder Nederlands, wervend maar feitelijk; proza met korte alinea’s.\n"
            "- Gebruik duidelijke Markdown-koppen (##/###) en waar nuttig korte tabellen.\n"
            "- Integreer voorbeelden/quotes uit CONTEXT alleen als ze aantoonbaar relevant zijn."
        )

        user = f"""
document_text: {document_text or ''}

OPDRACHT
Genereer een **scoregericht concept** dat exact aansluit op het gevraagde **gunningscriterium** en het beoordelingskader. 
Gebruik CONTEXT om passende voorbeelden, referenties en bewijs te verbinden. 
Houd je aan STRICT_FACT_MODE en volg **deze koppen exact**:

## Executive summary (scorefocus in 5 bullets)
- Wat vragen ze? Hoe scoren we maximaal? Belangrijkste 3 bewijsankers.

## Begrips- & beoordelingskader (facts)
- Herformuleer kort het criterium in eigen woorden (1 alinea).
- Beoordelingsaspecten/weging: noteer percentages indien genoemd; anders “Niet gespecificeerd in stukken”.

## Voorstelstructuur conform criterium
- Geef de exacte indeling (kopjes/subkopjes) die we in het Word-document kunnen overnemen.

## Uitwerking per subcriterium
Voor elk subcriterium (of thema) volg dit patroon:
- **Wat wordt gevraagd (quote/para-phrase)**  
- **Onze aanpak** (proces/stappen, verantwoordelijkheden)  
- **Architectuur & integraties** (Azure AD/SSO/MFA, Graph, SharePoint/Teams, Power BI, Common Ground/ZGW/STUF/OData/WCAG/EN 301 549 waar relevant)  
- **KPI’s (SMART)** – inclusief meetmethode, norm, frequentie, eigenaar  
- **Risico’s & mitigatie** – 1–2 concreet  
- **Bewijs** – referentie/type bewijs uit CONTEXT (link/naam indien aanwezig)

## Planning & mijlpalen
- Tabel met fasen, deliverables, afhankelijkheden en mijlpaaldata (indien bekend; anders “Niet gespecificeerd in stukken”).

## Rollen & RACI (compact)
- Tabel: Activiteit × (R/A/C/I) voor kernrollen (Projectmanager, Solution Architect, Integratie, Security/Privacy, Adoptie/Change, Beheer/SLA).

## KPI-overzicht (tabel)
| KPI | Definitie | Norm | Meting | Frequentie | Eigenaar |

## Paginabudget & opmaak
- Noteer limiet (indien opgegeven). Geef **een verdelingsvoorstel** per hoofdstuk met korte motivatie.
- Tips voor opmaak/illustraties (figuur-/tabelsuggesties) die scoren.

## Referenties & bewijs (gericht)
- 2–3 **typen** referenties met 1 zin “waarom dit scoort” per type. Koppel aan CONTEXT.

## Verhelderingsvragen aan aanbestedende dienst
- 6–10 vragen die scoren (duidelijkheid + optimalisatie van aanpak/KPI’s).

## Bronnen/quotes (compact)
- Noem gebruikte CONTEXT-bronnen of citaten (1 regel per bron).
""".strip() + "\n\n" + common_suffix

    return system, user

# ---------------- Chat call ----------------

def call_chat(system_text: str, user_text: str, mode_hint: str = "") -> str:
    """
    Robuuste AOAI call:
    - mode-aware max_completion_tokens (QUICKSCAN/DRAFT groter)
    - geen tool_choice/response_format
    - ondersteunt text of list-of-parts
    - retries: ruimer budget, daarna compacte fallback
    - logt token usage & finish_reason
    """
    client = get_aoai_client()

    def _extract(choice) -> str:
        if not choice:
            return ""
        msg = getattr(choice, "message", None)
        if msg is None:
            con = getattr(choice, "content", None)
            return con.strip() if isinstance(con, str) else ""
        con = getattr(msg, "content", None)
        if isinstance(con, str) and con.strip():
            return con.strip()
        if isinstance(con, list):
            parts = []
            for p in con:
                try:
                    if isinstance(p, dict):
                        if p.get("type") in ("text", "output_text") and isinstance(p.get("text"), str):
                            parts.append(p["text"])
                    else:
                        if getattr(p, "type", None) in ("text", "output_text") and isinstance(getattr(p, "text", None), str):
                            parts.append(getattr(p, "text"))
                except Exception:
                    pass
            if parts:
                return "\n".join(x for x in parts if x and x.strip())
        ref = getattr(msg, "refusal", None)
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
        return ""

    # Mode-aware budgets
    mode = (mode_hint or "").upper()
    if mode == "QUICKSCAN":
        max_out_1, max_out_2 = 5200, 7200
    elif mode == "DRAFT":
        max_out_1, max_out_2 = 4800, 6800
    else:  # REVIEW of onbekend
        max_out_1, max_out_2 = 3000, 4800

    def _call(messages, max_out):
        return client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=1,
            max_completion_tokens=max_out,
        )

    base_msgs = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]

    # Try 1
    resp = _call(base_msgs, max_out_1)
    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    finish = getattr(choice, "finish_reason", None) if choice else None
    try:
        usage = getattr(resp, "usage", None)
        if usage:
            logging.info(f"usage t={usage.total_tokens} p={usage.prompt_tokens} c={usage.completion_tokens}, finish={finish}")
        else:
            logging.info(f"finish_reason={finish}")
    except Exception:
        pass

    text = _extract(choice) if choice else ""
    if text and text.strip():
        return text.strip()

    # Try 2: groter budget of expliciete platte-tekst instructie
    retry_msgs = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text + "\n\nGeef het volledige antwoord als platte tekst in het Nederlands (geen tools/afbeeldingen/code)."}
    ]
    resp2 = _call(retry_msgs, max_out_2)
    choice2 = resp2.choices[0] if getattr(resp2, "choices", None) else None
    finish2 = getattr(choice2, "finish_reason", None) if choice2 else None
    try:
        usage2 = getattr(resp2, "usage", None)
        if usage2:
            logging.info(f"retry usage t={usage2.total_tokens} p={usage2.prompt_tokens} c={usage2.completion_tokens}, finish={finish2}")
        else:
            logging.info(f"retry finish_reason={finish2}")
    except Exception:
        pass

    text2 = _extract(choice2) if choice2 else ""
    if text2 and text2.strip():
        return text2.strip()

    # Try 3: compacte fallback
    compact_msgs = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text + "\n\nANTWOORD BEPERKT: Geef een compacte versie (max. ~600 woorden), alleen hoofdlijnen & tabellen, platte tekst."}
    ]
    resp3 = _call(compact_msgs, 1800)
    choice3 = resp3.choices[0] if getattr(resp3, "choices", None) else None
    text3 = _extract(choice3) if choice3 else ""

    if text3 and text3.strip():
        return text3.strip()

    return "Er kwam geen leesbare tekst terug van het AI-model. Probeer het nogmaals of verlaag de omvang (minder context of kortere vraag)."

# ---------------- HTTP Function ----------------

@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        if req.method == "GET":
            return func.HttpResponse("OK - TalkToTenderBot vA.26", status_code=200, mimetype="text/plain")

        # Logging intake
        try:
            ct = (req.headers.get("Content-Type") or "").lower()
            cl = req.headers.get("Content-Length") or "?"
            raw_len = len(req.get_body() or b"")
            logging.info(f"req.method={req.method}, ct={ct}, len={cl}, raw_len={raw_len}")
        except Exception:
            pass

        # ---- Parse input (JSON of multipart) ----
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
                return func.HttpResponse("Ongeldig multipart-verzoek (decode fout).", status_code=400, mimetype="text/plain")

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

            # Bestandsverwerking
            if file_part is not None:
                file_bytes = file_part.content or b""
                ext = (os.path.splitext(file_name or "")[1] or "").lower()
                try:
                    if ("pdf" in file_mt) or (ext == ".pdf"):
                        document_text = _read_pdf_text(file_bytes)
                    elif ("wordprocessingml" in file_mt) or (ext == ".docx") or (file_mt == "application/octet-stream" and ext == ".docx"):
                        document_text = _read_docx_text(file_bytes)
                    else:
                        return func.HttpResponse("Bestandstype niet ondersteund. Upload een PDF of DOCX.", status_code=415, mimetype="text/plain")
                except zipfile.BadZipFile:
                    sig = _hex_prefix(file_bytes)
                    return func.HttpResponse(
                        f"Kon DOCX niet lezen: bestand lijkt geen geldig DOCX/ZIP (magic={sig}). "
                        "Controleer IRM/wachtwoord-beveiliging of sla opnieuw op als .docx.",
                        status_code=422, mimetype="text/plain")
                except PackageNotFoundError:
                    return func.HttpResponse("Kon DOCX niet openen (geen geldig Office-package). Sla opnieuw op als .docx en probeer opnieuw.", status_code=422, mimetype="text/plain")
                except Exception as e:
                    return func.HttpResponse(f"Fout bij lezen van document: {repr(e)}", status_code=422, mimetype="text/plain")

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
                status_code=415, mimetype="text/plain"
            )

        # ---- Mode bepalen ----
        mode_final = detect_mode(mode, question)

        # ---- Zoek context in Azure AI Search (PF-achtig) ----
        queries = _expand_queries(question, mode_final, document_text)
        docs = []
        try:
            docs = hybrid_search_multi(queries, top_k=TOP_K)
        except Exception as e:
            logging.exception(f"Search error: {e}")
            docs = []

        context_text = _join_docs_for_context(
            docs,
            per_doc_chars=1600,
            max_docs=MAX_CONTEXT_DOCS,
            max_context_chars=16000
        )

        # Trim raw inputs to avoid blowing the context window
        document_text = _clip(document_text, 40000)
        if isinstance(conversation, list) and len(conversation) > 8:
            conversation = conversation[-8:]

        # ---- Chat history als PF-achtig blok ----
        chat_history_pf = format_history_for_prompt_flow(conversation)

        # ---- Render prompts ----
        system_text, user_text = render_system_and_user(
            mode=mode_final,
            document_text=document_text,
            context_text=context_text,
            chat_history_pf=chat_history_pf,
            chat_input=question or "Analyseer het bijgevoegde document."
        )

        # ---- Call Azure OpenAI Chat ----
        try:
            chat_output = call_chat(system_text, user_text, mode_hint=mode_final)
        except Exception as e:
            logging.exception(f"AOAI chat error: {e}")
            return func.HttpResponse(f"AI-dienst error: {repr(e)}", status_code=502, mimetype="text/plain")

        return func.HttpResponse(
            body=json.dumps({"chat_output": chat_output}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error("UNHANDLED", exc_info=True)
        body_text = f"Unhandled server error: {repr(e)}\n{traceback.format_exc()}"
        return func.HttpResponse(body_text, status_code=500, mimetype="text/plain")
