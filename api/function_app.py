# function_app.py — Serverless PF-vervanger (hybride search + AOAI chat), vA.1
# - Ondersteunt PDF/DOCX uploads
# - Hybride search tegen indexvelden: chunk/title/text_vector
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
TOP_K             = int(os.environ.get("TOP_K", "12"))

# Azure OpenAI
AOAI_ENDPOINT          = os.environ.get("AOAI_ENDPOINT", "").rstrip("/")
AOAI_API_KEY           = os.environ.get("AOAI_API_KEY", "")
AOAI_CHAT_DEPLOYMENT   = os.environ.get("AOAI_CHAT_DEPLOYMENT", "gpt-5")  # pas aan naar jouw deploymentnaam
AOAI_EMBED_DEPLOYMENT  = os.environ.get("AOAI_EMBED_DEPLOYMENT", "text-embedding-3-large")  # 3072-dim

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

def hybrid_search(query_text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Hybride search: keyword + vector (client-side embeddings), semantic rerank.
    Veldmapping: content=chunk, source=title, vector=text_vector
    """
    sc = get_search_client()
    vec = embed_text(query_text)
    vq = VectorizedQuery(vector=vec, k_nearest_neighbors=top_k, fields=TEXT_VECTOR_FIELD)

    results = sc.search(
        search_text=query_text,
        vector_queries=[vq],
        top=top_k,
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name=SEMANTIC_CONFIG
    )

    items = []
    for r in results:
        # r is a SearchResult; dict-like access
        items.append({
            "content":   r.get("chunk", "") or "",
            "source":    r.get("title", "") or "",
            "parent_id": r.get("parent_id", "") or ""
        })
    return items

def build_context(docs: List[Dict[str, Any]]) -> str:
    """
    Bouw RAG-context: Content + Source (titel) + parent_id (optioneel).
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
    Prompts zijn de (samengebalde) varianten die je hiervoor gebruikte.
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

    document_line = f"document_text: {document_text or ''}"

    if mode == "QUICKSCAN":
        system = (
            "Je bent TenderMate, de elite AI-tenderanalist en strategisch bid-adviseur van IT-Workz.\n\n"
            "Jouw taak: analyseer het volledige aanbestedingsdocument (document_text) en plaats dit altijd in de context "
            "van de meest relevante voorbeelden uit de IT-Workz kennisbank (context), zoals de Producten- en Diensten Catalogus, "
            "eerdere aanbestedingen en referentie-documenten."
        )
        user = f"""
{document_line}

Maak een strategische quickscan (kwalificatie / go-no-go) op basis van de CONTEXT. Gebruik deze koppen exact in deze volgorde en schrijf in proza, compact maar volledig. Gebruik visuele indicatoren (✅/❌ en 🟢/🔴) waar aangegeven. Waar informatie ontbreekt: “Niet gespecificeerd in stukken.”
Belangrijk: geef de koppen duidelijk in bold weer (grotere letter/duidelijke typografie) en onderscheid ze visueel van de inhoud.

## Kerngegevens aanbesteding
- **Naam aanbesteding / opdrachtgever**: [naam | Niet gespecificeerd in stukken]
- **Soort aanbesteding / procedure**: [type | Niet gespecificeerd in stukken]
- **Inkoopadviseur / contact**: [naam/organisatie | Niet gespecificeerd in stukken]
- **Vragen stellen – manier & kanaal**: [wijze | Niet gespecificeerd in stukken]
- **Aantal vragenronden**: [aantal | Niet gespecificeerd in stukken]

## Organisatieprofiel (onderwijs)
- **Soort organisatie / onderwijssoort**: [type | Niet gespecificeerd in stukken]
- **Aantal scholen/locaties**: [aantal | Niet gespecificeerd in stukken]
- **Aantal medewerkers**: [aantal | Niet gespecificeerd in stukken]
- **Aantal leerlingen/studenten**: [aantal | Niet gespecificeerd in stukken]

## Aanleiding en doel
- Duid kort het “waarom” van de uitvraag.

## Essentie van de uitvraag (scope)
- Scope/omvang, kernbegrippen; koppel aan CONTEXT.

## Contract & budget
- **Looptijd** (basis + opties); **Contractwaarde / richtprijs / prijsplafond**; bijzonderheden.

## KO-criteria en voorwaarden
- Uitsluitingsgronden / geschiktheidseisen (technisch / beroeps), normeringen/certificeringen (ISO/ISAE/NEN/ENSIA), juridische kaders (ARBIT/ARVODI/GIBIT, AVG).
- Match met IT-Workz Catalogus + haalbaarheidsconclusie.

## Gunningscriteria
- Overzicht + weging (indien beschikbaar) en scorefocus.

## Programma van Eisen/Wensen (samengevat)
- Cluster per thema (projectorganisatie, techniek/architectuur, privacy/AVG, beheer/SLA, adoptie/training, planning/migratie).

## Analyse van Vereiste Disciplines
- **Uitvoering** (rollen/skills) + **Bid-team** (rollen voor inschrijving).

## Weging, paginabudget en planning
- Wegingspercentages, paginabudget/verdeling, planning & deadlines.

## Budget en concurrentiepositie
- Signalen budget, concurrenten (indien af te leiden) + sterke/zwakke punten.

## Showstoppers / risico’s (en mitigatie)
- Lijst + mitigaties of verhelderingsvragen.

## Referenties (advies)
- 2–3 typen referenties + waarom ze passen.

## In te dienen bewijsstukken
- Gevraagde stukken (UEA, KvK, ISO/ISAE, verzekeringen, referenties, DPA, prijsbiljet, etc.) + vergelijking met IT-Workz standaardset en status.

## Standaard- en verdiepingsvragen
- Standaard en strategische vragen (incl. vragenprocedure).

## Actiechecklist (intern)
- Concrete to-do’s (taak; eigenaar; datum).

## Go/No-Go indicatie (conclusie)
- Samenvattend advies o.b.v. showstoppers, concurrentie, match portfolio, risico’s, beschikbaarheid bewijs/competenties.
""".strip() + "\n\n" + common_suffix

    elif mode == "REVIEW":
        system = (
            "Je bent TenderMate, AI-kwaliteitsauditor en kritische kwaliteitsmanager voor aanbestedingen van IT-Workz.\n\n"
            "Evalueer en verbeter een aangeleverde tekst (document_text) op volledigheid, relevantie en scoringspotentieel."
        )
        user = f"""
{document_line}

OPDRACHT
Beoordeel en verbeter een (impliciet) aangeleverd stuk t.o.v. gunningscriteria en beoordelingskader. Schrijf in proza; gebruik bullets alleen bij checklists/KPI’s. Noteer hiaten als “Niet gespecificeerd in stukken.” Volg deze koppen exact:

## Korte overall beoordeling
## Fit met gunningskader en eisen
## KO/voorwaarden en juridische punten
## SMART & KPI’s
## Stijl en toon (onderwijs)
## Paginabudget & structuur
## Boven verwachting scoren
## Herschrijfsuggesties (show, don’t tell)
## Referenties en bewijs
## Actiechecklist (intern)
""".strip() + "\n\n" + common_suffix

    else:  # DRAFT
        system = (
            "Je bent TenderMate, AI-tekstarchitect voor aanbestedingen van IT-Workz.\n\n"
            "Schrijf een SMART, wervend en praktisch conceptantwoord op een gunningscriterium. "
            "Gebruik relevante voorbeelden uit de kennisbank (context) en het aangeleverde document (document_text)."
        )
        user = f"""
{document_line}

OPDRACHT
Genereer een voorstelindeling + compacte conceptinhoud die exact aansluit op gunningscriteria en beoordelingskader. Gebruik deze koppen exact in deze volgorde. Schrijf in proza; bullets alleen bij checklists/KPI’s. Markeer gaten als “Niet gespecificeerd in stukken.”

## Inleiding
## Structuur conform gunningskader
## Onze aanpak in hoofdlijnen
## Uitwerking per thema/onderdeel
## Rollen en beschikbaarheid
## KPI’s en bewaking (SMART)
## Planning en mijlpalen
## Risico’s en beheersmaatregelen
## Referenties en bewijs
## Paginabudget en scorefocus
## Verhelderingsvragen richting aanbestedende dienst
## Gebruikte bronnen (compact)
""".strip() + "\n\n" + common_suffix

    return system, user

# ---------------- Chat call ----------------

def call_chat(system_text: str, user_text: str) -> str:
    """
    Roept Azure OpenAI Chat aan.
    Let op: voor (nieuwere) modellen is 'max_completion_tokens' vereist i.p.v. 'max_tokens'.
    """
    client = get_aoai_client()
    try:
        resp = client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
            temperature=0.3,
            max_completion_tokens=6000,  # <-- gebruik dit veld (niet max_tokens)
        )
    except TypeError:
        # fallback voor oudere SDK’s/modellen die nog max_tokens verwachten
        resp = client.chat.completions.create(
            model=AOAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
            temperature=0.3,
            max_tokens=6000,
        )
    msg = resp.choices[0].message.content if getattr(resp, "choices", None) else ""
    return msg or ""

# ---------------- HTTP Function ----------------

@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        if req.method == "GET":
            return func.HttpResponse("OK - TalkToTenderBot vA.3", status_code=200, mimetype="text/plain")

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
                except zipfile.BadZipFile as e:
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

        # ---- Zoek context in Azure AI Search ----
        # Gebruik de vraag + (optioneel) een hint uit document_text voor betere RAG
        search_query = question or (document_text[:300] if document_text else "aanbesteding onderwijs")
        docs = []
        try:
            docs = hybrid_search(search_query, top_k=TOP_K)
        except Exception as e:
            logging.exception(f"Search error: {e}")
            # We gaan door zonder RAG-context
            docs = []

        context_text = build_context(docs)

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
            chat_output = call_chat(system_text, user_text)
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
