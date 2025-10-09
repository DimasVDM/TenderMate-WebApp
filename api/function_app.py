# function_app.py — versie v12 (multipart + DOCX/PDF + mode + auto-router)

import azure.functions as func
import logging
import os
import requests
import json
import io
import traceback
import zipfile
import binascii
import re

from pypdf import PdfReader
from requests.exceptions import ReadTimeout
import docx
from docx.opc.exceptions import PackageNotFoundError
from requests_toolbelt.multipart.decoder import MultipartDecoder

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ---------------- Helpers ----------------

ALLOWED_MODES = {"QUICKSCAN", "REVIEW", "DRAFT"}

def format_history_for_prompt_flow(conversation):
    """
    Zet [{role, content}, ...] om naar Prompt Flow chat_history.
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


def extract_chat_output_from_json(pf_data: dict) -> str:
    """
    Probeer een bruikbaar antwoordveld uit de PF JSON te halen.
    """
    if pf_data is None:
        return ""
    for key in ("chat_output", "output", "result", "answer", "content"):
        val = pf_data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # eenvoudige fallback: niets gevonden
    return ""


def _hex_prefix(b: bytes, n: int = 16) -> str:
    return binascii.hexlify(b[:n]).decode().upper()


def _is_probably_docx(file_bytes: bytes) -> bool:
    # DOCX is een ZIP-container
    return zipfile.is_zipfile(io.BytesIO(file_bytes))


def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def _read_docx_text(file_bytes: bytes) -> str:
    """
    Lees tekst uit een DOCX. Gooit BadZipFile/PackageNotFoundError door bij corrupte/niet-DOCX bestanden.
    """
    if not _is_probably_docx(file_bytes):
        sig = _hex_prefix(file_bytes, 8)
        raise zipfile.BadZipFile(f"Not a DOCX/ZIP (magic={sig})")
    d = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in d.paragraphs])


def _normalize_mode(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().upper()
    # een paar synoniemen die uit de UI of typend kunnen komen
    synonyms = {
        "QUICK", "SCAN", "QUICK-SCAN", "QUICKSCAN",
        "REVIEW", "BEOORDELEN", "BEOORDELING", "FEEDBACK",
        "DRAFT", "OPZET", "SCHRIJVEN", "WRITE"
    }
    if v in ALLOWED_MODES:
        return v
    if v in {"QUICK", "SCAN", "QUICK-SCAN"}:
        return "QUICKSCAN"
    if v in {"BEOORDELEN", "BEOORDELING", "FEEDBACK"}:
        return "REVIEW"
    if v in {"OPZET", "SCHRIJVEN", "WRITE"}:
        return "DRAFT"
    # soms sturen we "TMMODE:REVIEW" etc. mee — haal uit patroon
    m = re.search(r"TMMODE\s*:\s*([A-Z_-]+)", v)
    if m:
        return _normalize_mode(m.group(1))
    return None


def _infer_mode_from_text(question: str) -> str:
    """
    Heel simpele heuristiek wanneer 'mode' niet expliciet is meegestuurd.
    """
    q = (question or "").lower()

    quick_keywords = [
        "quickscan", "quick scan", "go/no-go", "go nogo", "go no go",
        "samenvatting", "analyseer", "analyse", "kwalificatie"
    ]
    review_keywords = [
        "beoordeel", "beoordelen", "review", "feedback", "verbeter", "herschrijf",
        "audit", "kwaliteitscheck", "kwaliteitscontrole"
    ]
    draft_keywords = [
        "opzet", "concept", "schrijf", "uitwerken", "plan van aanpak",
        "concepttekst", "maak een antwoord", "draft"
    ]

    if any(k in q for k in review_keywords):
        return "REVIEW"
    if any(k in q for k in quick_keywords):
        return "QUICKSCAN"
    if any(k in q for k in draft_keywords):
        return "DRAFT"

    # default: opzet
    return "DRAFT"

# -------------- Eind helpers --------------


@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Healthcheck
        if req.method == "GET":
            return func.HttpResponse("OK - TalkToTenderBot v12", status_code=200, mimetype="text/plain")

        # Intake logging (hulp bij diagnose)
        try:
            ct = (req.headers.get("Content-Type") or "").lower()
            cl = req.headers.get("Content-Length") or "?"
            cookie_present = bool(req.headers.get("Cookie"))
            raw_body = req.get_body() or b""
            logging.info(
                f"req.method={req.method}, ct={ct}, len={cl}, raw_len={len(raw_body)}, cookie_present={cookie_present}"
            )
        except Exception as le:
            logging.warning(f"intake-log failed: {le}")

        # Config van Prompt Flow
        prompt_flow_url = os.environ.get("PROMPT_FLOW_URL")
        prompt_flow_api_key = os.environ.get("PROMPT_FLOW_API_KEY")
        logging.info(f"PFlow URL present={bool(prompt_flow_url)}, key present={bool(prompt_flow_api_key)}")
        if not prompt_flow_url or not prompt_flow_api_key:
            return func.HttpResponse("Server configuration error (PF URL/key missing).", status_code=500, mimetype="text/plain")

        # Invoer (JSON of multipart)
        question = ""
        conversation = []
        document_text = ""
        mode = None   # NEW

        content_type = (req.headers.get("Content-Type") or "").lower()

        if "multipart/form-data" in content_type:
            logging.info("parse-path=multipart")
            body_bytes = req.get_body() or b""

            # Parse multipart met requests-toolbelt (pakt boundary zelf uit de header)
            try:
                decoder = MultipartDecoder(body_bytes, content_type)
            except Exception as e:
                logging.exception("Multipart decode error")
                return func.HttpResponse("Ongeldig multipart-verzoek (decode fout).", status_code=400, mimetype="text/plain")

            text_fields = {}
            file_part = None
            file_mt = ""
            file_name = ""

            for p in decoder.parts:
                cd = p.headers.get(b'Content-Disposition', b'').decode(errors="ignore")
                ctype = p.headers.get(b'Content-Type', b'').decode(errors="ignore").lower()

                if 'filename=' in cd:
                    file_part = p
                    file_mt = ctype or "application/octet-stream"
                    # bestandsnaam uit Content-Disposition
                    m = re.search(r'filename\*?="?([^";]+)"?', cd, flags=re.IGNORECASE)
                    file_name = (m.group(1) if m else "").strip()
                else:
                    m = re.search(r'name="([^"]+)"', cd or "", flags=re.IGNORECASE)
                    if m:
                        text_fields[m.group(1)] = p.text

            question = (text_fields.get("question") or "").strip()
            conv_json = text_fields.get("conversation") or "[]"
            try:
                conversation = json.loads(conv_json) if conv_json else []
            except Exception:
                conversation = []

            # NEW: mode uit formveld (we ondersteunen 'mode' en 'tm_mode')
            mode = _normalize_mode(text_fields.get("mode") or text_fields.get("tm_mode"))

            has_file = file_part is not None
            logging.info(
                f"parsed(multipart): question_len={len(question)}, has_file={has_file}, conv_items={len(conversation)}, "
                f"file_ct={file_mt}, file_name={file_name}, mode={mode}"
            )

            if has_file:
                ext = (os.path.splitext(file_name or "")[1] or "").lower()
                file_bytes = file_part.content or b""

                try:
                    if ("pdf" in file_mt) or (ext == ".pdf"):
                        document_text = _read_pdf_text(file_bytes)
                    elif ("wordprocessingml" in file_mt) or (ext == ".docx") or (file_mt == "application/octet-stream" and ext == ".docx"):
                        document_text = _read_docx_text(file_bytes)
                    else:
                        logging.warning(f"Unsupported upload type: ctype={file_mt}, ext={ext}, name={file_name}")
                        return func.HttpResponse(
                            "Bestandstype niet ondersteund. Upload een PDF of DOCX.",
                            status_code=415, mimetype="text/plain"
                        )

                except zipfile.BadZipFile as e:
                    sig = _hex_prefix(file_bytes)
                    logging.exception(f"DOCX BadZipFile: {e} ctype={file_mt} ext={ext} name={file_name} sig={sig}")
                    return func.HttpResponse(
                        f"Kon DOCX niet lezen: bestand lijkt geen geldig DOCX/ZIP (magic={sig}). "
                        "Controleer of het geen .doc is, of dat IRM/wachtwoord-beveiliging is uitgezet.",
                        status_code=422, mimetype="text/plain"
                    )
                except PackageNotFoundError as e:
                    logging.exception(f"DOCX PackageNotFound: {e} ctype={file_mt} ext={ext} name={file_name}")
                    return func.HttpResponse(
                        "Kon DOCX niet openen (geen geldig Office-package). Sla opnieuw op als .docx en probeer opnieuw.",
                        status_code=422, mimetype="text/plain"
                    )
                except Exception as e:
                    logging.exception(f"Onbekende fout bij document lezen: {e} ctype={file_mt} ext={ext} name={file_name}")
                    return func.HttpResponse(
                        f"Fout bij lezen van document: {repr(e)}",
                        status_code=422, mimetype="text/plain"
                    )

        elif "application/json" in content_type:
            logging.info("parse-path=json")
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

            question = (body.get("question") or body.get("chat_input") or "").strip()
            conversation = body.get("conversation", []) or []
            document_text = body.get("document_text", "") or ""
            mode = _normalize_mode(body.get("mode") or body.get("tm_mode"))  # NEW

            logging.info(
                f"parsed(json): question_len={len(question)}, has_doc_text={bool(document_text)}, conv_items={len(conversation)}, mode={mode}"
            )
        else:
            return func.HttpResponse(
                f"Content-Type '{content_type}' niet ondersteund (verwacht JSON of multipart/form-data).",
                status_code=415, mimetype="text/plain"
            )

        # --- Mode fallback (auto-router) ---
        if not mode:
            mode = _infer_mode_from_text(question)
            logging.info(f"auto-inferred mode: {mode}")

        # ---- Payload naar Prompt Flow ----
        payload = {
            "chat_input":   question or "Analyseer het bijgevoegde document.",
            "chat_history": format_history_for_prompt_flow(conversation),
            "document_text": document_text or "",
            "mode": mode,  # NEW: naar jouw enkele Jinja-prompt
        }
        headers = {
            "Authorization": f"Bearer {prompt_flow_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Aanroep PF met royale timeout
        try:
            resp = requests.post(
                prompt_flow_url,
                headers=headers,
                json=payload,
                timeout=(10, 210),  # 10s connect, 210s read
            )
        except ReadTimeout:
            logging.error("PF ReadTimeout")
            return func.HttpResponse("AI-dienst duurde te lang (timeout).", status_code=504, mimetype="text/plain")
        except requests.exceptions.RequestException as e:
            logging.exception(f"PF request error: {e}")
            return func.HttpResponse(f"PF request error: {repr(e)}", status_code=502, mimetype="text/plain")

        logging.info(f"PF status={resp.status_code}")
        if not (200 <= resp.status_code < 300):
            snippet = (resp.text or "")[:500]
            logging.warning(f"PF non-2xx body snippet: {snippet}")
            return func.HttpResponse(
                f"AI-dienst error (status {resp.status_code}): {snippet}",
                status_code=502, mimetype="text/plain"
            )

        # ---- PF response interpreteren ----
        body_text = resp.text or ""
        try:
            pf_data = resp.json()
            chat_output = extract_chat_output_from_json(pf_data)
            if not chat_output:
                chat_output = json.dumps(pf_data)[:4000]
            return func.HttpResponse(
                body=json.dumps({"chat_output": chat_output}),
                status_code=200, mimetype="application/json"
            )
        except ValueError:
            txt = body_text.strip()
            if txt[:1] == "<" or txt.lower().startswith("<!doctype") or "html" in txt[:200].lower():
                logging.error("PF returned HTML; vermoedelijk auth/endpoint issue.")
                return func.HttpResponse(
                    "AI-dienst gaf HTML terug (mogelijk auth/endpoint fout).",
                    status_code=502, mimetype="text/plain"
                )
            return func.HttpResponse(
                body=json.dumps({"chat_output": txt[:4000]}),
                status_code=200, mimetype="application/json"
            )

    except Exception as e:
        logging.error("UNHANDLED", exc_info=True)

        # Debug-switch: ?debug=1 geeft 200 + fouttekst (handig tegen SWA maskering)
        try:
            qs = req.url.split("?", 1)[1] if "?" in req.url else ""
            debug_flag = "debug=1" in qs
        except Exception:
            debug_flag = False

        body_text = f"Unhandled server error: {repr(e)}\n{traceback.format_exc()}"
        if debug_flag:
            return func.HttpResponse(json.dumps({"error": body_text}), status_code=200, mimetype="application/json")
        return func.HttpResponse(body_text, status_code=500, mimetype="text/plain")
