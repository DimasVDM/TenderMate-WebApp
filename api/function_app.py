import azure.functions as func
import logging
import os
import requests
import json
import io
import traceback
from pypdf import PdfReader
from requests.exceptions import ReadTimeout
import docx

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# -------------------- Config --------------------
# Max aantal karakters dat we uit een document doorgeven aan Prompt Flow.
# Aanpasbaar via App Setting MAX_DOC_CHARS (string -> int).
DEFAULT_MAX_DOC_CHARS = 120_000
try:
    MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", DEFAULT_MAX_DOC_CHARS))
except Exception:
    MAX_DOC_CHARS = DEFAULT_MAX_DOC_CHARS
# ------------------------------------------------


# -------------------- Helpers --------------------
def format_history_for_prompt_flow(conversation):
    """
    Convert simple [{role, content}, ...] history into Prompt Flow chat_history.
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
                    "inputs": {"chat_input": conversation[i].get("content", "")},
                    "outputs": {"chat_output": conversation[i + 1].get("content", "")},
                }
            )
    return chat_history


def extract_chat_output_from_json(pf_data: dict) -> str:
    """
    Haal het meest waarschijnlijke veld met het antwoord uit diverse mogelijke JSON-vormen.
    """
    if pf_data is None:
        return ""

    # Meest gebruikelijke PF output
    for key in ("chat_output", "output", "result", "answer", "content"):
        val = pf_data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # OpenAI-achtige vorm
    try:
        choices = pf_data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        pass

    # Als 'output' een dict is met 'message'->'content'
    try:
        output = pf_data.get("output")
        if isinstance(output, dict):
            msg = output.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        pass

    # Fallback: niets bruikbaars gevonden
    return ""


def _extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """Lees tekst uit een PDF veilig uit."""
    try:
        reader = PdfReader(file_stream)
        parts = []
        for p in reader.pages:
            # p.extract_text() kan None geven
            parts.append(p.extract_text() or "")
        return "\n".join(parts)
    except Exception as e:
        logging.exception(f"PDF parse error: {e}")
        return ""


def _extract_text_from_docx(file_stream: io.BytesIO) -> str:
    """Lees tekst uit een DOCX veilig uit."""
    try:
        d = docx.Document(file_stream)
        return "\n".join([para.text for para in d.paragraphs])
    except Exception as e:
        logging.exception(f"DOCX parse error: {e}")
        return ""
# ------------------------------------------------


@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # --- health / version ---
        if req.method == "GET":
            return func.HttpResponse(
                "OK - TalkToTenderBot v7", status_code=200, mimetype="text/plain"
            )

        # ---------- intake logging ----------
        logging.info("TalkToTenderBot start")
        try:
            ct = (req.headers.get("Content-Type") or "").lower()
            cl = req.headers.get("Content-Length") or "?"
            cookie_present = bool(req.headers.get("Cookie"))
            raw_body = req.get_body() or b""
            logging.info(
                f"req.method={req.method}, ct={ct}, len={cl}, raw_len={len(raw_body)}, "
                f"cookie_present={cookie_present}"
            )
        except Exception as le:
            logging.warning(f"intake-log failed: {le}")
        # ------------------------------------

        prompt_flow_url = os.environ.get("PROMPT_FLOW_URL")
        prompt_flow_api_key = os.environ.get("PROMPT_FLOW_API_KEY")
        logging.info(
            f"PFlow URL present={bool(prompt_flow_url)}, key present={bool(prompt_flow_api_key)}"
        )
        if not prompt_flow_url or not prompt_flow_api_key:
            return func.HttpResponse(
                "Server configuration error (PF URL/key missing).",
                status_code=500,
                mimetype="text/plain",
            )

        # ---- parse input (JSON or multipart) ----
        question = ""
        conversation = []
        document_text = ""
        document_name = ""
        document_mime = ""
        doc_truncated = False

        try:
            ct = (req.headers.get("Content-Type") or "").lower()
            if "multipart/form-data" in ct:
                logging.info("parse-path=multipart")

                # Azure Functions Python kan form/files anders exposen; defensief zijn.
                try:
                    form = req.form or {}
                except AttributeError:
                    form = {}
                try:
                    files = req.files or {}
                except AttributeError:
                    files = {}

                question = (form.get("question") or "").strip()
                conv_json = form.get("conversation", "[]")
                conversation = json.loads(conv_json) if conv_json else []

                file = files.get("document")
                has_file = bool(file)
                logging.info(
                    f"parsed(multipart): question_len={len(question)}, has_file={has_file}, "
                    f"conv_items={len(conversation)}"
                )

                if file:
                    # meta
                    document_name = getattr(file, "filename", "") or ""
                    document_mime = (getattr(file, "mimetype", "") or "").lower()

                    file_bytes = file.read()
                    file_stream = io.BytesIO(file_bytes)

                    if document_mime == "application/pdf":
                        document_text = _extract_text_from_pdf(file_stream)
                    elif document_mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        document_text = _extract_text_from_docx(file_stream)
                    else:
                        # Probeer simpele fallback: treat as binary/unknown
                        document_text = ""
                        logging.info(f"Unsupported mimetype for inline extraction: {document_mime}")

                    # Truncate als extreem groot
                    if document_text and len(document_text) > MAX_DOC_CHARS:
                        document_text = document_text[:MAX_DOC_CHARS]
                        doc_truncated = True

            else:
                logging.info("parse-path=json")
                body = {}
                try:
                    body = req.get_json()
                except ValueError:
                    # fallback: als body raw JSON is maar content-type niet klopt
                    raw = req.get_body()
                    if raw:
                        try:
                            body = json.loads(raw.decode("utf-8", errors="ignore"))
                        except Exception:
                            body = {}

                question = (body.get("question") or body.get("chat_input") or "").strip()
                conversation = body.get("conversation", []) or []
                document_text = body.get("document_text", "") or ""
                document_name = body.get("document_name", "") or ""
                document_mime = body.get("document_mime", "") or ""

                if document_text and len(document_text) > MAX_DOC_CHARS:
                    document_text = document_text[:MAX_DOC_CHARS]
                    doc_truncated = True

                logging.info(
                    f"parsed(json): question_len={len(question)}, has_doc_text={bool(document_text)}, "
                    f"conv_items={len(conversation)}"
                )
        except Exception as pe:
            logging.exception(f"Invoer parse error: {pe}")
            return func.HttpResponse(
                "Ongeldig verzoek (parse error).",
                status_code=400,
                mimetype="text/plain",
            )

        # ---- build PF payload ----
        payload = {
            "chat_input": question or "Analyseer het bijgevoegde document.",
            "chat_history": format_history_for_prompt_flow(conversation),

            # Documentvelden voor de PF flow (beoordeling / context)
            "document_text": document_text or "",
            "document_name": document_name,
            "document_mime": document_mime,
            "document_length": len(document_text or ""),
            "document_truncated": doc_truncated,
            # Je kunt eventueel ook een 'mode' meegeven, als je PF flow dat ondersteunt:
            # "mode": "review" als er document_text is, anders "chat"
        }
        headers = {
            "Authorization": f"Bearer {prompt_flow_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # ---- call Prompt Flow ----
        try:
            resp = requests.post(
                prompt_flow_url,
                headers=headers,
                json=payload,
                timeout=(10, 210),  # 10s connect, 210s read
            )
        except ReadTimeout:
            logging.error("PF ReadTimeout")
            return func.HttpResponse(
                "AI-dienst duurde te lang (timeout).",
                status_code=504,
                mimetype="text/plain",
            )
        except requests.exceptions.RequestException as e:
            logging.exception(f"PF request error: {e}")
            return func.HttpResponse(
                f"PF request error: {repr(e)}",
                status_code=502,
                mimetype="text/plain",
            )

        logging.info(f"PF status={resp.status_code}")
        if not (200 <= resp.status_code < 300):
            snippet = (resp.text or "")[:500]
            logging.warning(f"PF non-2xx body snippet: {snippet}")
            return func.HttpResponse(
                f"AI-dienst error (status {resp.status_code}): {snippet}",
                status_code=502,
                mimetype="text/plain",
            )

        # ---- interpret PF response robustly ----
        body_text = resp.text or ""
        try:
            pf_data = resp.json()
            chat_output = extract_chat_output_from_json(pf_data)
            if not chat_output:
                # Als JSON is maar geen duidelijk veld, stuur de hele JSON compact terug
                chat_output = json.dumps(pf_data)[:4000]

            # Voeg bij truncatie een korte noot toe (zodat de gebruiker weet waarom)
            if doc_truncated:
                chat_output += (
                    "\n\n*Let op:* het geüploade document was erg groot; "
                    "alleen het eerste deel is beoordeeld."
                )

            return func.HttpResponse(
                body=json.dumps({"chat_output": chat_output}),
                status_code=200,
                mimetype="application/json",
            )
        except ValueError:
            # Niet-JSON: kan plain text of HTML zijn
            txt = body_text.strip()
            if txt[:1] == "<" or txt.lower().startswith("<!doctype") or "html" in txt[:200].lower():
                logging.error("PF returned HTML; vermoedelijk auth/endpoint issue.")
                return func.HttpResponse(
                    "AI-dienst gaf HTML terug (mogelijk auth/endpoint fout).",
                    status_code=502,
                    mimetype="text/plain",
                )
            # Plain text → geef het gewoon terug aan de UI
            return func.HttpResponse(
                body=json.dumps({"chat_output": txt[:4000]}),
                status_code=200,
                mimetype="application/json",
            )

    except Exception as e:
        logging.error("UNHANDLED", exc_info=True)
        # debug switch: ?debug=1 -> 200 met fouttekst zodat SWA niets maskeert
        try:
            qs = req.url.split("?", 1)[1] if "?" in req.url else ""
            debug_flag = "debug=1" in qs
        except Exception:
            debug_flag = False

        body_text = f"Unhandled server error: {repr(e)}\n{traceback.format_exc()}"
        if debug_flag:
            return func.HttpResponse(
                json.dumps({"error": body_text}),
                status_code=200,
                mimetype="application/json",
            )
        return func.HttpResponse(
            f"Unhandled server error: {repr(e)}\n{traceback.format_exc()}",
            status_code=500,
            mimetype="text/plain",
        )
