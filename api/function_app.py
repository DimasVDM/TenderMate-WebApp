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

# --- optionele fallback import voor multipart ---
try:
    from requests_toolbelt.multipart.decoder import MultipartDecoder
except Exception:
    MultipartDecoder = None

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Max aantal karakters dat we richting Prompt Flow sturen vanuit geuploade tekst
MAX_DOC_CHARS = 120_000


# -------- Helpers --------
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
    for key in ("chat_output", "output", "result", "answer", "content"):
        val = pf_data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    try:
        choices = pf_data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        pass
    try:
        output = pf_data.get("output")
        if isinstance(output, dict):
            msg = output.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        pass
    return ""


def _extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts)


def _extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    d = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in d.paragraphs])


def _parse_multipart_with_toolbelt(content_type: str, body_bytes: bytes):
    """
    Fallback: parse multipart met requests-toolbelt.
    Retourneert (question, conversation, document_text, debug_info)
    """
    if not MultipartDecoder:
        raise ValueError("requests-toolbelt niet beschikbaar (MultipartDecoder is None)")

    dec = MultipartDecoder(body_bytes, content_type)
    question = ""
    conversation = []
    document_text = ""
    debug_info = {"parts": []}

    def _get_header(headers: dict, key: bytes) -> str:
        return (headers.get(key, b"").decode(errors="ignore") or "").strip()

    # Probeer 'document' te vinden; anders pak eerste part met filename
    file_part = None
    for p in dec.parts:
        cd = _get_header(p.headers, b"Content-Disposition")
        ct = _get_header(p.headers, b"Content-Type")
        debug_info["parts"].append({"cd": cd, "ct": ct, "size": len(p.content)})

        if 'name="question"' in cd and 'filename="' not in cd:
            try:
                question = p.text.strip()
            except Exception:
                question = (p.content or b"").decode("utf-8", errors="ignore").strip()
        elif 'name="conversation"' in cd and 'filename="' not in cd:
            raw = ""
            try:
                raw = p.text
            except Exception:
                raw = (p.content or b"").decode("utf-8", errors="ignore")
            try:
                conversation = json.loads(raw) if raw else []
            except Exception:
                conversation = []
        elif 'filename="' in cd and ('name="document"' in cd or file_part is None):
            # neem expliciet 'document', anders 1e file
            file_part = p

    if file_part is not None:
        cd = _get_header(file_part.headers, b"Content-Disposition")
        ct = _get_header(file_part.headers, b"Content-Type").lower()
        file_bytes = file_part.content or b""
        # mimetype detectie
        is_pdf = ("pdf" in ct) or ('.pdf"' in cd.lower())
        is_docx = ("wordprocessingml" in ct) or (ct == "application/vnd.openxmlformats-officedocument.wordprocessingml.document") or ('.docx"' in cd.lower())

        if is_pdf:
            document_text = _extract_text_from_pdf_bytes(file_bytes)
        elif is_docx:
            document_text = _extract_text_from_docx_bytes(file_bytes)
        else:
            document_text = "Bestandstype niet ondersteund."

    return question, conversation, document_text, debug_info
# -------------------------


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

        try:
            ct = (req.headers.get("Content-Type") or "").lower()
            if "multipart/form-data" in ct:
                logging.info("parse-path=multipart")

                # 1) Probeer vriendelijke weg (kan op sommige workers werken)
                used_toolbelt = False
                try:
                    try:
                        form = req.form or {}
                    except AttributeError:
                        form = {}
                    try:
                        files = req.files or {}
                    except AttributeError:
                        files = {}

                    if form or files:
                        question = (form.get("question") or "").strip()
                        conv_json = form.get("conversation", "[]")
                        conversation = json.loads(conv_json) if conv_json else []

                        file = files.get("document")
                        has_file = bool(file)
                        logging.info(
                            f"parsed(multipart via req.form): q_len={len(question)}, has_file={has_file}, conv_items={len(conversation)}"
                        )

                        if file:
                            file_bytes = file.read()
                            mt = (getattr(file, "mimetype", "") or "").lower()
                            if "pdf" in mt:
                                document_text = _extract_text_from_pdf_bytes(file_bytes)
                            elif "wordprocessingml" in mt or mt == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                document_text = _extract_text_from_docx_bytes(file_bytes)
                            else:
                                document_text = "Bestandstype niet ondersteund."
                    else:
                        # 2) Fallback: toolbelt
                        body_bytes = req.get_body() or b""
                        if not body_bytes:
                            raise ValueError("Lege multipart body.")
                        if not MultipartDecoder:
                            raise ValueError("requests-toolbelt niet aanwezig voor multipart fallback.")

                        question, conversation, document_text, dbg = _parse_multipart_with_toolbelt(
                            req.headers.get("Content-Type", ""), body_bytes
                        )
                        used_toolbelt = True
                        logging.info(
                            f"parsed(multipart via toolbelt): q_len={len(question)}, doc_len={len(document_text)}, conv_items={len(conversation)}"
                        )

                except Exception as mpe:
                    logging.exception(f"Multipart parse error: {mpe}")
                    return func.HttpResponse(
                        "Ongeldig verzoek (parse error).",
                        status_code=400,
                        mimetype="text/plain",
                    )

            else:
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

                logging.info(
                    f"parsed(json): question_len={len(question)}, has_doc_text={bool(document_text)}, "
                    f"conv_items={len(conversation)}"
                )
        except Exception as pe:
            logging.exception(f"Invoer parse error outer: {pe}")
            return func.HttpResponse(
                "Ongeldig verzoek (parse error).",
                status_code=400,
                mimetype="text/plain",
            )

        # --- knip document_text af op MAX_DOC_CHARS ---
        if document_text and len(document_text) > MAX_DOC_CHARS:
            logging.info(f"document_text truncated from {len(document_text)} to {MAX_DOC_CHARS}")
            document_text = document_text[:MAX_DOC_CHARS]

        # ---- build PF payload ----
        payload = {
            "chat_input": question or "Analyseer het bijgevoegde document.",
            "chat_history": format_history_for_prompt_flow(conversation),
            "document_text": document_text or "",
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
                chat_output = json.dumps(pf_data)[:4000]
            return func.HttpResponse(
                body=json.dumps({"chat_output": chat_output}),
                status_code=200,
                mimetype="application/json",
            )
        except ValueError:
            txt = body_text.strip()
            if txt[:1] == "<" or txt.lower().startswith("<!doctype") or "html" in txt[:200].lower():
                logging.error("PF returned HTML; vermoedelijk auth/endpoint issue.")
                return func.HttpResponse(
                    "AI-dienst gaf HTML terug (mogelijk auth/endpoint fout).",
                    status_code=502,
                    mimetype="text/plain",
                )
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
