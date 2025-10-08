import azure.functions as func
import logging
import os
import requests
import json
import io
import traceback
import re
import zipfile

from pypdf import PdfReader
from requests.exceptions import ReadTimeout
import docx
from docx.opc.exceptions import PackageNotFoundError

from requests_toolbelt.multipart.decoder import MultipartDecoder

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

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


def _extract_filename_from_cd(content_disposition: str) -> str:
    """
    Probeer filename uit Content-Disposition te halen.
    """
    if not content_disposition:
        return ""
    # filename="something.docx" of filename=something.docx
    m = re.search(r'filename\*?="?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
    return (m.group(1) if m else "").strip()


def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def _read_docx_text(file_bytes: bytes) -> str:
    """
    Lees DOCX tekst; gooi specifieke fouten door zodat we nette 422 kunnen teruggeven.
    """
    try:
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in d.paragraphs])
    except PackageNotFoundError as e:
        # Geen geldig OPC-package (geen .docx)
        raise e
    except zipfile.BadZipFile as e:
        # Beschadigd .docx (zip container kapot)
        raise e
    except Exception as e:
        # Overige leesfouten
        raise e
# -------------------------


@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # --- health / version ---
        if req.method == "GET":
            return func.HttpResponse("OK - TalkToTenderBot v7", status_code=200, mimetype="text/plain")

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
        logging.info(f"PFlow URL present={bool(prompt_flow_url)}, key present={bool(prompt_flow_api_key)}")
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

        ct = (req.headers.get("Content-Type") or "").lower()
        if "multipart/form-data" in ct:
            logging.info("parse-path=multipart")

            body_bytes = req.get_body() or b""
            # Decoder zelf haalt de boundary uit Content-Type
            try:
                dec = MultipartDecoder(body_bytes, ct)
            except Exception as e:
                logging.exception("Multipart decode error")
                return func.HttpResponse("Ongeldig multipart-verzoek (decode fout).", status_code=400, mimetype="text/plain")

            text_fields = {}
            file_part = None
            file_mt = ""
            file_name = ""

            # Lees alle parts
            for p in dec.parts:
                cd = p.headers.get(b'Content-Disposition', b'').decode(errors="ignore")
                ctype = p.headers.get(b'Content-Type', b'').decode(errors="ignore").lower()

                if 'filename=' in cd:  # file
                    file_part = p
                    file_mt = ctype or "application/octet-stream"
                    file_name = _extract_filename_from_cd(cd)
                else:
                    # text veld
                    name = ""
                    m = re.search(r'name="([^"]+)"', cd or "", flags=re.IGNORECASE)
                    if m:
                        name = m.group(1)
                    if name:
                        text_fields[name] = p.text

            question = (text_fields.get("question") or "").strip()
            conv_json = text_fields.get("conversation") or "[]"
            try:
                conversation = json.loads(conv_json) if conv_json else []
            except Exception:
                conversation = []

            has_file = file_part is not None
            logging.info(
                f"parsed(multipart): question_len={len(question)}, has_file={has_file}, conv_items={len(conversation)}, "
                f"file_ct={file_mt}, file_name={file_name}"
            )

            # Bestandsverwerking in EIGEN try/except (zodat parse-fouten hier niet 400 opleveren)
            if has_file:
                file_bytes = file_part.content or b""
                ext = os.path.splitext(file_name or "")[1].lower()

                try:
                    if ("pdf" in file_mt) or (ext == ".pdf"):
                        document_text = _read_pdf_text(file_bytes)
                    elif ("wordprocessingml" in file_mt) or (ext == ".docx"):
                        document_text = _read_docx_text(file_bytes)
                    else:
                        # niet-ondersteund type
                        return func.HttpResponse(
                            "Bestandstype niet ondersteund. Upload een PDF of DOCX.",
                            status_code=415,
                            mimetype="text/plain",
                        )
                except (PackageNotFoundError, zipfile.BadZipFile):
                    logging.exception("DOCX container/zip fout")
                    return func.HttpResponse(
                        "Kon DOCX niet lezen (mogelijk beschadigd of geen geldig .docx-bestand).",
                        status_code=422,
                        mimetype="text/plain",
                    )
                except Exception as e:
                    logging.exception("Onbekende fout bij document lezen")
                    return func.HttpResponse(
                        f"Fout bij lezen van document: {repr(e)}",
                        status_code=422,
                        mimetype="text/plain",
                    )

        else:
            logging.info("parse-path=json")
            body = {}
            try:
                body = req.get_json()
            except ValueError:
                # fallback: probeer zelf te decoden
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
            return func.HttpResponse(json.dumps({"error": body_text}), status_code=200, mimetype="application/json")
        return func.HttpResponse(body_text, status_code=500, mimetype="text/plain")
