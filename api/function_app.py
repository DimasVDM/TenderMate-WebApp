# function_app.py (definitieve versie met handmatige parser)

import azure.functions as func
import logging
import os
import requests
import json
import io
import traceback
import zipfile
import re

from pypdf import PdfReader
from requests.exceptions import ReadTimeout
import docx
from docx.opc.exceptions import PackageNotFoundError
from requests_toolbelt.multipart.decoder import MultipartDecoder

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# --- Helpers blijven ongewijzigd ---
def format_history_for_prompt_flow(conversation):
    chat_history = []
    for i in range(0, len(conversation), 2):
        if (i + 1 < len(conversation) and isinstance(conversation[i], dict) and isinstance(conversation[i + 1], dict) and conversation[i].get("role") == "user" and conversation[i + 1].get("role") == "bot"):
            chat_history.append({"inputs": {"chat_input": conversation[i].get("content", "")},"outputs": {"chat_output": conversation[i + 1].get("content", "")},})
    return chat_history

def extract_chat_output_from_json(pf_data: dict) -> str:
    if pf_data is None: return ""
    for key in ("chat_output", "output", "result", "answer", "content"):
        val = pf_data.get(key)
        if isinstance(val, str) and val.strip(): return val.strip()
    return ""

def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def _read_docx_text(file_bytes: bytes) -> str:
    try:
        return "\n".join([p.text for p in docx.Document(io.BytesIO(file_bytes)).paragraphs])
    except (PackageNotFoundError, zipfile.BadZipFile) as e:
        raise e

@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        if req.method == "GET":
            return func.HttpResponse("OK - TalkToTenderBot v10 (manual_parser)", status_code=200, mimetype="text/plain")

        prompt_flow_url = os.environ.get("PROMPT_FLOW_URL")
        prompt_flow_api_key = os.environ.get("PROMPT_FLOW_API_KEY")

        if not prompt_flow_url or not prompt_flow_api_key:
            return func.HttpResponse("Server configuration error.", status_code=500, mimetype="text/plain")

        question = ""
        conversation = []
        document_text = ""
        
        content_type = req.headers.get("Content-Type", "").lower()

        # --- HANDMATIGE PARSING LOGICA OM DE BUG TE OMZEILEN ---
        try:
            if 'multipart/form-data' in content_type:
                # Lees de rauwe body, negeer de buggy req.form/req.files
                body_bytes = req.get_body()
                decoder = MultipartDecoder(body_bytes, content_type)

                # Loop door de onderdelen van het verzoek
                for part in decoder.parts:
                    disposition = part.headers.get(b'Content-Disposition', b'').decode()
                    # Zoek naar het bestandsdeel
                    if 'filename=' in disposition:
                        file_bytes = part.content
                        # Bepaal het bestandstype (MIME type is betrouwbaarder dan bestandsnaam)
                        part_content_type = part.headers.get(b'Content-Type', b'').decode().lower()
                        
                        if 'pdf' in part_content_type:
                            document_text = _read_pdf_text(file_bytes)
                        elif 'wordprocessingml' in part_content_type:
                            document_text = _read_docx_text(file_bytes)
                        else:
                             return func.HttpResponse("Bestandstype niet ondersteund. Upload een PDF of DOCX.", status_code=415, mimetype="text/plain")
                    else:
                        # Zoek naar de tekstvelden
                        name_match = re.search(r'name="([^"]+)"', disposition)
                        if name_match:
                            name = name_match.group(1)
                            if name == 'question':
                                question = part.text
                            elif name == 'conversation':
                                conversation = json.loads(part.text)
            
            elif 'application/json' in content_type:
                body = req.get_json()
                question = (body.get("question") or "").strip()
                conversation = body.get("conversation", [])
            else:
                return func.HttpResponse(f"Content-Type '{content_type}' niet ondersteund.", status_code=415, mimetype="text/plain")

        except (PackageNotFoundError, zipfile.BadZipFile) as e:
            logging.exception(f"DOCX is beschadigd: {e}")
            return func.HttpResponse("Kon DOCX niet lezen (mogelijk beschadigd).", status_code=422, mimetype="text/plain")
        except Exception as e:
            logging.exception(f"Fout bij parsen van input: {e}")
            return func.HttpResponse("Ongeldig verzoek (parse error).", status_code=400, mimetype="text/plain")
        # --- EINDE PARSING LOGICA ---

        payload = {
            "chat_input": question or "Analyseer het bijgevoegde document.",
            "chat_history": format_history_for_prompt_flow(conversation),
            "document_text": document_text or "",
        }
        headers = { "Authorization": f"Bearer {prompt_flow_api_key}", "Content-Type": "application/json" }

        # De rest van de code blijft hetzelfde...
        resp = requests.post(prompt_flow_url, headers=headers, json=payload, timeout=240)
        resp.raise_for_status()
        
        pf_data = resp.json()
        chat_output = extract_chat_output_from_json(pf_data) or "Kon geen antwoord extraheren."
        return func.HttpResponse(body=json.dumps({"chat_output": chat_output}), status_code=200, mimetype="application/json")

    except Exception as e:
        logging.error("ONVERWACHTE ALGEHELE FOUT", exc_info=True)
        return func.HttpResponse(f"Onverwachte serverfout: {repr(e)}", status_code=500, mimetype="text/plain")