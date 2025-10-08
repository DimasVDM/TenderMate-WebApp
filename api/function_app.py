# function_app.py (definitieve, robuuste versie)

import azure.functions as func
import logging
import os
import requests
import json
import io
import traceback
import zipfile

from pypdf import PdfReader
from requests.exceptions import ReadTimeout
import docx
from docx.opc.exceptions import PackageNotFoundError

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# --- De Helper functies blijven ongewijzigd ---
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
    return "" # Simpel gehouden voor duidelijkheid

def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def _read_docx_text(file_bytes: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in d.paragraphs])
    except (PackageNotFoundError, zipfile.BadZipFile) as e:
        raise e

@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        if req.method == "GET":
            return func.HttpResponse("OK - TalkToTenderBot v9 (stable)", status_code=200, mimetype="text/plain")

        logging.info("TalkToTenderBot POST request received.")

        prompt_flow_url = os.environ.get("PROMPT_FLOW_URL")
        prompt_flow_api_key = os.environ.get("PROMPT_FLOW_API_KEY")

        if not prompt_flow_url or not prompt_flow_api_key:
            logging.error("Server config error: Missing PF URL/key.")
            return func.HttpResponse("Server configuration error.", status_code=500, mimetype="text/plain")

        question = ""
        conversation = []
        document_text = ""
        
        # --- ROBUUSTE PARSING LOGICA ---
        # Bepaal het type verzoek op basis van de Content-Type header
        content_type = req.headers.get("Content-Type", "").lower()

        try:
            # Pad 1: Verzoek met bestandsupload
            if 'multipart/form-data' in content_type:
                logging.info("Parsing request as multipart/form-data.")
                question = (req.form.get("question") or "").strip()
                conv_json = req.form.get("conversation", "[]")
                conversation = json.loads(conv_json) if conv_json else []
                
                file = req.files.get('document')
                if file:
                    logging.info(f"File found: {file.filename}, type: {file.mimetype}")
                    file_bytes = file.read()
                    if file.mimetype == 'application/pdf':
                        document_text = _read_pdf_text(file_bytes)
                    elif file.mimetype == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                        document_text = _read_docx_text(file_bytes)
                    else:
                        return func.HttpResponse(f"Bestandstype '{file.mimetype}' niet ondersteund.", status_code=415, mimetype="text/plain")

            # Pad 2: Verzoek zonder bestand (pure JSON)
            elif 'application/json' in content_type:
                logging.info("Parsing request as application/json.")
                body = req.get_json()
                question = (body.get("question") or "").strip()
                conversation = body.get("conversation", [])
            
            else:
                logging.error(f"Unsupported Content-Type: {content_type}")
                return func.HttpResponse(f"Content-Type '{content_type}' niet ondersteund.", status_code=415, mimetype="text/plain")

        except (PackageNotFoundError, zipfile.BadZipFile):
            logging.exception("DOCX is beschadigd of ongeldig.")
            return func.HttpResponse("Kon DOCX niet lezen (mogelijk beschadigd).", status_code=422, mimetype="text/plain")
        except Exception as e:
            logging.exception("Fout bij het parsen van de input.")
            return func.HttpResponse("Ongeldig verzoek (parse error).", status_code=400, mimetype="text/plain")
        # --- EINDE PARSING LOGICA ---

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

        # De rest van de code voor de call naar Prompt Flow blijft ongewijzigd...
        try:
            resp = requests.post(prompt_flow_url, headers=headers, json=payload, timeout=(10, 210))
            resp.raise_for_status() # Werpt een error voor 4xx/5xx status codes
            
            pf_data = resp.json()
            chat_output = extract_chat_output_from_json(pf_data)
            if not chat_output:
                chat_output = f"Kon geen duidelijk antwoord extraheren uit de AI-response."
            return func.HttpResponse(body=json.dumps({"chat_output": chat_output}), status_code=200, mimetype="application/json")

        except requests.exceptions.HTTPError as e:
             logging.error(f"HTTP Fout van Prompt Flow: {e.response.status_code} {e.response.text[:500]}")
             return func.HttpResponse("Fout in communicatie met de AI-dienst.", status_code=502)
        except requests.exceptions.RequestException as e:
            logging.exception(f"Netwerkfout bij aanroepen Prompt Flow: {e}")
            return func.HttpResponse("Netwerkfout naar de AI-dienst.", status_code=504)
        except ValueError: # JSONDecodeError
            logging.error("Prompt Flow gaf geen geldige JSON terug.")
            return func.HttpResponse("De AI-dienst gaf een onverwacht antwoord.", status_code=502)

    except Exception as e:
        logging.error("ONVERWACHTE ALGEHELE FOUT", exc_info=True)
        return func.HttpResponse(f"Onverwachte serverfout: {repr(e)}", status_code=500, mimetype="text/plain")