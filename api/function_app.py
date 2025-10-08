# function_app.py (verbeterde versie)

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

# -------- Helpers (deze blijven ongewijzigd) --------
def format_history_for_prompt_flow(conversation):
    chat_history = []
    # ... (deze functie blijft exact hetzelfde) ...
    for i in range(0, len(conversation), 2):
        if (i + 1 < len(conversation) and isinstance(conversation[i], dict) and isinstance(conversation[i + 1], dict) and conversation[i].get("role") == "user" and conversation[i + 1].get("role") == "bot"):
            chat_history.append({"inputs": {"chat_input": conversation[i].get("content", "")},"outputs": {"chat_output": conversation[i + 1].get("content", "")},})
    return chat_history

def extract_chat_output_from_json(pf_data: dict) -> str:
    # ... (deze functie blijft exact hetzelfde) ...
    if pf_data is None: return ""
    for key in ("chat_output", "output", "result", "answer", "content"):
        val = pf_data.get(key)
        if isinstance(val, str) and val.strip(): return val.strip()
    try:
        choices = pf_data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}; content = msg.get("content")
            if isinstance(content, str) and content.strip(): return content.strip()
    except Exception: pass
    try:
        output = pf_data.get("output")
        if isinstance(output, dict):
            msg = output.get("message") or {}; content = msg.get("content")
            if isinstance(content, str) and content.strip(): return content.strip()
    except Exception: pass
    return ""

def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def _read_docx_text(file_bytes: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in d.paragraphs])
    except (PackageNotFoundError, zipfile.BadZipFile) as e:
        raise e # Gooi specifieke fouten door voor betere error handling
# -------------------------


@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        if req.method == "GET":
            return func.HttpResponse("OK - TalkToTenderBot v8", status_code=200, mimetype="text/plain")

        logging.info("TalkToTenderBot POST request received.")

        prompt_flow_url = os.environ.get("PROMPT_FLOW_URL")
        prompt_flow_api_key = os.environ.get("PROMPT_FLOW_API_KEY")

        if not prompt_flow_url or not prompt_flow_api_key:
            logging.error("Server configuration error: PROMPT_FLOW_URL or PROMPT_FLOW_API_KEY is missing.")
            return func.HttpResponse("Server configuration error.", status_code=500, mimetype="text/plain")

        # ---- NIEUWE, VEREENVOUDIGDE PARSING ----
        question = ""
        conversation = []
        document_text = ""
        
        try:
            # Haal tekstvelden direct uit het formulier
            question = (req.form.get("question") or "").strip()
            conv_json = req.form.get("conversation", "[]")
            conversation = json.loads(conv_json) if conv_json else []

            # Haal het bestand op via de ingebouwde 'files' methode
            file = req.files.get('document')

            if file:
                logging.info(f"File received: {file.filename}, type: {file.mimetype}, size: {file.length}")
                
                file_bytes = file.read()

                if file.mimetype == 'application/pdf':
                    document_text = _read_pdf_text(file_bytes)
                elif file.mimetype == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    document_text = _read_docx_text(file_bytes)
                else:
                    return func.HttpResponse("Bestandstype niet ondersteund. Upload een PDF of DOCX.", status_code=415, mimetype="text/plain")

        except (PackageNotFoundError, zipfile.BadZipFile):
            logging.exception("DOCX container/zip fout")
            return func.HttpResponse("Kon DOCX niet lezen (mogelijk beschadigd of geen geldig .docx-bestand).", status_code=422, mimetype="text/plain")
        except Exception as e:
            # Dit vangt nu alle parse-fouten, inclusief ongeldige JSON in 'conversation'
            logging.exception("Fout bij het parsen van de input request.")
            return func.HttpResponse(f"Ongeldig verzoek (parse error).", status_code=400, mimetype="text/plain")
        # ----------------------------------------

        # ---- build PF payload (ongewijzigd) ----
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

        # ---- call Prompt Flow (ongewijzigd) ----
        # ... (de rest van je functie vanaf de 'requests.post' call blijft exact hetzelfde) ...
        try:
            resp = requests.post(prompt_flow_url, headers=headers, json=payload, timeout=(10, 210))
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
            return func.HttpResponse(f"AI-dienst error (status {resp.status_code}): {snippet}", status_code=502, mimetype="text/plain")

        try:
            pf_data = resp.json()
            chat_output = extract_chat_output_from_json(pf_data)
            if not chat_output:
                chat_output = f"Kon geen duidelijk antwoord extraheren uit de AI-response: {json.dumps(pf_data)[:1000]}"
            return func.HttpResponse(body=json.dumps({"chat_output": chat_output}), status_code=200, mimetype="application/json")
        except ValueError:
            logging.error("PF returned non-JSON response.")
            return func.HttpResponse("AI-dienst gaf een onverwacht (niet-JSON) antwoord.", status_code=502, mimetype="text/plain")

    except Exception as e:
        logging.error("UNHANDLED global exception", exc_info=True)
        return func.HttpResponse(f"Unhandled server error: {repr(e)}\n{traceback.format_exc()}", status_code=500, mimetype="text/plain")