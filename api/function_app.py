import azure.functions as func
import logging
import os
import requests
import json
import io
from pypdf import PdfReader
import docx

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# --- NIEUWE HULPFUNCTIE HIER ---
# Deze functie neemt een simpele conversatielijst en zet deze om naar het 
# exacte formaat dat de Prompt Flow verwacht voor de 'chat_history' input.
def format_history_for_prompt_flow(conversation):
    chat_history = []
    # Ga door de lijst van berichten en groepeer ze per paar (gebruiker + bot)
    for i in range(0, len(conversation), 2):
        # Zorg ervoor dat we niet buiten de lijst gaan en dat de rollen kloppen
        if i + 1 < len(conversation) and conversation[i]['role'] == 'user' and conversation[i+1]['role'] == 'bot':
            user_turn = conversation[i]
            bot_turn = conversation[i+1]
            chat_history.append({
                "inputs": {"chat_input": user_turn['content']},
                "outputs": {"chat_output": bot_turn['content']}
            })
    return chat_history
# ---------------------------------

@app.route(route="TalkToTenderBot")
def TalkToTenderBot(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("TalkToTenderBot start")

    prompt_flow_url = os.environ.get("PROMPT_FLOW_URL")
    prompt_flow_api_key = os.environ.get("PROMPT_FLOW_API_KEY")
    logging.info(f"PFlow URL present={bool(prompt_flow_url)}, key present={bool(prompt_flow_api_key)}")

    if not prompt_flow_url or not prompt_flow_api_key:
        return func.HttpResponse("Server configuration error (missing PF URL/key).", status_code=500)

    # ---- Invoer robuust parsen (multipart én JSON) ----
    question = None
    conversation = []
    document_text = ""

    try:
        ct = req.headers.get('Content-Type', '')
        if 'multipart/form-data' in ct:
            try:
                form = req.form
            except AttributeError:
                form = {}
            try:
                files = req.files
            except AttributeError:
                files = {}

            question = (form.get('question') or '').strip()
            conversation_json = form.get('conversation', '[]')
            conversation = json.loads(conversation_json)

            file = files.get('document')
            if file:
                file_bytes = file.read()
                file_stream = io.BytesIO(file_bytes)
                mt = file.mimetype or ''
                if mt == 'application/pdf':
                    reader = PdfReader(file_stream)
                    parts = []
                    for p in reader.pages:
                        txt = p.extract_text() or ""
                        parts.append(txt)
                    document_text = "\n".join(parts)
                elif mt == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    doc = docx.Document(file_stream)
                    document_text = "\n".join([para.text for para in doc.paragraphs])
                else:
                    document_text = "Bestandstype niet ondersteund."
        else:
            body = req.get_json()
            question = (body.get('question') or body.get('chat_input') or '').strip()
            conversation = body.get('conversation', [])
            document_text = body.get('document_text', '')
    except Exception as e:
        logging.exception(f"Invoer parse error: {e}")
        return func.HttpResponse("Ongeldig verzoek (parse error).", status_code=400)

    # ---- History naar Prompt Flow-formaat ----
    prompt_flow_history = format_history_for_prompt_flow(conversation)

    payload = {
        "chat_input": question or "Analyseer het bijgevoegde document.",
        "chat_history": prompt_flow_history,
        "document_text": document_text or ""
    }

    headers = {
        "Authorization": f"Bearer {prompt_flow_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # ---- Call naar Prompt Flow met goede foutafhandeling ----
    try:
        resp = requests.post(prompt_flow_url, headers=headers, json=payload, timeout=30)
        status = resp.status_code
        text = resp.text
        logging.info(f"PF status={status}")

        if status < 200 or status >= 300:
            logging.error(f"PF non-2xx: body[:500]={text[:500]}")
            return func.HttpResponse(f"AI-dienst error (status {status}): {text}", status_code=502, mimetype="text/plain")

        try:
            pf_data = resp.json()
        except ValueError:
            logging.error(f"PF gaf non-JSON terug: {text[:500]}")
            return func.HttpResponse("AI-dienst gaf ongeldige JSON terug.", status_code=502)

        return func.HttpResponse(
            body=json.dumps({"chat_output": pf_data.get("chat_output")}),
            status_code=200,
            mimetype="application/json"
        )

    except requests.exceptions.RequestException as e:
        logging.exception(f"PF request error: {e}")
        # Tijdelijk de exception doorgeven om te zien wat er misgaat
        return func.HttpResponse(f"PF request error: {repr(e)}", status_code=502, mimetype="text/plain")

