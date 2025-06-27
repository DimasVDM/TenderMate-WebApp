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
    logging.info('Python HTTP trigger function processed a request.')

    prompt_flow_url = os.environ.get("PROMPT_FLOW_URL")
    prompt_flow_api_key = os.environ.get("PROMPT_FLOW_API_KEY")

    if not prompt_flow_url or not prompt_flow_api_key:
        return func.HttpResponse("Server configuration error.", status_code=500)

    try:
        form = req.form
        question = form.get('question')
        
        # We ontvangen nu een simpele conversatielijst
        conversation_json = form.get('conversation', '[]')
        conversation = json.loads(conversation_json)
        
        document_text = ""
        file = req.files.get('document')
        if file:
            # ... (de logica voor bestandsverwerking blijft hetzelfde) ...
            file_bytes = file.read()
            file_stream = io.BytesIO(file_bytes)
            if file.mimetype == 'application/pdf':
                reader = PdfReader(file_stream)
                document_text = "\n".join([page.extract_text() for page in reader.pages])
            elif file.mimetype == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                doc = docx.Document(file_stream)
                document_text = "\n".join([para.text for para in doc.paragraphs])
            else:
                document_text = "Bestandstype niet ondersteund."

    except Exception as e:
        logging.error(f"Fout bij het verwerken van het verzoek: {e}")
        return func.HttpResponse("Ongeldig verzoek.", status_code=400)

    headers = {
        'Authorization': f'Bearer {prompt_flow_api_key}',
        'Content-Type': 'application/json'
    }
    
    # Gebruik de hulpfunctie om de geschiedenis correct te formatteren
    prompt_flow_history = format_history_for_prompt_flow(conversation)
    
    payload = {
        'chat_input': question,
        'chat_history': prompt_flow_history,
        'document_text': document_text
    }

    try:
        response = requests.post(prompt_flow_url, headers=headers, json=payload)
        response.raise_for_status()
        
        # --- AANGEPASTE OUTPUT ---
        # We sturen alleen het antwoord terug. De frontend beheert de geschiedenis.
        pf_data = response.json()
        return_payload = {"chat_output": pf_data.get("chat_output")}
        
        return func.HttpResponse(
            body=json.dumps(return_payload),
            status_code=200,
            mimetype="application/json"
        )
        # -------------------------
    except requests.exceptions.RequestException as e:
        # Dit is de error die je in de terminal ziet!
        logging.error(f"Fout bij het aanroepen van Prompt Flow: {e}")
        return func.HttpResponse("Fout in communicatie met de AI-dienst.", status_code=502)
