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
    try:
        # ---- health check / versie check ----
        if req.method == "GET":
            # Pas dit versienummer aan bij iedere deploy om te zien of je live code draait
            return func.HttpResponse("OK - TalkToTenderBot v7", status_code=200, mimetype="text/plain")

        logging.info("TalkToTenderBot start")
        prompt_flow_url = os.environ.get("PROMPT_FLOW_URL")
        prompt_flow_api_key = os.environ.get("PROMPT_FLOW_API_KEY")
        logging.info(f"PFlow URL present={bool(prompt_flow_url)}, key present={bool(prompt_flow_api_key)}")

        if not prompt_flow_url or not prompt_flow_api_key:
            return func.HttpResponse("Server configuration error (PF URL/key missing).", status_code=500)

        # ---- invoer parsen (multipart én JSON) ----
        question = ""
        conversation = []
        document_text = ""

        try:
            ct = (req.headers.get("Content-Type") or "").lower()
            if "multipart/form-data" in ct:
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
                conversation = json.loads(conv_json)

                file = files.get("document")
                if file:
                    file_bytes = file.read()
                    file_stream = io.BytesIO(file_bytes)
                    mt = (file.mimetype or "").lower()
                    if mt == "application/pdf":
                        reader = PdfReader(file_stream)
                        parts = []
                        for p in reader.pages:
                            parts.append(p.extract_text() or "")
                        document_text = "\n".join(parts)
                    elif mt == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        doc = docx.Document(file_stream)
                        document_text = "\n".join([para.text for para in doc.paragraphs])
                    else:
                        document_text = "Bestandstype niet ondersteund."
            else:
                body = req.get_json()
                question = (body.get("question") or body.get("chat_input") or "").strip()
                conversation = body.get("conversation", [])
                document_text = body.get("document_text", "")
        except Exception as pe:
            logging.exception(f"Invoer parse error: {pe}")
            return func.HttpResponse("Ongeldig verzoek (parse error).", status_code=400, mimetype="text/plain")

        # ---- history mappen ----
        def format_history_for_prompt_flow(conv):
            mapped = []
            for i in range(0, len(conv), 2):
                if i + 1 < len(conv) and conv[i].get("role") == "user" and conv[i+1].get("role") == "bot":
                    mapped.append({
                        "inputs": {"chat_input": conv[i]["content"]},
                        "outputs": {"chat_output": conv[i+1]["content"]}
                    })
            return mapped

        payload = {
            "chat_input": question or "Analyseer het bijgevoegde document.",
            "chat_history": format_history_for_prompt_flow(conversation),
            "document_text": document_text or ""
        }
        headers = {
            "Authorization": f"Bearer {prompt_flow_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # ---- call naar PF met ruime timeout ----
        try:
            resp = requests.post(
                prompt_flow_url,
                headers=headers,
                json=payload,
                timeout=(10, 210)  # 10s connect, 210s read
            )
        except ReadTimeout:
            logging.error("PF ReadTimeout")
            return func.HttpResponse("AI-dienst duurde te lang (timeout).", status_code=504, mimetype="text/plain")
        except requests.exceptions.RequestException as e:
            logging.exception(f"PF request error: {e}")
            return func.HttpResponse(f"PF request error: {repr(e)}", status_code=502, mimetype="text/plain")

        logging.info(f"PF status={resp.status_code}")
        if resp.status_code < 200 or resp.status_code >= 300:
            # toon maximaal 500 chars van de body om HTML-spam te vermijden
            return func.HttpResponse(
                f"AI-dienst error (status {resp.status_code}): {resp.text[:500]}",
                status_code=502,
                mimetype="text/plain"
            )

        try:
            pf_data = resp.json()
        except ValueError:
            logging.error(f"PF gaf non-JSON terug: {resp.text[:400]}")
            return func.HttpResponse("AI-dienst gaf ongeldige JSON terug.", status_code=502, mimetype="text/plain")

        return func.HttpResponse(
            body=json.dumps({"chat_output": pf_data.get("chat_output")}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        # last-resort: ALLES afvangen en nooit kale 500 zonder tekst sturen
        logging.error("UNHANDLED", exc_info=True)
        return func.HttpResponse(f"Unhandled server error: {repr(e)}\n{traceback.format_exc()}", status_code=500, mimetype="text/plain")