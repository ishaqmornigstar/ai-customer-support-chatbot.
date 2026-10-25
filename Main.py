from ai_service import generate_ai_reply
from pathlib import Path
import json
import pickle
import random
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from Database import (
    create_database,
    track_order,
    save_chat,
    get_chat_history,
    create_ticket,
    get_tickets
)


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Customer Support Chatbot",
    version="2.0"
)

BASE_DIR = Path(__file__).resolve().parent
print("PROJECT FOLDER:", BASE_DIR)
print("HTML FILE LOCATION:", BASE_DIR / "index.html")

# Initialize SQLite database
create_database()


# ==========================================
# 2. CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# ==========================================
# 3. LOAD MACHINE LEARNING MODEL
# ==========================================

with open(BASE_DIR / "chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# 4. LOAD INTENTS AND RESPONSES
# ==========================================

with open(
    BASE_DIR / "intents.json",
    "r",
    encoding="utf-8"
) as file:
    data = json.load(file)

responses = {
    intent["tag"]: intent["responses"]
    for intent in data["intents"]
}


# ==========================================
# 5. REQUEST MODELS
# ==========================================

class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(
        default="demo",
        min_length=1,
        max_length=100
    )


class TicketRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=100
    )
    issue: str = Field(
        min_length=1,
        max_length=2000
    )


# ==========================================
# 6. HELPER FUNCTION: SAVE CHAT
# ==========================================

def chat_response(
    session_id,
    message,
    reply,
    intent,
    confidence
):
    # Save the conversation in SQLite
    save_chat(
        session_id,
        message,
        reply
    )

    return {
        "reply": reply,
        "intent": intent,
        "confidence": confidence
    }


# ==========================================
# 7. HOME PAGE
# ==========================================

@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "index.html"
    )


# ==========================================
# 8. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "AI Chatbot API is running"
    }


# ==========================================
# 9. ORDER TRACKING
# ==========================================

@app.get("/order/{order_id}")
def get_order(order_id: str):
    return track_order(order_id)


# ==========================================
# 10. CHAT HISTORY API
# ==========================================

@app.get("/history/{session_id}")
def history(session_id: str):
    return {
        "session_id": session_id,
        "history": get_chat_history(session_id)
    }


# ==========================================
# 11. CREATE SUPPORT TICKET
# ==========================================

@app.post("/ticket")
def new_ticket(request: TicketRequest):

    issue = request.issue.strip()

    if not issue:
        raise HTTPException(
            status_code=400,
            detail="Please describe your issue."
        )

    ticket_id = create_ticket(
        request.session_id,
        issue
    )

    return {
        "ticket_id": ticket_id,
        "message": "Support ticket created successfully.",
        "status": "Open"
    }


# ==========================================
# 12. VIEW SUPPORT TICKETS
# ==========================================

@app.get("/tickets/{session_id}")
def tickets(session_id: str):
    return {
        "session_id": session_id,
        "tickets": get_tickets(session_id)
    }


# ==========================================
# 13. AI CHATBOT API
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()
    session_id = request.session_id

    # --------------------------------------
    # CHECK EMPTY MESSAGE
    # --------------------------------------

    if not message:
        return {
            "reply": "Please enter a message.",
            "intent": "empty",
            "confidence": 0.0
        }

    # --------------------------------------
    # CHECK ORDER NUMBER
    # --------------------------------------

    order_match = re.search(
        r"\bORD\d+\b",
        message,
        re.IGNORECASE
    )

    if order_match:

        order_id = order_match.group().upper()

        order = track_order(order_id)

        if order["found"]:
            reply = (
                f"Your {order['product']} order "
                f"({order_id}) is currently "
                f"{order['status']}."
            )
        else:
            reply = (
                "Sorry, I couldn't find that order "
                "number. Please check and try again."
            )

        return chat_response(
            session_id,
            message,
            reply,
            "order_tracking",
            1.0
        )

    # --------------------------------------
    # HANDLE GOODBYE
    # --------------------------------------

    if message.lower() in [
        "bye",
        "goodbye",
        "exit"
    ]:

        return chat_response(
            session_id,
            message,
            "Thank you for contacting us! Goodbye!",
            "goodbye",
            1.0
        )

    # --------------------------------------
    # TRANSFORM MESSAGE INTO FEATURES
    # --------------------------------------

    vectorizer = model.named_steps["tfidf"]
    features = vectorizer.transform([message])

    if features.nnz == 0:

        return chat_response(
            session_id,
            message,
            (
                "Sorry, I didn't understand your "
                "question. Could you please rephrase?"
            ),
            "unknown",
            0.0
        )

    # --------------------------------------
    # PREDICT INTENT
    # --------------------------------------

    probabilities = model.predict_proba(
        [message]
    )[0]

    confidence = float(max(probabilities))

    intent = model.classes_[
        probabilities.argmax()
    ]

    # --------------------------------------
    # LOW CONFIDENCE HANDLING
    # --------------------------------------

    if confidence < 0.35:

        return chat_response(
            session_id,
            message,
            (
                "I'm not sure I understood. "
                "Please rephrase your question "
                "or contact customer support."
            ),
            "unknown",
            round(confidence, 2)
        )


    # --------------------------------------
    # DAY 8: GENERATIVE AI RESPONSE
    # --------------------------------------

    ai_reply = None

    try:
        previous_chats = get_chat_history(session_id)

        ai_reply = generate_ai_reply(
            message,
            previous_chats
        )

    except Exception as error:
        print("AI API error:", type(error).__name__)

    # Use AI response if available
    if ai_reply:
        return chat_response(
            session_id,
            message,
            ai_reply,
            "generative_ai",
            1.0
        )

    # Fall back to existing ML responses
    if intent in responses and responses[intent]:
        reply = random.choice(responses[intent])
    else:
        reply = (
            "Sorry, I don't have an answer "
            "for that question yet."
        )

    return chat_response(
        session_id,
        message,
        reply,
        str(intent),
        round(confidence, 2)
    )


    # --------------------------------------
    # SAVE AND RETURN CHAT RESPONSE
    # --------------------------------------

    return chat_response(
        session_id,
        message,
        reply,
        str(intent),
        round(confidence, 2)
    )
