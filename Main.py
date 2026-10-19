from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import pickle
import random
import re

app = FastAPI(title="AI Customer Support Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500",
                   "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

with open("chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

responses = {
    intent["tag"]: intent["responses"]
    for intent in data["intents"]
}

# Words the model learned during training
vocabulary = model.named_steps["tfidf"].vocabulary_

class ChatRequest(BaseModel):
    message: str


BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        return {
            "reply": "Please enter a message.",
            "intent": "empty",
            "confidence": 0.0
        }

    # Handle clear goodbye commands
    if message.lower() in ["bye", "goodbye", "exit"]:
        return {
            "reply": "Thank you for contacting us!",
            "intent": "goodbye",
            "confidence": 1.0
        }

    # Convert message to text features
    vectorizer = model.named_steps["tfidf"]
    features = vectorizer.transform([message])

    # Reject messages with no known text features
    if features.nnz == 0:
        return {
            "reply": "Sorry, I didn't understand. Could you rephrase your customer-support question?",
            "intent": "unknown",
            "confidence": 0.0
        }

    probabilities = model.predict_proba([message])[0]
    confidence = float(max(probabilities))
    intent = model.classes_[probabilities.argmax()]

    # Confidence threshold
    if confidence < 0.35:
        return {
            "reply": "I'm not sure I understood. Please rephrase your question or contact customer support.",
            "intent": "unknown",
            "confidence": round(confidence, 2)
        }

    reply = random.choice(responses[intent])

    return {
        "reply": reply,
        "intent": intent,
        "confidence": round(confidence, 2)
    }