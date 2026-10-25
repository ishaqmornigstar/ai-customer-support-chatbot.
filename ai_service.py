
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(
    api_key=API_KEY,
    timeout=20.0,
    max_retries=1
) if API_KEY else None


def generate_ai_reply(message, history=None):
    if client is None:
        return None

    conversation = []

    # Include a small amount of previous conversation
    for chat in (history or [])[-5:]:
        conversation.append({
            "role": "user",
            "content": chat["user"]
        })

        conversation.append({
            "role": "assistant",
            "content": chat["bot"]
        })

    conversation.append({
        "role": "user",
        "content": message
    })

    response = client.responses.create(
        model=MODEL,
        instructions=(
            "You are a helpful customer support chatbot. "
            "Use simple, friendly English. "
            "Keep answers concise and practical. "
            "Do not invent order statuses, refund approvals, "
            "company policies, or support ticket numbers. "
            "You cannot perform real account actions. "
            "For order tracking, ask the customer to use "
            "the order tracking feature. "
            "For unresolved issues, suggest creating "
            "a support ticket."
        ),
        input=conversation,
        max_output_tokens=250,
        store=False
    )

    return response.output_text.strip() or None
