
import json
import pickle
import random

# Load trained AI model
with open("chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load chatbot responses
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Create response dictionary
responses = {}

for intent in data["intents"]:
    responses[intent["tag"]] = intent["responses"]

print("================================")
print("   AI CUSTOMER SUPPORT CHATBOT")
print("================================")
print("Bot: Hello! How can I help you?")
print("Type 'bye' to exit.")

while True:

    user_message = input("You: ").strip()

    if not user_message:
        print("Bot: Please type a message.")
        continue

    if user_message.lower() in ["bye", "goodbye", "exit"]:
        print("Bot: Thank you for contacting us!")
        break

    # Predict customer's intent
    intent = model.predict([user_message])[0]

    # Get model confidence
    confidence = max(model.predict_proba([user_message])[0])

    print("Predicted Intent:", intent)
    print("Confidence:", round(confidence * 100, 2), "%")

    if confidence < 0.35:
        print("Bot: Sorry, I didn't understand. Could you rephrase?")
    else:
        answer = random.choice(responses[intent])
        print("Bot:", answer)
