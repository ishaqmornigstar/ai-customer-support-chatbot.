import json


with open("intents.json","r", encoding="utf-8") as file:
    data = json.load(file)


    for intent in data["intents"]:
        print("intent:",intent["tag"])
        print("questions:",intent["patterns"])
        print("responses:",intent["responses"]) 
        print("------------------------------")