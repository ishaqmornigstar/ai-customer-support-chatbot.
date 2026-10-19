
import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Step 1: Load training data
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

questions = []
labels = []

# Step 2: Collect questions and intent labels
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        questions.append(pattern)
        labels.append(intent["tag"])

# Step 3: Create machine-learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Step 4: Train model
model.fit(questions, labels)

# Step 5: Save trained model
with open("chatbot_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("AI chatbot model trained successfully!")
print("Total training questions:", len(questions))
print("Total intents:", len(set(labels)))
