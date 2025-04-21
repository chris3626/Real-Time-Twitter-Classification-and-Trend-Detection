import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

INPUT_FILE = "data/cleaned_tweets.jsonl"

with open(INPUT_FILE, 'r') as f:
    docs = [" ".join(json.loads(line)["tokens"]) for line in f]

#initialize and fit TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(docs)
#saving vectorized data and vectorizer
with open("data/tfidf_vectors.pkl", "wb") as f:
    pickle.dump((X, vectorizer), f)
