import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string
import json
from datetime import datetime
import os

# Set up stopwords and tokenizer
stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer()

# Paths
DATA_PATH = "training.1600000.processed.noemoticon.csv"
OUTPUT_JSON = "topic_history.json"

# Load data (limited batch size for simulation)
def load_data(batch_size=5000, offset=0):
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1', header=None, skiprows=offset, nrows=batch_size)
    df.columns = ["sentiment", "id", "date", "query", "user", "text"]
    return df[["date", "text"]]

# Preprocess tweet text
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    return [t for t in tokens if t not in stop_words and t not in string.punctuation and len(t) > 2]

# Main online-simulated LDA using sklearn
def run_incremental_lda():
    topic_history = {}
    offset = 0
    batch_size = 5000
    n_topics = 5

    print("Preparing warm-up vocabulary...")
    warmup_df = load_data(batch_size=batch_size, offset=offset)
    warmup_texts = [" ".join(preprocess(t)) for t in warmup_df["text"] if t.strip() != ""]

    vectorizer = CountVectorizer()
    vectorizer.fit(warmup_texts)

    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=42)

    # Begin incremental updates
    for i in range(10):
        print(f"Processing batch {i+1}")
        df = load_data(batch_size=batch_size, offset=offset)
        offset += batch_size

        processed_texts = [" ".join(preprocess(text)) for text in df["text"] if text.strip() != ""]
        if not processed_texts:
            continue

        X = vectorizer.transform(processed_texts)
        lda.partial_fit(X)

        # Extract top words per topic
        feature_names = vectorizer.get_feature_names_out()
        ts = datetime.now().isoformat()
        topic_words = {
            i: [feature_names[idx] for idx in topic.argsort()[:-6:-1]]
            for i, topic in enumerate(lda.components_)
        }

        topic_history[ts] = topic_words

    output_dir = os.path.dirname(OUTPUT_JSON)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(topic_history, f, indent=2)

    print("Incremental LDA complete. Saved to topic_history.json")

if __name__ == "__main__":
    run_incremental_lda()
