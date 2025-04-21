# preprocess.py - Preprocessing pipeline
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required NLTK datasets are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Define stopwords
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)             # Remove non-letters
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def preprocess_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            tweet = json.loads(line)
            if 'text' in tweet:
                tokens = clean_text(tweet['text'])
                if tokens:
                    json.dump({"tokens": tokens}, outfile)
                    outfile.write("\n")

if __name__ == "__main__":
    preprocess_file("data/raw_tweets.jsonl", "data/cleaned_tweets.jsonl")
