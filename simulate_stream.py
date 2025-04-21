#simulate streaming from Sentiment140 dataset
import pandas as pd
import json
import time
import os

#reading the csv with correct encoding and no header
df = pd.read_csv("data/Sentiment140.csv", encoding="ISO-8859-1", header=None)
df.columns = ["sentiment", "id", "date", "query", "user", "text"]
os.makedirs("data", exist_ok=True) #only occurs if it doesnt exist
#writes all tweets to a JSONL file
with open("data/raw_tweets.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        tweet = {"text": row["text"]}
        f.write(json.dumps(tweet) + "\n")
