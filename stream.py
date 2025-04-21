import tweepy
import json
import os
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

class TweetStreamer(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        print(tweet.text)
        os.makedirs("data", exist_ok=True)
        with open("data/raw_tweets.jsonl", "a") as f:
            f.write(json.dumps(tweet.data) + "\n")

if __name__ == "__main__":
    stream = TweetStreamer(BEARER_TOKEN)
    #define keyword-based filtering rules
    stream.add_rules(tweepy.StreamRule("AI OR climate OR education"))
    #start streaming tweets with certain fields included
    stream.filter(tweet_fields=["created_at", "lang"])
