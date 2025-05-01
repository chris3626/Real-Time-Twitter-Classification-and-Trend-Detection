# Real-Time-Twitter-Classification-and-Trend-Detection
Real-Time-Twitter-Classification-and-Trend-Detection/
│
├── stream.py              # Collect tweets using Twitter API
├── preprocess.py          # Clean and tokenize tweets
├── vectorize.py           # Convert cleaned tokens into TF-IDF vectors
├── data/
│   ├── raw_tweets.jsonl        # Raw tweet data (1 tweet per line)
│   ├── cleaned_tweets.jsonl    # Preprocessed tokenized tweets
│   └── tfidf_vectors.pkl       # Pickled TF-IDF vectors and vectorizer

Requirements:
Python 3.8+
Install dependencies with:
nginx
Copy
Edit
pip install tweepy nltk scikit-learn

CURRENT:
Using 1.6M Tweets from https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
renamed dataset to "Sentiment140"
FUTURE:
1. Stream Live Tweets
Set your Twitter API Bearer Token as an environment variable:
bash
Copy
Edit
export TWITTER_BEARER_TOKEN="your_token"  # Mac/Linux
set TWITTER_BEARER_TOKEN="your_token"     # Windows
Then run:
bash
Copy
Edit
python stream.py
This collects tweets containing keywords like AI, climate, and education, and appends them to data/raw_tweets.jsonl.

2. Preprocess Tweets
bash
Copy
Edit
python preprocess.py
This loads each tweet, converts to lowercase, removes punctuation and URLs, and tokenizes the words using NLTK.
The result is written to data/cleaned_tweets.jsonl.

3. Vectorize with TF-IDF
bash
Copy
Edit
python vectorize.py
This reads the cleaned tokens and transforms them into TF-IDF vectors (limited to 5000 features).
Vectors and the vectorizer are saved to data/tfidf_vectors.pkl.
