import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as TV, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def vectorize_tokenizer_train(text, vectorizer):
  return vectorizer.fit_transform(text)

def vectorize_tokenizer_test(text, vectorizer):
  return vectorizer.transform(text)

def run_tfidf_naives_bayes(data_etl):
    X = data_etl['text']
    y = data_etl['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TV(max_features = 5000, stop_words='english')

    X_train_vec = vectorize_tokenizer_train(X_train, vectorizer)
    X_test_vec = vectorize_tokenizer_test(X_test, vectorizer)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    print("TF-IDF Model Score: ", model.score(X_test_vec, y_test))

    y_hay = model.predict(X_test_vec)
    print(classification_report(y_test, y_hay))
    print(f"Accuracy Score: {accuracy_score(y_test,y_hay)}")

    # Random Prediction
    text = [
        'Sending positive vibes and good energy to everyone starting a new chapter, whether its a new job, relationship, or project. Embrace the unknown and trust in your journey!',
        'I really believe that being president is the greatest job in the world.'
        ]

    text_NB = vectorize_tokenizer_test(text, vectorizer)
    y_predict_NB = model.predict(text_NB)
    print("Predictions:",  *text)
    print("TF-IDF Predictions:", ['Negative' if x == 0 else 'Positive' for x in y_predict_NB])

def run_count_naives_bayes(data_etl):
    X = data_etl['text']
    y = data_etl['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cvectorizer = CountVectorizer()
    X_train_vec = vectorize_tokenizer_train(X_train, cvectorizer)
    X_test_vec = vectorize_tokenizer_test(X_test, cvectorizer)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    print("CountVectorizer Model Score:", model.score(X_test_vec, y_test))

    y_hat = model.predict(X_test_vec)
    print(classification_report(y_test, y_hat))

    test_texts = [
        "I actually don't think anybody is innocent of anything.",
        "I don't have any regrets about my previous life.",
        "Life is good"
    ]
    test_vec = vectorize_tokenizer_test(test_texts, cvectorizer)
    predictions = model.predict(test_vec)
    print("Predictions: ", test_texts)
    print("CountVectorizer Predictions:", ['Negative' if x == 0 else 'Positive' for x in predictions])