import pickle
import pandas as pd

N = 1000  #number of rows to load for preview

with open("data/tfidf_vectors.pkl", "rb") as f:
    X, vectorizer = pickle.load(f)

#converts only the first N rows to a DataFrame
X_sample = X[:N].toarray()
feature_names = vectorizer.get_feature_names_out()
df = pd.DataFrame(X_sample, columns=feature_names)
print(df.head())
df.to_csv("data/tfidf_sample.csv", index=False)
