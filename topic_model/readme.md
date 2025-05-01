# Topic Models---

## 📂 Files

### `topic_model.py`
Simulates real-time processing of tweets using **batch-wise incremental LDA**. It:
- Loads tweets in chunks from a CSV dataset.
- Preprocesses text (tokenization, stopword and punctuation removal).
- Uses `scikit-learn`'s `LatentDirichletAllocation` with `partial_fit` to update topics per batch.
- Logs top-5 words per topic with timestamps.
- Saves all topic evolution data to `topic_history.json`.

### `plot_topics.py`
Reads the output from `topic_model.py` and generates **timeline-style visualizations**:
- For each topic, it creates a PNG table showing how top words change over time.
- Saves all visualizations to the `visuals/` folder.

## 🛠 Requirements
Install required Python libraries:
- pandas	
- scikit-learn
- matplotlib
- nltk
```bash
import nltk
nltk.download('stopwords')
```

## ▶️ Order of Execution
1. python topic_model.py
2. python plot_topics.py
