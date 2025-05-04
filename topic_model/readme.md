# Topic Models---

## 📂 Files

### `topic_model.py`
Simulates real-time processing of tweets using **batch-wise incremental LDA**. It:
- Loads tweets in chunks from a CSV dataset.
- Preprocesses text (tokenization, stopword and punctuation removal).
- Uses `scikit-learn`'s `LatentDirichletAllocation` with `partial_fit` to update topics per batch.
- Logs top-5 words per topic with timestamps.
- Computes perplexity after each batch and saves to `perplexity_log.json`.
- Saves topic evolution data to `topic_history.json`.

### `plot_topics.py`
Reads the output from `topic_model.py` and generates **timeline-style visualizations**:
- For each topic, it creates a PNG table showing how top words change over time.
- Saves all visualizations to the `visuals/` folder.

### `plot_perplexity.py`
Plots the model’s **perplexity over tweet batches**, using `perplexity_log.json`:
- Helps evaluate how well the LDA model generalizes as it sees more data.
- Saves a line chart in the `visuals/` folder as `perplexity_over_time.png`.

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
3. python plot_perplexity.py
