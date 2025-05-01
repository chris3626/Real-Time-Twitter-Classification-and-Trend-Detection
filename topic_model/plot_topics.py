import json
import matplotlib.pyplot as plt
import os

HISTORY_JSON = "topic_history.json"
OUTPUT_DIR = "visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_topic_history():
    with open(HISTORY_JSON, "r") as f:
        return json.load(f)

def plot_topic_drift_table(topic_history):
    timestamps = list(topic_history.keys())
    num_topics = len(next(iter(topic_history.values())))

    for topic_id in range(num_topics):
        fig, ax = plt.subplots(figsize=(12, 0.5 * len(timestamps)))
        ax.axis('off')
        table_data = []

        for ts in timestamps:
            words = topic_history[ts][str(topic_id)]
            table_data.append([ts, ", ".join(words)])

        # Create table with matplotlib
        table = plt.table(
            cellText=table_data,
            colLabels=["Time", f"Top Words for Topic {topic_id}"],
            loc="center",
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        plt.title(f"Topic {topic_id} Drift Over Time", fontsize=10)
        plt.savefig(os.path.join(OUTPUT_DIR, f"topic_{topic_id}_table.png"))
        plt.close()

if __name__ == "__main__":
    topic_history = load_topic_history()
    plot_topic_drift_table(topic_history)
    print("Saved readable topic drift visualizations in 'visuals/' folder.")
