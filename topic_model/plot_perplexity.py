# Sujal More
# 1001926735

import matplotlib.pyplot as plt
import json
import os

# Load perplexity log
with open("perplexity_log.json", "r") as f:
    perplexity_data = json.load(f)

# Sort by batch order
batches = list(perplexity_data.keys())
scores = [perplexity_data[batch] for batch in batches]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(batches, scores, marker='o', linewidth=2)
plt.title("LDA Perplexity Over Tweet Batches")
plt.xlabel("Batch")
plt.ylabel("Perplexity (lower is better)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save to file
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/perplexity_over_time.png")
plt.show()
