import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get data from classification report
df = pd.read_csv("example/example_classification_report.csv")
aircraft_df = df.iloc[:20]
summary_df = df.iloc[20:]

# Display summary
print(f"Accuracy:            {summary_df.iloc[0]["precision"]:.4f}")
print()
print(f"Macro precision:     {summary_df.iloc[1]["precision"]:.4f}")
print(f"Macro recall:        {summary_df.iloc[1]["recall"]:.4f}")
print(f"Macro f1:            {summary_df.iloc[1]["f1-score"]:.4f}")
print()
print(f"Weighted precision:  {summary_df.iloc[2]["precision"]:.4f}")
print(f"Weighted recall:     {summary_df.iloc[2]["recall"]:.4f}")
print(f"Weighted f1:         {summary_df.iloc[2]["f1-score"]:.4f}")

# Extract aircraft data
aircraft = aircraft_df["aircraft"].tolist()
precision = aircraft_df["precision"].to_numpy()
recall = aircraft_df["recall"].to_numpy()
f1 = aircraft_df["f1-score"].to_numpy()
support = aircraft_df["support"].to_numpy()

# Set up grouped bars
x = np.arange(len(aircraft))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x - width, precision, width, label="Precision")
ax.bar(x, recall, width, label="Recall")
ax.bar(x + width, f1, width, label="F1-score")

# Label each aircraft group with its sample count
for i, count in enumerate(support):
  ax.text(
    x[i], 1.05,
    int(count),
    ha="center",
    fontsize=8,
  )

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(aircraft, rotation=45, ha="right")
ax.set_ylim(0, 1.3)
ax.grid(axis="y", linestyle="dotted", alpha=0.5)
ax.legend()

# Save diagram
plt.savefig("classification_report_diagram.png", bbox_inches="tight")