from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

VISUALIZATION_DIR = (
    BASE_DIR / "visualizations"
)

VISUALIZATION_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# Load category results
# ==================================================

df = pd.read_csv(
    RESULTS_DIR / "category_performance.csv"
)


# ==================================================
# Sort by accuracy
# ==================================================

df = df.sort_values(
    by="accuracy_percent",
    ascending=True
)


# ==================================================
# Plot
# ==================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    df["category"],
    df["accuracy_percent"]
)

plt.xlabel(
    "Accuracy (%)"
)

plt.ylabel(
    "Test Category"
)

plt.title(
    "Gemini Performance by Test Category"
)

plt.xlim(
    0,
    100
)


for index, value in enumerate(
    df["accuracy_percent"]
):

    plt.text(
        value + 1,
        index,
        f"{value:.0f}%"
    )


plt.tight_layout()


output_file = (
    VISUALIZATION_DIR /
    "category_performance.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "Saved:",
    output_file
)