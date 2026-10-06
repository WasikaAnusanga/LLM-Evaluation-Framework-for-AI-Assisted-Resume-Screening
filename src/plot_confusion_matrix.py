from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


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
# Load results
# ==================================================

df = pd.read_csv(
    RESULTS_DIR / "model_outputs.csv"
)

df = df[
    df["predicted_decision"].notna()
].copy()


labels = [
    "Shortlist",
    "Review",
    "Reject"
]


# ==================================================
# Confusion matrix
# ==================================================

cm = confusion_matrix(
    df["expected_decision"],
    df["predicted_decision"],
    labels=labels
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)

display.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Gemini Resume Screening Confusion Matrix"
)

plt.tight_layout()


output_file = (
    VISUALIZATION_DIR /
    "confusion_matrix.png"
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