from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ==================================================
# Project paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

input_file = RESULTS_DIR / "model_outputs.csv"


# ==================================================
# Load results
# ==================================================

df = pd.read_csv(input_file)

print("=" * 70)
print("CLASSIFICATION EVALUATION")
print("=" * 70)

print("\nTotal test cases:", len(df))


# ==================================================
# Remove unusable predictions if necessary
# ==================================================

valid_df = df[
    df["predicted_decision"].notna()
].copy()

print(
    "Cases with usable predicted decisions:",
    len(valid_df)
)


# ==================================================
# Expected and predicted labels
# ==================================================

y_true = valid_df["expected_decision"]

y_pred = valid_df["predicted_decision"]


# ==================================================
# Fixed label order
# ==================================================

labels = [
    "Shortlist",
    "Review",
    "Reject"
]


# ==================================================
# Accuracy
# ==================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\nOverall Accuracy:")
print(f"{accuracy:.4f}")
print(f"{accuracy * 100:.2f}%")


# ==================================================
# Precision, Recall and F1
# ==================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


metrics_df = pd.DataFrame({
    "class": labels,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "support": support
})


print("\n" + "=" * 70)
print("PER-CLASS METRICS")
print("=" * 70)

print(
    metrics_df.to_string(
        index=False
    )
)


# ==================================================
# Macro and weighted averages
# ==================================================

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


print("\n" + "=" * 70)
print("AVERAGE METRICS")
print("=" * 70)

print(
    f"Macro Precision: {macro_precision:.4f}"
)

print(
    f"Macro Recall:    {macro_recall:.4f}"
)

print(
    f"Macro F1:        {macro_f1:.4f}"
)

print()

print(
    f"Weighted Precision: {weighted_precision:.4f}"
)

print(
    f"Weighted Recall:    {weighted_recall:.4f}"
)

print(
    f"Weighted F1:        {weighted_f1:.4f}"
)


# ==================================================
# Confusion matrix
# ==================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

confusion_df = pd.DataFrame(
    cm,
    index=[
        "Actual Shortlist",
        "Actual Review",
        "Actual Reject"
    ],
    columns=[
        "Predicted Shortlist",
        "Predicted Review",
        "Predicted Reject"
    ]
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(confusion_df)


# ==================================================
# Classification report
# ==================================================

print("\n" + "=" * 70)
print("FULL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


# ==================================================
# Save metrics
# ==================================================

metrics_output = (
    RESULTS_DIR / "classification_metrics.csv"
)

metrics_df.to_csv(
    metrics_output,
    index=False
)


summary_df = pd.DataFrame([{
    "accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_precision": weighted_precision,
    "weighted_recall": weighted_recall,
    "weighted_f1": weighted_f1
}])

summary_output = (
    RESULTS_DIR / "classification_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)


confusion_output = (
    RESULTS_DIR / "confusion_matrix.csv"
)

confusion_df.to_csv(
    confusion_output
)


print("\nSaved:")

print(metrics_output)
print(summary_output)
print(confusion_output)