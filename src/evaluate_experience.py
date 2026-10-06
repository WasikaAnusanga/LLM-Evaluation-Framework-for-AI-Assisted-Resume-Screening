from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = (
    RESULTS_DIR / "model_outputs.csv"
)


# ==================================================
# Load results
# ==================================================

df = pd.read_csv(INPUT_FILE)

df = df[
    df["predicted_experience_match"].notna()
].copy()


# ==================================================
# Labels
# ==================================================

labels = [
    "High",
    "Medium",
    "Low"
]


# ==================================================
# Ground truth and predictions
# ==================================================

y_true = df[
    "expected_experience_match"
]

y_pred = df[
    "predicted_experience_match"
]


# ==================================================
# Accuracy
# ==================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


print("=" * 75)
print("EXPERIENCE MATCH EVALUATION")
print("=" * 75)

print(
    "\nTotal cases:",
    len(df)
)

print(
    f"Overall Accuracy: "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


# ==================================================
# Per-class metrics
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

    "class":
        labels,

    "precision":
        precision,

    "recall":
        recall,

    "f1_score":
        f1,

    "support":
        support
})


print("\n" + "=" * 75)
print("PER-CLASS METRICS")
print("=" * 75)

print(
    metrics_df.to_string(
        index=False
    )
)


# ==================================================
# Macro metrics
# ==================================================

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)


print("\n" + "=" * 75)
print("MACRO METRICS")
print("=" * 75)

print(
    f"Macro Precision: "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall:    "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1:        "
    f"{macro_f1:.4f}"
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
        "Actual High",
        "Actual Medium",
        "Actual Low"
    ],

    columns=[
        "Predicted High",
        "Predicted Medium",
        "Predicted Low"
    ]
)


print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(confusion_df)


# ==================================================
# Full classification report
# ==================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


# ==================================================
# Incorrect experience predictions
# ==================================================

df["experience_correct"] = (
    df["expected_experience_match"]
    ==
    df["predicted_experience_match"]
)


incorrect = df[
    df["experience_correct"] == False
][
    [
        "test_id",
        "category",
        "expected_experience_match",
        "predicted_experience_match",
        "reason"
    ]
]


print("\n" + "=" * 75)
print("INCORRECT EXPERIENCE PREDICTIONS")
print("=" * 75)


if len(incorrect) == 0:

    print(
        "No incorrect experience predictions."
    )

else:

    print(
        incorrect.to_string(
            index=False
        )
    )


# ==================================================
# Save outputs
# ==================================================

metrics_output = (
    RESULTS_DIR /
    "experience_metrics.csv"
)

confusion_output = (
    RESULTS_DIR /
    "experience_confusion_matrix.csv"
)

incorrect_output = (
    RESULTS_DIR /
    "incorrect_experience_predictions.csv"
)


metrics_df.to_csv(
    metrics_output,
    index=False
)

confusion_df.to_csv(
    confusion_output
)

incorrect.to_csv(
    incorrect_output,
    index=False
)


summary_df = pd.DataFrame([
    {
        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1
    }
])


summary_output = (
    RESULTS_DIR /
    "experience_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)


print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(metrics_output)
print(summary_output)
print(confusion_output)
print(incorrect_output)