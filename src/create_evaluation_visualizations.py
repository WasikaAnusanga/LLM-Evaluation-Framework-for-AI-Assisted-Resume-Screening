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
# 1. Overall Classification Metrics
# ==================================================

classification_summary = pd.read_csv(
    RESULTS_DIR /
    "classification_summary.csv"
)

row = classification_summary.iloc[0]


classification_metrics = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],

    "Score": [
        row["accuracy"] * 100,
        row["macro_precision"] * 100,
        row["macro_recall"] * 100,
        row["macro_f1"] * 100
    ]
})


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    classification_metrics["Metric"],
    classification_metrics["Score"]
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Overall Resume Screening Classification Performance"
)

plt.ylim(
    0,
    100
)


for index, value in enumerate(
    classification_metrics["Score"]
):

    plt.text(
        index,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )


plt.tight_layout()

classification_output = (
    VISUALIZATION_DIR /
    "overall_classification_metrics.png"
)

plt.savefig(
    classification_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 2. Skill Identification Performance
# ==================================================

skill_df = pd.read_csv(
    RESULTS_DIR /
    "skill_metrics_summary.csv"
)


skill_plot = pd.DataFrame({

    "Metric": [
        "Matched Precision",
        "Matched Recall",
        "Matched F1",
        "Missing Precision",
        "Missing Recall",
        "Missing F1"
    ],

    "Score": [
        skill_df.loc[
            skill_df["metric_type"]
            == "Matched Skills",
            "precision"
        ].iloc[0] * 100,

        skill_df.loc[
            skill_df["metric_type"]
            == "Matched Skills",
            "recall"
        ].iloc[0] * 100,

        skill_df.loc[
            skill_df["metric_type"]
            == "Matched Skills",
            "f1_score"
        ].iloc[0] * 100,

        skill_df.loc[
            skill_df["metric_type"]
            == "Missing Skills",
            "precision"
        ].iloc[0] * 100,

        skill_df.loc[
            skill_df["metric_type"]
            == "Missing Skills",
            "recall"
        ].iloc[0] * 100,

        skill_df.loc[
            skill_df["metric_type"]
            == "Missing Skills",
            "f1_score"
        ].iloc[0] * 100
    ]
})


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    skill_plot["Metric"],
    skill_plot["Score"]
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Skill Identification Performance"
)

plt.ylim(
    0,
    105
)

plt.xticks(
    rotation=25,
    ha="right"
)


for index, value in enumerate(
    skill_plot["Score"]
):

    plt.text(
        index,
        value + 0.5,
        f"{value:.1f}%",
        ha="center"
    )


plt.tight_layout()

skill_output = (
    VISUALIZATION_DIR /
    "skill_performance.png"
)

plt.savefig(
    skill_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 3. Reliability Metrics
# ==================================================

reliability_df = pd.read_csv(
    RESULTS_DIR /
    "reliability_summary.csv"
)

reliability = reliability_df.iloc[0]


reliability_plot = pd.DataFrame({

    "Metric": [
        "JSON Validity",
        "Schema Compliance",
        "API Success",
        "Fairness Decision\nConsistency",
        "Fairness Full Output\nConsistency",
        "Prompt Injection\nResistance",
        "Adversarial Decision\nAccuracy"
    ],

    "Score": [
        reliability[
            "json_validity_rate"
        ] * 100,

        reliability[
            "schema_compliance_rate"
        ] * 100,

        reliability[
            "api_success_rate"
        ] * 100,

        reliability[
            "fairness_decision_consistency"
        ] * 100,

        reliability[
            "fairness_full_output_consistency"
        ] * 100,

        reliability[
            "prompt_injection_resistance_rate"
        ] * 100,

        reliability[
            "adversarial_decision_accuracy"
        ] * 100
    ]
})


plt.figure(
    figsize=(11, 6)
)

plt.bar(
    reliability_plot["Metric"],
    reliability_plot["Score"]
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "LLM Reliability and Robustness Metrics"
)

plt.ylim(
    0,
    105
)

plt.xticks(
    rotation=25,
    ha="right"
)


for index, value in enumerate(
    reliability_plot["Score"]
):

    plt.text(
        index,
        value + 0.5,
        f"{value:.0f}%",
        ha="center"
    )


plt.tight_layout()

reliability_output = (
    VISUALIZATION_DIR /
    "reliability_metrics.png"
)

plt.savefig(
    reliability_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 4. Failure Pattern Visualization
# ==================================================

failure_df = pd.read_csv(
    RESULTS_DIR /
    "failure_pattern_summary.csv"
)

failure_df = failure_df.sort_values(
    by="count",
    ascending=True
)


plt.figure(
    figsize=(10, 6)
)

plt.barh(
    failure_df["failure_type"],
    failure_df["count"]
)

plt.xlabel(
    "Number of Decision Failures"
)

plt.ylabel(
    "Failure Type"
)

plt.title(
    "LLM Decision Failure Patterns"
)


for index, value in enumerate(
    failure_df["count"]
):

    plt.text(
        value + 0.05,
        index,
        str(value)
    )


plt.tight_layout()

failure_output = (
    VISUALIZATION_DIR /
    "failure_patterns.png"
)

plt.savefig(
    failure_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# Complete
# ==================================================

print("=" * 75)
print("EVALUATION VISUALIZATIONS CREATED")
print("=" * 75)

print(classification_output)
print(skill_output)
print(reliability_output)
print(failure_output)