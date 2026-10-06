from pathlib import Path

import pandas as pd


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"


# ==================================================
# Load result files
# ==================================================

classification = pd.read_csv(
    RESULTS_DIR / "classification_summary.csv"
).iloc[0]

skill_metrics = pd.read_csv(
    RESULTS_DIR / "skill_metrics_summary.csv"
)

experience = pd.read_csv(
    RESULTS_DIR / "experience_summary.csv"
).iloc[0]

reliability = pd.read_csv(
    RESULTS_DIR / "reliability_summary.csv"
).iloc[0]

skill_failure = pd.read_csv(
    RESULTS_DIR / "skill_failure_summary.csv"
).iloc[0]

failure_patterns = pd.read_csv(
    RESULTS_DIR / "failure_pattern_summary.csv"
)

category_performance = pd.read_csv(
    RESULTS_DIR / "category_performance.csv"
)


# ==================================================
# Extract skill metrics
# ==================================================

matched = skill_metrics[
    skill_metrics["metric_type"] == "Matched Skills"
].iloc[0]

missing = skill_metrics[
    skill_metrics["metric_type"] == "Missing Skills"
].iloc[0]


# ==================================================
# Find worst-performing category
# ==================================================

worst_category = category_performance.sort_values(
    "accuracy_percent",
    ascending=True
).iloc[0]


# ==================================================
# Find most common failure pattern
# ==================================================

top_failure = failure_patterns.sort_values(
    "count",
    ascending=False
).iloc[0]


# ==================================================
# Build final metric table
# ==================================================

summary_rows = [

    {
        "metric_group": "Classification",
        "metric": "Overall Accuracy",
        "value_percent": classification["accuracy"] * 100
    },

    {
        "metric_group": "Classification",
        "metric": "Macro Precision",
        "value_percent": classification["macro_precision"] * 100
    },

    {
        "metric_group": "Classification",
        "metric": "Macro Recall",
        "value_percent": classification["macro_recall"] * 100
    },

    {
        "metric_group": "Classification",
        "metric": "Macro F1",
        "value_percent": classification["macro_f1"] * 100
    },

    {
        "metric_group": "Skill Identification",
        "metric": "Matched Skill Precision",
        "value_percent": matched["precision"] * 100
    },

    {
        "metric_group": "Skill Identification",
        "metric": "Matched Skill Recall",
        "value_percent": matched["recall"] * 100
    },

    {
        "metric_group": "Skill Identification",
        "metric": "Matched Skill F1",
        "value_percent": matched["f1_score"] * 100
    },

    {
        "metric_group": "Skill Identification",
        "metric": "Matched Skill Exact Match",
        "value_percent": matched["exact_match_rate"] * 100
    },

    {
        "metric_group": "Missing Skill Detection",
        "metric": "Missing Skill Precision",
        "value_percent": missing["precision"] * 100
    },

    {
        "metric_group": "Missing Skill Detection",
        "metric": "Missing Skill Recall",
        "value_percent": missing["recall"] * 100
    },

    {
        "metric_group": "Missing Skill Detection",
        "metric": "Missing Skill F1",
        "value_percent": missing["f1_score"] * 100
    },

    {
        "metric_group": "Missing Skill Detection",
        "metric": "Missing Skill Exact Match",
        "value_percent": missing["exact_match_rate"] * 100
    },

    {
        "metric_group": "Experience",
        "metric": "Experience Match Accuracy",
        "value_percent": experience["accuracy"] * 100
    },

    {
        "metric_group": "Reliability",
        "metric": "JSON Validity",
        "value_percent": reliability["json_validity_rate"] * 100
    },

    {
        "metric_group": "Reliability",
        "metric": "Schema Compliance",
        "value_percent": reliability["schema_compliance_rate"] * 100
    },

    {
        "metric_group": "Reliability",
        "metric": "API Success",
        "value_percent": reliability["api_success_rate"] * 100
    },

    {
        "metric_group": "Fairness",
        "metric": "Counterfactual Decision Consistency",
        "value_percent":
            reliability["fairness_decision_consistency"] * 100
    },

    {
        "metric_group": "Fairness",
        "metric": "Counterfactual Full Output Consistency",
        "value_percent":
            reliability["fairness_full_output_consistency"] * 100
    },

    {
        "metric_group": "Security / Robustness",
        "metric": "Prompt Injection Resistance",
        "value_percent":
            reliability["prompt_injection_resistance_rate"] * 100
    },

    {
        "metric_group": "Security / Robustness",
        "metric": "Adversarial Decision Accuracy",
        "value_percent":
            reliability["adversarial_decision_accuracy"] * 100
    },

    {
        "metric_group": "Hallucination / Attribution",
        "metric": "Unsupported Skill Attribution Rate",
        "value_percent":
            skill_failure["unsupported_skill_rate_percent"]
    }
]


summary_df = pd.DataFrame(summary_rows)


# ==================================================
# Save main summary
# ==================================================

summary_output = (
    RESULTS_DIR / "evaluation_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)


# ==================================================
# Create high-level findings file
# ==================================================

findings = pd.DataFrame([
    {
        "finding": "Total Test Cases",
        "value": "60"
    },

    {
        "finding": "Correct Screening Decisions",
        "value": "51/60"
    },

    {
        "finding": "Decision Failures",
        "value": "9"
    },

    {
        "finding": "Worst Performing Category",
        "value":
            f"{worst_category['category']} "
            f"({worst_category['accuracy_percent']:.2f}%)"
    },

    {
        "finding": "Most Common Failure Pattern",
        "value":
            f"{top_failure['failure_type']} "
            f"({top_failure['percentage']:.2f}% of decision failures)"
    },

    {
        "finding": "Unsupported Skill Claims",
        "value":
            str(
                int(
                    skill_failure[
                        "unsupported_skill_claims"
                    ]
                )
            )
    },

    {
        "finding": "Model",
        "value": "gemini-3.1-flash-lite"
    }
])


findings_output = (
    RESULTS_DIR / "key_findings.csv"
)

findings.to_csv(
    findings_output,
    index=False
)


# ==================================================
# Display
# ==================================================

print("=" * 80)
print("FINAL EVALUATION SUMMARY")
print("=" * 80)

display_df = summary_df.copy()

display_df["value_percent"] = (
    display_df["value_percent"]
    .apply(
        lambda x: f"{x:.2f}%"
    )
)

print(
    display_df.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print(
    findings.to_string(
        index=False
    )
)


print("\nSaved:")
print(summary_output)
print(findings_output)