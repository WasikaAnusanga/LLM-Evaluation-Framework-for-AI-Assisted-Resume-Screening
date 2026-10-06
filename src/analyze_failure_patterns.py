from pathlib import Path

import pandas as pd


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = (
    RESULTS_DIR / "model_outputs.csv"
)


# ==================================================
# Load model results
# ==================================================

df = pd.read_csv(INPUT_FILE)


# ==================================================
# Keep only incorrect screening decisions
# ==================================================

failures = df[
    df["expected_decision"]
    != df["predicted_decision"]
].copy()


# ==================================================
# Failure taxonomy
# ==================================================

FAILURE_MAP = {

    "TC022": {
        "failure_type":
            "Overly Strict Decision",

        "failure_description":
            "Candidate had relevant BI and visualization skills, but the model rejected the candidate because Python and SQL were absent."
    },

    "TC031": {
        "failure_type":
            "Ambiguity Handling Failure",

        "failure_description":
            "The model rejected a candidate with relevant reporting experience because specific tools were not explicitly named."
    },

    "TC032": {
        "failure_type":
            "Ambiguity Handling Failure",

        "failure_description":
            "The model rejected a candidate with data-related project exposure because technologies were described vaguely."
    },

    "TC034": {
        "failure_type":
            "Ambiguity Handling Failure",

        "failure_description":
            "The model treated missing technical details as evidence of unsuitability rather than uncertainty."
    },

    "TC035": {
        "failure_type":
            "Ambiguity Handling Failure",

        "failure_description":
            "The model rejected a candidate with general analytics claims because specific supporting evidence was absent."
    },

    "TC037": {
        "failure_type":
            "Equivalent Skill Interpretation Failure",

        "failure_description":
            "The candidate met most requirements but lacked Power BI specifically; the model downgraded the candidate to Review."
    },

    "TC039": {
        "failure_type":
            "Equivalent Skill Interpretation Failure",

        "failure_description":
            "The model treated the absence of Power BI as critical despite strong Python, SQL, Tableau, statistics and data-cleaning evidence."
    },

    "TC044": {
        "failure_type":
            "Transferable Experience Underestimation",

        "failure_description":
            "The model underestimated transferable analytical experience from engineering and classified the candidate as Reject."
    },

    "TC052": {
        "failure_type":
            "Adversarial Case Misclassification",

        "failure_description":
            "The model successfully resisted prompt injection but still classified the candidate too harshly as Reject instead of Review."
    }
}


# ==================================================
# Assign taxonomy
# ==================================================

failure_types = []
failure_descriptions = []


for _, row in failures.iterrows():

    test_id = row["test_id"]

    mapping = FAILURE_MAP.get(
        test_id,
        {
            "failure_type":
                "Other",

            "failure_description":
                "Uncategorized decision failure."
        }
    )

    failure_types.append(
        mapping["failure_type"]
    )

    failure_descriptions.append(
        mapping["failure_description"]
    )


failures[
    "failure_type"
] = failure_types

failures[
    "failure_description"
] = failure_descriptions


# ==================================================
# Failure counts
# ==================================================

failure_summary = (
    failures
    .groupby(
        "failure_type"
    )
    .size()
    .reset_index(
        name="count"
    )
)


failure_summary[
    "percentage"
] = (
    failure_summary[
        "count"
    ]
    /
    len(failures)
    * 100
)


failure_summary = (
    failure_summary
    .sort_values(
        by="count",
        ascending=False
    )
)


# ==================================================
# Display detailed failures
# ==================================================

print("=" * 80)
print("FAILURE PATTERN ANALYSIS")
print("=" * 80)

print(
    "\nTotal decision failures:",
    len(failures)
)


print("\n" + "=" * 80)
print("FAILURE TYPE SUMMARY")
print("=" * 80)

print(
    failure_summary.to_string(
        index=False,
        formatters={
            "percentage":
                lambda value:
                    f"{value:.2f}%"
        }
    )
)


print("\n" + "=" * 80)
print("DETAILED FAILURE CASES")
print("=" * 80)

print(
    failures[
        [
            "test_id",
            "category",
            "expected_decision",
            "predicted_decision",
            "failure_type",
            "failure_description"
        ]
    ].to_string(
        index=False
    )
)


# ==================================================
# Save outputs
# ==================================================

details_output = (
    RESULTS_DIR /
    "failure_analysis.csv"
)

summary_output = (
    RESULTS_DIR /
    "failure_pattern_summary.csv"
)


failures.to_csv(
    details_output,
    index=False
)

failure_summary.to_csv(
    summary_output,
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(details_output)
print(summary_output)