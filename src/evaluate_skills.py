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
# Valid required skills
# ==================================================

VALID_SKILLS = {
    "Python",
    "SQL",
    "Power BI",
    "Basic Statistics",
    "Data Cleaning",
    "Data Visualization"
}


# ==================================================
# Helper function
# ==================================================

def parse_skills(value):
    """
    Convert a pipe-separated skill string into a Python set.

    Example:
    Python|SQL|Power BI

    becomes:

    {"Python", "SQL", "Power BI"}
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if value == "":
        return set()

    skills = {
        skill.strip()
        for skill in value.split("|")
        if skill.strip()
    }

    return skills


# ==================================================
# Metric calculation
# ==================================================

def calculate_skill_metrics(
    expected_column,
    predicted_column,
    metric_name,
    df
):

    true_positive = 0
    false_positive = 0
    false_negative = 0

    exact_matches = 0

    per_case_results = []


    for _, row in df.iterrows():

        expected = parse_skills(
            row[expected_column]
        )

        predicted = parse_skills(
            row[predicted_column]
        )


        # ------------------------------------------
        # Set operations
        # ------------------------------------------

        correct = (
            expected.intersection(
                predicted
            )
        )

        extra = (
            predicted.difference(
                expected
            )
        )

        missed = (
            expected.difference(
                predicted
            )
        )


        tp = len(correct)

        fp = len(extra)

        fn = len(missed)


        true_positive += tp

        false_positive += fp

        false_negative += fn


        # ------------------------------------------
        # Exact set match
        # ------------------------------------------

        exact_match = (
            expected == predicted
        )

        if exact_match:
            exact_matches += 1


        # ------------------------------------------
        # Save per-case result
        # ------------------------------------------

        per_case_results.append({
            "test_id":
                row["test_id"],

            "category":
                row["category"],

            "expected_skills":
                "|".join(
                    sorted(expected)
                ),

            "predicted_skills":
                "|".join(
                    sorted(predicted)
                ),

            "correct_skills":
                "|".join(
                    sorted(correct)
                ),

            "extra_skills":
                "|".join(
                    sorted(extra)
                ),

            "missed_skills":
                "|".join(
                    sorted(missed)
                ),

            "true_positives":
                tp,

            "false_positives":
                fp,

            "false_negatives":
                fn,

            "exact_match":
                exact_match
        })


    # ==================================================
    # Precision
    # ==================================================

    if (
        true_positive
        + false_positive
    ) > 0:

        precision = (
            true_positive
            /
            (
                true_positive
                + false_positive
            )
        )

    else:
        precision = 0


    # ==================================================
    # Recall
    # ==================================================

    if (
        true_positive
        + false_negative
    ) > 0:

        recall = (
            true_positive
            /
            (
                true_positive
                + false_negative
            )
        )

    else:
        recall = 0


    # ==================================================
    # F1
    # ==================================================

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:
        f1 = 0


    # ==================================================
    # Exact-match rate
    # ==================================================

    exact_match_rate = (
        exact_matches
        / len(df)
    )


    summary = {

        "metric_type":
            metric_name,

        "true_positives":
            true_positive,

        "false_positives":
            false_positive,

        "false_negatives":
            false_negative,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1,

        "exact_matches":
            exact_matches,

        "total_cases":
            len(df),

        "exact_match_rate":
            exact_match_rate
    }


    return (
        summary,
        pd.DataFrame(
            per_case_results
        )
    )


# ==================================================
# Load model results
# ==================================================

df = pd.read_csv(
    INPUT_FILE
)


print("=" * 75)
print("SKILL IDENTIFICATION EVALUATION")
print("=" * 75)


# ==================================================
# Matched skill evaluation
# ==================================================

matched_summary, matched_cases = (
    calculate_skill_metrics(
        expected_column=
            "expected_matched_skills",

        predicted_column=
            "predicted_matched_skills",

        metric_name=
            "Matched Skills",

        df=df
    )
)


# ==================================================
# Missing skill evaluation
# ==================================================

missing_summary, missing_cases = (
    calculate_skill_metrics(
        expected_column=
            "expected_missing_skills",

        predicted_column=
            "predicted_missing_skills",

        metric_name=
            "Missing Skills",

        df=df
    )
)


# ==================================================
# Create summary
# ==================================================

summary_df = pd.DataFrame(
    [
        matched_summary,
        missing_summary
    ]
)


# ==================================================
# Display results
# ==================================================

for summary in [
    matched_summary,
    missing_summary
]:

    print(
        "\n" + "=" * 75
    )

    print(
        summary["metric_type"].upper()
    )

    print("=" * 75)

    print(
        "True Positives:",
        summary[
            "true_positives"
        ]
    )

    print(
        "False Positives:",
        summary[
            "false_positives"
        ]
    )

    print(
        "False Negatives:",
        summary[
            "false_negatives"
        ]
    )

    print()

    print(
        f"Precision: "
        f"{summary['precision']:.4f} "
        f"({summary['precision'] * 100:.2f}%)"
    )

    print(
        f"Recall:    "
        f"{summary['recall']:.4f} "
        f"({summary['recall'] * 100:.2f}%)"
    )

    print(
        f"F1 Score:  "
        f"{summary['f1_score']:.4f} "
        f"({summary['f1_score'] * 100:.2f}%)"
    )

    print(
        f"Exact Match Rate: "
        f"{summary['exact_match_rate']:.4f} "
        f"({summary['exact_match_rate'] * 100:.2f}%)"
    )

    print(
        "Exact Matches:",
        f"{summary['exact_matches']}"
        f"/{summary['total_cases']}"
    )


# ==================================================
# Save outputs
# ==================================================

summary_output = (
    RESULTS_DIR /
    "skill_metrics_summary.csv"
)

matched_output = (
    RESULTS_DIR /
    "matched_skill_analysis.csv"
)

missing_output = (
    RESULTS_DIR /
    "missing_skill_analysis.csv"
)


summary_df.to_csv(
    summary_output,
    index=False
)

matched_cases.to_csv(
    matched_output,
    index=False
)

missing_cases.to_csv(
    missing_output,
    index=False
)


print("\n" + "=" * 75)

print("FILES SAVED")

print("=" * 75)

print(summary_output)

print(matched_output)

print(missing_output)