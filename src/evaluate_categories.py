from pathlib import Path

import pandas as pd


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

input_file = RESULTS_DIR / "model_outputs.csv"


# ==================================================
# Load results
# ==================================================

df = pd.read_csv(input_file)


# ==================================================
# Determine whether decision was correct
# ==================================================

df["decision_correct"] = (
    df["expected_decision"]
    == df["predicted_decision"]
)


# ==================================================
# Category-level evaluation
# ==================================================

category_results = (
    df.groupby("category")
    .agg(
        total_cases=("test_id", "count"),
        correct_decisions=("decision_correct", "sum")
    )
    .reset_index()
)


category_results["accuracy"] = (
    category_results["correct_decisions"]
    / category_results["total_cases"]
)


category_results["accuracy_percent"] = (
    category_results["accuracy"] * 100
)


# ==================================================
# Display results
# ==================================================

print("=" * 80)
print("CATEGORY PERFORMANCE")
print("=" * 80)

print(
    category_results[
        [
            "category",
            "total_cases",
            "correct_decisions",
            "accuracy_percent"
        ]
    ].to_string(
        index=False,
        formatters={
            "accuracy_percent":
                lambda value: f"{value:.2f}%"
        }
    )
)


# ==================================================
# Find incorrect cases
# ==================================================

incorrect_cases = df[
    df["decision_correct"] == False
][
    [
        "test_id",
        "category",
        "expected_decision",
        "predicted_decision",
        "reason"
    ]
]


print("\n" + "=" * 80)
print("INCORRECT DECISIONS")
print("=" * 80)

if len(incorrect_cases) == 0:

    print("No incorrect decisions.")

else:

    print(
        incorrect_cases.to_string(
            index=False
        )
    )


# ==================================================
# Save results
# ==================================================

category_output = (
    RESULTS_DIR / "category_performance.csv"
)

incorrect_output = (
    RESULTS_DIR / "incorrect_decisions.csv"
)

category_results.to_csv(
    category_output,
    index=False
)

incorrect_cases.to_csv(
    incorrect_output,
    index=False
)


print("\nSaved:")
print(category_output)
print(incorrect_output)