from pathlib import Path

import pandas as pd


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

MATCHED_FILE = (
    RESULTS_DIR / "matched_skill_analysis.csv"
)

MISSING_FILE = (
    RESULTS_DIR / "missing_skill_analysis.csv"
)


# ==================================================
# Load analysis files
# ==================================================

matched_df = pd.read_csv(MATCHED_FILE)

missing_df = pd.read_csv(MISSING_FILE)


# ==================================================
# Helper
# ==================================================

def has_content(value):
    """
    Returns True if the CSV cell contains
    a meaningful non-empty value.
    """

    if pd.isna(value):
        return False

    return str(value).strip() != ""


# ==================================================
# Matched skill failures
# ==================================================

matched_failures = matched_df[
    matched_df["exact_match"] == False
].copy()


# Cases where Gemini added unsupported skills
unsupported_skills = matched_df[
    matched_df["extra_skills"].apply(
        has_content
    )
].copy()


# Cases where Gemini missed valid skills
missed_matched_skills = matched_df[
    matched_df["missed_skills"].apply(
        has_content
    )
].copy()


# ==================================================
# Missing skill failures
# ==================================================

missing_failures = missing_df[
    missing_df["exact_match"] == False
].copy()


# Extra missing skills:
# Gemini said a skill was missing even though
# ground truth did not consider it missing.
incorrect_missing_claims = missing_df[
    missing_df["extra_skills"].apply(
        has_content
    )
].copy()


# Gemini failed to identify an expected missing skill.
missed_missing_skills = missing_df[
    missing_df["missed_skills"].apply(
        has_content
    )
].copy()


# ==================================================
# Calculate hallucination / unsupported-skill rate
# ==================================================

total_tp = matched_df[
    "true_positives"
].sum()

total_fp = matched_df[
    "false_positives"
].sum()

total_fn = matched_df[
    "false_negatives"
].sum()


total_predicted_skill_claims = (
    total_tp + total_fp
)


if total_predicted_skill_claims > 0:

    unsupported_skill_rate = (
        total_fp
        / total_predicted_skill_claims
    )

else:

    unsupported_skill_rate = 0


# ==================================================
# Display summary
# ==================================================

print("=" * 78)
print("SKILL FAILURE ANALYSIS")
print("=" * 78)

print(
    "\nTotal correct matched-skill claims:",
    total_tp
)

print(
    "Unsupported matched-skill claims:",
    total_fp
)

print(
    "Missed valid matched skills:",
    total_fn
)

print(
    "Total predicted matched-skill claims:",
    total_predicted_skill_claims
)

print(
    f"\nUnsupported Skill Rate: "
    f"{unsupported_skill_rate:.4f} "
    f"({unsupported_skill_rate * 100:.2f}%)"
)


# ==================================================
# Unsupported skill cases
# ==================================================

print("\n" + "=" * 78)
print("UNSUPPORTED SKILL ATTRIBUTIONS")
print("=" * 78)

if len(unsupported_skills) == 0:

    print(
        "No unsupported skill attributions found."
    )

else:

    print(
        unsupported_skills[
            [
                "test_id",
                "category",
                "expected_skills",
                "predicted_skills",
                "extra_skills"
            ]
        ].to_string(index=False)
    )


# ==================================================
# Missed valid skills
# ==================================================

print("\n" + "=" * 78)
print("MISSED VALID MATCHED SKILLS")
print("=" * 78)

if len(missed_matched_skills) == 0:

    print(
        "No valid matched skills were missed."
    )

else:

    print(
        missed_matched_skills[
            [
                "test_id",
                "category",
                "expected_skills",
                "predicted_skills",
                "missed_skills"
            ]
        ].to_string(index=False)
    )


# ==================================================
# Incorrect missing-skill claims
# ==================================================

print("\n" + "=" * 78)
print("INCORRECT MISSING-SKILL CLAIMS")
print("=" * 78)

if len(incorrect_missing_claims) == 0:

    print(
        "No incorrect missing-skill claims found."
    )

else:

    print(
        incorrect_missing_claims[
            [
                "test_id",
                "category",
                "expected_skills",
                "predicted_skills",
                "extra_skills"
            ]
        ].to_string(index=False)
    )


# ==================================================
# Missed missing skills
# ==================================================

print("\n" + "=" * 78)
print("EXPECTED MISSING SKILLS NOT IDENTIFIED")
print("=" * 78)

if len(missed_missing_skills) == 0:

    print(
        "No expected missing skills were missed."
    )

else:

    print(
        missed_missing_skills[
            [
                "test_id",
                "category",
                "expected_skills",
                "predicted_skills",
                "missed_skills"
            ]
        ].to_string(index=False)
    )


# ==================================================
# Save failure files
# ==================================================

unsupported_output = (
    RESULTS_DIR /
    "unsupported_skill_attributions.csv"
)

missed_output = (
    RESULTS_DIR /
    "missed_matched_skills.csv"
)

incorrect_missing_output = (
    RESULTS_DIR /
    "incorrect_missing_skill_claims.csv"
)

missed_missing_output = (
    RESULTS_DIR /
    "missed_missing_skills.csv"
)


unsupported_skills.to_csv(
    unsupported_output,
    index=False
)

missed_matched_skills.to_csv(
    missed_output,
    index=False
)

incorrect_missing_claims.to_csv(
    incorrect_missing_output,
    index=False
)

missed_missing_skills.to_csv(
    missed_missing_output,
    index=False
)


# ==================================================
# Save summary
# ==================================================

summary_df = pd.DataFrame([
    {
        "correct_skill_claims":
            total_tp,

        "unsupported_skill_claims":
            total_fp,

        "missed_valid_skills":
            total_fn,

        "total_predicted_skill_claims":
            total_predicted_skill_claims,

        "unsupported_skill_rate":
            unsupported_skill_rate,

        "unsupported_skill_rate_percent":
            unsupported_skill_rate * 100
    }
])


summary_output = (
    RESULTS_DIR /
    "skill_failure_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)


print("\n" + "=" * 78)
print("FILES SAVED")
print("=" * 78)

print(summary_output)
print(unsupported_output)
print(missed_output)
print(incorrect_missing_output)
print(missed_missing_output)