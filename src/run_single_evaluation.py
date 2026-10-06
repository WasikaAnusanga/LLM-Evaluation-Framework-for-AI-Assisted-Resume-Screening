import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load API key
# --------------------------------------------------

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your .env file."
    )


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# Load files
# --------------------------------------------------

job_description = (
    DATA_DIR / "job_description.txt"
).read_text(encoding="utf-8")

prompt_template = (
    PROMPT_DIR / "resume_screening_prompt.txt"
).read_text(encoding="utf-8")

df = pd.read_csv(DATA_DIR / "test_cases.csv")


# --------------------------------------------------
# Select one test case
# --------------------------------------------------

test_case = df.loc[df["test_id"] == "TC001"].iloc[0]


# --------------------------------------------------
# Build prompt
# --------------------------------------------------

final_prompt = (
    prompt_template
    .replace("{job_description}", job_description)
    .replace("{resume}", test_case["resume"])
)


# --------------------------------------------------
# Call Gemini
# --------------------------------------------------

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=final_prompt
)

raw_response = response.text.strip()


# --------------------------------------------------
# Validate JSON
# --------------------------------------------------

valid_json = False

predicted_decision = ""
predicted_matched_skills = ""
predicted_missing_skills = ""
predicted_experience_match = ""
reason = ""

try:
    parsed_response = json.loads(raw_response)

    valid_json = True

    predicted_decision = parsed_response.get(
        "decision",
        ""
    )

    predicted_matched_skills = "|".join(
        parsed_response.get("matched_skills", [])
    )

    predicted_missing_skills = "|".join(
        parsed_response.get("missing_skills", [])
    )

    predicted_experience_match = parsed_response.get(
        "experience_match",
        ""
    )

    reason = parsed_response.get(
        "reason",
        ""
    )

except json.JSONDecodeError as error:
    print("\nJSON parsing failed:")
    print(error)


# --------------------------------------------------
# Prepare result row
# --------------------------------------------------

result = {
    "test_id": test_case["test_id"],
    "category": test_case["category"],

    "expected_decision":
        test_case["expected_decision"],

    "predicted_decision":
        predicted_decision,

    "expected_matched_skills":
        test_case["expected_matched_skills"],

    "predicted_matched_skills":
        predicted_matched_skills,

    "expected_missing_skills":
        (
            ""
            if pd.isna(test_case["expected_missing_skills"])
            else test_case["expected_missing_skills"]
        ),

    "predicted_missing_skills":
        predicted_missing_skills,

    "expected_experience_match":
        test_case["expected_experience_match"],

    "predicted_experience_match":
        predicted_experience_match,

    "valid_json":
        valid_json,

    "reason":
        reason,

    "raw_response":
        raw_response
}


# --------------------------------------------------
# Save result
# --------------------------------------------------

result_df = pd.DataFrame([result])

output_file = RESULTS_DIR / "model_outputs.csv"

result_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("\n" + "=" * 70)
print("TC001 EVALUATION RESULT")
print("=" * 70)

print(
    "\nExpected Decision:",
    test_case["expected_decision"]
)

print(
    "Predicted Decision:",
    predicted_decision
)

print(
    "Expected Experience:",
    test_case["expected_experience_match"]
)

print(
    "Predicted Experience:",
    predicted_experience_match
)

print(
    "Valid JSON:",
    valid_json
)

print(
    "\nPredicted Matched Skills:",
    predicted_matched_skills
)

print(
    "Predicted Missing Skills:",
    predicted_missing_skills
)   

print(
    "\nSaved to:",
    output_file
)