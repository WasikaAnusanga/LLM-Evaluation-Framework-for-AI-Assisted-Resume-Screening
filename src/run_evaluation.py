import os
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "gemini-3.1-flash-lite"

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ==================================================
# Load environment variables
# ==================================================

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your .env file."
    )


# ==================================================
# Gemini client
# ==================================================

client = genai.Client(api_key=api_key)


# ==================================================
# Generation configuration
# ==================================================

generation_config = types.GenerateContentConfig(
    automatic_function_calling=
        types.AutomaticFunctionCallingConfig(
            disable=True
        )
)


# ==================================================
# Load project files
# ==================================================

job_description = (
    DATA_DIR / "job_description.txt"
).read_text(encoding="utf-8")

prompt_template = (
    PROMPT_DIR / "resume_screening_prompt.txt"
).read_text(encoding="utf-8")

test_cases = pd.read_csv(
    DATA_DIR / "test_cases.csv"
)


# ==================================================
# Helper function
# ==================================================

def safe_expected_value(value):
    """
    Convert empty CSV/NaN values to empty strings.
    """
    if pd.isna(value):
        return ""

    return str(value)


# ==================================================
# Store evaluation results
# ==================================================

results = []


# ==================================================
# Run every test case
# ==================================================

total_cases = len(test_cases)

print("=" * 70)
print("LLM RESUME SCREENING EVALUATION")
print("=" * 70)

print("Model:", MODEL_NAME)
print("Total test cases:", total_cases)

print("=" * 70)


for index, test_case in test_cases.iterrows():

    test_id = test_case["test_id"]

    print(
        f"\nRunning {test_id} "
        f"({index + 1}/{total_cases})..."
    )


    # ----------------------------------------------
    # Build prompt
    # ----------------------------------------------

    final_prompt = (
        prompt_template
        .replace(
            "{job_description}",
            job_description
        )
        .replace(
            "{resume}",
            test_case["resume"]
        )
    )


    # ----------------------------------------------
    # Default values
    # ----------------------------------------------

    raw_response = ""

    valid_json = False

    predicted_decision = ""

    predicted_matched_skills = ""

    predicted_missing_skills = ""

    predicted_experience_match = ""

    reason = ""

    api_error = ""


    # ----------------------------------------------
    # Call Gemini
    # ----------------------------------------------

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=final_prompt,
            config=generation_config
        )

        raw_response = (
            response.text.strip()
            if response.text
            else ""
        )


        # ------------------------------------------
        # Validate JSON
        # ------------------------------------------

        try:

            parsed_response = json.loads(
                raw_response
            )

            valid_json = True


            predicted_decision = (
                parsed_response.get(
                    "decision",
                    ""
                )
            )


            matched_skills = (
                parsed_response.get(
                    "matched_skills",
                    []
                )
            )

            if isinstance(
                matched_skills,
                list
            ):
                predicted_matched_skills = "|".join(
                    matched_skills
                )


            missing_skills = (
                parsed_response.get(
                    "missing_skills",
                    []
                )
            )

            if isinstance(
                missing_skills,
                list
            ):
                predicted_missing_skills = "|".join(
                    missing_skills
                )


            predicted_experience_match = (
                parsed_response.get(
                    "experience_match",
                    ""
                )
            )


            reason = (
                parsed_response.get(
                    "reason",
                    ""
                )
            )


        except json.JSONDecodeError:

            valid_json = False

            print(
                f"  JSON parsing failed for {test_id}"
            )


    except Exception as error:

        api_error = str(error)

        print(
            f"  API error for {test_id}: "
            f"{error}"
        )


    # ----------------------------------------------
    # Store result
    # ----------------------------------------------

    result = {

        "test_id":
            test_id,

        "category":
            test_case["category"],

        "model":
            MODEL_NAME,


        "expected_decision":
            safe_expected_value(
                test_case[
                    "expected_decision"
                ]
            ),

        "predicted_decision":
            predicted_decision,


        "expected_matched_skills":
            safe_expected_value(
                test_case[
                    "expected_matched_skills"
                ]
            ),

        "predicted_matched_skills":
            predicted_matched_skills,


        "expected_missing_skills":
            safe_expected_value(
                test_case[
                    "expected_missing_skills"
                ]
            ),

        "predicted_missing_skills":
            predicted_missing_skills,


        "expected_experience_match":
            safe_expected_value(
                test_case[
                    "expected_experience_match"
                ]
            ),

        "predicted_experience_match":
            predicted_experience_match,


        "valid_json":
            valid_json,

        "reason":
            reason,

        "raw_response":
            raw_response,

        "api_error":
            api_error
    }


    results.append(result)


    # ----------------------------------------------
    # Save progress after every case
    # ----------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_DIR / "model_outputs.csv",
        index=False
    )


    # ----------------------------------------------
    # Display quick result
    # ----------------------------------------------

    print(
        "  Expected:",
        test_case["expected_decision"]
    )

    print(
        "  Predicted:",
        predicted_decision
        if predicted_decision
        else "N/A"
    )

    print(
        "  Valid JSON:",
        valid_json
    )


    # Small delay between API calls
    time.sleep(1)


# ==================================================
# Final summary
# ==================================================

results_df = pd.DataFrame(results)

output_file = (
    RESULTS_DIR / "model_outputs.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)

print("EVALUATION COMPLETE")

print("=" * 70)

print(
    "Total cases processed:",
    len(results_df)
)

print(
    "Valid JSON responses:",
    results_df["valid_json"].sum()
)

print(
    "API errors:",
    (
        results_df["api_error"] != ""
    ).sum()
)

print(
    "\nResults saved to:"
)

print(output_file)