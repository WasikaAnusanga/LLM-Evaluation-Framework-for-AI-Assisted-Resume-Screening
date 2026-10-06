import os
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


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your .env file."
    )


# --------------------------------------------------
# Create Gemini client
# --------------------------------------------------

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# Load job description
# --------------------------------------------------

job_description = (
    DATA_DIR / "job_description.txt"
).read_text(encoding="utf-8")


# --------------------------------------------------
# Load prompt template
# --------------------------------------------------

prompt_template = (
    PROMPT_DIR / "resume_screening_prompt.txt"
).read_text(encoding="utf-8")


# --------------------------------------------------
# Load test cases
# --------------------------------------------------

df = pd.read_csv(DATA_DIR / "test_cases.csv")

test_case = df.loc[df["test_id"] == "TC001"].iloc[0]


# --------------------------------------------------
# Build final prompt
# --------------------------------------------------

final_prompt = prompt_template.replace(
    "{job_description}",
    job_description
).replace(
    "{resume}",
    test_case["resume"]
)


# --------------------------------------------------
# Run Gemini
# --------------------------------------------------

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=final_prompt
)


# --------------------------------------------------
# Display test information
# --------------------------------------------------

print("=" * 70)
print("TEST CASE:", test_case["test_id"])
print("=" * 70)

print("\nEXPECTED DECISION:")
print(test_case["expected_decision"])

print("\nEXPECTED MATCHED SKILLS:")
print(test_case["expected_matched_skills"])

print("\nEXPECTED MISSING SKILLS:")
print(test_case["expected_missing_skills"])

print("\nEXPECTED EXPERIENCE MATCH:")
print(test_case["expected_experience_match"])

print("\n" + "=" * 70)
print("GEMINI RESPONSE")
print("=" * 70)

print(response.text)