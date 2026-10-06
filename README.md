# LLM Evaluation Framework for AI-Assisted Resume Screening

## Overview

This project develops and evaluates an LLM-based resume screening framework for a real-world HR recruitment application.

The selected business application is **HR Resume Screening**.

The framework evaluates how reliably a Large Language Model can compare candidate resumes against a fixed Junior Data Analyst job description and produce a structured screening recommendation.

The evaluated model is:

`gemini-3.1-flash-lite`

The project focuses on systematic LLM evaluation rather than building a fully autonomous hiring system.

---

## Project Objective

The main objective of this project is to design and implement an LLM evaluation framework that:

- Uses more than 50 independently designed test cases
- Defines expected outputs before testing the LLM
- Evaluates screening decisions quantitatively
- Measures matched-skill and missing-skill identification
- Evaluates experience-level classification
- Tests ambiguous and borderline cases
- Tests counterfactual consistency
- Tests prompt-injection resistance
- Measures structured-output reliability
- Identifies repeated failure patterns

---

## Business Application

The LLM evaluates candidates for a **Junior Data Analyst** role.

For each evaluation case, the model receives:

1. A fixed Job Description
2. A Candidate Resume

The model returns a structured JSON response containing:

```json
{
  "decision": "Shortlist | Review | Reject",
  "matched_skills": [],
  "missing_skills": [],
  "experience_match": "High | Medium | Low",
  "reason": ""
}
```

The final screening decision is classified as:

- **Shortlist** — candidate satisfies most essential requirements
- **Review** — candidate partially satisfies the requirements or important information is unclear
- **Reject** — candidate clearly lacks sufficient evidence for the role

The six standardized required skills are:

- Python
- SQL
- Power BI
- Basic Statistics
- Data Cleaning
- Data Visualization

---

## Test Dataset

A total of **60 synthetic test cases** were independently designed before running the LLM.

| Category | Test Cases |
|---|---:|
| Strong Match | 10 |
| Weak Match | 10 |
| Borderline Match | 10 |
| Missing or Ambiguous Information | 5 |
| Skill Synonyms | 5 |
| Irrelevant Experience | 5 |
| Fairness / Counterfactual | 5 |
| Prompt Injection / Adversarial | 5 |
| Formatting / Unusual Resume | 5 |
| **Total** | **60** |

Expected outputs were manually defined before model testing using predefined ground-truth rules.

---

## Evaluation Metrics

The framework evaluates the LLM using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Matched-skill precision, recall and F1
- Missing-skill precision, recall and F1
- Skill exact-match rate
- Experience-match accuracy
- JSON validity
- Schema compliance
- API success rate
- Unsupported skill attribution rate
- Counterfactual decision consistency
- Prompt-injection resistance
- Adversarial decision accuracy
- Performance by test category
- Failure-pattern analysis

---

## Key Results

| Metric | Result |
|---|---:|
| Overall Screening Accuracy | 85.00% |
| Macro Precision | 85.41% |
| Macro Recall | 85.56% |
| Macro F1 | 84.00% |
| Matched-Skill Precision | 99.49% |
| Matched-Skill Recall | 96.57% |
| Matched-Skill F1 | 98.01% |
| Missing-Skill Precision | 95.68% |
| Missing-Skill Recall | 99.36% |
| Missing-Skill F1 | 97.48% |
| Experience-Match Accuracy | 88.33% |
| JSON Validity | 100.00% |
| Schema Compliance | 100.00% |
| API Success | 100.00% |
| Counterfactual Decision Consistency | 100.00% |
| Prompt-Injection Resistance | 100.00% |
| Adversarial Decision Accuracy | 80.00% |
| Unsupported Skill Attribution Rate | 0.51% |

The model correctly classified **51 out of 60** screening decisions.

---

## Main Findings

The evaluation produced several important findings:

- Skill identification performance was stronger than final screening classification.
- Matched-skill F1 reached **98.01%**, compared with **85.00%** final decision accuracy.
- The model produced only one unsupported matched-skill attribution.
- The unsupported skill attribution rate was **0.51%**.
- All 60 responses were valid JSON and followed the required schema.
- All counterfactual fairness variants produced consistent core outputs.
- All five prompt-injection attempts were resisted.
- The weakest category was **Missing or Ambiguous Information**, with **20% classification accuracy**.
- The most common decision failure was **Ambiguity Handling Failure**, accounting for **44.44% of all decision failures**.
- The model frequently classified expected `Review` cases as `Reject`, indicating a conservative tendency for incomplete or partially suitable candidates.

---

## Project Structure

```text
LLM_Resume_Evaluation/
│
├── data/
│   ├── job_description.txt
│   ├── ground_truth_rules.md
│   ├── test_plan.md
│   └── test_cases.csv
│
├── prompts/
│   └── resume_screening_prompt.txt
│
├── src/
│   ├── test_setup.py
│   ├── test_gemini.py
│   ├── run_single_evaluation.py
│   ├── run_evaluation.py
│   ├── evaluate_classification.py
│   ├── plot_confusion_matrix.py
│   ├── evaluate_categories.py
│   ├── plot_category_performance.py
│   ├── evaluate_skills.py
│   ├── analyze_skill_failures.py
│   ├── evaluate_experience.py
│   ├── evaluate_reliability.py
│   ├── analyze_failure_patterns.py
│   ├── create_evaluation_visualizations.py
│   └── create_final_summary.py
│
├── results/
│   ├── model_outputs.csv
│   ├── classification_metrics.csv
│   ├── classification_summary.csv
│   ├── confusion_matrix.csv
│   ├── category_performance.csv
│   ├── incorrect_decisions.csv
│   ├── skill_metrics_summary.csv
│   ├── matched_skill_analysis.csv
│   ├── missing_skill_analysis.csv
│   ├── skill_failure_summary.csv
│   ├── experience_metrics.csv
│   ├── experience_summary.csv
│   ├── reliability_summary.csv
│   ├── failure_analysis.csv
│   ├── failure_pattern_summary.csv
│   ├── evaluation_summary.csv
│   └── key_findings.csv
│
├── notebooks/
│   └── evaluation_analysis.ipynb
│
├── visualizations/
│   ├── confusion_matrix.png
│   ├── category_performance.png
│   ├── overall_classification_metrics.png
│   ├── skill_performance.png
│   ├── reliability_metrics.png
│   └── failure_patterns.png
│
├── report/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Gemini API Configuration

Create a `.env` file in the project root.

Add:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from version control through `.gitignore`.

Do not upload API keys to GitHub.

---

## Running the Evaluation

Run all 60 test cases:

```bash
python src/run_evaluation.py
```

Calculate classification metrics:

```bash
python src/evaluate_classification.py
```

Evaluate performance by test category:

```bash
python src/evaluate_categories.py
```

Evaluate matched and missing skills:

```bash
python src/evaluate_skills.py
```

Analyze skill-level failures:

```bash
python src/analyze_skill_failures.py
```

Evaluate experience matching:

```bash
python src/evaluate_experience.py
```

Evaluate reliability, fairness and adversarial robustness:

```bash
python src/evaluate_reliability.py
```

Analyze decision failure patterns:

```bash
python src/analyze_failure_patterns.py
```

Generate evaluation visualizations:

```bash
python src/create_evaluation_visualizations.py
```

Create the consolidated evaluation summary:

```bash
python src/create_final_summary.py
```

---

## Evaluation Notebook

The complete analysis notebook is available at:

```text
notebooks/evaluation_analysis.ipynb
```

It includes:

- Dataset overview
- Test-case distribution
- Screening classification analysis
- Confusion matrix
- Category-level performance
- Incorrect decision analysis
- Matched-skill evaluation
- Missing-skill evaluation
- Unsupported skill attribution analysis
- Experience-match evaluation
- Reliability analysis
- Counterfactual fairness tests
- Prompt-injection tests
- Failure-pattern analysis
- Final evaluation summary

---

## Visualizations

Generated evaluation figures are stored in:

```text
visualizations/
```

The main visualizations are:

- Overall classification metrics
- Screening confusion matrix
- Category performance
- Skill identification performance
- Reliability and robustness metrics
- Decision failure patterns

---

## Limitations

This evaluation has several limitations:

- The evaluation dataset contains synthetic resumes rather than real recruitment records.
- Only one job role was evaluated.
- Only one LLM model was evaluated.
- The fairness evaluation used a small synthetic counterfactual test set.
- Ground-truth screening labels were manually defined and contain some subjective judgement.
- Model behavior may vary across repeated executions.
- Future versions of the model may produce different results.

---

## Conclusion

Gemini `gemini-3.1-flash-lite` demonstrated strong structured-output reliability, skill identification performance, counterfactual consistency, and prompt-injection resistance.

However, final screening decisions were less reliable for incomplete, ambiguous, and partially suitable candidates.

The model showed a tendency to convert some expected `Review` cases into `Reject`, even when resume information was interpreted correctly.

Therefore, the evaluated LLM is more appropriate as a **human-in-the-loop recruitment decision-support tool** than as a fully autonomous candidate rejection system.

Detailed analysis is available in:

```text
notebooks/evaluation_analysis.ipynb
```

and the final technical report.