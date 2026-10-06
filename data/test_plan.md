# LLM Resume Screening Evaluation Test Plan

## Total Test Cases

60

## Test Categories

| Category | Test IDs | Count |
|---|---|---:|
| Strong Match | TC001-TC010 | 10 |
| Weak Match | TC011-TC020 | 10 |
| Borderline Match | TC021-TC030 | 10 |
| Missing or Ambiguous Information | TC031-TC035 | 5 |
| Skill Synonyms | TC036-TC040 | 5 |
| Irrelevant Experience | TC041-TC045 | 5 |
| Fairness / Counterfactual | TC046-TC050 | 5 |
| Prompt Injection / Adversarial | TC051-TC055 | 5 |
| Formatting / Unusual Resume | TC056-TC060 | 5 |

## Evaluation Dimensions

- Screening decision accuracy
- Precision
- Recall
- F1-score
- Skill extraction performance
- Missing-skill detection
- Experience-match accuracy
- JSON validity
- Hallucination rate
- Robustness
- Counterfactual consistency
- Prompt-injection resistance