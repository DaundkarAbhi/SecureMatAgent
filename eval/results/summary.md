# SecureMatAgent — Evaluation Summary

_Generated: 2026-03-18 12:00:40_

---

## RAGAS Scores

| Metric | Score |
|--------|-------|
| Faithfulness | 0.527 |
| Answer Relevancy | 0.691 |
| Context Precision | 0.300 |
| Context Recall | 0.042 |

_Evaluated 10/10 questions in 1529s_

### Per-Domain Breakdown

| Domain | Count | Faithfulness | Answer Relevancy | Keyword Overlap |
|--------|-------|-------------|-----------------|-----------------|
| materials_science | 5 | 0.250 | 0.000 | 0.294 |
| cybersecurity | 3 | 0.929 | 0.952 | 0.205 |
| cross_domain | 2 | 0.000 | 0.860 | 0.268 |

### Worst-Performing Questions (by Keyword Overlap)

| ID | Domain | Difficulty | Keyword Overlap | Faithfulness |
|----|--------|------------|-----------------|--------------|
| MS-007 | materials_science | medium | 0.194 | N/A |
| CY-001 | cybersecurity | easy | 0.245 | 0.857 |
| MS-001 | materials_science | easy | 0.261 | 0.250 |
| MS-010 | materials_science | hard | 0.315 | N/A |
| XD-001 | cross_domain | hard | 0.321 | 0.000 |

---

## Tool Selection Accuracy

**Overall accuracy:** 80.0%  (8/10)

- Positive cases: 100.0%
- Negative cases (abstain): 33.3%

| Domain | Accuracy |
|--------|----------|
| materials_science | 80.0% |
| cybersecurity | 100.0% |
| cross_domain | 50.0% |


### Per-Tool Precision / Recall / F1

| Tool | Precision | Recall | F1 |
|------|-----------|--------|----|
| document_search | 0.667 | 1.000 | 0.800 |
| materials_calculator | 1.000 | 1.000 | 1.000 |
| data_anomaly_checker | 1.000 | 1.000 | 1.000 |
| none | 1.000 | 0.333 | 0.500 |

### Tool Confusion Matrix

```
Expected\Actual         data_anomaly_  document_sear  materials_cal  none          
-----------------------------------------------------------------------------------
data_anomaly_checker    1              0              0              0
document_search         0              4              0              0
materials_calculator    0              0              2              0
none                    0              2              0              1
```


---

## Improvement Recommendations

- **Negative case accuracy is low.** The agent is using tools for questions outside its knowledge base. Add explicit 'I don't know' instructions to the system prompt.

- **Low recall for `none` (33.3%).** The agent often uses a different tool when `none` is expected. Review the tool description and few-shot examples.

- **Faithfulness is 0.53 (< 0.70).** The agent is generating answers not grounded in retrieved context. Consider reducing temperature, adding explicit grounding instructions, or increasing `top_k`.

- **Answer relevancy is 0.69 (< 0.70).** Answers are drifting from the question. Review the ReAct prompt template and consider adding answer format constraints.

- **Context recall is 0.04 (< 0.60).** The retriever is missing relevant chunks. Consider increasing `TOP_K`, re-tuning chunk size, or using a higher-dimensional embedding model.
