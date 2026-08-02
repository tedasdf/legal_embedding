# Embedding Model Validation Results

## Evaluation scope

The comparison uses the same 214-query validation split for all four
fine-tuned models. Because the split is small, one query changes recall by:

```text
1 / 214 = approximately 0.00467
```

Small aggregate differences may therefore represent only one or two queries.
Exact query counts should be reported alongside metrics wherever possible.

## Overall conclusion

All four fine-tuning strategies achieved strong retrieval performance. The V1
in-batch model retrieved 203 relevant passages at rank one and 213 within the
top five. BM25 hard-negative training increased top-one retrieval to 204
queries. Dense hard-negative training achieved the strongest top-one result,
retrieving 206 queries at rank one and 212 within the top five. The distilled
model performed worse overall, retrieving 198 queries at rank one and 208
within the top five.

Dense hard-negative training achieved the highest Recall@1, MRR and NDCG@10
and is therefore the preferred checkpoint for subsequent release experiments.
Its advantage is nevertheless small in absolute query count. Its slightly
lower Recall@5 and Recall@10 than V1 also shows that it does not dominate V1
on every query.

The dense result is consistent with the hypothesis that training against
semantically similar passages helps the model distinguish the correct legal
passage from difficult alternatives. It is not, by itself, sufficient evidence
of a general causal advantage. The difference should be examined through
paired per-query comparisons and, where appropriate, bootstrap confidence
intervals.

Distillation did not improve aggregate retrieval performance. It did,
however, improve the hardest shared failure from rank 50 under dense mining to
rank 7. This single-query improvement does not offset the lower aggregate
performance, but it suggests that teacher supervision may help selected
difficult examples. Further experiments could vary the teacher, temperature
and distillation weight.

## BM25 versus dense hard-negative training

BM25 and dense hard-negative training achieved the same Recall@5 score of
0.9907: both retrieved the relevant passage within the top five for 212 of the
214 validation queries. They did not succeed on exactly the same samples.
Their top-five successes overlapped on 211 queries, and each model retrieved
one query that the other missed.

| Category | Query ID |
|---|---|
| Missed by both | `q_00105c0c288bb5609a70` |
| Missed only by BM25 | `q_d1f6e1e30ab5e342f8dd` |
| Missed only by dense | `q_0b5eafb20d26174fbb29` |

| Query | BM25 rank | Dense rank | Top-five outcome |
|---|---:|---:|---|
| `q_00105c0c288bb5609a70` | 123 | 50 | Both fail |
| `q_d1f6e1e30ab5e342f8dd` | 11 | 4 | Dense succeeds |
| `q_0b5eafb20d26174fbb29` | 4 | 25 | BM25 succeeds |

### Lancaster and Canny

The Lancaster and Canny query was ranked eleventh by the BM25-trained model
and fourth by the dense-trained model. It asks whether Rule 73(b)(x)
prescribes an objective standard.

One hypothesis is that the supporting passage expresses the answer through
semantic or interpretative legal language rather than the exact phrase
“objective standard.” Dense hard-negative training may therefore have helped
on this example. This explanation remains provisional until the gold passage
and the models' higher-ranked alternatives are inspected.

### Du v Feng

The Du v Feng query was ranked fourth by the BM25-trained model and
twenty-fifth by the dense-trained model. It asks what the dispute between the
parties concerned.

One hypothesis is that many unrelated legal passages describe disputes using
similar semantic language, while exact party names, citations and
case-specific terminology help the BM25-trained model preserve the identity
of the relevant judgment. This should be confirmed by inspecting the dense
model's higher-ranked passages.

### Dawson v Howard

The Dawson v Howard query was the only sample that both models failed to
retrieve within the top five. BM25 ranked the relevant passage at 123, while
dense training improved it to rank 50. Distillation subsequently moved it to
rank 7.

The query asks for the main issue considered in the case. It should be
manually checked for:

- an annotation mismatch;
- evidence fragmented across several passages;
- a positive passage that does not directly support the answer;
- wording that requires document-level summarisation rather than local
  passage retrieval.

## Interpretation and next steps

Equal aggregate recall does not imply identical model behaviour. The paired
results are consistent with BM25 hard-negative training helping when exact
legal identifiers and case-specific vocabulary matter, while dense
hard-negative training may help when relevance depends on semantic paraphrase
or legal interpretation. These explanations remain hypotheses until the
retrieved passages are inspected.

Recommended next steps:

1. Select V3 dense as the primary checkpoint for release evaluation.
2. Retain V1 and V2 as comparison baselines.
3. Inspect the gold and higher-ranked passages for the three disagreement
   queries.
4. Run paired bootstrap confidence intervals before making a strong claim
   about small metric differences.
5. Evaluate lexical-dense score fusion as a separate hybrid retrieval
   experiment.
6. Treat V4 as an ablation and tune its teacher, temperature and distillation
   weight before reconsidering it as the primary model.

Hybrid retrieval is a proposed next experiment; the current results do not yet
show that score fusion improves the validation set.
