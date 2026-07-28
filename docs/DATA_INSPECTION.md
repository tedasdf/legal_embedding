# Dataset inspection outputs and decision strategy

This guide documents the inspection results produced by
`src/data/inspect_dataset.py` for `isaacus/open-australian-legal-qa`. The
checks distinguish repeated QA rows, duplicate-document candidates, template
similarity, and shared boilerplate.

The dataset contains QA passages rather than guaranteed complete document
text. Text matches are therefore evidence about the sampled passages, not
automatic proof that the complete legal documents are duplicates.

## Output map

| Check | Output | Result | Intended use |
| --- | --- | ---: | --- |
| Repeated `version_id` | `repeated_version_id_samples.jsonl` | 12 rows across 6 repeated IDs | Confirm that multiple QA rows can belong to one document version. |
| Normalized URL across IDs | No candidate JSONL | 0 duplicate normalized URLs | A shared normalized URL would be strong evidence, but none was found. |
| Normalized citation across IDs | `citation_multi_version_candidates.jsonl` | 3 candidate rows | Inspect citations shared by different IDs; do not merge automatically. |
| Citation plus metadata across IDs | `metadata_multi_version_candidates.jsonl` | 3 candidate rows | Strengthen citation candidates using jurisdiction, type, and derived year. |
| Exact normalized passage hash | `exact_passage_duplicate_groups.jsonl` | 2 groups | Treat as exact passage-duplicate groups and inspect before removal. |
| High-Jaccard and similar length | `template_similar_passage_pairs.jsonl` | 2 pairs | Inspect as template-similar documents; never merge automatically. |
| High containment only | `boilerplate_overlap_warnings.jsonl` | 55 pairs | Record shared boilerplate only; ignore for deduplication. |

Each line in a JSONL file is one independent JSON object. Group outputs use
one line per group; pair outputs use one line per directly compared pair.

`text_similarity_candidates.jsonl` was a legacy broad candidate export from
before Check 4 was narrowed. The integrated `--all` command intentionally does
not regenerate it.

## Check 1: repeated `version_id`

Strategy:

1. Group rows by normalized `source.version_id`.
2. Count rows, passages, questions, citations, and URLs for each ID.
3. Inspect IDs occurring more than once for inconsistent citation or URL
   metadata.

Repeated IDs are not duplicate documents. They normally mean that more than
one QA example was sampled from the same document version. Preserve these
rows unless the question-passage pair itself is duplicated.

Output: `repeated_version_id_samples.jsonl`.

## Check 2: normalized URL across different IDs

URL normalization:

- lowercase the hostname only;
- standardize the scheme to HTTPS;
- remove fragments and trailing slashes;
- remove known tracking parameters such as `utm_*` and `source`;
- preserve path capitalization;
- preserve identity-bearing query parameters.

The dataset contains no source URL shared by multiple `version_id` values
after safe normalization. Consequently, this check has no candidate JSONL.
Do not remove all query parameters: some legislation URLs use them to identify
the document.

## Check 3: citation and metadata across different IDs

The citation check normalizes case, whitespace, quotes, and dashes. For
judgments, it also attempts to extract a neutral citation such as
`[2018] NSWCATAD 242`.

The metadata check combines:

```text
normalized citation
+ jurisdiction
+ document type
+ derived citation year
```

The three rows found by both checks are separate Tasmanian proclamations with
the same generic title but different statutory-rule identifiers. They should
remain separate documents. A shared citation or metadata combination is an
inspection signal, not an automatic merge rule.

Outputs:

- `citation_multi_version_candidates.jsonl`
- `metadata_multi_version_candidates.jsonl`

## Check 4: duplicate and repeated-text analysis

### A. Exact normalized passage duplicates

Passages are normalized and hashed with SHA-256. Different IDs are placed in
the same group only when their normalized passage hashes are identical.

Decision:

```text
identical normalized passage hash
→ create exact-duplicate group
→ inspect every question-answer-passage triple
→ keep when the answer is supported by the passage alone
→ remove only an unsupported positive pair
```

This produces two groups, including the three identical airworthiness
directive passages:

```text
F2006B04469
F2006B07349
F2006B08555
```

All five QA rows in the two exact-hash groups are supported:

- rows 447, 1490, and 1543 ask about the generic purpose and operation of an
  Airworthiness Directive, which their shared passage directly explains;
- rows 1230 and 1661 ask what legislative endnotes provide, which their shared
  passage directly explains.

The recommended action for all five rows is `keep`. Do not retain only one
arbitrary row per hash group: the questions are valid positive pairs even
though their passages contain the same boilerplate.

Output: `exact_passage_duplicate_groups.jsonl`.

### B. High-Jaccard, similar-length passages

Five-token shingles are compared using:

```python
possible_template_similarity = (
    jaccard_similarity >= 0.85
    and length_ratio >= 0.80
)
```

where:

```python
length_ratio = min(length_a, length_b) / max(length_a, length_b)
```

These pairs are labelled `template_similar_documents` and
`inspect_only_do_not_merge`. The tariff orders are separate legal instruments:
their dates and order numbers differ even though they use the same template.

The four rows in the two candidate pairs were manually checked:

- row 1159 is supported by its passage, including order number `0405836`,
  its original date, the revocation date, and the replacement action;
- row 1351 is supported by its passage, including order number `0406369`,
  its original date, the revocation date, and the replacement action;
- row 295 is supported by its passage, including the `30.05.11` effective
  date and the conditions under which the order ceases to operate;
- row 1261 is supported by its passage, including the `03.11.11` effective
  date and the conditions under which the order ceases to operate.

All four rows should be kept as separate QA pairs. Their passages have similar
templates, but their document-specific facts differ.

Output: `template_similar_passage_pairs.jsonl`.

### C. Containment and shared boilerplate

A short generic introduction or disclaimer contained in a longer passage is
not a duplicate. Containment is therefore never a grouping edge.

Decision:

```text
containment >= 0.90 without the near-match rule
→ possible shared boilerplate
→ report only
→ ignore for deduplication
```

NSW judgments linked by the standard tribunal disclaimer belong here. They
are unrelated cases and must not be grouped, merged, or removed.

Output: `boilerplate_overlap_warnings.jsonl`.

## Final deduplication policy

| Signal | Action |
| --- | --- |
| Repeated `version_id` | Preserve; it represents multiple QA examples from one document version. |
| Same safely normalized URL across IDs | Manually verify as a strong document-duplicate candidate. |
| Same citation or metadata across IDs | Inspect; never merge solely on this signal. |
| Exact normalized passage hash | Review each QA pair; keep supported rows and remove only unsupported positive pairs. |
| Jaccard ≥ 0.85 and length ratio ≥ 0.80 | Label template-similar; do not merge automatically. |
| High containment or shared disclaimer | Record as boilerplate and ignore for deduplication. |

No connected-component grouping is used for near or containment matches.
This avoids large false-positive clusters joined transitively through common
legal templates, introductions, endnotes, or disclaimers.
