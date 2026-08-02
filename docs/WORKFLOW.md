# Legal Retrieval Dataset Inspection and Split Strategy

## 1. Retrieval objective

A retrieval system contains three important elements:

* the user query;
* the passage being retrieved;
* the larger document containing that passage.

Different retrieval tasks require different forms of generalisation. A system may need to answer new queries over a fixed document collection, retrieve an unseen passage from a document represented during training, or process entirely new documents and queries.

For this project, the primary objective is:

> Given a new legal query, retrieve the correct passage from a legal document that was not used during model fine-tuning.

This reflects a realistic legal-retrieval setting. New judgments, contracts, court filings, legislative amendments and client documents can be divided into passages, embedded and added to the retrieval index without retraining the embedding model.

Legal documents reuse recurring concepts and structures, including:

* breach;
* negligence;
* causation;
* termination;
* confidentiality;
* procedural fairness;
* limitation periods;
* statutory interpretation.

This repetition allows the model to learn relationships between differently worded passages that concern the same legal issue. However, small wording differences can materially change legal meaning, such as:

* `may` compared with `must`;
* an obligation compared with a prohibition;
* immediate termination compared with termination after notice;
* a general rule compared with an exception;
* capped liability compared with uncapped liability.

The model must therefore learn both:

1. broad legal-semantic similarity across different documents and wording; and
2. fine-grained distinctions that may affect legal interpretation.

The main evaluation should test generalisation to both new queries and passages from held-out legal documents. This motivates a document-level split in which every row associated with a given `source.version_id` remains entirely within the train, validation or test partition.

---

## 2. Dataset structure

The inspected dataset is:

```text
isaacus/open-australian-legal-qa
```

The local source file contains 2,124 JSONL rows. Each row contains a synthetic legal question-and-answer pair generated from a passage of an Australian legal document.

A representative row has the following structure:

```python
{
    "question": "In the case of Nasr v NRMA Insurance [2006] NSWSC 1018, "
                "why was the plaintiff's appeal lodged out of time?",

    "answer": "The summons was filed approximately seven months after the "
              "Local Court decision, and no explanation was provided for the delay.",

    "text": "Question: ...\nAnswer: ...",

    "prompt": "The generation prompt containing document metadata, the source "
              "snippet and QA-generation instructions.",

    "source": {
        "version_id": "nsw_caselaw:549fc6183004262463bb648a",
        "type": "decision",
        "jurisdiction": "new_south_wales",
        "source": "nsw_caselaw",
        "citation": "Nasr v NRMA Insurance [2006] NSWSC 1018",
        "url": "https://www.caselaw.nsw.gov.au/decision/549fc6183004262463bb648a",
        "text": "The original legal passage from which the QA pair was generated."
    }
}
```

### Retrieval fields

```python
query = row["question"]
positive_passage = row["source"]["text"]
document_id = row["source"]["version_id"]
```

The embedding model is trained to place the query close to its corresponding positive passage.

| Field                 | Meaning                                                                        | Retrieval use                                                                |
| --------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `question`            | Synthetic legal query generated from the source passage                        | Query                                                                        |
| `answer`              | Answer generated from the source passage                                       | QA validation; not normally required for bi-encoder training                 |
| `text`                | Combined question-and-answer string                                            | Do not use as the retrieval passage because it contains the query and answer |
| `prompt`              | Full QA-generation prompt, including metadata, source snippet and instructions | Provenance and debugging                                                     |
| `source.text`         | Original legal passage                                                         | Positive passage                                                             |
| `source.version_id`   | Identifier for the underlying legal document or document version               | Primary document-level grouping key                                          |
| `source.source`       | Dataset or source provider, such as `nsw_caselaw`                              | Metadata                                                                     |
| `source.url`          | Original document URL                                                          | Duplicate-document inspection and provenance                                 |
| `source.jurisdiction` | Jurisdiction associated with the document                                      | Metadata and stratified analysis                                             |
| `source.citation`     | Formal citation or title                                                       | Duplicate-document inspection and provenance                                 |
| `source.type`         | Legal-document type, such as `decision`                                        | Metadata and stratified analysis                                             |

### Document ID versus passage ID

`source.version_id` identifies the source document or document version. It does not necessarily identify a unique passage.

```text
Legal document
└── source.version_id
    ├── passage 1
    ├── passage 2
    ├── passage 3
    └── passage 4
```

The dataset does not provide a separate passage identifier. A stable passage ID can therefore be generated from the normalised document ID and passage text:

```python
import hashlib


def create_passage_id(row: dict) -> str:
    document_id = row["source"]["version_id"].strip().lower()
    passage = " ".join(row["source"]["text"].split())

    passage_hash = hashlib.sha256(
        passage.encode("utf-8")
    ).hexdigest()[:16]

    return f"{document_id}:{passage_hash}"
```

This gives each field a distinct responsibility:

```text
source.version_id
    → identifies and groups the legal document

passage_id
    → identifies one unique passage from that document

source.text
    → contains the passage content
```

---

## 3. Leakage model

The desired evaluation setting is a new query over a held-out legal document. Leakage can occur on the query side, passage side or document side.

| Validation question compared with training question | Same passage                                                                     | Near-same or overlapping passage                                            | Different passage from same document  | Different passage from different document                                 |
| --------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| **Exact same question**                             | Critical direct leakage: the query–passage pair may already have been trained on | High leakage: the query is memorised and much of the content was seen       | Query leakage and document dependence | Query leakage, valid multi-positive query, ambiguity or incorrect pairing |
| **Near-same question**                              | High leakage: near-duplicate supervision over a seen passage                     | High leakage on both query and passage sides                                | Moderate-to-high contamination        | Query-side contamination that weakens an unseen-query claim               |
| **Different question**                              | Passage leakage: new query over a passage used in training                       | Content leakage: validation content substantially overlaps training content | Document-level contamination          | Cleanest case, assuming no hidden duplicate documents                     |

The primary split policy must therefore prevent:

1. rows from the same `source.version_id` appearing in different splits; and
2. exactly identical passages associated with different IDs appearing across different splits.

Approximate template similarity and shared boilerplate are recorded for audit purposes but are not sufficient evidence for merging documents or forcing them into the same split.

---

## 4. Inspection outputs

The inspection distinguishes repeated IDs, duplicate-document candidates, exact passage reuse, template similarity and shared boilerplate.

The dataset contains sampled QA passages rather than guaranteed complete document text. Consequently:

> A passage match is evidence about the sampled text, not automatic proof that the complete legal documents are duplicates.

### Output summary

| Check                                    | Output                                    |                        Result | Final action                                                                   |
| ---------------------------------------- | ----------------------------------------- | ----------------------------: | ------------------------------------------------------------------------------ |
| 1A. Repeated `version_id`                | `repeated_version_id_samples.jsonl`       | 12 rows across 6 repeated IDs | Keep valid rows and keep each `version_id` within one split                    |
| 1B. Same normalised URL across IDs       | No candidate JSONL                        |   0 duplicate normalised URLs | No action required                                                             |
| 2. Same normalised citation across IDs   | `citation_multi_version_candidates.jsonl` |              3 candidate rows | Retain as separate documents after inspection                                  |
| 3. Same citation and metadata across IDs | `metadata_multi_version_candidates.jsonl` |              3 candidate rows | Retain as separate documents after inspection                                  |
| 4A. Exact normalised passage hash        | `exact_passage_duplicate_groups.jsonl`    |    2 groups containing 5 rows | Keep all supported rows and keep each exact-text group within one split        |
| 4B. High Jaccard and similar length      | `template_similar_passage_pairs.jsonl`    |                       2 pairs | Retain as distinct legal instruments                                           |
| 4C. High containment or shared text      | `boilerplate_overlap_warnings.jsonl`      |                      55 pairs | Retain all rows and ignore these relationships for deduplication and splitting |

Each line in a JSONL file is an independent JSON object:

* group outputs contain one object per group;
* pair outputs contain one object per directly compared pair.

The legacy file `text_similarity_candidates.jsonl` was generated before Check 4 was divided into exact matches, template similarity and containment warnings. It is retained only for audit history and must not be used for deduplication, grouping or split assignment.

---

## 5. Check 1A: repeated `version_id`

### Purpose

This check identifies rows that share the same normalised `source.version_id`.

Repeated IDs are not evidence of duplicate documents. They normally indicate that multiple QA examples were sampled from the same legal document version.

### Result

```text
Total rows:                  2,124
Non-missing version IDs:     2,124
Missing version IDs:         0
Unique version IDs:          2,118
Repeated version-ID groups:  6
Extra rows from repeats:      6
```

Repeated IDs:

```text
federal_court_of_australia:fca/single/1983/1983fca0149
federal_court_of_australia:fca/single/1995/1995fca1173
federal_court_of_australia:fca/single/1997/1997fca0188
federal_court_of_australia:fca/single/1997/1997fca0669
nsw_caselaw:549f7a493004262463a9533b
nsw_caselaw:599f6857e4b058596cba9963
```

Each repeated ID occurs twice, producing 12 rows across 6 document versions.

### Decision

```text
repeated version_id
→ multiple QA examples from one document version
→ retain valid rows
→ keep all rows for the document in the same split
```

A row should only be removed if the exact question–passage pair is duplicated or the passage does not support the generated answer.

### Output

```text
repeated_version_id_samples.jsonl
```

---

## 6. Check 1B: same normalised URL across different IDs

### Purpose

This check identifies different `version_id` values that resolve to the same safely normalised source URL. A shared source URL would be strong evidence that two IDs may refer to the same underlying legal document.

### URL normalisation

The normalisation procedure:

* lowercases the hostname;
* standardises the scheme to HTTPS;
* removes fragments and trailing slashes;
* removes known non-identity tracking parameters such as `utm_*` and `source`;
* preserves path capitalisation;
* preserves query parameters that may identify the document.

Removing all query parameters would be unsafe. For example, 30 Western Australian legislation records use the shared path:

```text
https://www.legislation.wa.gov.au/legislation/statutes.nsf/RedirectURL
```

but are distinguished by identity-bearing query parameters such as:

```text
?OpenAgent&query=mrdoc_3678.docx
?OpenAgent&query=mrdoc_1299.docx
?OpenAgent&query=mrdoc_40983.docx
```

### Result

```text
Dataset URLs inspected:                     2,124
Missing URLs:                               0
Exact URLs linked to multiple IDs:          0
Safely normalised URLs with multiple IDs:   0
Path case-variant groups:                   0
```

### Decision

No exact or safely normalised URL was associated with more than one `version_id`, so no candidate JSONL was produced.

This result supports the use of `source.version_id` as the document-level key, but it does not rule out duplicates appearing through:

* different providers;
* HTML and PDF copies;
* original and corrected versions;
* mirrored legal databases;
* different URLs for substantively identical documents.

---

## 7. Check 2: same normalised citation across different IDs

### Purpose

This check identifies different IDs that share the same normalised `source.citation`.

Citation normalisation includes:

* case normalisation;
* whitespace normalisation;
* quote and dash normalisation;
* neutral-citation extraction where possible.

For judgments, a repeated neutral citation such as `[2006] NSWSC 1018` would be strong evidence that multiple IDs may refer to the same judgment. Legislative and secondary-instrument titles are less reliable because generic titles may legitimately identify different instruments.

### Result

One normalised citation was associated with three different IDs:

```text
Proclamation under the National Parks and Wildlife Act 1970 (Tas)
```

Candidate IDs:

```text
tasmanian_legislation:2017-07-05/sr-1999-091
tasmanian_legislation:2017-07-05/sr-2000-117
tasmanian_legislation:2017-07-05/sr-2001-086
```

Manual inspection showed that these were separate statutory instruments:

| Statutory rule | Subject                                                         |
| -------------- | --------------------------------------------------------------- |
| `sr-1999-091`  | Long Spit Private Nature Reserve                                |
| `sr-2000-117`  | Deal Island Conservation Area                                   |
| `sr-2001-086`  | Lime Bay, Peter Murrell and Three Hummock Island State Reserves |

They differ in statutory-rule number, year, URL, affected land, legal provision and operative effect.

### Decision

The citation collision is a false-positive duplicate signal caused by a generic legislative title.

```text
same normalised citation
→ generate candidate
→ inspect manually
→ never merge solely on citation
```

### Output

```text
citation_multi_version_candidates.jsonl
```

---

## 8. Check 3: same citation and metadata across different IDs

### Purpose

This check strengthens the citation comparison by combining:

```text
normalised citation
+ jurisdiction
+ document type
+ derived citation year
```

### Result

The same three Tasmanian proclamations were identified. The additional metadata did not establish that they were duplicates because their statutory-rule identifiers and substantive legal effects remained different.

### Decision

A shared citation-and-metadata key is a stronger candidate-generation signal than citation alone, but it is still not proof of document duplication.

All three instruments are retained separately.

### Output

```text
metadata_multi_version_candidates.jsonl
```

---

## 9. Check 4: passage-level similarity

Text similarity was divided into three operationally different cases:

1. exact normalised passage duplicates;
2. high-Jaccard, similar-length passages;
3. high-containment or shared-text warnings.

This separation prevents common legal wording from creating large false-positive duplicate clusters.

### Similarity measures

Passages are represented using five-token shingles.

For shingle sets (A) and (B), Jaccard similarity is:

[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
]

Shorter-passage containment is:

[
C(A,B)=\frac{|A\cap B|}{\min(|A|,|B|)}
]

Jaccard similarity measures overall overlap. Containment detects cases in which a short generic passage appears almost entirely inside a longer passage.

LSH is used only to generate likely candidate pairs efficiently. Exact Jaccard, containment and length-ratio values are then calculated for those candidates.

### 9.1 Check 4A: exact normalised passage duplicates

#### Method

```text
normalise passage
→ calculate SHA-256 hash
→ group different version_ids with the same hash
```

This is the strongest passage-level duplication signal. It proves exact reuse of the sampled passage, but not necessarily duplication of the complete legal documents.

#### Result

Two exact-text groups were identified.

**Group 1: CASA Airworthiness Directive wording**

Rows:

```text
447
1490
1543
```

Document IDs:

```text
federal_register_of_legislation:F2006B04469
federal_register_of_legislation:F2006B07349
federal_register_of_legislation:F2006B08555
```

The shared passage explains the general purpose and operation of CASA Airworthiness Directives.

**Group 2: legislative endnotes**

Rows:

```text
1230
1661
```

Document IDs:

```text
federal_register_of_legislation:C2016C01044
federal_register_of_legislation:F2018C00828
```

The shared passage explains the information contained in legislative endnotes.

#### Manual review

| Rows            | Shared content                                         | Supported by passage? | Action |
| --------------- | ------------------------------------------------------ | --------------------: | ------ |
| 447, 1490, 1543 | Purpose and operation of CASA Airworthiness Directives |                   Yes | Keep   |
| 1230, 1661      | Information provided by legislative endnotes           |                   Yes | Keep   |

All five QA pairs are supported by their passages, so no rows are removed.

#### Evaluation treatment

Members of an exact-text group must remain within the same split.

During evaluation, identical passages must not be treated as negatives for one another. Two valid approaches are available:

* **Equivalent positives:** treat every identical passage in the group as a valid positive;
* **Canonical passage:** store the passage once and associate it with all supported questions and source metadata.

#### Output

```text
exact_passage_duplicate_groups.jsonl
```

### 9.2 Check 4B: high-Jaccard, similar-length passages

#### Rule

A pair is labelled as template-similar when:

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

The length-ratio condition reduces false positives caused by a short disclaimer or introduction being contained within a much longer passage.

#### Result

Two pairs were identified.

**Pair 1: Tariff Concession Revocation Orders**

Rows:

```text
1159
1351
```

The passages reuse the same statutory template but differ in order number, original order date, citation and source URL.

**Pair 2: Tariff Concession Orders**

Rows:

```text
295
1261
```

The passages reuse the same statutory template but differ in concession-order number, effective date, citation and source URL.

#### Decision

These are separate legal instruments.

```text
high Jaccard + similar length
→ shared drafting template
→ retain both documents
→ do not merge or deduplicate
```

Template similarity is not used as a split constraint. Generalisation to a new legal document may legitimately involve a drafting structure that was seen during training.

#### Output

```text
template_similar_passage_pairs.jsonl
```

### 9.3 Check 4C: high containment and shared-text warnings

#### Rule

```text
containment >= 0.90
and the pair does not satisfy the Check 4B rule
→ record a shared-text warning
→ do not use as a grouping edge
```

This output includes shared boilerplate and highly templated text that falls below the stricter 4B Jaccard threshold.

#### Result

The output contains 55 pairs. The main patterns are:

* NSW tribunal certification and publication disclaimers;
* standard CASA Airworthiness Directive introductions;
* repeated statutory wording in tariff instruments.

These similarities arise from recurring legal language rather than confirmed duplicate documents.

#### Decision

```text
high containment or shared boilerplate
→ retain all rows
→ do not merge
→ do not deduplicate
→ do not construct connected components
→ do not use for split assignment
```

The pairs do not require individual manual review. The file is retained as an audit record showing why containment alone cannot establish duplication.

#### Output

```text
boilerplate_overlap_warnings.jsonl
```

---

## 10. Final deduplication policy

| Signal                                 | Interpretation                                      | Dataset action                                    | Split action                                |
| -------------------------------------- | --------------------------------------------------- | ------------------------------------------------- | ------------------------------------------- |
| Repeated `version_id`                  | Multiple QA examples from one document version      | Keep valid rows                                   | Keep the same `version_id` within one split |
| Same safely normalised URL across IDs  | Strong duplicate-document candidate                 | Verify manually                                   | Group only if confirmed                     |
| Same citation across IDs               | Citation collision or possible version relationship | Inspect only                                      | No automatic grouping                       |
| Same citation and metadata             | Stronger candidate, but not proof                   | Inspect only                                      | No automatic grouping                       |
| Exact normalised passage hash          | Exact passage reuse                                 | Validate question support and keep supported rows | Keep the exact-text group within one split  |
| Jaccard ≥ 0.85 and length ratio ≥ 0.80 | Shared legal template                               | Keep as distinct documents                        | Split independently                         |
| High containment or shared boilerplate | Repeated generic legal language                     | Keep and ignore for deduplication                 | Split independently                         |

No connected-component grouping is used for approximate or containment matches. This avoids large false-positive clusters joined transitively through common legal introductions, disclaimers, endnotes and drafting templates.

---

## 11. Split policy

The default split unit is the normalised document ID:

```text
source.version_id
```

All QA rows associated with one document version must remain in the same split.

The only additional constraint discovered by the inspection applies to exact passage groups:

```text
same version_id
or
same exact normalised passage group
→ same split
```

Conceptually:

```python
split_group_id = normalised_version_id

if normalised_version_id in exact_passage_group:
    split_group_id = exact_passage_group_id
```

Template-similar pairs and containment warnings remain independent because they represent distinct legal documents rather than confirmed duplicates.

After constructing the split, a final leakage audit should verify that:

* no `version_id` occurs in more than one split;
* no exact normalised passage hash occurs in more than one split;
* no exact question–passage pair occurs in more than one split.

---