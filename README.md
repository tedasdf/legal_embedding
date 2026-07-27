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
