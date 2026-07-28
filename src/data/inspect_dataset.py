#!/usr/bin/env python3
"""Run reproducible integrity checks over the Open Australian Legal QA data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


DEFAULT_INPUT = Path(
    "data/raw/embed/open-australian-legal-qa-v2.0.0/qa.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/inspection")


def inspect_version_ids(input_path: Path) -> dict:
    """Count version IDs and retain row indices for repeated IDs."""
    version_rows: dict[str, list[int]] = defaultdict(list)
    total_rows = 0
    missing_rows: list[int] = []

    with input_path.open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue

            total_rows += 1
            row = json.loads(line)
            source = row.get("source")
            source = source if isinstance(source, dict) else {}
            version_id = source.get("version_id")

            if not isinstance(version_id, str) or not version_id.strip():
                missing_rows.append(row_index)
                continue

            # Casefolding prevents capitalization-only differences from being
            # counted as distinct source documents.
            normalized_id = version_id.strip().casefold()
            version_rows[normalized_id].append(row_index)

    repeated = {
        version_id: indices
        for version_id, indices in sorted(version_rows.items())
        if len(indices) > 1
    }
    unique_once = {
        version_id: indices
        for version_id, indices in version_rows.items()
        if len(indices) == 1
    }

    return {
        "input": str(input_path),
        "total_rows": total_rows,
        "non_missing_version_ids": total_rows - len(missing_rows),
        "missing_version_ids": len(missing_rows),
        "missing_row_indices": missing_rows,
        "unique_version_ids": len(version_rows),
        "version_ids_appearing_once": len(unique_once),
        "rows_with_version_ids_appearing_once": sum(
            len(indices) for indices in unique_once.values()
        ),
        "version_ids_appearing_more_than_once": len(repeated),
        "rows_with_repeated_version_ids": sum(
            len(indices) for indices in repeated.values()
        ),
        "repeated_version_id_groups": len(repeated),
        "extra_rows_from_repeated_ids": sum(
            len(indices) - 1 for indices in repeated.values()
        ),
        "repeated_version_ids": repeated,
    }


def load_repeated_records(
    input_path: Path,
    repeated_version_ids: set[str],
) -> list[dict]:
    """Load compact samples for every row belonging to a repeated ID."""
    samples = []
    with input_path.open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            source = source if isinstance(source, dict) else {}
            version_id = source.get("version_id")
            normalized_id = (
                version_id.strip().casefold()
                if isinstance(version_id, str)
                else ""
            )
            if normalized_id not in repeated_version_ids:
                continue
            samples.append(
                {
                    "row_index": row_index,
                    "version_id": version_id,
                    "citation": source.get("citation"),
                    "source_type": source.get("type"),
                    "jurisdiction": source.get("jurisdiction"),
                    "url": source.get("url"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "passage": source.get("text"),
                }
            )
    return samples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_repeated_samples(samples: list[dict], text_limit: int) -> None:
    """Print repeated records grouped by normalized version ID."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["version_id"].strip().casefold()].append(sample)

    for version_id, records in sorted(grouped.items()):
        print("\n" + "=" * 100)
        print(f"Version ID: {version_id}")
        print(f"Rows: {[record['row_index'] for record in records]}")
        for record in records:
            print("\n" + "-" * 100)
            print(f"Row index:    {record['row_index']}")
            print(f"Citation:     {record['citation']}")
            print(f"Type:         {record['source_type']}")
            print(f"Jurisdiction: {record['jurisdiction']}")
            print(f"URL:          {record['url']}")
            print(f"\nQuestion:\n{record['question']}")
            print(f"\nAnswer:\n{record['answer']}")
            passage = record["passage"] or ""
            suffix = "..." if len(passage) > text_limit else ""
            print(f"\nPassage preview:\n{passage[:text_limit]}{suffix}")


def normalize_url_safely(value: str) -> str:
    """Normalize URL syntax without changing path case or identity parameters."""
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path.rstrip("/") or "/"

    # Remove only clearly non-identity tracking parameters. Other query
    # parameters may identify the document, as with WA RedirectURL records.
    query_items = [
        (name, item)
        for name, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not name.casefold().startswith("utm_")
        and name.casefold() not in {"source"}
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit(("https", host + port, path, query, ""))


def inspect_urls(input_path: Path) -> dict:
    """Audit URL normalization and path-capitalization evidence."""
    exact: dict[str, set[str]] = defaultdict(set)
    safe: dict[str, set[str]] = defaultdict(set)
    path_folded: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"paths": set(), "version_ids": set()}
    )
    path_without_query: dict[str, set[str]] = defaultdict(set)
    total_urls = 0

    with input_path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            source = source if isinstance(source, dict) else {}
            url = source.get("url")
            version_id = source.get("version_id")
            if not isinstance(url, str) or not url.strip():
                continue
            normalized_id = (
                version_id.strip().casefold()
                if isinstance(version_id, str)
                else ""
            )
            total_urls += 1
            parsed = urlsplit(url.strip())
            host = (parsed.hostname or "").lower()
            path = parsed.path.rstrip("/") or "/"
            exact[url.strip()].add(normalized_id)
            safe[normalize_url_safely(url)].add(normalized_id)
            folded_key = f"{host}{path.casefold()}"
            path_folded[folded_key]["paths"].add(path)
            path_folded[folded_key]["version_ids"].add(normalized_id)
            path_without_query[f"{host}{path}"].add(normalized_id)

    case_variants = {
        key: {
            "paths": sorted(values["paths"]),
            "version_ids": sorted(values["version_ids"]),
        }
        for key, values in path_folded.items()
        if len(values["paths"]) > 1
    }
    unsafe_query_groups = {
        key: sorted(version_ids)
        for key, version_ids in path_without_query.items()
        if len(version_ids) > 1
    }

    return {
        "total_non_missing_urls": total_urls,
        "exact_urls_with_multiple_version_ids": sum(
            len(ids) > 1 for ids in exact.values()
        ),
        "safely_normalized_urls_with_multiple_version_ids": sum(
            len(ids) > 1 for ids in safe.values()
        ),
        "path_case_variant_groups": len(case_variants),
        "path_case_variants": case_variants,
        "paths_that_become_ambiguous_if_all_query_parameters_are_removed": len(
            unsafe_query_groups
        ),
        "unsafe_query_removal_groups": unsafe_query_groups,
        "conclusion": (
            "No path case variants occur in this dataset; preserve path case "
            "because provider case-insensitivity is not established. Preserve "
            "identity-bearing query parameters."
        ),
    }


def normalize_citation(value: str) -> str:
    """Normalize citation formatting without removing meaningful words."""
    value = value.replace("’", "'").replace("‘", "'")
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


NEUTRAL_CITATION_PATTERN = re.compile(
    r"\[(?:18|19|20)\d{2}\]\s+[A-Z][A-Z0-9]{1,19}\s+\d+",
    flags=re.IGNORECASE,
)


def extract_neutral_citation(value: str) -> str | None:
    """Extract a judgment's neutral citation, when present."""
    match = NEUTRAL_CITATION_PATTERN.search(value)
    return normalize_citation(match.group(0)) if match else None


def inspect_citations(input_path: Path) -> dict:
    """Find own citations and neutral citations shared by different IDs."""
    full_groups: dict[str, list[dict]] = defaultdict(list)
    neutral_groups: dict[str, list[dict]] = defaultdict(list)
    total_non_missing = 0

    with input_path.open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            source = source if isinstance(source, dict) else {}
            citation = source.get("citation")
            if not isinstance(citation, str) or not citation.strip():
                continue
            total_non_missing += 1
            record = {
                "row_index": row_index,
                "version_id": source.get("version_id"),
                "provider": source.get("source"),
                "citation": citation,
                "source_type": source.get("type"),
                "jurisdiction": source.get("jurisdiction"),
                "url": source.get("url"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "passage": source.get("text"),
            }
            full_groups[normalize_citation(citation)].append(record)
            neutral = extract_neutral_citation(citation)
            if neutral:
                neutral_groups[neutral].append(record)

    def multiple_id_groups(groups: dict[str, list[dict]]) -> dict:
        result = {}
        for citation, records in sorted(groups.items()):
            version_ids = {
                str(record["version_id"]).strip().casefold()
                for record in records
            }
            if len(version_ids) > 1:
                result[citation] = {
                    "version_ids": sorted(version_ids),
                    "providers": sorted(
                        {
                            str(record["provider"])
                            for record in records
                        }
                    ),
                    "rows": records,
                }
        return result

    full_multi_id = multiple_id_groups(full_groups)
    neutral_multi_id = multiple_id_groups(neutral_groups)
    return {
        "total_non_missing_citations": total_non_missing,
        "unique_normalized_full_citations": len(full_groups),
        "full_citations_with_multiple_version_ids": len(full_multi_id),
        "full_citation_candidates": full_multi_id,
        "rows_with_extracted_neutral_citations": sum(
            len(records) for records in neutral_groups.values()
        ),
        "unique_neutral_citations": len(neutral_groups),
        "neutral_citations_with_multiple_version_ids": len(neutral_multi_id),
        "neutral_citation_candidates": neutral_multi_id,
    }


YEAR_PATTERN = re.compile(r"\b(?:18|19|20)\d{2}\b")
ISO_DATE_PATTERN = re.compile(
    r"(?<!\d)((?:18|19|20)\d{2}-\d{2}-\d{2})(?!\d)"
)


def derive_metadata_dates(source: dict) -> dict:
    """Derive transparent date evidence without claiming a publication date."""
    citation = source.get("citation")
    citation = citation if isinstance(citation, str) else ""
    neutral = extract_neutral_citation(citation)
    neutral_year_match = YEAR_PATTERN.search(neutral or "")
    citation_year_match = YEAR_PATTERN.search(citation)
    version_id = source.get("version_id")
    version_id = version_id if isinstance(version_id, str) else ""
    version_date_match = ISO_DATE_PATTERN.search(version_id)

    if neutral_year_match:
        citation_year = neutral_year_match.group(0)
        citation_year_source = "neutral_citation"
    elif citation_year_match:
        citation_year = citation_year_match.group(0)
        citation_year_source = "first_year_in_source_citation"
    else:
        citation_year = None
        citation_year_source = None

    return {
        "citation_year": citation_year,
        "citation_year_source": citation_year_source,
        "version_date": (
            version_date_match.group(1)
            if version_date_match
            else None
        ),
    }


def inspect_metadata_combinations(input_path: Path) -> dict:
    """Compare coarse and citation-strengthened metadata across IDs."""
    coarse_groups: dict[tuple, list[dict]] = defaultdict(list)
    strong_groups: dict[tuple, list[dict]] = defaultdict(list)
    rows_without_derived_year = 0
    rows_with_version_date = 0

    with input_path.open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            source = source if isinstance(source, dict) else {}
            dates = derive_metadata_dates(source)
            if dates["citation_year"] is None:
                rows_without_derived_year += 1
            if dates["version_date"] is not None:
                rows_with_version_date += 1

            jurisdiction = normalize_citation(
                str(source.get("jurisdiction") or "")
            )
            document_type = normalize_citation(
                str(source.get("type") or "")
            )
            citation = str(source.get("citation") or "")
            normalized_own_citation = normalize_citation(citation)
            record = {
                "row_index": row_index,
                "version_id": source.get("version_id"),
                "provider": source.get("source"),
                "citation": citation,
                "normalized_citation": normalized_own_citation,
                "jurisdiction": source.get("jurisdiction"),
                "document_type": source.get("type"),
                **dates,
                "url": source.get("url"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "passage": source.get("text"),
            }
            coarse_key = (
                jurisdiction,
                document_type,
                dates["citation_year"],
            )
            strong_key = (
                normalized_own_citation,
                jurisdiction,
                document_type,
                dates["citation_year"],
            )
            coarse_groups[coarse_key].append(record)
            strong_groups[strong_key].append(record)

    def groups_with_multiple_ids(groups: dict[tuple, list[dict]]) -> list[dict]:
        candidates = []
        for key, records in groups.items():
            version_ids = sorted(
                {
                    str(record["version_id"]).strip().casefold()
                    for record in records
                }
            )
            if len(version_ids) < 2:
                continue
            candidates.append(
                {
                    "group_key": list(key),
                    "version_ids": version_ids,
                    "providers": sorted(
                        {str(record["provider"]) for record in records}
                    ),
                    "row_count": len(records),
                    "rows": records,
                }
            )
        return sorted(
            candidates,
            key=lambda group: (
                -len(group["version_ids"]),
                str(group["group_key"]),
            ),
        )

    coarse_candidates = groups_with_multiple_ids(coarse_groups)
    strong_candidates = groups_with_multiple_ids(strong_groups)
    return {
        "explicit_publication_date_field": False,
        "date_warning": (
            "The QA schema has no publication-date field. citation_year is "
            "derived from a neutral citation or the first year in "
            "source.citation; version_date is separately extracted from "
            "version_id where available and is not treated as publication."
        ),
        "rows_without_derived_citation_year": rows_without_derived_year,
        "rows_with_version_date_in_version_id": rows_with_version_date,
        "coarse_group_definition": [
            "jurisdiction",
            "document_type",
            "derived_citation_year",
        ],
        "coarse_groups_with_multiple_version_ids": len(coarse_candidates),
        "coarse_candidates": coarse_candidates,
        "strong_group_definition": [
            "normalized_source_citation",
            "jurisdiction",
            "document_type",
            "derived_citation_year",
        ],
        "strong_groups_with_multiple_version_ids": len(strong_candidates),
        "strong_candidates": strong_candidates,
    }


def normalize_passage(value: str) -> str:
    """Normalize passage text for exact and shingle-based comparison."""
    value = value.replace("’", "'").replace("‘", "'")
    value = value.replace("“", '"').replace("”", '"')
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def token_shingles(value: str, size: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE)
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def inspect_text_similarity(
    input_path: Path,
    shingle_size: int = 5,
    jaccard_threshold: float = 0.85,
    length_ratio_threshold: float = 0.8,
    containment_threshold: float = 0.9,
    maximum_shingle_frequency: int = 50,
) -> dict:
    """Compare aggregated QA snippets across different version IDs."""
    documents: dict[str, dict] = {}
    with input_path.open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            source = source if isinstance(source, dict) else {}
            version_id = str(source.get("version_id") or "").strip()
            passage = str(source.get("text") or "")
            if not version_id or not passage.strip():
                continue
            normalized_id = version_id.casefold()
            document = documents.setdefault(
                normalized_id,
                {
                    "version_id": version_id,
                    "providers": set(),
                    "citations": set(),
                    "urls": set(),
                    "row_indices": [],
                    "passages": set(),
                    "qa_examples": [],
                },
            )
            document["providers"].add(str(source.get("source")))
            document["citations"].add(str(source.get("citation")))
            document["urls"].add(str(source.get("url")))
            document["row_indices"].append(row_index)
            document["passages"].add(normalize_passage(passage))
            document["qa_examples"].append(
                {
                    "row_index": row_index,
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "citation": source.get("citation"),
                    "original_passage": passage,
                }
            )

    for document in documents.values():
        document["combined_text"] = "\n".join(
            sorted(document["passages"])
        )
        document["sha256"] = hashlib.sha256(
            document["combined_text"].encode("utf-8")
        ).hexdigest()
        document["shingles"] = token_shingles(
            document["combined_text"],
            shingle_size,
        )

    exact_hash_groups: dict[str, list[str]] = defaultdict(list)
    for version_id, document in documents.items():
        exact_hash_groups[document["sha256"]].append(version_id)

    exact_candidates = []
    for digest, version_ids in sorted(exact_hash_groups.items()):
        if len(version_ids) < 2:
            continue
        exact_candidates.append(
            {
                "match_type": "exact_aggregated_qa_snippet_hash",
                "sha256": digest,
                "version_ids": sorted(version_ids),
                "documents": [
                    _serializable_document(documents[version_id])
                    for version_id in sorted(version_ids)
                ],
            }
        )

    shingle_documents: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for version_id, document in documents.items():
        for shingle in document["shingles"]:
            shingle_documents[shingle].append(version_id)

    intersections: Counter[tuple[str, str]] = Counter()
    ignored_common_shingles = 0
    for version_ids in shingle_documents.values():
        if len(version_ids) > maximum_shingle_frequency:
            ignored_common_shingles += 1
            continue
        for left, right in itertools.combinations(sorted(version_ids), 2):
            intersections[(left, right)] += 1

    exact_pairs = {
        tuple(sorted((left, right)))
        for group in exact_candidates
        for left, right in itertools.combinations(group["version_ids"], 2)
    }
    template_candidates = []
    boilerplate_warnings = []
    for (left, right), shared_count in intersections.items():
        if (left, right) in exact_pairs:
            continue
        left_shingles = documents[left]["shingles"]
        right_shingles = documents[right]["shingles"]
        # Recalculate the true intersection because frequent indexing shingles
        # were skipped only for candidate generation.
        true_shared = len(left_shingles & right_shingles)
        union = len(left_shingles | right_shingles)
        shorter = min(len(left_shingles), len(right_shingles))
        longer = max(len(left_shingles), len(right_shingles))
        jaccard = true_shared / union if union else 0.0
        containment = true_shared / shorter if shorter else 0.0
        length_ratio = shorter / longer if longer else 0.0
        left_text = documents[left]["combined_text"]
        right_text = documents[right]["combined_text"]
        is_nsw_disclaimer = (
            "registrar disclaimer" in left_text
            and "registrar disclaimer" in right_text
        )
        is_template_candidate = (
            jaccard >= jaccard_threshold
            and length_ratio >= length_ratio_threshold
            and not is_nsw_disclaimer
        )
        is_boilerplate_warning = (
            containment >= containment_threshold
            and not is_template_candidate
        )
        if not is_template_candidate and not is_boilerplate_warning:
            continue
        candidate = {
            "match_type": (
                "template_similar_documents"
                if is_template_candidate
                else "possible_shared_boilerplate"
            ),
            "decision": (
                "inspect_only_do_not_merge"
                if is_template_candidate
                else "ignore_for_deduplication"
            ),
            "left_version_id": left,
            "right_version_id": right,
            "shared_shingles": true_shared,
            "jaccard_similarity": round(jaccard, 8),
            "length_ratio": round(length_ratio, 8),
            "shorter_text_containment": round(containment, 8),
            "left": _serializable_document(documents[left]),
            "right": _serializable_document(documents[right]),
        }
        if is_nsw_disclaimer:
            candidate["boilerplate_reason"] = "NSW tribunal disclaimer"
        if is_template_candidate:
            template_candidates.append(candidate)
        else:
            boilerplate_warnings.append(candidate)

    candidate_sort_key = lambda candidate: (
        -candidate["jaccard_similarity"],
        -candidate["length_ratio"],
        candidate["left_version_id"],
        candidate["right_version_id"],
    )
    template_candidates.sort(key=candidate_sort_key)
    boilerplate_warnings.sort(
        key=lambda candidate: (
            -candidate["shorter_text_containment"],
            -candidate["jaccard_similarity"],
            candidate["left_version_id"],
            candidate["right_version_id"],
        )
    )

    return {
        "full_document_text_available": False,
        "evidence_warning": (
            "source.text contains sampled QA snippets, not complete legal "
            "documents. Exact and near matches are strong snippet-level "
            "evidence, but non-matches cannot rule out duplicate documents."
        ),
        "version_ids_compared": len(documents),
        "normalization": (
            "casefold; collapse whitespace; standardize quotes and dashes; "
            "sort and join unique snippets per version_id"
        ),
        "parameters": {
            "shingle_size": shingle_size,
            "jaccard_threshold": jaccard_threshold,
            "length_ratio_threshold": length_ratio_threshold,
            "containment_threshold": containment_threshold,
            "maximum_shingle_frequency_for_candidate_generation": (
                maximum_shingle_frequency
            ),
        },
        "ignored_common_shingles_during_candidate_generation": (
            ignored_common_shingles
        ),
        "exact_cross_id_groups": len(exact_candidates),
        "exact_candidates": exact_candidates,
        "template_similar_pairs": len(template_candidates),
        "template_candidates": template_candidates,
        "boilerplate_overlap_pairs": len(boilerplate_warnings),
        "boilerplate_warnings": boilerplate_warnings,
    }


def _serializable_document(document: dict) -> dict:
    return {
        "version_id": document["version_id"],
        "providers": sorted(document["providers"]),
        "citations": sorted(document["citations"]),
        "urls": sorted(document["urls"]),
        "row_indices": sorted(document["row_indices"]),
        "unique_passage_count": len(document["passages"]),
        "combined_normalized_text": document["combined_text"],
        "qa_examples": sorted(
            document["qa_examples"],
            key=lambda row: row["row_index"],
        ),
    }


def build_text_similarity_examples(
    text_report: dict,
    example_count: int = 3,
) -> list[dict]:
    """Create compact, pairwise examples from the detailed candidate report."""
    examples: list[dict] = []

    for candidate in text_report["exact_candidates"]:
        documents = candidate["documents"]
        if len(documents) < 2:
            continue
        left, right = documents[:2]
        examples.append(
            {
                "example_type": "exact_passage_match",
                "explanation": (
                    "Different version IDs contain the same normalized QA "
                    "passage. This may be shared boilerplate and does not, by "
                    "itself, prove that the full documents are duplicates."
                ),
                "metrics": {
                    "exact_normalized_text": True,
                    "sha256": candidate["sha256"],
                },
                "left": {
                    "version_id": left["version_id"],
                    "citation": left["citations"][0],
                    "url": left["urls"][0],
                    "row_indices": left["row_indices"],
                    "passage_text": left["combined_normalized_text"],
                },
                "right": {
                    "version_id": right["version_id"],
                    "citation": right["citations"][0],
                    "url": right["urls"][0],
                    "row_indices": right["row_indices"],
                    "passage_text": right["combined_normalized_text"],
                },
            }
        )
        if len(examples) >= example_count:
            return examples

    ranked_near = sorted(
        text_report["template_candidates"],
        key=lambda row: (
            row["jaccard_similarity"],
            row["shorter_text_containment"],
            row["shared_shingles"],
        ),
        reverse=True,
    )
    for candidate in ranked_near:
        left = candidate["left"]
        right = candidate["right"]
        examples.append(
            {
                "example_type": "near_passage_match",
                "explanation": (
                    "Different version IDs contain highly similar normalized "
                    "QA passages. Inspect the differing text before treating "
                    "the source documents as duplicates."
                ),
                "metrics": {
                    "shared_shingles": candidate["shared_shingles"],
                    "jaccard_similarity": candidate["jaccard_similarity"],
                    "shorter_text_containment": candidate[
                        "shorter_text_containment"
                    ],
                },
                "left": {
                    "version_id": left["version_id"],
                    "citation": left["citations"][0],
                    "url": left["urls"][0],
                    "row_indices": left["row_indices"],
                    "passage_text": left["combined_normalized_text"],
                },
                "right": {
                    "version_id": right["version_id"],
                    "citation": right["citations"][0],
                    "url": right["urls"][0],
                    "row_indices": right["row_indices"],
                    "passage_text": right["combined_normalized_text"],
                },
            }
        )
        if len(examples) >= example_count:
            break

    return examples


def build_exact_duplicate_groups(text_report: dict) -> list[dict]:
    """Return only identical normalized-passage hash groups."""
    reviewed_rows = {
        447: {
            "answer_supported_by_passage": True,
            "recommended_action": "keep",
            "reason": (
                "The question asks about the generic purpose and operation "
                "of an Airworthiness Directive, which the passage explains."
            ),
        },
        1490: {
            "answer_supported_by_passage": True,
            "recommended_action": "keep",
            "reason": (
                "The question asks about the generic purpose and operation "
                "of an Airworthiness Directive, which the passage explains."
            ),
        },
        1543: {
            "answer_supported_by_passage": True,
            "recommended_action": "keep",
            "reason": (
                "The question asks about the generic purpose and operation "
                "of an Airworthiness Directive, which the passage explains."
            ),
        },
        1230: {
            "answer_supported_by_passage": True,
            "recommended_action": "keep",
            "reason": (
                "The question asks what the endnotes provide, and every "
                "substantive part of the answer appears in the passage."
            ),
        },
        1661: {
            "answer_supported_by_passage": True,
            "recommended_action": "keep",
            "reason": (
                "The question asks what the endnotes provide, and every "
                "substantive part of the answer appears in the passage."
            ),
        },
    }
    groups = []
    for index, candidate in enumerate(
        text_report["exact_candidates"],
        start=1,
    ):
        documents = candidate["documents"]
        for document in documents:
            for qa_example in document["qa_examples"]:
                qa_example["support_review"] = reviewed_rows.get(
                    qa_example["row_index"],
                    {
                        "answer_supported_by_passage": None,
                        "recommended_action": "manual_review",
                        "reason": (
                            "Check whether the answer can be obtained from "
                            "the positive passage alone."
                        ),
                    },
                )
        groups.append(
            {
                "group_id": f"exact-text-group-{index:04d}",
                "grouping_method": "identical_normalized_passage_sha256",
                "decision": (
                    "review_each_qa_pair_for_support_do_not_deduplicate_by_hash"
                ),
                "sha256": candidate["sha256"],
                "version_ids": candidate["version_ids"],
                "documents": documents,
            }
        )
    return groups


def add_template_support_reviews(candidates: list[dict]) -> list[dict]:
    """Attach the manual support decisions for reviewed template pairs."""
    reviewed_rows = {
        1159: (
            "The passage states order 0405836, its original date, revocation "
            "date, and replacement by the TABLE B orders."
        ),
        1351: (
            "The passage states order 0406369, its original date, revocation "
            "date, replacement orders, and the applicable tariff table."
        ),
        295: (
            "The passage states the 30.05.11 effective date and both ways the "
            "order can cease to be in force."
        ),
        1261: (
            "The passage states the 03.11.11 effective date and both ways the "
            "order can cease to be in force."
        ),
    }
    for candidate in candidates:
        for side in ("left", "right"):
            for qa_example in candidate[side].get("qa_examples", []):
                reason = reviewed_rows.get(qa_example["row_index"])
                qa_example["support_review"] = (
                    {
                        "answer_supported_by_passage": True,
                        "recommended_action": "keep_as_separate_qa_pair",
                        "reason": reason,
                    }
                    if reason
                    else {
                        "answer_supported_by_passage": None,
                        "recommended_action": "manual_review",
                        "reason": (
                            "Check whether the answer can be obtained from "
                            "this positive passage alone."
                        ),
                    }
                )
    return candidates


def _obsolete_connected_grouping_not_used(text_report: dict) -> list[dict]:
    """Legacy implementation retained temporarily; do not call."""
    documents: dict[str, dict] = {}
    edges: list[dict] = []
    neighbours: dict[str, set[str]] = defaultdict(set)

    def remember(document: dict) -> str:
        version_id = document["version_id"]
        documents[version_id] = {
            "version_id": version_id,
            "providers": document["providers"],
            "citations": document["citations"],
            "urls": document["urls"],
            "row_indices": document["row_indices"],
            "passage_text": document["combined_normalized_text"],
        }
        return version_id

    for candidate in text_report["exact_candidates"]:
        candidate_documents = candidate["documents"]
        for left, right in itertools.combinations(candidate_documents, 2):
            left_id = remember(left)
            right_id = remember(right)
            neighbours[left_id].add(right_id)
            neighbours[right_id].add(left_id)
            edges.append(
                {
                    "match_type": "exact",
                    "left_version_id": left_id,
                    "right_version_id": right_id,
                    "sha256": candidate["sha256"],
                    "jaccard_similarity": 1.0,
                    "shorter_text_containment": 1.0,
                }
            )

    for candidate in text_report["near_candidates"]:
        left_id = remember(candidate["left"])
        right_id = remember(candidate["right"])
        neighbours[left_id].add(right_id)
        neighbours[right_id].add(left_id)
        edges.append(
            {
                "match_type": "near_or_containment",
                "left_version_id": left_id,
                "right_version_id": right_id,
                "shared_shingles": candidate["shared_shingles"],
                "jaccard_similarity": candidate["jaccard_similarity"],
                "shorter_text_containment": candidate[
                    "shorter_text_containment"
                ],
            }
        )

    groups: list[dict] = []
    visited: set[str] = set()
    for start_id in sorted(neighbours):
        if start_id in visited:
            continue
        stack = [start_id]
        component: set[str] = set()
        while stack:
            version_id = stack.pop()
            if version_id in visited:
                continue
            visited.add(version_id)
            component.add(version_id)
            stack.extend(neighbours[version_id] - visited)

        component_edges = [
            edge
            for edge in edges
            if edge["left_version_id"] in component
            and edge["right_version_id"] in component
        ]
        groups.append(
            {
                "group_id": "",
                "grouping_method": "connected_component",
                "warning": (
                    "This is a duplicate-candidate group built from passage "
                    "matches. Membership may be transitive, and matching QA "
                    "snippets do not prove full-document duplication."
                ),
                "version_id_count": len(component),
                "direct_match_count": len(component_edges),
                "contains_exact_match": any(
                    edge["match_type"] == "exact"
                    for edge in component_edges
                ),
                "version_ids": sorted(component),
                "documents": [
                    documents[version_id]
                    for version_id in sorted(component)
                ],
                "matches": component_edges,
            }
        )

    groups.sort(
        key=lambda group: (
            group["version_id_count"],
            group["direct_match_count"],
        ),
        reverse=True,
    )
    for index, group in enumerate(groups, start=1):
        group["group_id"] = f"text-group-{index:04d}"
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run every maintained check and write the six standard inspection "
            "JSONL outputs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory used by --all "
            "(default: data/inspection)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete result as JSON.",
    )
    parser.add_argument(
        "--show-repeated-samples",
        action="store_true",
        help="Print every row belonging to a repeated version ID.",
    )
    parser.add_argument(
        "--sample-text-limit",
        type=int,
        default=600,
        help="Maximum passage characters printed for each sample.",
    )
    parser.add_argument(
        "--write-repeated-jsonl",
        type=Path,
        help="Write all repeated-version-ID rows to this JSONL path.",
    )
    parser.add_argument(
        "--inspect-urls",
        action="store_true",
        help="Audit URL case variants and normalization safety.",
    )
    parser.add_argument(
        "--inspect-citations",
        action="store_true",
        help="Find full and neutral citations shared by different version IDs.",
    )
    parser.add_argument(
        "--write-citation-candidates-jsonl",
        type=Path,
        help=(
            "Write rows whose normalized full citation belongs to multiple "
            "version IDs."
        ),
    )
    parser.add_argument(
        "--inspect-metadata",
        action="store_true",
        help="Compare jurisdiction/type/year metadata across version IDs.",
    )
    parser.add_argument(
        "--write-metadata-candidates-jsonl",
        type=Path,
        help="Write strong multi-version metadata candidate rows.",
    )
    parser.add_argument(
        "--inspect-text-similarity",
        action="store_true",
        help="Compare aggregated QA snippets across different version IDs.",
    )
    parser.add_argument(
        "--write-text-candidates-jsonl",
        type=Path,
        help="Write exact groups and template-similar passage pairs.",
    )
    parser.add_argument(
        "--write-text-examples-jsonl",
        type=Path,
        help="Write a small, human-readable set of pairwise match examples.",
    )
    parser.add_argument(
        "--write-exact-text-groups-jsonl",
        type=Path,
        help="Write groups only where normalized passage hashes are identical.",
    )
    parser.add_argument(
        "--write-template-similar-jsonl",
        type=Path,
        help="Write high-Jaccard, similar-length pairs for inspection only.",
    )
    parser.add_argument(
        "--write-boilerplate-warnings-jsonl",
        type=Path,
        help="Write containment-only overlaps that must not be deduplicated.",
    )
    parser.add_argument(
        "--text-example-count",
        type=int,
        default=3,
        help="Number of pairwise examples to write (default: 3).",
    )
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.85)
    parser.add_argument("--length-ratio-threshold", type=float, default=0.8)
    parser.add_argument("--containment-threshold", type=float, default=0.9)
    args = parser.parse_args()

    if args.all:
        args.inspect_urls = True
        args.write_repeated_jsonl = (
            args.write_repeated_jsonl
            or args.output_dir / "repeated_version_id_samples.jsonl"
        )
        args.write_citation_candidates_jsonl = (
            args.write_citation_candidates_jsonl
            or args.output_dir / "citation_multi_version_candidates.jsonl"
        )
        args.write_metadata_candidates_jsonl = (
            args.write_metadata_candidates_jsonl
            or args.output_dir / "metadata_multi_version_candidates.jsonl"
        )
        args.write_exact_text_groups_jsonl = (
            args.write_exact_text_groups_jsonl
            or args.output_dir / "exact_passage_duplicate_groups.jsonl"
        )
        args.write_template_similar_jsonl = (
            args.write_template_similar_jsonl
            or args.output_dir / "template_similar_passage_pairs.jsonl"
        )
        args.write_boilerplate_warnings_jsonl = (
            args.write_boilerplate_warnings_jsonl
            or args.output_dir / "boilerplate_overlap_warnings.jsonl"
        )

    report = inspect_version_ids(args.input)
    samples = load_repeated_records(
        args.input,
        set(report["repeated_version_ids"]),
    )

    if args.write_repeated_jsonl:
        write_jsonl(args.write_repeated_jsonl, samples)
        print(f"Wrote {len(samples)} rows to {args.write_repeated_jsonl}")

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Input:                       {report['input']}")
    print(f"Total rows:                  {report['total_rows']}")
    print(f"Non-missing version IDs:     {report['non_missing_version_ids']}")
    print(f"Missing version IDs:         {report['missing_version_ids']}")
    print(f"Unique version IDs:          {report['unique_version_ids']}")
    print(
        "IDs appearing once:          "
        f"{report['version_ids_appearing_once']}"
    )
    print(
        "Rows with once-only IDs:     "
        f"{report['rows_with_version_ids_appearing_once']}"
    )
    print(
        "IDs appearing more than once:"
        f" {report['version_ids_appearing_more_than_once']}"
    )
    print(
        "Rows with repeated IDs:      "
        f"{report['rows_with_repeated_version_ids']}"
    )
    print(
        "Repeated version-ID groups: "
        f"{report['repeated_version_id_groups']}"
    )
    print(
        "Extra rows from repeats:     "
        f"{report['extra_rows_from_repeated_ids']}"
    )

    if report["repeated_version_ids"]:
        print("\nRepeated version IDs:")
        for version_id, indices in report["repeated_version_ids"].items():
            print(f"- {version_id}: {len(indices)} rows at indices {indices}")

    if args.show_repeated_samples:
        print_repeated_samples(samples, args.sample_text_limit)

    if args.inspect_urls:
        url_report = inspect_urls(args.input)
        print("\nURL analysis:")
        print(json.dumps(url_report, indent=2))

    if args.inspect_citations or args.write_citation_candidates_jsonl:
        citation_report = inspect_citations(args.input)
        if args.write_citation_candidates_jsonl:
            candidate_rows = []
            for normalized_citation, group in citation_report[
                "full_citation_candidates"
            ].items():
                for row in group["rows"]:
                    candidate_rows.append(
                        {
                            "candidate_reason": (
                                "same normalized source.citation under "
                                "different version_ids"
                            ),
                            "normalized_citation": normalized_citation,
                            "group_version_ids": group["version_ids"],
                            "group_providers": group["providers"],
                            **row,
                        }
                    )
            write_jsonl(
                args.write_citation_candidates_jsonl,
                candidate_rows,
            )
            print(
                f"Wrote {len(candidate_rows)} rows to "
                f"{args.write_citation_candidates_jsonl}"
            )
        if args.inspect_citations:
            print("\nCitation analysis:")
            print(json.dumps(citation_report, indent=2, ensure_ascii=False))

    if args.inspect_metadata or args.write_metadata_candidates_jsonl:
        metadata_report = inspect_metadata_combinations(args.input)
        if args.write_metadata_candidates_jsonl:
            candidate_rows = []
            for group in metadata_report["strong_candidates"]:
                for row in group["rows"]:
                    candidate_rows.append(
                        {
                            "candidate_reason": (
                                "same normalized citation, jurisdiction, "
                                "document type, and derived citation year "
                                "under different version_ids"
                            ),
                            "group_key": group["group_key"],
                            "group_version_ids": group["version_ids"],
                            "group_providers": group["providers"],
                            **row,
                        }
                    )
            write_jsonl(
                args.write_metadata_candidates_jsonl,
                candidate_rows,
            )
            print(
                f"Wrote {len(candidate_rows)} rows to "
                f"{args.write_metadata_candidates_jsonl}"
            )
        if args.inspect_metadata:
            summary = {
                key: value
                for key, value in metadata_report.items()
                if key not in {"coarse_candidates", "strong_candidates"}
            }
            print("\nMetadata-combination analysis:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("\nStrong candidates:")
            print(
                json.dumps(
                    metadata_report["strong_candidates"],
                    indent=2,
                    ensure_ascii=False,
                )
            )

    if (
        args.inspect_text_similarity
        or args.write_text_candidates_jsonl
        or args.write_text_examples_jsonl
        or args.write_exact_text_groups_jsonl
        or args.write_template_similar_jsonl
        or args.write_boilerplate_warnings_jsonl
    ):
        text_report = inspect_text_similarity(
            args.input,
            shingle_size=args.shingle_size,
            jaccard_threshold=args.jaccard_threshold,
            length_ratio_threshold=args.length_ratio_threshold,
            containment_threshold=args.containment_threshold,
        )
        add_template_support_reviews(text_report["template_candidates"])
        if args.write_text_candidates_jsonl:
            rows = [
                *text_report["exact_candidates"],
                *text_report["template_candidates"],
            ]
            write_jsonl(args.write_text_candidates_jsonl, rows)
            print(
                f"Wrote {len(rows)} text-similarity candidates to "
                f"{args.write_text_candidates_jsonl}"
            )
        if args.write_text_examples_jsonl:
            example_rows = build_text_similarity_examples(
                text_report,
                example_count=max(1, args.text_example_count),
            )
            write_jsonl(args.write_text_examples_jsonl, example_rows)
            print(
                f"Wrote {len(example_rows)} text-similarity examples to "
                f"{args.write_text_examples_jsonl}"
            )
        if args.write_exact_text_groups_jsonl:
            group_rows = build_exact_duplicate_groups(text_report)
            write_jsonl(args.write_exact_text_groups_jsonl, group_rows)
            print(
                f"Wrote {len(group_rows)} exact duplicate groups to "
                f"{args.write_exact_text_groups_jsonl}"
            )
        if args.write_template_similar_jsonl:
            write_jsonl(
                args.write_template_similar_jsonl,
                text_report["template_candidates"],
            )
            print(
                f"Wrote {len(text_report['template_candidates'])} "
                f"template-similar pairs to "
                f"{args.write_template_similar_jsonl}"
            )
        if args.write_boilerplate_warnings_jsonl:
            write_jsonl(
                args.write_boilerplate_warnings_jsonl,
                text_report["boilerplate_warnings"],
            )
            print(
                f"Wrote {len(text_report['boilerplate_warnings'])} "
                f"boilerplate warnings to "
                f"{args.write_boilerplate_warnings_jsonl}"
            )
        if args.inspect_text_similarity:
            summary = {
                key: value
                for key, value in text_report.items()
                if key
                not in {
                    "exact_candidates",
                    "template_candidates",
                    "boilerplate_warnings",
                }
            }
            print("\nText-similarity analysis:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("\nExact candidates:")
            print(
                json.dumps(
                    text_report["exact_candidates"],
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print("\nTemplate-similar candidates:")
            print(
                json.dumps(
                    text_report["template_candidates"],
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print("\nBoilerplate-overlap warnings:")
            print(
                json.dumps(
                    text_report["boilerplate_warnings"],
                    indent=2,
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
