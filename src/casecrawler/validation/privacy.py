from __future__ import annotations

import re

from casecrawler.models.synthetic import Modality, SyntheticRecord, ValidationIssue

PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
MRN_RE = re.compile(
    r"\b(?:MRN|medical record(?: number)?|record number)\s*[:#-]?\s*[A-Z0-9-]{5,}\b",
    flags=re.IGNORECASE,
)
DOB_RE = re.compile(
    r"\b(?:DOB|date of birth|birth date)\s*[:#-]?\s*"
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    flags=re.IGNORECASE,
)
STREET_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct)\b",
    flags=re.IGNORECASE,
)
MEMORIZATION_NGRAM_WORDS = 12
SOURCE_TEXT_KEYS = {
    "abstract",
    "chunk",
    "content",
    "excerpt",
    "source_text",
    "snippet",
    "text",
}


def _extract_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for key, nested in value.items():
            strings.extend(_extract_strings(key))
            strings.extend(_extract_strings(nested))
        return strings
    if isinstance(value, (list, tuple, set)):
        strings = []
        for nested in value:
            strings.extend(_extract_strings(nested))
        return strings
    return []


def _text_blobs(record: SyntheticRecord) -> list[str]:
    """Return strings the privacy regex set should scan.

    The earlier implementation walked ``record.model_dump(mode="python")``
    and pulled every string anywhere in the record. That produced
    false positives on fields that are NOT patient PHI by definition:

    - the tool's own contact email in ``provenance.source_refs`` and
      ``provenance.generator`` would trip the EMAIL regex
    - URLs, DOIs, and snippets in ``metadata.grounding.citations``
      could match the address / MRN heuristics

    The fix: dump the record without those subtrees, then run
    ``_extract_strings`` on the remaining structure. Patient
    demographics, encounters, free-text metadata, etc. are still
    scanned -- only the always-non-PHI subtrees are excluded.
    """

    payload = record.model_dump(mode="python")
    payload.pop("provenance", None)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("grounding", None)
    return _extract_strings(payload)


def _generated_text_blobs(record: SyntheticRecord) -> list[str]:
    blobs: list[str] = []
    for document in record.documents:
        blobs.append(document.clean_text)
        if document.messy_text:
            blobs.append(document.messy_text)
    for asset in record.imaging:
        blobs.append(asset.report_text)
    return [blob for blob in blobs if blob]


def _source_text_blobs(record: SyntheticRecord) -> list[str]:
    refs = record.provenance.source_refs
    blobs: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        blobs.extend(_source_texts_from_mapping(ref))
    return [blob for blob in blobs if blob]


def _source_texts_from_mapping(value: dict) -> list[str]:
    blobs: list[str] = []
    for key, nested in value.items():
        normalized_key = str(key).lower()
        if isinstance(nested, str) and normalized_key in SOURCE_TEXT_KEYS:
            blobs.append(nested)
            continue
        if isinstance(nested, dict):
            blobs.extend(_source_texts_from_mapping(nested))
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    blobs.extend(_source_texts_from_mapping(item))
    return blobs


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngram_set(text: str, size: int) -> set[tuple[str, ...]]:
    words = _normalized_words(text)
    if len(words) < size:
        return set()
    return {
        tuple(words[index:index + size])
        for index in range(0, len(words) - size + 1)
    }


def _has_long_source_overlap(generated_text: str, source_text: str) -> bool:
    generated_ngrams = _ngram_set(generated_text, MEMORIZATION_NGRAM_WORDS)
    if not generated_ngrams:
        return False
    return bool(generated_ngrams & _ngram_set(source_text, MEMORIZATION_NGRAM_WORDS))


def validate_privacy(record: SyntheticRecord) -> list[ValidationIssue]:
    text = "\n".join(_text_blobs(record))
    issues: list[ValidationIssue] = []
    for regex, label in [
        (PHONE_RE, "phone number"),
        (SSN_RE, "SSN"),
        (EMAIL_RE, "email"),
        (MRN_RE, "medical record number"),
        (DOB_RE, "date of birth"),
        (STREET_ADDRESS_RE, "street address"),
    ]:
        if regex.search(text):
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.CLINICAL_TEXT,
                    field="privacy",
                    message=f"Potential PHI-like {label} detected.",
                )
            )
    source_texts = _source_text_blobs(record)
    if source_texts:
        for generated_text in _generated_text_blobs(record):
            if any(
                _has_long_source_overlap(generated_text, source_text)
                for source_text in source_texts
            ):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.CLINICAL_TEXT,
                        field="privacy.memorization_risk",
                        message=(
                            "Generated text contains a long verbatim span from "
                            "a source reference."
                        ),
                    )
                )
                break
    return issues
