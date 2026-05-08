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
    blobs = _extract_strings(record.model_dump(mode="python"))
    return blobs


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
    return issues
