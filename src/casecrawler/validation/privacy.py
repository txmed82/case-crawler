from __future__ import annotations

import re

from casecrawler.models.synthetic import Modality, SyntheticRecord, ValidationIssue

PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


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
