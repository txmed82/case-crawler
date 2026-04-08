from __future__ import annotations

import hashlib
import re

from casecrawler.models.document import Chunk, Document

# Drug label section headers
_DRUG_LABEL_HEADERS = re.compile(
    r"(?m)^("
    r"INDICATIONS AND USAGE|CONTRAINDICATIONS|WARNINGS AND PRECAUTIONS|"
    r"ADVERSE REACTIONS|DRUG INTERACTIONS|USE IN SPECIFIC POPULATIONS|"
    r"OVERDOSAGE|DESCRIPTION|CLINICAL PHARMACOLOGY|NONCLINICAL TOXICOLOGY|"
    r"CLINICAL STUDIES|REFERENCES|HOW SUPPLIED|WARNINGS|PRECAUTIONS|"
    r"DOSAGE AND ADMINISTRATION"
    r")$"
)

# Trial protocol section headers
_TRIAL_HEADERS = re.compile(
    r"(?m)^("
    r"BACKGROUND|OBJECTIVES?|STUDY DESIGN|ELIGIBILITY CRITERIA|"
    r"INCLUSION CRITERIA|EXCLUSION CRITERIA|ENDPOINTS?|INTERVENTIONS?|"
    r"STATISTICAL ANALYSIS|RESULTS?|CONCLUSIONS?|METHODS?|PROTOCOL"
    r")$"
)


def _chunk_id(source: str, source_id: str, position: int) -> str:
    key = f"{source}:{source_id}:{position}"
    return hashlib.sha256(key.encode()).hexdigest()


class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        ct = doc.content_type

        if ct == "abstract":
            texts = [doc.content]
        elif ct == "drug_label":
            texts = self._split_by_sections(doc.content, _DRUG_LABEL_HEADERS)
        elif ct == "trial_protocol":
            texts = self._split_by_sections(doc.content, _TRIAL_HEADERS)
        else:
            texts = self._split_by_size(doc.content)

        source_doc_id = f"{doc.source}:{doc.source_id}"
        chunks = []
        for pos, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.source, doc.source_id, pos),
                    source_document_id=source_doc_id,
                    text=text,
                    position=pos,
                    metadata=doc.metadata.model_copy(),
                )
            )
        return chunks

    def _split_by_sections(self, content: str, pattern: re.Pattern) -> list[str]:
        parts = pattern.split(content)
        # re.split with a capturing group alternates: pre-header, header, content, header, content...
        # parts[0] is content before first header (possibly empty)
        # then pairs: parts[1]=header, parts[2]=content, parts[3]=header, parts[4]=content ...
        results = []

        # Handle preamble (content before any header)
        if parts[0].strip():
            results.append(parts[0])

        # Pair up headers with their content
        i = 1
        while i < len(parts) - 1:
            header = parts[i]
            body = parts[i + 1]
            combined = f"{header}\n{body}".strip()
            if combined:
                results.append(combined)
            i += 2

        # Fallback: if no sections matched, return whole content
        if not results:
            results = [content]

        return results

    def _split_by_size(self, content: str) -> list[str]:
        paragraphs = re.split(r"\n\n+", content)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para)

            if current_len + para_len > self.chunk_size and current_parts:
                # Emit current chunk
                chunks.append("\n\n".join(current_parts))
                # Keep overlap: retain last N chars worth of paragraphs
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) <= self.overlap:
                        overlap_parts.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append(para)
            current_len += para_len

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks if chunks else [content]
