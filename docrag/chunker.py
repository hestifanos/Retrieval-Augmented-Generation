"""Split documents into overlapping chunks that respect sentence boundaries."""
import re
from dataclasses import dataclass
from textwrap import wrap


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_document(doc_id: str, text: str, max_chars: int = 800, overlap_sentences: int = 1) -> list[Chunk]:
    """Greedy chunking: pack sentences until max_chars, then start a new chunk
    that repeats the last `overlap_sentences` sentences for context continuity."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    sentences = [
        fragment
        for sentence in split_sentences(text)
        for fragment in (wrap(sentence, max_chars) if len(sentence) > max_chars else [sentence])
    ]
    chunks: list[Chunk] = []
    current: list[str] = []
    length = 0
    overlap_only = False

    for sentence in sentences:
        if current and length + len(sentence) + 1 > max_chars:
            if not overlap_only:
                chunks.append(Chunk(doc_id, len(chunks), " ".join(current)))
                current = current[-overlap_sentences:] if overlap_sentences else []
                length = len(" ".join(current))
                overlap_only = bool(current)
            if current and length + len(sentence) + 1 > max_chars:
                current = []
                length = 0
                overlap_only = False
        current.append(sentence)
        length += len(sentence) + (1 if len(current) > 1 else 0)
        overlap_only = False
    if current:
        chunks.append(Chunk(doc_id, len(chunks), " ".join(current)))
    return chunks
