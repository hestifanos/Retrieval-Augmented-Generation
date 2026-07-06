"""Split documents into overlapping chunks that respect sentence boundaries."""
import re
from dataclasses import dataclass


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
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    length = 0

    def flush():
        nonlocal current, length
        if current:
            chunks.append(Chunk(doc_id, len(chunks), " ".join(current)))
            current = current[-overlap_sentences:] if overlap_sentences else []
            length = sum(len(s) + 1 for s in current)

    for sentence in sentences:
        if length + len(sentence) > max_chars and current:
            flush()
        current.append(sentence)
        length += len(sentence) + 1
    flush()
    return chunks
