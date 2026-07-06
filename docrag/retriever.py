"""A TF-IDF retriever implemented from scratch.

The vectorizer is written by hand rather than imported from scikit-learn so
the scoring behaviour (log-scaled term frequency, smoothed inverse document
frequency, cosine similarity) is explicit and easy to inspect.
"""
import math
import re
from collections import Counter

from docrag.chunker import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were which will with what where who how i".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._doc_tokens = [tokenize(c.text) for c in chunks]
        n = len(chunks)
        df = Counter(token for tokens in self._doc_tokens for token in set(tokens))
        # Smoothed IDF so unseen query terms do not divide by zero.
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vectors = [self._vectorize(tokens) for tokens in self._doc_tokens]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        qvec = self._vectorize(tokenize(query))
        scored = []
        for chunk, dvec in zip(self.chunks, self._vectors):
            score = sum(qvec[t] * dvec.get(t, 0.0) for t in qvec)  # sparse cosine
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
