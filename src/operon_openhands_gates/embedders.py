"""Zero-dependency character-n-gram embedder.

Default for ``StagnationGate`` so users can adopt the gate without installing
a neural embedding model. Accuracy is lower than a real sentence encoder but
sufficient to detect the most common stagnation pathology: an agent repeating
itself verbatim or with trivial edits.

Power users can swap in any callable ``(text: str) -> list[float]`` via the
``embedder=`` parameter on the gate.
"""

from __future__ import annotations

import hashlib
import math

_DEFAULT_NGRAM = 3
_DEFAULT_DIM = 512


class NGramEmbedder:
    """Character n-gram embedder using hashed-bag-of-n-grams.

    Produces a fixed-dimension L2-normalized vector; cosine similarity equals
    the dot product. Not as expressive as a neural embedder but deterministic,
    dependency-free, and good enough to catch verbatim-repeat stagnation.
    """

    def __init__(self, n: int = _DEFAULT_NGRAM, dim: int = _DEFAULT_DIM) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.n = n
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        counts = [0.0] * self.dim
        for gram in _ngrams(text, self.n):
            bucket = _hash_to_bucket(gram, self.dim)
            counts[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in counts))
        if norm == 0.0:
            return counts
        return [x / norm for x in counts]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Assumes equal dim; zero vectors -> 0.0."""
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))


def _ngrams(text: str, n: int) -> list[str]:
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _hash_to_bucket(gram: str, dim: int) -> int:
    # MD5 is fine here: not cryptographic use, just stable hashing across
    # Python processes (unlike hash(), which is salted per interpreter).
    digest = hashlib.md5(gram.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % dim
