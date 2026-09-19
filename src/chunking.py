from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        # raise NotImplementedError("Implement SentenceChunker.chunk")
        if not text.strip():
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        step = self.max_sentences_per_chunk

        for i in range(0, len(sentences), step):
            chunk = " ".join(sentences[i : i + step])
            chunks.append(chunk)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        # raise NotImplementedError("Implement RecursiveChunker.chunk")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if not text.strip():
            return []

        return self._split(text.strip(), self.separators)


    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        # raise NotImplementedError("Implement RecursiveChunker._split")
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []

        # Hết dấu phân cách: cắt trực tiếp theo số thứ tự
        if not remaining_separators:
            return [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Chuỗi rỗng biểu thị bước cắt theo kí tự
        if separator == "":
            return [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        # Giữ dấu phân cách để không làm mất dấu câu
        parts = current_text.split(separator)
        pieces = [
            part + separator if i < len(parts) - 1 else part
            for i, part in enumerate(parts)
        ]

        chunks = []
        buffer = ""

        for piece in pieces:
            if not piece:
                continue

            if len(piece) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split(piece, next_separators))

            elif len(buffer) + len(piece) <= self.chunk_size:
                buffer += piece
            else:
                chunks.append(buffer)
                buffer = piece

        if buffer:
            chunks.append(buffer)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    # raise NotImplementedError("Implement compute_similarity")
    magnitude_a = math.sqrt(sum(x * x for x in vec_a))
    magnitude_b = math.sqrt(sum(x * x for x in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        # raise NotImplementedError("Implement ChunkingStrategyComparator.compare")
        if chunk_size <= 0:
            raise ValueError("chunk_size phải lớn hơn 0")

        strategies = {
            "fixed_size": FixedSizeChunker(
                chunk_size=chunk_size,
                overlap=min(50, chunk_size - 1),
            ),
            "by_sentences": SentenceChunker(
                max_sentences_per_chunk=3,
            ),
            "recursive": RecursiveChunker(
                chunk_size=chunk_size,
            ),
        }

        results = {}

        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)

            results[name] = {
                "count": count,
                "avg_length": (
                    sum(len(chunk) for chunk in chunks) / count
                    if count > 0 else 0.0
                ),
                "chunks": chunks,
            }

        return results
