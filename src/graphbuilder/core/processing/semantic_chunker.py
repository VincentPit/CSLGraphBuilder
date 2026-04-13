"""
Semantic Text Chunker.

Splits text into chunks at natural semantic boundaries (sentence/paragraph
breaks) instead of using fixed character windows.  Each chunk stays below
a configurable token budget while preserving coherent meaning units.

Algorithm
---------
1.  Split the document into sentences (regex-based, no NLTK dependency).
2.  Compute sentence embeddings using the project's existing
    ``sentence-transformers`` model.
3.  Walk the sentence list and start a *new* chunk whenever:
    a.  adding the next sentence would exceed ``max_chunk_tokens``, **or**
    b.  the cosine similarity between the current sentence and the
        running centroid of the current chunk drops below
        ``similarity_threshold`` (a semantic breakpoint).
4.  Return ``DocumentChunk`` objects with accurate positions and token
    counts.

The class also exposes a fast ``split_fixed()`` fallback that reproduces
the original character-window behaviour for callers that don't need
semantic awareness.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ...domain.models.graph_models import DocumentChunk

logger = logging.getLogger(__name__)

# ── Sentence boundary regex ──────────────────────────────────────────────
# Splits after sentence-ending punctuation followed by whitespace.
_SENT_RE = re.compile(r"(?<=[.!?])\s+", re.UNICODE)


@dataclass
class SemanticChunkerConfig:
    """Settings for the semantic chunker."""

    max_chunk_tokens: int = 512
    min_chunk_tokens: int = 30
    similarity_threshold: float = 0.5
    embedding_model_name: str = "all-MiniLM-L6-v2"


class SemanticChunker:
    """
    Semantic-boundary-aware text chunker.

    Parameters
    ----------
    config:
        A ``SemanticChunkerConfig`` with size and similarity settings.
    """

    def __init__(self, config: Optional[SemanticChunkerConfig] = None) -> None:
        self.config = config or SemanticChunkerConfig()
        self._model = None  # lazy-loaded

    # ── Public API ────────────────────────────────────────────────────────

    def chunk(
        self,
        content: str,
        document_id: str,
    ) -> List[DocumentChunk]:
        """
        Split *content* into semantic chunks, returning ``DocumentChunk``
        objects with proper metadata.
        """
        sentences = self._split_sentences(content)
        if not sentences:
            return []

        embeddings = self._embed(sentences)
        groups = self._group_sentences(sentences, embeddings, content)
        groups = self._merge_small_groups(groups)
        return self._to_chunks(groups, document_id)

    # ── Fixed-size fallback ───────────────────────────────────────────────

    def split_fixed(
        self,
        content: str,
        document_id: str,
        chunk_size: int = 1000,
        overlap_size: int = 100,
    ) -> List[DocumentChunk]:
        """Character-window chunking (original behaviour)."""
        chunks: List[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]

            if chunk_content.strip():
                chunks.append(
                    DocumentChunk(
                        content=chunk_content,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        token_count=len(chunk_content.split()),
                        character_count=len(chunk_content),
                        start_position=start,
                        end_position=end,
                    )
                )
                chunk_index += 1

            start = end - overlap_size if end < len(content) else end

        return chunks

    # ── Internals ─────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving whitespace tokens."""
        parts = _SENT_RE.split(text)
        # Also split on double-newlines (paragraph breaks)
        sentences: List[str] = []
        for part in parts:
            subs = re.split(r"\n{2,}", part)
            sentences.extend(s for s in subs if s.strip())
        return sentences

    def _embed(self, sentences: List[str]) -> np.ndarray:
        """Return (N, D) embedding matrix for *sentences*."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model_name)
            logger.info(
                "Loaded embedding model '%s' for semantic chunking",
                self.config.embedding_model_name,
            )
        return self._model.encode(sentences, show_progress_bar=False)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _group_sentences(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        full_text: str,
    ) -> List[Tuple[str, int, int]]:
        """
        Group sentences into chunks.

        Returns a list of (chunk_text, start_position, end_position) tuples.
        """
        groups: List[Tuple[str, int, int]] = []

        current_sents: List[str] = []
        current_embeds: List[np.ndarray] = []
        current_tokens = 0
        current_start = 0

        for idx, sent in enumerate(sentences):
            sent_tokens = len(sent.split())
            embed = embeddings[idx]

            # Check semantic break
            should_break = False
            if current_sents:
                centroid = np.mean(current_embeds, axis=0)
                sim = self._cosine_similarity(centroid, embed)
                if sim < self.config.similarity_threshold:
                    should_break = True

            # Check size break
            if current_tokens + sent_tokens > self.config.max_chunk_tokens and current_sents:
                should_break = True

            # Suppress break if current chunk is still below minimum size
            if should_break and current_tokens < self.config.min_chunk_tokens:
                should_break = False

            if should_break:
                chunk_text = " ".join(current_sents)
                start_pos = full_text.find(current_sents[0], current_start)
                if start_pos == -1:
                    start_pos = current_start
                end_pos = start_pos + len(chunk_text)
                groups.append((chunk_text, start_pos, end_pos))
                current_start = end_pos
                current_sents = []
                current_embeds = []
                current_tokens = 0

            current_sents.append(sent)
            current_embeds.append(embed)
            current_tokens += sent_tokens

        # Flush remaining
        if current_sents:
            chunk_text = " ".join(current_sents)
            start_pos = full_text.find(current_sents[0], current_start)
            if start_pos == -1:
                start_pos = current_start
            end_pos = start_pos + len(chunk_text)
            groups.append((chunk_text, start_pos, end_pos))

        return groups

    def _merge_small_groups(
        self,
        groups: List[Tuple[str, int, int]],
    ) -> List[Tuple[str, int, int]]:
        """Merge any group below ``min_chunk_tokens`` into its neighbour."""
        if len(groups) <= 1:
            return groups

        merged: List[Tuple[str, int, int]] = []
        for text, start, end in groups:
            tok_count = len(text.split())
            if merged and tok_count < self.config.min_chunk_tokens:
                # Merge into the previous group
                prev_text, prev_start, _ = merged[-1]
                merged[-1] = (prev_text + " " + text, prev_start, end)
            elif (
                merged
                and len(merged[-1][0].split()) < self.config.min_chunk_tokens
            ):
                # Previous group was undersized — merge forward into this one
                prev_text, prev_start, _ = merged[-1]
                merged[-1] = (prev_text + " " + text, prev_start, end)
            else:
                merged.append((text, start, end))

        return merged

    def _to_chunks(
        self,
        groups: List[Tuple[str, int, int]],
        document_id: str,
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for i, (text, start, end) in enumerate(groups):
            if not text.strip():
                continue
            chunks.append(
                DocumentChunk(
                    content=text,
                    document_id=document_id,
                    chunk_index=i,
                    token_count=len(text.split()),
                    character_count=len(text),
                    start_position=start,
                    end_position=end,
                )
            )
        return chunks
