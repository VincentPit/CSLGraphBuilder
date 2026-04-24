"""
EmbeddingVerifier — Stage 2 of the cascading verification pipeline.

Computes the cosine similarity between a sentence-transformers embedding of
the relationship description and the context, then compares it against a
configurable threshold.

When a *graph_repo* is provided, the verifier also performs a Neo4j vector
search to find semantically similar relationships already stored in the graph.
This catches cases where the same relationship exists but is worded differently
(e.g. "TNF-alpha" vs "Tumor Necrosis Factor Alpha").

Dependency: ``sentence-transformers`` (optional — if not installed the verifier
returns a SKIPPED result with a clear message instead of raising).

The model is lazy-loaded on first call and shared across instances that use
the same model name (module-level cache).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .models import VerificationResult, VerificationStage, VerificationStatus
from ...domain.models.graph_models import GraphRelationship

if TYPE_CHECKING:
    from ...infrastructure.repositories.graph_repository import GraphRepositoryInterface

logger = logging.getLogger(__name__)

# Module-level model cache: model_name → SentenceTransformer instance
_MODEL_CACHE: Dict[str, object] = {}


@dataclass
class EmbeddingConfig:
    """Configuration for ``EmbeddingVerifier``."""

    model_name: str = "all-MiniLM-L6-v2"
    """sentence-transformers model to use."""

    threshold: float = 0.5
    """Minimum cosine similarity to PASS."""

    max_context_chars: int = 2_000
    """Truncate context to this many characters before embedding."""

    enable_vector_search: bool = True
    """When a graph_repo is available, also search Neo4j for similar entities."""

    vector_search_top_k: int = 5
    """Number of nearest-neighbour entities to retrieve from the vector index."""

    vector_search_min_score: float = 0.6
    """Minimum cosine score for a vector search hit to count."""


class EmbeddingVerifier:
    """
    Semantic similarity verifier using sentence-transformers.

    Parameters
    ----------
    config:
        ``EmbeddingConfig``.  Defaults to ``EmbeddingConfig()`` if not given.
    graph_repo:
        Optional graph repository for Neo4j vector search.  When supplied the
        verifier will query the ``entity_name_vector`` index to find entities
        with similar names, then check whether matching relationships exist.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        graph_repo: Optional['GraphRepositoryInterface'] = None,
    ) -> None:
        self._cfg = config or EmbeddingConfig()
        self._graph_repo = graph_repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        relationship: GraphRelationship,
        context: str,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a relationship via cosine similarity.

        The *query* is constructed from the relationship description, entity
        names, and relationship type; the *document* is the trimmed context.

        When a graph_repo is available and ``enable_vector_search`` is True,
        the verifier additionally queries the Neo4j vector index for entities
        whose name embeddings are similar to the source/target names.  If
        matching entities have a matching relationship type, the verification
        passes with a boost — catching cases where the same fact exists under
        different wording.
        """
        if not context and not self._graph_repo:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                stage=VerificationStage.EMBEDDING,
                confidence=0.0,
                reasoning="Empty context and no graph repository for vector search.",
            )

        try:
            model = self._get_model()
        except ImportError:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                stage=VerificationStage.EMBEDDING,
                confidence=0.0,
                reasoning=(
                    "sentence-transformers is not installed. "
                    "Run `pip install sentence-transformers` to enable this stage."
                ),
            )

        query = self._build_query(relationship, source_name, target_name)

        # ---- Context-based cosine similarity (original behaviour) ----
        context_similarity = 0.0
        if context:
            doc = context[: self._cfg.max_context_chars]
            context_similarity = self._cosine_similarity(model, query, doc)

        # ---- Neo4j vector search for similar existing relationships ----
        vector_hits: List[Dict[str, Any]] = []
        vector_best_score = 0.0
        if (
            self._graph_repo is not None
            and self._cfg.enable_vector_search
        ):
            try:
                vector_hits, vector_best_score = self._run_vector_search(
                    model, relationship, source_name, target_name
                )
            except Exception as exc:
                logger.warning("Vector search failed (non-fatal): %s", exc)

        # ---- Combine scores ----
        # Take the best of context similarity and vector search score
        combined = max(context_similarity, vector_best_score)
        passed = combined >= self._cfg.threshold

        # Build reasoning
        parts = []
        if context:
            parts.append(f"context cosine={context_similarity:.4f}")
        if vector_hits:
            parts.append(
                f"vector search best={vector_best_score:.4f} "
                f"({len(vector_hits)} graph hit(s))"
            )
        detail = "; ".join(parts) if parts else "no signal"
        reasoning = (
            f"Combined similarity {combined:.4f} "
            f"{'≥' if passed else '<'} threshold {self._cfg.threshold}. "
            f"[{detail}]"
        )

        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            stage=VerificationStage.EMBEDDING,
            confidence=round(float(combined), 4),
            reasoning=reasoning,
            metadata={
                "context_similarity": float(context_similarity),
                "vector_best_score": float(vector_best_score),
                "vector_hits": vector_hits,
                "combined_similarity": float(combined),
                "threshold": self._cfg.threshold,
                "model": self._cfg.model_name,
                "query": query,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> object:
        """Resolve the embedding model.

        If the verifier was configured with a custom ``model_name`` we
        respect it (tests can pin a specific model). Otherwise we delegate
        to ``embedding_factory.get_model()`` so the verifier shares the
        same instance — and the same dim — as the rest of the app.
        """
        name = self._cfg.model_name
        # Default → use the shared factory (env-driven, biomedical default).
        if name == EmbeddingConfig.__dataclass_fields__["model_name"].default:
            from ...infrastructure.services.embedding_factory import get_model
            shared = get_model()
            if shared is not None:
                return shared
        # Explicit override → load and cache by name as before.
        if name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info("Loading sentence-transformers model '%s' …", name)
            _MODEL_CACHE[name] = SentenceTransformer(name)
        return _MODEL_CACHE[name]

    def _run_vector_search(
        self,
        model: object,
        relationship: GraphRelationship,
        source_name: Optional[str],
        target_name: Optional[str],
    ) -> tuple:
        """Query Neo4j vector index for similar entities, then check relationships.

        Returns ``(hits_list, best_score)`` where each hit is a dict with
        ``entity_id``, ``entity_name``, ``score``, and optional ``matching_rel_type``.
        """
        import numpy as np  # type: ignore

        hits: List[Dict[str, Any]] = []
        best_score = 0.0
        repo = self._graph_repo
        top_k = self._cfg.vector_search_top_k
        min_score = self._cfg.vector_search_min_score

        # Search for entities similar to source and target names
        for role, name in [("source", source_name), ("target", target_name)]:
            if not name:
                continue
            embedding = model.encode(name, convert_to_numpy=True).tolist()  # type: ignore

            # Run the async repo method from sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    entity_hits = pool.submit(
                        asyncio.run,
                        repo.vector_search_entities(embedding, top_k=top_k, min_score=min_score),
                    ).result()
            else:
                entity_hits = asyncio.run(
                    repo.vector_search_entities(embedding, top_k=top_k, min_score=min_score)
                )

            for entity, score in entity_hits:
                hit = {
                    "role": role,
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "score": round(score, 4),
                }
                hits.append(hit)
                if score > best_score:
                    best_score = score

        # If we found similar entities for both source and target, check if
        # a relationship exists between any pair — regardless of type, since
        # the same fact can be expressed with different relationship types
        # (e.g. ACTIVATES vs STIMULATES vs UPREGULATES).
        source_ids = {h["entity_id"] for h in hits if h["role"] == "source"}
        target_ids = {h["entity_id"] for h in hits if h["role"] == "target"}

        if source_ids and target_ids:
            rel_type = relationship.relationship_type.value
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            # Fetch ALL relationships between matched entities (no type filter)
            query = (
                "MATCH (s:Entity)-[r:RELATES]->(t:Entity) "
                "WHERE s.id IN $src_ids AND t.id IN $tgt_ids "
                "RETURN s.id as sid, t.id as tid, "
                "       r.relationship_type as rtype, "
                "       r.description as rdesc "
                "LIMIT 20"
            )
            params = {
                "src_ids": list(source_ids),
                "tgt_ids": list(target_ids),
            }

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    matches = pool.submit(
                        asyncio.run,
                        repo.execute_cypher_query(query, params),
                    ).result()
            else:
                matches = asyncio.run(repo.execute_cypher_query(query, params))

            if matches:
                # Score each matched relationship by semantic similarity
                # between the new relationship type/desc and the existing one
                new_rel_text = self._build_query(relationship, source_name, target_name)

                for m in matches:
                    existing_parts = []
                    if m.get("rtype"):
                        existing_parts.append(m["rtype"].replace("_", " "))
                    if m.get("rdesc"):
                        existing_parts.append(m["rdesc"])
                    existing_rel_text = " ".join(existing_parts) if existing_parts else m.get("rtype", "")

                    # Compute cosine between new and existing relationship descriptions
                    rel_sim = self._cosine_similarity(model, new_rel_text, existing_rel_text)

                    # Exact type match = 0.95, semantic match ≥ 0.6 = scaled score
                    if m["rtype"] == rel_type:
                        rel_score = 0.95
                    elif rel_sim >= 0.6:
                        # Scale: 0.6 → 0.75, 1.0 → 0.95
                        rel_score = round(0.75 + (rel_sim - 0.6) * 0.5, 4)
                    else:
                        continue  # too dissimilar, skip this match

                    hits.append({
                        "role": "relationship_match",
                        "source_entity_id": m["sid"],
                        "target_entity_id": m["tid"],
                        "relationship_type": m["rtype"],
                        "relationship_desc": m.get("rdesc", ""),
                        "rel_similarity": round(rel_sim, 4),
                        "score": rel_score,
                    })
                    if rel_score > best_score:
                        best_score = rel_score

                rel_matches = [h for h in hits if h["role"] == "relationship_match"]
                if rel_matches:
                    logger.info(
                        "Vector search found %d semantically matching relationship(s) "
                        "(query='%s', best_rel_sim=%.3f)",
                        len(rel_matches), rel_type,
                        max(h.get("rel_similarity", 0) for h in rel_matches),
                    )

        return hits, best_score

    @staticmethod
    def _build_query(
        rel: GraphRelationship,
        source_name: Optional[str],
        target_name: Optional[str],
    ) -> str:
        parts = []
        if source_name:
            parts.append(source_name)
        parts.append(rel.relationship_type.value.replace("_", " "))
        if target_name:
            parts.append(target_name)
        if rel.description:
            parts.append(rel.description)
        return " ".join(parts)

    @staticmethod
    def _cosine_similarity(model: object, query: str, document: str) -> float:
        import numpy as np  # type: ignore

        embeddings = model.encode([query, document], convert_to_numpy=True)  # type: ignore
        a, b = embeddings[0], embeddings[1]
        norm = (np.linalg.norm(a) * np.linalg.norm(b))
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)
