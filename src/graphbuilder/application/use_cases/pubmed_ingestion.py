"""
PubMed Ingestion Use Case.

Searches PubMed for articles matching a query, maps each article to domain
graph entities/relationships, and persists them via GraphRepositoryInterface.

Pipeline
--------
1. Search PubMed → list of PMIDs.
2. EFetch → structured ``PubMedArticle`` objects.
3. Map each article to:
   - One ``GraphEntity`` (EntityType.DOCUMENT) representing the article.
   - One ``GraphEntity`` per unique author (EntityType.PERSON).
   - ``GraphRelationship`` (RelationshipType.AUTHORED_BY) linking article → author.
   - One ``GraphEntity`` per unique MeSH term / keyword (EntityType.CONCEPT).
   - ``GraphRelationship`` (RelationshipType.CATEGORIZED_AS) linking article → concept.
4. Save all entities and relationships.
5. Return ``ProcessingResult`` with metrics.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ...domain.models.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)
from ...domain.models.processing_models import ProcessingResult
from ...infrastructure.config.settings import GraphBuilderConfig
from ...infrastructure.external.pubmed_client import PubMedArticle, PubMedClient
from ...infrastructure.repositories.graph_repository import GraphRepositoryInterface


@dataclass
class PubMedIngestionConfig:
    """Runtime parameters for a single PubMed ingestion run."""

    query: str
    max_articles: int = 100
    # NCBI requires an email for all API usage
    email: str = field(
        default_factory=lambda: os.getenv("NCBI_EMAIL", "graphbuilder@example.com")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("NCBI_API_KEY")
    )
    tag: str = "pubmed"
    include_mesh: bool = True
    include_keywords: bool = True


class PubMedIngestionUseCase:
    """
    Ingest PubMed articles into the knowledge graph.

    Parameters
    ----------
    config:
        Application-level configuration.
    graph_repo:
        Repository for persisting entities and relationships.
    client:
        Optional pre-constructed ``PubMedClient``.  When ``None``, one is
        created per ``execute`` call using ``ingestion_config.email`` and
        ``ingestion_config.api_key``.
    """

    def __init__(
        self,
        config: GraphBuilderConfig,
        graph_repo: GraphRepositoryInterface,
        client: Optional[PubMedClient] = None,
    ) -> None:
        self.config = config
        self.graph_repo = graph_repo
        self._client = client
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def execute(
        self, ingestion_config: PubMedIngestionConfig
    ) -> ProcessingResult:
        """
        Run the full ingestion pipeline.

        Returns a ``ProcessingResult`` whose ``data`` dict contains:

        * ``query`` — the PubMed search string
        * ``total_hits`` — articles found by ESearch
        * ``articles_fetched`` — articles successfully parsed
        * ``entities_created`` — total graph entities persisted
        * ``relationships_created`` — total graph relationships persisted
        """
        start = datetime.now(timezone.utc)

        try:
            fetch_result = await self._fetch(ingestion_config)

            if not fetch_result.success:
                return ProcessingResult(
                    success=False,
                    message=f"PubMed fetch failed: {'; '.join(fetch_result.errors)}",
                    errors=fetch_result.errors,
                )

            entities_saved = 0
            rels_saved = 0

            # De-duplicate authors and concepts across articles
            author_cache: Dict[str, GraphEntity] = {}
            concept_cache: Dict[str, GraphEntity] = {}

            for article in fetch_result.articles:
                article_entity = self._build_article_entity(article, ingestion_config)
                await self.graph_repo.save_entity(article_entity)
                entities_saved += 1

                # Authors
                for author_name in article.authors:
                    author_entity = self._get_or_create_person(
                        author_name, author_cache, ingestion_config
                    )
                    if author_name not in author_cache:
                        await self.graph_repo.save_entity(author_entity)
                        author_cache[author_name] = author_entity
                        entities_saved += 1

                    rel = GraphRelationship(
                        source_entity_id=article_entity.id,
                        target_entity_id=author_entity.id,
                        relationship_type=RelationshipType.AUTHORED_BY,
                        description=f"{article.pmid} authored by {author_name}",
                    )
                    await self.graph_repo.save_relationship(rel)
                    rels_saved += 1

                # Concepts (MeSH + keywords)
                concepts: List[str] = []
                if ingestion_config.include_mesh:
                    concepts.extend(article.mesh_terms)
                if ingestion_config.include_keywords:
                    concepts.extend(article.keywords)

                for concept_name in dict.fromkeys(concepts):  # unique, stable order
                    concept_entity = self._get_or_create_concept(
                        concept_name, concept_cache, ingestion_config
                    )
                    if concept_name not in concept_cache:
                        await self.graph_repo.save_entity(concept_entity)
                        concept_cache[concept_name] = concept_entity
                        entities_saved += 1

                    rel = GraphRelationship(
                        source_entity_id=article_entity.id,
                        target_entity_id=concept_entity.id,
                        relationship_type=RelationshipType.CATEGORIZED_AS,
                        description=f"{article.pmid} categorised as {concept_name}",
                    )
                    await self.graph_repo.save_relationship(rel)
                    rels_saved += 1

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()

            result = ProcessingResult(
                success=True,
                message=(
                    f"Ingested {len(fetch_result.articles)} PubMed articles "
                    f"({entities_saved} entities, {rels_saved} relationships)"
                ),
                data={
                    "query": ingestion_config.query,
                    "total_hits": fetch_result.total_hits,
                    "articles_fetched": fetch_result.fetched_count,
                    "entities_created": entities_saved,
                    "relationships_created": rels_saved,
                },
                processing_time=elapsed,
            )
            result.add_metric("articles_fetched", fetch_result.fetched_count)
            result.add_metric("entities_created", entities_saved)
            result.add_metric("relationships_created", rels_saved)
            result.add_metric("processing_time", elapsed)
            return result

        except Exception as exc:
            self.logger.error("PubMed ingestion error: %s", exc, exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"PubMed ingestion failed: {exc}",
                errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch(self, cfg: PubMedIngestionConfig):
        if self._client is not None:
            return await self._client.fetch_articles(cfg.query, cfg.max_articles)
        async with PubMedClient(email=cfg.email, api_key=cfg.api_key) as client:
            return await client.fetch_articles(cfg.query, cfg.max_articles)

    @staticmethod
    def _build_article_entity(
        article: PubMedArticle, cfg: PubMedIngestionConfig
    ) -> GraphEntity:
        entity = GraphEntity(
            name=article.title,
            entity_type=EntityType.DOCUMENT,
            description=article.abstract[:500] if article.abstract else None,
        )
        entity.add_external_id("pubmed", article.pmid)
        if article.doi:
            entity.add_external_id("doi", article.doi)
        entity.metadata.add_tag(cfg.tag)
        entity.properties["journal"] = article.journal
        entity.properties["publication_date"] = article.publication_date
        entity.properties["pmid"] = article.pmid
        return entity

    @staticmethod
    def _get_or_create_person(
        name: str,
        cache: Dict[str, GraphEntity],
        cfg: PubMedIngestionConfig,
    ) -> GraphEntity:
        if name in cache:
            return cache[name]
        entity = GraphEntity(name=name, entity_type=EntityType.PERSON)
        entity.metadata.add_tag(cfg.tag)
        entity.metadata.add_tag("author")
        return entity

    @staticmethod
    def _get_or_create_concept(
        name: str,
        cache: Dict[str, GraphEntity],
        cfg: PubMedIngestionConfig,
    ) -> GraphEntity:
        if name in cache:
            return cache[name]
        entity = GraphEntity(name=name, entity_type=EntityType.CONCEPT)
        entity.metadata.add_tag(cfg.tag)
        entity.metadata.add_tag("mesh")
        return entity
