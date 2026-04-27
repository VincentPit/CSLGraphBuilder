"""
Open Targets Ingestion Use Case.

Pulls a root entity from the Open Targets Platform — disease, target,
drug, variant, or study — together with all of its associated edges,
and persists every neighbor as its own ``GraphEntity`` linked to the
root by a typed ``GraphRelationship``.

Materialized edges by root kind:

    DISEASE  → associatedTargets (RELATED_TO),
                knownDrugs       (INFLUENCES drug → disease)
    TARGET   → associatedDiseases (RELATED_TO),
                knownDrugs        (INFLUENCES drug → target),
                pathways          (PART_OF)
    DRUG     → mechanismsOfAction (INFLUENCES drug → target),
                indications        (INFLUENCES drug → disease),
                linkedTargets      (RELATED_TO),
                linkedDiseases     (RELATED_TO)
    VARIANT  → transcriptConsequences (PART_OF variant → target)
    STUDY    → diseases (RELATED_TO study → disease)

No live API or database call is made in this module; both are injected
as interfaces so the use case is fully unit-testable.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ...domain.models.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)
from ...domain.models.processing_models import ProcessingResult
from ...infrastructure.config.settings import GraphBuilderConfig
from ...infrastructure.external.open_targets_client import (
    DiseaseRef,
    DiseaseRoot,
    DrugRef,
    DrugRoot,
    EntityKind,
    IngestResult,
    KnownDrug,
    OpenTargetsClient,
    PathwayRef,
    StudyRoot,
    TargetRef,
    TargetRoot,
    VariantRoot,
    detect_entity_kind,
)
from ...infrastructure.repositories.graph_repository import GraphRepositoryInterface


@dataclass
class IngestionConfig:
    """Runtime parameters for a single Open Targets ingestion run."""

    entity_id: str
    entity_kind: Optional[EntityKind] = None  # auto-detect when None
    max_associations: int = 500
    max_known_drugs: int = 100
    min_association_score: float = 0.0  # only used for disease/target associations
    tag: str = "open-targets"

    # ── back-compat: older callers passed ``disease_id=`` ──────────────
    @property
    def disease_id(self) -> str:
        return self.entity_id

    @classmethod
    def for_disease(cls, disease_id: str, **kw: Any) -> "IngestionConfig":
        return cls(entity_id=disease_id, entity_kind=EntityKind.DISEASE, **kw)


class OpenTargetsIngestionUseCase:
    """
    Ingest an Open Targets entity (any kind) and its neighbor edges.

    Parameters
    ----------
    config:
        Application-level configuration (used for client construction).
    graph_repo:
        Repository for persisting entities and relationships.
    client:
        Optional pre-constructed ``OpenTargetsClient``. When ``None`` a new
        client is created from settings on each ``execute`` call.
    """

    def __init__(
        self,
        config: GraphBuilderConfig,
        graph_repo: GraphRepositoryInterface,
        client: Optional[OpenTargetsClient] = None,
    ) -> None:
        self.config = config
        self.graph_repo = graph_repo
        self._client = client
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def execute(self, ingestion_config: IngestionConfig) -> ProcessingResult:
        start = datetime.now(timezone.utc)

        try:
            # Resolve the entity kind up-front so error messages are useful.
            kind = ingestion_config.entity_kind or detect_entity_kind(
                ingestion_config.entity_id
            )

            ingest_result = await self._fetch(ingestion_config, kind)

            if not ingest_result.success:
                return ProcessingResult(
                    success=False,
                    message=f"Open Targets fetch failed: {'; '.join(ingest_result.errors)}",
                    errors=ingest_result.errors,
                )

            # Build root + all neighbor entities and edges
            entities, relationships = self._build_graph(
                ingest_result, ingestion_config
            )

            # Persist entities first so save_relationship can resolve IDs
            # (save_entity dedupes on (name, type) and may rewrite entity.id
            # to point at an existing record — relationships referencing the
            # original UUIDs would otherwise miss).
            #
            # Both calls go through the batched repo paths so embeddings
            # for all neighbors are computed in one model.encode() call —
            # for OT-scale fetches (hundreds of entities) this is the
            # difference between sub-second and tens-of-seconds persistence.
            saved_entities = await self.graph_repo.save_entities_batch(entities)

            # Hand the relationship batch a {id → name} map so it can skip
            # the per-rel endpoint MATCH lookup. ``save_entities_batch`` may
            # have rewritten entity.id (existing-match dedup), so use the
            # post-save IDs.
            id_to_name = {e.id: e.name for e in saved_entities}
            saved_rels = await self.graph_repo.save_relationships_batch(
                relationships, entity_names=id_to_name
            )
            rels_saved = len(saved_rels)

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()

            root_label = _root_display_name(ingest_result)

            result = ProcessingResult(
                success=True,
                message=(
                    f"Ingested {len(entities)} entities and {rels_saved} "
                    f"relationships for {ingest_result.kind.value} "
                    f"{ingestion_config.entity_id} ({root_label})"
                ),
                data={
                    "entity_id": ingestion_config.entity_id,
                    "entity_kind": ingest_result.kind.value,
                    "entity_name": root_label,
                    # Disease-shaped fields kept for back-compat with older UI/CLI
                    "disease_id": ingestion_config.entity_id,
                    "disease_name": root_label,
                    "entities_created": len(entities),
                    "relationships_created": rels_saved,
                    "associations_fetched": len(relationships),
                    "total_associations_available": _total_neighbors(ingest_result),
                },
                processing_time=elapsed,
            )
            result.add_metric("entities_created", len(entities))
            result.add_metric("relationships_created", rels_saved)
            result.add_metric("associations_fetched", len(relationships))
            result.add_metric("processing_time", elapsed)
            return result

        except Exception as exc:
            self.logger.error(
                "Open Targets ingestion error: %s", exc, exc_info=True
            )
            return ProcessingResult(
                success=False,
                message=f"Ingestion failed: {exc}",
                errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch(
        self, cfg: IngestionConfig, kind: Optional[EntityKind]
    ) -> IngestResult:
        if self._client is not None:
            return await self._client.fetch_entity(
                cfg.entity_id,
                kind=kind,
                max_neighbors=cfg.max_associations,
                max_known_drugs=cfg.max_known_drugs,
            )
        async with OpenTargetsClient() as client:
            return await client.fetch_entity(
                cfg.entity_id,
                kind=kind,
                max_neighbors=cfg.max_associations,
                max_known_drugs=cfg.max_known_drugs,
            )

    def _build_graph(
        self, ingest_result: IngestResult, cfg: IngestionConfig
    ) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
        """Materialize the root and every neighbor edge for the result."""
        builder = _GraphBuilder(cfg)
        root = ingest_result.root

        if isinstance(root, DiseaseRoot):
            builder.from_disease(root)
        elif isinstance(root, TargetRoot):
            builder.from_target(root)
        elif isinstance(root, DrugRoot):
            builder.from_drug(root)
        elif isinstance(root, VariantRoot):
            builder.from_variant(root)
        elif isinstance(root, StudyRoot):
            builder.from_study(root)
        else:  # pragma: no cover
            raise RuntimeError(f"Unknown Open Targets root type: {type(root)!r}")

        return builder.entities, builder.relationships


# ── Internal builder ──────────────────────────────────────────────────────


class _GraphBuilder:
    """Accumulates entities + relationships for one ingestion run.

    Deduplicates neighbor entities by (entity_type, external_id) so that, e.g.,
    the same target referenced by both ``mechanismsOfAction`` and
    ``linkedTargets`` is materialized only once.
    """

    def __init__(self, cfg: IngestionConfig) -> None:
        self.cfg = cfg
        self.entities: List[GraphEntity] = []
        self.relationships: List[GraphRelationship] = []
        self._by_external: Dict[Tuple[EntityType, str], GraphEntity] = {}
        self._seen_edges: set = set()

    # ── Roots ────────────────────────────────────────────────────────

    def from_disease(self, root: DiseaseRoot) -> None:
        disease_entity = self._add_disease_entity(
            DiseaseRef(
                disease_id=root.id,
                name=root.name,
                description=root.description,
                therapeutic_areas=root.therapeutic_areas,
            ),
            extra_aliases=root.synonyms,
            mark_root=True,
        )

        for assoc in root.associated_targets:
            if assoc.score < self.cfg.min_association_score:
                continue
            target_entity = self._add_target_entity(assoc.target)
            self._add_rel(
                disease_entity,
                target_entity,
                RelationshipType.RELATED_TO,
                strength=_clamp(assoc.score),
                description=f"Open Targets disease–target association (score {assoc.score:.3f})",
                properties={
                    "association_score": assoc.score,
                    "datatype_scores": assoc.datatype_scores,
                    "edge": "disease_target_association",
                },
            )

        for kd in root.known_drugs:
            self._add_known_drug_edges(kd, disease_entity=disease_entity)

    def from_target(self, root: TargetRoot) -> None:
        target_entity = self._add_target_entity(
            TargetRef(
                target_id=root.id,
                symbol=root.symbol,
                name=root.name,
                biotype=root.biotype,
                function_descriptions=root.function_descriptions,
            ),
            mark_root=True,
        )

        for assoc in root.associated_diseases:
            if assoc.score < self.cfg.min_association_score:
                continue
            disease_entity = self._add_disease_entity(assoc.disease)
            self._add_rel(
                disease_entity,
                target_entity,
                RelationshipType.RELATED_TO,
                strength=_clamp(assoc.score),
                description=f"Open Targets disease–target association (score {assoc.score:.3f})",
                properties={
                    "association_score": assoc.score,
                    "datatype_scores": assoc.datatype_scores,
                    "edge": "disease_target_association",
                },
            )

        for pathway in root.pathways:
            pathway_entity = self._add_pathway_entity(pathway)
            self._add_rel(
                target_entity,
                pathway_entity,
                RelationshipType.PART_OF,
                description="Target participates in pathway",
                properties={"edge": "target_pathway"},
            )

        for kd in root.known_drugs:
            self._add_known_drug_edges(kd, target_entity=target_entity)

    def from_drug(self, root: DrugRoot) -> None:
        drug_entity = self._add_drug_entity(
            DrugRef(drug_id=root.id, name=root.name),
            description=root.description,
            properties={
                "drug_type": root.drug_type,
                "max_clinical_stage": root.max_clinical_stage,
                "trade_names": root.trade_names,
            },
            extra_aliases=list(root.synonyms) + list(root.trade_names),
            mark_root=True,
        )

        for moa in root.mechanisms_of_action:
            for target in moa.targets:
                if not target.target_id:
                    continue
                target_entity = self._add_target_entity(target)
                self._add_rel(
                    drug_entity,
                    target_entity,
                    RelationshipType.INFLUENCES,
                    description=moa.description or "Mechanism of action",
                    properties={
                        "mechanism_of_action": moa.description,
                        "action_type": moa.action_type,
                        "target_name": moa.target_name,
                        "edge": "drug_mechanism_of_action",
                    },
                )

        for indication in root.indications:
            if not indication.disease.disease_id:
                continue
            disease_entity = self._add_disease_entity(indication.disease)
            self._add_rel(
                drug_entity,
                disease_entity,
                RelationshipType.INFLUENCES,
                description=(
                    f"Indicated for {indication.disease.name}"
                    + (f" ({indication.clinical_stage})" if indication.clinical_stage else "")
                ),
                properties={
                    "clinical_stage": indication.clinical_stage,
                    "edge": "drug_indication",
                },
            )

    def from_variant(self, root: VariantRoot) -> None:
        variant_entity = self._add_variant_entity(root)

        for tc in root.transcript_consequences:
            if not tc.target.target_id:
                continue
            target_entity = self._add_target_entity(tc.target)
            self._add_rel(
                variant_entity,
                target_entity,
                RelationshipType.PART_OF,
                description=(
                    "Transcript consequence: "
                    + ", ".join(tc.consequence_terms)
                    if tc.consequence_terms
                    else "Variant maps to target"
                ),
                properties={
                    "consequence_terms": tc.consequence_terms,
                    "edge": "variant_transcript_consequence",
                },
            )

    def from_study(self, root: StudyRoot) -> None:
        study_entity = self._add_study_entity(root)

        for disease in root.diseases:
            if not disease.disease_id:
                continue
            disease_entity = self._add_disease_entity(disease)
            self._add_rel(
                study_entity,
                disease_entity,
                RelationshipType.RELATED_TO,
                description="Study trait",
                properties={"edge": "study_disease"},
            )

    # ── Per-kind entity factories ────────────────────────────────────

    def _add_disease_entity(
        self,
        ref: DiseaseRef,
        extra_aliases: Optional[List[str]] = None,
        mark_root: bool = False,
    ) -> GraphEntity:
        if not ref.name:
            ref.name = ref.disease_id
        existing = self._by_external.get((EntityType.DISEASE, ref.disease_id))
        if existing:
            for alias in extra_aliases or []:
                existing.add_alias(alias)
            if mark_root:
                existing.metadata.add_tag("root")
            return existing

        entity = GraphEntity(
            name=ref.name,
            entity_type=EntityType.DISEASE,
            description=ref.description or None,
        )
        entity.add_external_id("open_targets", ref.disease_id)
        self._tag(entity, "disease", mark_root)
        if ref.therapeutic_areas:
            entity.metadata.add_annotation(
                "therapeutic_areas",
                [ta.get("name") for ta in ref.therapeutic_areas],
            )
        for alias in extra_aliases or []:
            entity.add_alias(alias)
        self._register(entity, EntityType.DISEASE, ref.disease_id)
        return entity

    def _add_target_entity(
        self, ref: TargetRef, mark_root: bool = False
    ) -> GraphEntity:
        existing = self._by_external.get((EntityType.GENE, ref.target_id))
        if existing:
            if mark_root:
                existing.metadata.add_tag("root")
            return existing

        display_name = ref.symbol or ref.name or ref.target_id
        entity = GraphEntity(
            name=display_name,
            entity_type=EntityType.GENE,
            description=(
                ref.function_descriptions[0]
                if ref.function_descriptions
                else None
            ),
        )
        entity.add_external_id("ensembl", ref.target_id)
        entity.add_external_id("open_targets", ref.target_id)
        self._tag(entity, "target", mark_root)
        if ref.biotype:
            entity.metadata.add_tag(ref.biotype)
            entity.properties["biotype"] = ref.biotype
        if ref.name:
            entity.properties["approved_name"] = ref.name
            if ref.symbol and ref.symbol != ref.name:
                entity.add_alias(ref.name)
        self._register(entity, EntityType.GENE, ref.target_id)
        return entity

    def _add_drug_entity(
        self,
        ref: DrugRef,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        extra_aliases: Optional[List[str]] = None,
        mark_root: bool = False,
    ) -> GraphEntity:
        existing = self._by_external.get((EntityType.DRUG, ref.drug_id))
        if existing:
            for alias in extra_aliases or []:
                existing.add_alias(alias)
            if mark_root:
                existing.metadata.add_tag("root")
            return existing

        entity = GraphEntity(
            name=ref.name or ref.drug_id,
            entity_type=EntityType.DRUG,
            description=description or None,
        )
        entity.add_external_id("chembl", ref.drug_id)
        entity.add_external_id("open_targets", ref.drug_id)
        self._tag(entity, "drug", mark_root)
        for k, v in (properties or {}).items():
            if v is not None and v != "":
                entity.properties[k] = v
        for alias in extra_aliases or []:
            if alias:
                entity.add_alias(alias)
        self._register(entity, EntityType.DRUG, ref.drug_id)
        return entity

    def _add_pathway_entity(self, ref: PathwayRef) -> GraphEntity:
        existing = self._by_external.get((EntityType.PATHWAY, ref.pathway_id))
        if existing:
            return existing
        entity = GraphEntity(
            name=ref.name or ref.pathway_id,
            entity_type=EntityType.PATHWAY,
        )
        entity.add_external_id("reactome", ref.pathway_id)
        entity.add_external_id("open_targets", ref.pathway_id)
        self._tag(entity, "pathway", root=False)
        self._register(entity, EntityType.PATHWAY, ref.pathway_id)
        return entity

    def _add_variant_entity(self, root: VariantRoot) -> GraphEntity:
        existing = self._by_external.get((EntityType.CONCEPT, root.id))
        if existing:
            return existing

        # Variant is not a first-class EntityType; tag it explicitly so the
        # graph layer can distinguish it from generic concepts.
        rs_label = root.rs_ids[0] if root.rs_ids else root.id
        entity = GraphEntity(
            name=rs_label,
            entity_type=EntityType.CONCEPT,
            description=(
                f"Variant {root.id} ({root.most_severe_consequence})"
                if root.most_severe_consequence
                else f"Variant {root.id}"
            ),
        )
        entity.add_external_id("open_targets", root.id)
        self._tag(entity, "variant", root=True)
        if root.chromosome:
            entity.properties["chromosome"] = root.chromosome
        if root.position is not None:
            entity.properties["position"] = root.position
        if root.reference_allele:
            entity.properties["reference_allele"] = root.reference_allele
        if root.alternate_allele:
            entity.properties["alternate_allele"] = root.alternate_allele
        if root.most_severe_consequence:
            entity.properties["most_severe_consequence"] = root.most_severe_consequence
        for rs in root.rs_ids:
            entity.add_alias(rs)
        self._register(entity, EntityType.CONCEPT, root.id)
        return entity

    def _add_study_entity(self, root: StudyRoot) -> GraphEntity:
        existing = self._by_external.get((EntityType.DOCUMENT, root.id))
        if existing:
            return existing

        entity = GraphEntity(
            name=root.trait or root.id,
            entity_type=EntityType.DOCUMENT,
            description=(
                f"GWAS study {root.id}"
                + (f" — {root.first_author}" if root.first_author else "")
            ),
        )
        entity.add_external_id("open_targets", root.id)
        if root.pubmed_id:
            entity.add_external_id("pubmed", root.pubmed_id)
        self._tag(entity, "study", root=True)
        if root.publication_date:
            entity.properties["publication_date"] = root.publication_date
        if root.first_author:
            entity.properties["first_author"] = root.first_author
        if root.n_samples is not None:
            entity.properties["n_samples"] = root.n_samples
        self._register(entity, EntityType.DOCUMENT, root.id)
        return entity

    # ── Edge helpers ─────────────────────────────────────────────────

    def _add_known_drug_edges(
        self,
        kd: KnownDrug,
        disease_entity: Optional[GraphEntity] = None,
        target_entity: Optional[GraphEntity] = None,
        drug_entity_override: Optional[GraphEntity] = None,
    ) -> None:
        """Emit drug↔target and/or drug↔disease edges from a knownDrugs row."""
        if not kd.drug.drug_id and drug_entity_override is None:
            return

        drug_entity = drug_entity_override or self._add_drug_entity(kd.drug)

        # Resolve target/disease endpoints from either the row itself or
        # the caller-supplied root.
        edge_target = target_entity
        if edge_target is None and kd.target and kd.target.target_id:
            edge_target = self._add_target_entity(kd.target)

        edge_disease = disease_entity
        if edge_disease is None and kd.disease and kd.disease.disease_id:
            edge_disease = self._add_disease_entity(kd.disease)

        stage_strength = _stage_to_strength(kd.clinical_stage)

        if edge_target is not None:
            self._add_rel(
                drug_entity,
                edge_target,
                RelationshipType.INFLUENCES,
                description=f"Clinical candidate for target ({kd.clinical_stage})" if kd.clinical_stage else "Known drug → target",
                strength=stage_strength,
                properties={
                    "clinical_stage": kd.clinical_stage,
                    "edge": "known_drug_target",
                },
                dedupe_key=("known_drug_target", drug_entity.id, edge_target.id),
            )

        if edge_disease is not None:
            self._add_rel(
                drug_entity,
                edge_disease,
                RelationshipType.INFLUENCES,
                description=f"Known drug for {edge_disease.name}",
                strength=stage_strength,
                properties={
                    "clinical_stage": kd.clinical_stage,
                    "edge": "known_drug_disease",
                },
                dedupe_key=("known_drug_disease", drug_entity.id, edge_disease.id),
            )

    def _add_rel(
        self,
        source: GraphEntity,
        target: GraphEntity,
        rel_type: RelationshipType,
        description: str = "",
        strength: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
        dedupe_key: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        if source.id == target.id:
            return  # Self-loops are rejected by GraphRelationship.validate
        if dedupe_key is not None and dedupe_key in self._seen_edges:
            return
        rel = GraphRelationship(
            source_entity_id=source.id,
            target_entity_id=target.id,
            relationship_type=rel_type,
            description=description or None,
            strength=_clamp(strength),
        )
        for k, v in (properties or {}).items():
            if v is not None:
                rel.properties[k] = v
        rel.properties["source"] = "open_targets"
        rel.metadata.source_trust = "reviewed"
        rel.metadata.source_system = "open_targets"
        rel.metadata.add_tag(self.cfg.tag)
        self.relationships.append(rel)
        if dedupe_key is not None:
            self._seen_edges.add(dedupe_key)

    # ── Bookkeeping ──────────────────────────────────────────────────

    def _tag(self, entity: GraphEntity, kind_tag: str, root: bool) -> None:
        entity.metadata.add_tag(self.cfg.tag)
        entity.metadata.add_tag(kind_tag)
        if root:
            entity.metadata.add_tag("root")
        entity.metadata.source_trust = "reviewed"
        entity.metadata.source_system = "open_targets"

    def _register(
        self, entity: GraphEntity, entity_type: EntityType, external_id: str
    ) -> None:
        self._by_external[(entity_type, external_id)] = entity
        self.entities.append(entity)


# ── Free helpers ─────────────────────────────────────────────────────────


def _clamp(x: float) -> float:
    return min(max(x, 0.0), 1.0)


_STAGE_STRENGTH = {
    "phase i": 0.25,
    "phase i/ii": 0.4,
    "phase ii": 0.5,
    "phase ii/iii": 0.6,
    "phase iii": 0.75,
    "phase iv": 1.0,
    "approved": 1.0,
    "preclinical": 0.1,
}


def _stage_to_strength(stage: str) -> float:
    """Map an Open Targets ``maxClinicalStage`` label to an edge strength."""
    if not stage:
        return 0.5
    return _STAGE_STRENGTH.get(stage.strip().lower(), 0.5)


def _root_display_name(result: IngestResult) -> str:
    root = result.root
    if isinstance(root, DiseaseRoot):
        return root.name or root.id
    if isinstance(root, TargetRoot):
        return root.symbol or root.name or root.id
    if isinstance(root, DrugRoot):
        return root.name or root.id
    if isinstance(root, VariantRoot):
        return (root.rs_ids[0] if root.rs_ids else root.id)
    if isinstance(root, StudyRoot):
        return root.trait or root.id
    return ""


def _total_neighbors(result: IngestResult) -> int:
    root = result.root
    if isinstance(root, DiseaseRoot):
        return root.total_associated_targets
    if isinstance(root, TargetRoot):
        return root.total_associated_diseases
    return 0
