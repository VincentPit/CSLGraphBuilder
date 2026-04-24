"""Dev-only routes — seed the in-memory graph with sample biomedical data."""

import os
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_graph_repo

router = APIRouter(prefix="/dev", tags=["dev"])


"""Dev-only routes — seed the graph with sample biomedical data from PubMed and Open Targets."""

import os
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_graph_repo

router = APIRouter(prefix="/dev", tags=["dev"])

# ---------------------------------------------------------------------------
# PubMed-sourced entities (literature-derived facts)
# ---------------------------------------------------------------------------
PUBMED_ENTITIES = [
    # Diseases
    ("pm-d1", "Lung Adenocarcinoma", "DISEASE", "Most common subtype of non-small-cell lung cancer; driven by EGFR, KRAS, and ALK mutations. PMID:29625055"),
    ("pm-d2", "Parkinson's Disease", "DISEASE", "Progressive neurodegenerative disorder characterised by loss of dopaminergic neurons in the substantia nigra. PMID:30291251"),
    ("pm-d3", "Rheumatoid Arthritis", "DISEASE", "Chronic autoimmune inflammatory disease primarily affecting synovial joints. PMID:27156434"),
    # Genes
    ("pm-g1", "EGFR", "GENE", "Epidermal growth factor receptor; tyrosine kinase driving proliferation in lung adenocarcinoma. PMID:29625055"),
    ("pm-g2", "KRAS", "GENE", "GTPase proto-oncogene; KRAS G12C mutation found in ~13 % of lung adenocarcinomas. PMID:34919752"),
    ("pm-g3", "LRRK2", "GENE", "Leucine-rich repeat kinase 2; most common cause of autosomal-dominant Parkinson's disease. PMID:29847546"),
    ("pm-g4", "SNCA", "GENE", "Alpha-synuclein gene; aggregation of its product forms Lewy bodies in Parkinson's disease. PMID:30291251"),
    ("pm-g5", "TNF", "GENE", "Tumour necrosis factor; key pro-inflammatory cytokine central to rheumatoid arthritis pathogenesis. PMID:27156434"),
    # Proteins
    ("pm-pr1", "PD-L1", "PROTEIN", "Programmed death-ligand 1; immune checkpoint protein exploited by lung adenocarcinoma to evade T cells. PMID:29625055"),
    ("pm-pr2", "Alpha-synuclein", "PROTEIN", "Neuronal protein encoded by SNCA; misfolded aggregates are the hallmark of Parkinson's Lewy body pathology. PMID:30291251"),
    # Drugs
    ("pm-dr1", "Osimertinib", "DRUG", "Third-generation EGFR tyrosine kinase inhibitor approved for EGFR-mutant lung adenocarcinoma. PMID:29151359"),
    ("pm-dr2", "Sotorasib", "DRUG", "First covalent KRAS G12C inhibitor; FDA-approved for KRAS G12C-mutant non-small-cell lung cancer. PMID:34919752"),
    ("pm-dr3", "Levodopa", "DRUG", "Dopamine precursor; gold-standard treatment for motor symptoms of Parkinson's disease. PMID:29847546"),
    ("pm-dr4", "Adalimumab", "DRUG", "Anti-TNF monoclonal antibody used as disease-modifying therapy in rheumatoid arthritis. PMID:27156434"),
    # Pathways
    ("pm-pw1", "EGFR-RAS-MAPK Signalling", "PATHWAY", "Receptor tyrosine kinase cascade regulating cell proliferation; frequently dysregulated in lung adenocarcinoma. PMID:29625055"),
    ("pm-pw2", "Dopamine Biosynthesis", "PATHWAY", "Metabolic pathway converting tyrosine to dopamine via tyrosine hydroxylase; impaired in Parkinson's disease. PMID:29847546"),
]

PUBMED_RELATIONSHIPS = [
    # Gene → Disease
    ("pm-r1",  "pm-g1", "pm-d1", "RELATED_TO", 0.96, "Activating EGFR mutations are the primary oncogenic driver of lung adenocarcinoma. PMID:29625055"),
    ("pm-r2",  "pm-g2", "pm-d1", "RELATED_TO", 0.93, "KRAS G12C mutation is found in approximately 13 % of lung adenocarcinomas. PMID:34919752"),
    ("pm-r3",  "pm-g3", "pm-d2", "RELATED_TO", 0.91, "LRRK2 G2019S is the most frequent genetic cause of Parkinson's disease. PMID:29847546"),
    ("pm-r4",  "pm-g4", "pm-d2", "RELATED_TO", 0.94, "SNCA encodes alpha-synuclein whose aggregation causes Parkinson's Lewy body pathology. PMID:30291251"),
    ("pm-r5",  "pm-g5", "pm-d3", "RELATED_TO", 0.95, "TNF is the central pro-inflammatory cytokine driving joint destruction in rheumatoid arthritis. PMID:27156434"),
    # Drug → Disease (treats)
    ("pm-r6",  "pm-dr1", "pm-d1", "RELATED_TO", 0.97, "Osimertinib is the preferred first-line treatment for EGFR-mutant lung adenocarcinoma. PMID:29151359"),
    ("pm-r7",  "pm-dr2", "pm-d1", "RELATED_TO", 0.90, "Sotorasib provides clinical benefit in KRAS G12C-mutant non-small-cell lung cancer. PMID:34919752"),
    ("pm-r8",  "pm-dr3", "pm-d2", "RELATED_TO", 0.98, "Levodopa is the gold-standard treatment for motor symptoms of Parkinson's disease. PMID:29847546"),
    ("pm-r9",  "pm-dr4", "pm-d3", "RELATED_TO", 0.94, "Adalimumab (anti-TNF) significantly reduces disease activity in rheumatoid arthritis. PMID:27156434"),
    # Drug → Gene (target)
    ("pm-r10", "pm-dr1", "pm-g1", "INFLUENCES", 0.97, "Osimertinib covalently inhibits EGFR tyrosine kinase activity. PMID:29151359"),
    ("pm-r11", "pm-dr2", "pm-g2", "INFLUENCES", 0.95, "Sotorasib covalently binds and locks KRAS G12C in the GDP-bound inactive state. PMID:34919752"),
    ("pm-r12", "pm-dr4", "pm-g5", "INFLUENCES", 0.96, "Adalimumab neutralises TNF, blocking downstream inflammatory signalling. PMID:27156434"),
    # Protein relationships
    ("pm-r13", "pm-pr1", "pm-d1", "RELATED_TO", 0.88, "PD-L1 overexpression allows lung adenocarcinoma to evade anti-tumour immunity. PMID:29625055"),
    ("pm-r14", "pm-pr2", "pm-d2", "RELATED_TO", 0.93, "Alpha-synuclein aggregates form Lewy bodies, the pathological hallmark of Parkinson's. PMID:30291251"),
    # Gene → Pathway
    ("pm-r15", "pm-g1", "pm-pw1", "PART_OF", 0.92, "EGFR activation initiates the RAS-MAPK signalling cascade. PMID:29625055"),
    ("pm-r16", "pm-g2", "pm-pw1", "PART_OF", 0.90, "KRAS is a core component of the EGFR-RAS-MAPK signalling pathway. PMID:29625055"),
]

# ---------------------------------------------------------------------------
# Open Targets-sourced entities (target-disease association evidence)
# ---------------------------------------------------------------------------
OT_ENTITIES = [
    # Diseases
    ("ot-d1", "Ulcerative Colitis", "DISEASE", "Chronic inflammatory bowel disease affecting the colon and rectum. Open Targets: EFO_0000729"),
    ("ot-d2", "Atopic Dermatitis", "DISEASE", "Chronic inflammatory skin condition characterised by pruritic eczematous lesions. Open Targets: EFO_0000274"),
    ("ot-d3", "Chronic Kidney Disease", "DISEASE", "Progressive loss of kidney function over months to years; eGFR < 60 mL/min. Open Targets: EFO_0003884"),
    ("ot-d4", "Multiple Myeloma", "DISEASE", "Haematological malignancy of plasma cells in bone marrow. Open Targets: EFO_0001378"),
    # Genes / Targets
    ("ot-g1", "JAK2", "GENE", "Janus kinase 2; non-receptor tyrosine kinase involved in cytokine signalling. Open Targets overall association with ulcerative colitis: 0.72"),
    ("ot-g2", "IL13", "GENE", "Interleukin-13; type-2 cytokine driving Th2-mediated inflammation in atopic dermatitis. Open Targets overall association: 0.81"),
    ("ot-g3", "UMOD", "GENE", "Uromodulin; most abundant urinary protein; genetic variants associated with chronic kidney disease risk. Open Targets: 0.65"),
    ("ot-g4", "APOL1", "GENE", "Apolipoprotein L1; high-risk variants G1/G2 strongly predispose to chronic kidney disease in individuals of African ancestry. Open Targets: 0.74"),
    ("ot-g5", "BRAF", "GENE", "Serine/threonine kinase; V600E mutation found in hairy cell leukaemia, melanoma, and some myelomas. Open Targets association with multiple myeloma: 0.44"),
    ("ot-g6", "CRBN", "GENE", "Cereblon; E3 ubiquitin ligase substrate receptor; molecular target of lenalidomide in multiple myeloma. Open Targets: 0.82"),
    # Proteins
    ("ot-pr1", "IL-4Rα", "PROTEIN", "Interleukin-4 receptor alpha subunit; shared receptor for IL-4 and IL-13 signalling in atopic dermatitis."),
    # Drugs
    ("ot-dr1", "Tofacitinib", "DRUG", "Pan-JAK inhibitor approved for ulcerative colitis; preferentially inhibits JAK1 and JAK3. Open Targets clinical phase 4"),
    ("ot-dr2", "Dupilumab", "DRUG", "Monoclonal antibody blocking IL-4Rα; first biologic approved for moderate-to-severe atopic dermatitis. Open Targets clinical phase 4"),
    ("ot-dr3", "Dapagliflozin", "DRUG", "SGLT2 inhibitor shown to slow progression of chronic kidney disease regardless of diabetes status. Open Targets clinical phase 4"),
    ("ot-dr4", "Lenalidomide", "DRUG", "Immunomodulatory drug targeting CRBN; backbone of multiple myeloma treatment regimens. Open Targets clinical phase 4"),
    # Pathways
    ("ot-pw1", "JAK-STAT Signalling", "PATHWAY", "Cytokine-activated signalling cascade; JAK phosphorylates STAT transcription factors regulating immune gene expression."),
    ("ot-pw2", "NF-κB Inflammatory Pathway", "PATHWAY", "Master transcription factor pathway driving inflammatory cytokine production in autoimmune diseases."),
]

OT_RELATIONSHIPS = [
    # Gene → Disease (Open Targets associations)
    ("ot-r1",  "ot-g1", "ot-d1", "RELATED_TO",  0.72, "JAK2 is associated with ulcerative colitis (Open Targets overall score 0.72)."),
    ("ot-r2",  "ot-g2", "ot-d2", "RELATED_TO",  0.81, "IL13 is a key driver of type-2 inflammation in atopic dermatitis (Open Targets score 0.81)."),
    ("ot-r3",  "ot-g3", "ot-d3", "RELATED_TO",  0.65, "UMOD variants are associated with chronic kidney disease risk (Open Targets score 0.65)."),
    ("ot-r4",  "ot-g4", "ot-d3", "RELATED_TO",  0.74, "APOL1 G1/G2 risk variants strongly predispose to chronic kidney disease (Open Targets score 0.74)."),
    ("ot-r5",  "ot-g5", "ot-d4", "RELATED_TO",  0.44, "BRAF mutations have a low but detectable association with multiple myeloma (Open Targets score 0.44)."),
    ("ot-r6",  "ot-g6", "ot-d4", "RELATED_TO",  0.82, "CRBN is the molecular target of IMiDs and is critical in multiple myeloma therapy (Open Targets score 0.82)."),
    # Drug → Disease (treats)
    ("ot-r7",  "ot-dr1", "ot-d1", "RELATED_TO",  0.90, "Tofacitinib is an approved JAK inhibitor for moderate-to-severe ulcerative colitis."),
    ("ot-r8",  "ot-dr2", "ot-d2", "RELATED_TO",  0.95, "Dupilumab is the first-line biologic for moderate-to-severe atopic dermatitis."),
    ("ot-r9",  "ot-dr3", "ot-d3", "RELATED_TO",  0.88, "Dapagliflozin slows CKD progression regardless of diabetes status (DAPA-CKD trial)."),
    ("ot-r10", "ot-dr4", "ot-d4", "RELATED_TO",  0.96, "Lenalidomide is the backbone of multiple myeloma treatment regimens."),
    # Drug → Gene (target)
    ("ot-r11", "ot-dr1", "ot-g1", "INFLUENCES",  0.93, "Tofacitinib inhibits JAK2 kinase activity, suppressing cytokine signalling."),
    ("ot-r12", "ot-dr2", "ot-pr1","INFLUENCES",  0.96, "Dupilumab blocks IL-4Rα, preventing IL-4 and IL-13 signalling."),
    ("ot-r13", "ot-dr4", "ot-g6", "INFLUENCES",  0.95, "Lenalidomide binds CRBN to recruit neo-substrates for proteasomal degradation."),
    # Gene → Pathway
    ("ot-r14", "ot-g1", "ot-pw1", "PART_OF",     0.88, "JAK2 is a central kinase in the JAK-STAT signalling cascade."),
    ("ot-r15", "ot-g2", "ot-pw2", "INFLUENCES",   0.75, "IL13 activates NF-κB inflammatory signalling in epithelial cells."),
    ("ot-r16", "ot-g5", "ot-pw1", "INFLUENCES",   0.60, "BRAF cross-talks with JAK-STAT signalling in haematological malignancies."),
]

# ---------------------------------------------------------------------------
# Cross-source relationships — link PubMed and Open Targets sub-graphs so the
# combined graph is connected and the force-graph viewer shows interplay.
# ---------------------------------------------------------------------------
CROSS_RELATIONSHIPS = [
    ("xr-1", "pm-g5",  "ot-d1", "RELATED_TO",  0.70, "TNF signalling drives mucosal inflammation in ulcerative colitis (cross-source)."),
    ("xr-2", "pm-g5",  "ot-pw2", "PART_OF",    0.85, "TNF is an upstream activator of the NF-κB inflammatory pathway."),
    ("xr-3", "pm-dr4", "ot-d1", "RELATED_TO",  0.80, "Adalimumab (anti-TNF) is approved for moderate-to-severe ulcerative colitis."),
    ("xr-4", "ot-g1",  "pm-d3", "RELATED_TO",  0.65, "JAK2 contributes to cytokine signalling implicated in rheumatoid arthritis."),
    ("xr-5", "ot-dr1", "pm-d3", "RELATED_TO",  0.78, "Tofacitinib is also approved for rheumatoid arthritis as a JAK inhibitor."),
    ("xr-6", "pm-g1",  "ot-pw1", "INFLUENCES", 0.72, "EGFR cross-talks with JAK-STAT signalling in epithelial tumours."),
    ("xr-7", "ot-g5",  "pm-pw1", "PART_OF",    0.74, "BRAF acts downstream of RAS in the EGFR-RAS-MAPK signalling cascade."),
    ("xr-8", "pm-pr1", "ot-d4", "RELATED_TO",  0.55, "PD-L1 expression on multiple-myeloma cells contributes to immune escape (emerging evidence)."),
]

# Combine all seed data
SEED_ENTITIES = PUBMED_ENTITIES + OT_ENTITIES
SEED_RELATIONSHIPS = PUBMED_RELATIONSHIPS + OT_RELATIONSHIPS + CROSS_RELATIONSHIPS


@router.post("/seed", summary="Seed sample biomedical data (dev only)")
async def seed_graph(graph_repo=Depends(get_graph_repo)):
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        raise HTTPException(status_code=403, detail="Seed endpoint is disabled in production.")

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.domain.models.graph_models import (
        GraphEntity, EntityType, GraphRelationship, RelationshipType,
    )
    from graphbuilder.infrastructure.repositories.graph_repository import Neo4jGraphRepository

    TYPE_MAP = {v.value: v for v in EntityType}
    REL_MAP = {v.value: v for v in RelationshipType}

    use_cypher = isinstance(graph_repo, Neo4jGraphRepository)
    added_e, added_r = 0, 0

    if use_cypher:
        # Seed directly via Cypher — store biomedical entity_type strings
        # so the frontend TYPE_COLORS map (GENE, DISEASE, DRUG, etc.) works
        for eid, name, etype, desc in SEED_ENTITIES:
            await graph_repo.execute_cypher_query(
                """
                MERGE (e:Entity {id: $id})
                ON CREATE SET e.name = $name, e.entity_type = $entity_type,
                              e.description = $desc,
                              e.created_at = datetime(), e.version = 1
                """,
                {"id": eid, "name": name, "entity_type": etype,
                 "desc": desc},
            )
            added_e += 1

        for rid, src, tgt, rtype, strength, desc in SEED_RELATIONSHIPS:
            await graph_repo.execute_cypher_query(
                """
                MATCH (s:Entity {id: $src}), (t:Entity {id: $tgt})
                MERGE (s)-[r:RELATES {id: $rid}]->(t)
                ON CREATE SET r.relationship_type = $rtype, r.description = $desc,
                              r.strength = $strength, r.created_at = datetime(),
                              r.source_entity_id = $src, r.target_entity_id = $tgt
                """,
                {"rid": rid, "src": src, "tgt": tgt,
                 "rtype": REL_MAP.get(rtype, RelationshipType.RELATED_TO).value,
                 "desc": desc, "strength": strength},
            )
            added_r += 1

        stats = await graph_repo.get_graph_statistics()
    else:
        # In-memory path
        for eid, name, etype, desc in SEED_ENTITIES:
            existing = await graph_repo.get_entity_by_id(eid)
            if existing is None:
                e = GraphEntity(name=name, entity_type=TYPE_MAP.get(etype, EntityType.CONCEPT), description=desc)
                e.id = eid
                await graph_repo.save_entity(e)
                added_e += 1

        for rid, src, tgt, rtype, strength, desc in SEED_RELATIONSHIPS:
            existing = await graph_repo.get_relationship_by_id(rid)
            if existing is None:
                r = GraphRelationship(
                    source_entity_id=src,
                    target_entity_id=tgt,
                    relationship_type=REL_MAP.get(rtype, RelationshipType.RELATED_TO),
                    description=desc,
                    strength=strength,
                )
                r.id = rid
                await graph_repo.save_relationship(r)
                added_r += 1

        stats = {"total_entities": len(graph_repo.entities),
                 "total_relationships": len(graph_repo.relationships)}
    return {
        "message": f"Seeded {added_e} entities and {added_r} relationships.",
        "total_entities": stats.get("total_entities", added_e),
        "total_relationships": stats.get("total_relationships", added_r),
    }


@router.post("/seed-curation", summary="Tag existing items for the curation queue (dev only)")
async def seed_curation(graph_repo=Depends(get_graph_repo)):
    """Mark a sample of existing entities/relationships as ``unverified``,
    ``flagged``, or ``rejected`` so the Curation page is populated.

    Why we need this: pipeline-extracted items now land as ``unverified``
    automatically, but seeded reference data (PubMed/OpenTargets dev seed,
    bulk ingests) is intentionally tagged ``verified`` and won't show up.
    This route stamps a representative slice of the existing graph with
    a mix of statuses so the queue, status badges, and approve/reject
    actions all have something to act on.
    """
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        raise HTTPException(status_code=403, detail="Disabled in production.")

    all_entities = await graph_repo.get_all_entities()
    all_rels = await graph_repo.get_all_relationships()

    ent_list = list(all_entities.values())
    rel_list = list(all_rels.values())

    if not ent_list and not rel_list:
        return {
            "message": "Graph is empty — run /dev/seed first or process a document.",
            "tagged": {"entities": 0, "relationships": 0},
        }

    # Round-robin three statuses across the first ~12 of each so the table
    # shows a representative mix.
    statuses = ["unverified", "flagged", "rejected"]
    notes = {
        "unverified": "Awaiting reviewer confirmation",
        "flagged": "Low confidence — please double-check sources",
        "rejected": "Conflicts with curated reference data",
    }

    tagged_e = 0
    for i, ent in enumerate(ent_list[:12]):
        status = statuses[i % len(statuses)]
        ent.metadata.add_annotation("verification_status", status)
        ent.metadata.add_annotation("verification_notes", notes[status])
        await graph_repo.save_entity(ent)
        tagged_e += 1

    tagged_r = 0
    for i, rel in enumerate(rel_list[:8]):
        status = statuses[i % len(statuses)]
        rel.metadata.add_annotation("verification_status", status)
        rel.metadata.add_annotation("verification_notes", notes[status])
        await graph_repo.save_relationship(rel)
        tagged_r += 1

    return {
        "message": f"Tagged {tagged_e} entities and {tagged_r} relationships for curation.",
        "tagged": {"entities": tagged_e, "relationships": tagged_r},
        "tip": "Open /curation in the frontend to see them.",
    }


@router.post("/reembed", summary="Recompute every embedding with the current model")
async def reembed_graph(graph_repo=Depends(get_graph_repo)):
    """Migrate the graph to the currently-configured embedding model.

    What it does, in order:
      1. Detects the current model + dim from ``embedding_factory``.
      2. Drops the two vector indexes (``entity_name_vector`` and
         ``rel_desc_vector``) — they may have been created with a
         different dim, which would silently break new searches.
      3. Re-embeds every entity (name + description) and every
         relationship (description) with the current model and saves.
      4. ``save_entity`` / ``save_relationship`` call the repo's index
         setup on first write, so the indexes get recreated with the
         new dim automatically.

    Use this when you switch ``EMBEDDING_MODEL`` (e.g. MiniLM 384-d →
    SapBERT 768-d). Safe to run multiple times — it's idempotent.
    """
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        raise HTTPException(status_code=403, detail="Reembed endpoint is disabled in production.")

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.infrastructure.services.embedding_factory import (
        get_model_name, get_embedding_dim,
    )
    from graphbuilder.infrastructure.repositories.graph_repository import Neo4jGraphRepository

    model_name = get_model_name() or "(unknown)"
    new_dim = get_embedding_dim() or 0
    if new_dim == 0:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not loaded yet — try again in a few seconds.",
        )

    # Drop old vector indexes so they get recreated with the right dim
    # on the first save below. Best-effort; a missing index is fine.
    if isinstance(graph_repo, Neo4jGraphRepository):
        for idx in ("entity_name_vector", "rel_desc_vector"):
            try:
                await graph_repo.execute_cypher_query(f"DROP INDEX `{idx}` IF EXISTS", {})
            except Exception:
                pass
        # Reset the in-memory dim cache so the next save reads the new dim.
        try:
            graph_repo._embedding_dim = new_dim   # type: ignore[attr-defined]
        except Exception:
            pass

    # Re-embed and save every entity + relationship.
    all_entities = await graph_repo.get_all_entities()
    all_rels = await graph_repo.get_all_relationships()

    re_e, re_r, errors = 0, 0, 0
    for ent in list(all_entities.values()):
        try:
            await graph_repo.save_entity(ent)   # save_entity recomputes the embedding
            re_e += 1
        except Exception:
            errors += 1
    for rel in list(all_rels.values()):
        try:
            await graph_repo.save_relationship(rel)
            re_r += 1
        except Exception:
            errors += 1

    return {
        "model": model_name,
        "dim": new_dim,
        "reembedded_entities": re_e,
        "reembedded_relationships": re_r,
        "errors": errors,
        "tip": "Run a Process job — vector pre-filter will now use the new model.",
    }


@router.delete("/reset", summary="Clear all data (dev only)")
async def reset_graph(graph_repo=Depends(get_graph_repo)):
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        raise HTTPException(status_code=403, detail="Reset endpoint is disabled in production.")
    # Works for both Neo4j and in-memory repos
    try:
        await graph_repo.execute_cypher_query("MATCH (n) DETACH DELETE n", {})
    except Exception:
        # Fallback for in-memory
        if hasattr(graph_repo, "entities"):
            graph_repo.entities.clear()
        if hasattr(graph_repo, "relationships"):
            graph_repo.relationships.clear()
    return {"message": "Graph cleared."}
