"""
Open Targets GraphQL API Client.

Wraps the Open Targets Platform GraphQL API
(https://api.platform.opentargets.org/api/v4/graphql), providing typed,
async access to all primary entity kinds — diseases, targets (genes),
drugs, variants, and studies — along with their associated edges.

The client auto-detects entity kind from the ID prefix:

    EFO_/MONDO_/Orphanet_/HP_/DOID_/OTAR_  → DISEASE
    ENSG…                                  → TARGET
    CHEMBL…                                → DRUG
    rs… or chr-pos-ref-alt                 → VARIANT
    GCST…                                  → STUDY

A single ``fetch_entity(entity_id, kind=None)`` call returns the root
entity plus all neighbor edges, so the use-case layer can persist them
uniformly regardless of which entity type the user submitted.

All network I/O is async; the caller is responsible for providing an
aiohttp.ClientSession or letting this module manage one per call.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    AIOHTTP_AVAILABLE = False


_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"


class EntityKind(str, Enum):
    """Open Targets root entity kinds supported by this client."""

    DISEASE = "disease"
    TARGET = "target"
    DRUG = "drug"
    VARIANT = "variant"
    STUDY = "study"


_DISEASE_PREFIXES = ("EFO_", "MONDO_", "Orphanet_", "HP_", "DOID_", "OTAR_", "MP_", "GO_")
_VARIANT_RE = re.compile(r"^(rs\d+|\d+_\d+_[ACGT]+_[ACGT]+|[XYM]+_\d+_[ACGT]+_[ACGT]+)$", re.IGNORECASE)


def detect_entity_kind(entity_id: str) -> Optional[EntityKind]:
    """
    Best-effort identification of an Open Targets ID's kind from its prefix.

    Returns ``None`` when the ID does not match any known pattern, in which
    case callers should require the user to pass an explicit ``EntityKind``.
    """
    if not entity_id:
        return None
    if entity_id.startswith("ENSG"):
        return EntityKind.TARGET
    if entity_id.startswith("CHEMBL"):
        return EntityKind.DRUG
    if entity_id.startswith(_DISEASE_PREFIXES):
        return EntityKind.DISEASE
    if entity_id.startswith("GCST"):
        return EntityKind.STUDY
    if _VARIANT_RE.match(entity_id):
        return EntityKind.VARIANT
    return None


# ── GraphQL queries ──────────────────────────────────────────────────────

# GraphQL query field names verified against the live Open Targets v4
# Platform schema (introspection 2026-04). Notable schema quirks:
#   - Disease/Target use `drugAndClinicalCandidates`, not `knownDrugs`.
#   - Drug.indications returns clinicalIndicationsFromDrugImp (no
#     pagination args, no `linkedTargets/linkedDiseases/knownDrugs`).
#   - Drug has `maximumClinicalStage` (String), not the older
#     `maximumClinicalTrialPhase / isApproved / yearOfFirstApproval` fields.
#   - Variant's most-severe-consequence is a SequenceOntologyTerm object,
#     and transcript-consequence sequence terms are exposed as
#     `variantConsequences { id label }` — NOT `consequenceTerms`.
#   - Study identifier field is `id`, not `studyId`.

_DISEASE_QUERY = """
query DiseaseRoot($id: String!, $size: Int!) {
  disease(efoId: $id) {
    id
    name
    description
    therapeuticAreas { id name }
    synonyms { relation terms }
    associatedTargets(page: { index: 0, size: $size }) {
      count
      rows {
        score
        datatypeScores { id score }
        target { id approvedSymbol approvedName biotype functionDescriptions }
      }
    }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug { id name }
      }
    }
  }
}
"""

_TARGET_QUERY = """
query TargetRoot($id: String!, $size: Int!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    pathways { pathway pathwayId }
    associatedDiseases(page: { index: 0, size: $size }) {
      count
      rows {
        score
        datatypeScores { id score }
        disease { id name description therapeuticAreas { id name } }
      }
    }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug { id name }
        diseases { disease { id name } }
      }
    }
  }
}
"""

_DRUG_QUERY = """
query DrugRoot($id: String!) {
  drug(chemblId: $id) {
    id
    name
    description
    synonyms
    tradeNames
    drugType
    maximumClinicalStage
    mechanismsOfAction {
      rows {
        mechanismOfAction
        actionType
        targetName
        targets { id approvedSymbol approvedName biotype }
      }
    }
    indications {
      count
      rows {
        id
        maxClinicalStage
        disease { id name description therapeuticAreas { id name } }
      }
    }
  }
}
"""

_VARIANT_QUERY = """
query VariantRoot($id: String!) {
  variant(variantId: $id) {
    id
    rsIds
    chromosome
    position
    referenceAllele
    alternateAllele
    mostSevereConsequence { id label }
    transcriptConsequences {
      target { id approvedSymbol approvedName }
      variantConsequences { id label }
    }
  }
}
"""

_STUDY_QUERY = """
query StudyRoot($id: String!) {
  study(studyId: $id) {
    id
    traitFromSource
    publicationDate
    pubmedId
    publicationFirstAuthor
    nSamples
    diseases { id name }
  }
}
"""


# ── Normalized neighbor dataclasses ──────────────────────────────────────


@dataclass
class TargetRef:
    """A neighbor target/gene reference."""

    target_id: str
    symbol: str = ""
    name: str = ""
    biotype: str = ""
    function_descriptions: List[str] = field(default_factory=list)


@dataclass
class DiseaseRef:
    """A neighbor disease reference."""

    disease_id: str
    name: str = ""
    description: str = ""
    therapeutic_areas: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DrugRef:
    """A neighbor drug reference."""

    drug_id: str
    name: str = ""


@dataclass
class PathwayRef:
    """A pathway a target participates in."""

    pathway_id: str
    name: str = ""


@dataclass
class TargetAssociation:
    """Disease ↔ target association edge with scores."""

    target: TargetRef
    score: float = 0.0
    datatype_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class DiseaseAssociation:
    """Target ↔ disease association edge with scores."""

    disease: DiseaseRef
    score: float = 0.0
    datatype_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class MechanismOfAction:
    """Drug → target(s) mechanism edge."""

    description: str
    action_type: str = ""
    target_name: str = ""
    targets: List[TargetRef] = field(default_factory=list)


@dataclass
class Indication:
    """Drug → disease indication edge."""

    disease: DiseaseRef
    clinical_stage: str = ""  # Open Targets `maxClinicalStage` (e.g. "Phase IV")


@dataclass
class KnownDrug:
    """Disease/target ↔ drug clinical-candidate edge.

    Backed by Open Targets' ``drugAndClinicalCandidates`` field — each row
    is one indication-level link from a (disease|target) to a drug, with
    the clinical stage that the drug has reached for that link.
    """

    drug: DrugRef
    clinical_stage: str = ""
    disease: Optional[DiseaseRef] = None
    target: Optional[TargetRef] = None


# ── Root entity dataclasses (one per kind) ───────────────────────────────


@dataclass
class DiseaseRoot:
    id: str
    name: str = ""
    description: str = ""
    therapeutic_areas: List[Dict[str, str]] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    associated_targets: List[TargetAssociation] = field(default_factory=list)
    total_associated_targets: int = 0
    known_drugs: List[KnownDrug] = field(default_factory=list)


@dataclass
class TargetRoot:
    id: str
    symbol: str = ""
    name: str = ""
    biotype: str = ""
    function_descriptions: List[str] = field(default_factory=list)
    pathways: List[PathwayRef] = field(default_factory=list)
    associated_diseases: List[DiseaseAssociation] = field(default_factory=list)
    total_associated_diseases: int = 0
    known_drugs: List[KnownDrug] = field(default_factory=list)


@dataclass
class DrugRoot:
    id: str
    name: str = ""
    description: str = ""
    synonyms: List[str] = field(default_factory=list)
    trade_names: List[str] = field(default_factory=list)
    drug_type: str = ""
    max_clinical_stage: str = ""
    mechanisms_of_action: List[MechanismOfAction] = field(default_factory=list)
    indications: List[Indication] = field(default_factory=list)


@dataclass
class VariantTranscriptConsequence:
    target: TargetRef
    consequence_terms: List[str] = field(default_factory=list)


@dataclass
class VariantRoot:
    id: str
    rs_ids: List[str] = field(default_factory=list)
    chromosome: str = ""
    position: Optional[int] = None
    reference_allele: str = ""
    alternate_allele: str = ""
    most_severe_consequence: str = ""
    transcript_consequences: List[VariantTranscriptConsequence] = field(default_factory=list)


@dataclass
class StudyRoot:
    id: str
    trait: str = ""
    publication_date: str = ""
    pubmed_id: str = ""
    first_author: str = ""
    n_samples: Optional[int] = None
    diseases: List[DiseaseRef] = field(default_factory=list)


@dataclass
class IngestResult:
    """Outcome of a single ``fetch_entity`` call."""

    kind: Optional[EntityKind]
    root: Optional[Any]  # one of DiseaseRoot/TargetRoot/DrugRoot/VariantRoot/StudyRoot
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.root is not None and not self.errors


# ── Client ───────────────────────────────────────────────────────────────


class OpenTargetsClient:
    """
    Async GraphQL client for the Open Targets Platform API.

    Usage::

        async with OpenTargetsClient() as client:
            result = await client.fetch_entity("ENSG00000048462", max_neighbors=200)

    Parameters
    ----------
    endpoint:
        GraphQL endpoint URL. Defaults to the public Open Targets API.
    page_size:
        Default page size for paginated edges (associatedTargets, etc.).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str = _ENDPOINT,
        page_size: int = 100,
        timeout: int = 30,
    ) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is required for OpenTargetsClient. "
                "Install it with: pip install aiohttp"
            )
        self._endpoint = endpoint
        self._page_size = page_size
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def __aenter__(self) -> "OpenTargetsClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_entity(
        self,
        entity_id: str,
        kind: Optional[EntityKind] = None,
        max_neighbors: int = 500,
        max_known_drugs: int = 100,  # accepted for back-compat; unused — drugAndClinicalCandidates is non-paginated
    ) -> IngestResult:
        """
        Fetch the root entity and its associated edges.

        ``kind`` is auto-detected from the ID prefix when omitted. Pass it
        explicitly when the prefix is ambiguous.
        """
        del max_known_drugs  # silence linter — kept in signature for back-compat
        resolved_kind = kind or detect_entity_kind(entity_id)
        if resolved_kind is None:
            return IngestResult(
                kind=None,
                root=None,
                errors=[
                    f"Could not detect Open Targets entity kind for ID '{entity_id}'. "
                    "Pass an explicit kind (disease/target/drug/variant/study)."
                ],
            )

        try:
            if resolved_kind == EntityKind.DISEASE:
                root = await self._fetch_disease(entity_id, max_neighbors)
            elif resolved_kind == EntityKind.TARGET:
                root = await self._fetch_target(entity_id, max_neighbors)
            elif resolved_kind == EntityKind.DRUG:
                root = await self._fetch_drug(entity_id)
            elif resolved_kind == EntityKind.VARIANT:
                root = await self._fetch_variant(entity_id)
            elif resolved_kind == EntityKind.STUDY:
                root = await self._fetch_study(entity_id)
            else:  # pragma: no cover
                raise RuntimeError(f"Unhandled entity kind: {resolved_kind}")
        except Exception as exc:
            self.logger.error("Open Targets fetch error: %s", exc, exc_info=True)
            return IngestResult(kind=resolved_kind, root=None, errors=[str(exc)])

        if root is None:
            return IngestResult(
                kind=resolved_kind,
                root=None,
                errors=[
                    f"{resolved_kind.value.title()} '{entity_id}' not found in Open Targets"
                ],
            )

        return IngestResult(kind=resolved_kind, root=root)

    # Back-compat shim — older callers (CLI, tests) used fetch_disease()
    # which returned a (disease, associations, total) shaped result. We
    # now return the unified IngestResult; callers have been updated.
    async def fetch_disease(
        self,
        disease_id: str,
        max_associations: int = 500,
    ) -> IngestResult:
        """Deprecated alias kept for back-compat — prefer ``fetch_entity``."""
        return await self.fetch_entity(
            disease_id, kind=EntityKind.DISEASE, max_neighbors=max_associations
        )

    # ------------------------------------------------------------------
    # Per-kind fetchers
    # ------------------------------------------------------------------

    async def _fetch_disease(
        self, disease_id: str, size: int
    ) -> Optional[DiseaseRoot]:
        data = await self._query(
            _DISEASE_QUERY, {"id": disease_id, "size": size}
        )
        node = (data.get("data") or {}).get("disease")
        if node is None:
            return None

        synonyms: List[str] = []
        for syn_group in node.get("synonyms") or []:
            synonyms.extend(syn_group.get("terms") or [])

        assoc_block = node.get("associatedTargets") or {}
        associations = [
            TargetAssociation(
                target=_target_ref(row.get("target") or {}),
                score=row.get("score") or 0.0,
                datatype_scores={
                    d["id"]: d["score"]
                    for d in (row.get("datatypeScores") or [])
                    if d.get("id") is not None
                },
            )
            for row in assoc_block.get("rows") or []
        ]

        candidates = (node.get("drugAndClinicalCandidates") or {}).get("rows") or []
        known_drugs = [
            KnownDrug(
                drug=DrugRef(
                    drug_id=(row.get("drug") or {}).get("id") or "",
                    name=(row.get("drug") or {}).get("name") or "",
                ),
                clinical_stage=row.get("maxClinicalStage") or "",
            )
            for row in candidates
            if (row.get("drug") or {}).get("id")
        ]

        return DiseaseRoot(
            id=node["id"],
            name=node.get("name") or "",
            description=node.get("description") or "",
            therapeutic_areas=node.get("therapeuticAreas") or [],
            synonyms=synonyms,
            associated_targets=associations,
            total_associated_targets=assoc_block.get("count") or 0,
            known_drugs=known_drugs,
        )

    async def _fetch_target(
        self, target_id: str, size: int
    ) -> Optional[TargetRoot]:
        data = await self._query(
            _TARGET_QUERY, {"id": target_id, "size": size}
        )
        node = (data.get("data") or {}).get("target")
        if node is None:
            return None

        assoc_block = node.get("associatedDiseases") or {}
        associations = [
            DiseaseAssociation(
                disease=_disease_ref(row.get("disease") or {}),
                score=row.get("score") or 0.0,
                datatype_scores={
                    d["id"]: d["score"]
                    for d in (row.get("datatypeScores") or [])
                    if d.get("id") is not None
                },
            )
            for row in assoc_block.get("rows") or []
        ]

        candidate_rows = (node.get("drugAndClinicalCandidates") or {}).get("rows") or []
        known_drugs: List[KnownDrug] = []
        for row in candidate_rows:
            drug = row.get("drug") or {}
            if not drug.get("id"):
                continue
            stage = row.get("maxClinicalStage") or ""
            disease_list = row.get("diseases") or []
            if disease_list:
                # One KnownDrug per (drug, disease) pair so the use case
                # can materialize a drug→disease edge for each indication.
                # Each item is a ClinicalDiseaseListItem wrapping a Disease.
                for d_item in disease_list:
                    d = (d_item or {}).get("disease") or {}
                    if not d.get("id"):
                        continue
                    known_drugs.append(
                        KnownDrug(
                            drug=DrugRef(drug_id=drug["id"], name=drug.get("name") or ""),
                            clinical_stage=stage,
                            disease=DiseaseRef(
                                disease_id=d["id"],
                                name=d.get("name") or "",
                            ),
                        )
                    )
            else:
                known_drugs.append(
                    KnownDrug(
                        drug=DrugRef(drug_id=drug["id"], name=drug.get("name") or ""),
                        clinical_stage=stage,
                    )
                )

        pathways = [
            PathwayRef(
                pathway_id=p.get("pathwayId") or "",
                name=p.get("pathway") or "",
            )
            for p in node.get("pathways") or []
            if p.get("pathwayId")
        ]

        return TargetRoot(
            id=node["id"],
            symbol=node.get("approvedSymbol") or "",
            name=node.get("approvedName") or "",
            biotype=node.get("biotype") or "",
            function_descriptions=node.get("functionDescriptions") or [],
            pathways=pathways,
            associated_diseases=associations,
            total_associated_diseases=assoc_block.get("count") or 0,
            known_drugs=known_drugs,
        )

    async def _fetch_drug(self, drug_id: str) -> Optional[DrugRoot]:
        data = await self._query(_DRUG_QUERY, {"id": drug_id})
        node = (data.get("data") or {}).get("drug")
        if node is None:
            return None

        moas = [
            MechanismOfAction(
                description=row.get("mechanismOfAction") or "",
                action_type=row.get("actionType") or "",
                target_name=row.get("targetName") or "",
                targets=[_target_ref(t) for t in row.get("targets") or []],
            )
            for row in (node.get("mechanismsOfAction") or {}).get("rows") or []
        ]

        indications = [
            Indication(
                disease=_disease_ref(row.get("disease") or {}),
                clinical_stage=row.get("maxClinicalStage") or "",
            )
            for row in (node.get("indications") or {}).get("rows") or []
            if (row.get("disease") or {}).get("id")
        ]

        return DrugRoot(
            id=node["id"],
            name=node.get("name") or "",
            description=node.get("description") or "",
            synonyms=list(node.get("synonyms") or []),
            trade_names=list(node.get("tradeNames") or []),
            drug_type=node.get("drugType") or "",
            max_clinical_stage=node.get("maximumClinicalStage") or "",
            mechanisms_of_action=moas,
            indications=indications,
        )

    async def _fetch_variant(self, variant_id: str) -> Optional[VariantRoot]:
        data = await self._query(_VARIANT_QUERY, {"id": variant_id})
        node = (data.get("data") or {}).get("variant")
        if node is None:
            return None

        consequences = [
            VariantTranscriptConsequence(
                target=_target_ref(row.get("target") or {}),
                consequence_terms=[
                    c.get("label") or c.get("id") or ""
                    for c in row.get("variantConsequences") or []
                    if c
                ],
            )
            for row in node.get("transcriptConsequences") or []
            if (row.get("target") or {}).get("id")
        ]

        msc = node.get("mostSevereConsequence") or {}
        most_severe = (msc.get("label") or msc.get("id") or "") if isinstance(msc, dict) else ""

        return VariantRoot(
            id=node["id"],
            rs_ids=list(node.get("rsIds") or []),
            chromosome=str(node.get("chromosome") or ""),
            position=node.get("position"),
            reference_allele=node.get("referenceAllele") or "",
            alternate_allele=node.get("alternateAllele") or "",
            most_severe_consequence=most_severe,
            transcript_consequences=consequences,
        )

    async def _fetch_study(self, study_id: str) -> Optional[StudyRoot]:
        data = await self._query(_STUDY_QUERY, {"id": study_id})
        node = (data.get("data") or {}).get("study")
        if node is None:
            return None

        return StudyRoot(
            id=node.get("id") or study_id,
            trait=node.get("traitFromSource") or "",
            publication_date=node.get("publicationDate") or "",
            pubmed_id=str(node.get("pubmedId") or ""),
            first_author=node.get("publicationFirstAuthor") or "",
            n_samples=node.get("nSamples"),
            diseases=[_disease_ref(d) for d in node.get("diseases") or []],
        )

    # ------------------------------------------------------------------
    # GraphQL transport
    # ------------------------------------------------------------------

    async def _query(
        self, query: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError(
                "OpenTargetsClient must be used as an async context manager."
            )

        payload = {"query": query, "variables": variables}

        try:
            async with self._session.post(
                self._endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                # Open Targets returns GraphQL validation errors with HTTP 400
                # but a JSON body that contains an `errors` array. Read the
                # body first so we can surface the actual GraphQL message
                # rather than a generic "Bad Request".
                try:
                    result: Dict[str, Any] = await response.json()
                except (aiohttp.ContentTypeError, ValueError):
                    response.raise_for_status()
                    raise RuntimeError(
                        f"Open Targets returned non-JSON response (HTTP {response.status})"
                    )

                if isinstance(result, dict) and "errors" in result:
                    messages = [e.get("message", "unknown") for e in result["errors"]]
                    raise RuntimeError(f"GraphQL errors: {'; '.join(messages)}")

                if response.status >= 400:
                    raise RuntimeError(
                        f"Open Targets API HTTP {response.status}: {result}"
                    )

                return result

        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Open Targets API request failed: {exc}") from exc


# ── Module-level helpers ─────────────────────────────────────────────────


def _target_ref(node: Dict[str, Any]) -> TargetRef:
    return TargetRef(
        target_id=node.get("id") or "",
        symbol=node.get("approvedSymbol") or "",
        name=node.get("approvedName") or "",
        biotype=node.get("biotype") or "",
        function_descriptions=list(node.get("functionDescriptions") or []),
    )


def _disease_ref(node: Dict[str, Any]) -> DiseaseRef:
    return DiseaseRef(
        disease_id=node.get("id") or "",
        name=node.get("name") or "",
        description=node.get("description") or "",
        therapeutic_areas=list(node.get("therapeuticAreas") or []),
    )
