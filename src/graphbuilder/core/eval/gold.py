"""Gold-set loader for the RAG eval harness.

The gold set lives in ``tests/eval/rag_gold.yaml``. Each entry is one
curated question/expected-source triple (§9.1 of docs/RAG_QA_PLAN.md):

```yaml
- id: q001
  question: "What kinases does imatinib inhibit?"
  intent: relational
  gold_entity_ids:        [ent_imatinib, ent_bcr_abl, ent_kit]
  gold_relationship_ids:  [rel_imatinib_bcr_abl, rel_imatinib_kit]
  gold_chunk_ids:         [chunk_42]
  gold_answer_substrings: ["BCR-ABL"]   # any-of: at least one must appear
  notes: "from curator review #c123"
```

The format is a top-level list of mappings. We support both YAML and
JSON; the ``.json`` extension switches loaders. Validation is strict —
unknown keys become a load error so typos can't silently disable a
question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional


# Intents are advisory — they let the report group results, but the
# harness does not branch on them. We accept any string here; the
# common values are documented in the gold-set README.
KNOWN_INTENTS = {"lookup", "relational", "multi_hop", "definitional", "out_of_graph"}

_VALID_KEYS = {
    "id",
    "question",
    "intent",
    "gold_entity_ids",
    "gold_relationship_ids",
    "gold_chunk_ids",
    "gold_answer_substrings",
    "notes",
}


@dataclass(frozen=True)
class GoldQuestion:
    """One curated question with the sources we expect retrieval to surface.

    All ``gold_*_ids`` fields are independent — a question may pin only
    entities, only relationships, only chunks, or any combination. The
    ``gold_answer_substrings`` list is "any-of": the answer passes the
    coverage check if at least one substring (case-insensitive) appears.
    """

    id: str
    question: str
    intent: Optional[str] = None
    gold_entity_ids: List[str] = field(default_factory=list)
    gold_relationship_ids: List[str] = field(default_factory=list)
    gold_chunk_ids: List[str] = field(default_factory=list)
    gold_answer_substrings: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def all_gold_source_ids(self) -> set[str]:
        """Composite-id form used by retrieval metrics — matches the
        orchestrator's ``"<kind>:<id>"`` keying so we can compare apples
        to apples without double-counting an entity that also appears
        as a chunk reference."""
        out: set[str] = set()
        for eid in self.gold_entity_ids:
            out.add(f"entity:{eid}")
        for rid in self.gold_relationship_ids:
            out.add(f"relationship:{rid}")
        for cid in self.gold_chunk_ids:
            out.add(f"chunk:{cid}")
        return out


def load_gold(path: str | Path) -> List[GoldQuestion]:
    """Load and validate a gold set from YAML or JSON.

    Raises ``ValueError`` on any structural problem (bad type, missing
    required field, duplicate ids, unknown key) so misformatted
    additions are caught at import time, not at metric time.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        # PyYAML is already an implicit dependency via
        # ``infrastructure/config/settings.py``; importing here keeps the
        # eval package self-contained.
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(raw)

    if not isinstance(data, list):
        raise ValueError(
            f"gold set must be a top-level list of questions, got {type(data).__name__}"
        )

    seen_ids: set[str] = set()
    out: List[GoldQuestion] = []
    for i, raw_item in enumerate(data):
        if not isinstance(raw_item, dict):
            raise ValueError(f"gold[{i}] must be a mapping, got {type(raw_item).__name__}")

        unknown = set(raw_item.keys()) - _VALID_KEYS
        if unknown:
            raise ValueError(
                f"gold[{i}] has unknown keys {sorted(unknown)}; "
                f"valid keys are {sorted(_VALID_KEYS)}"
            )

        qid = raw_item.get("id")
        question = raw_item.get("question")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"gold[{i}] missing or empty 'id'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"gold[{qid}] missing or empty 'question'")
        if qid in seen_ids:
            raise ValueError(f"gold set has duplicate id {qid!r}")
        seen_ids.add(qid)

        out.append(
            GoldQuestion(
                id=qid,
                question=question,
                intent=_optional_str(raw_item, "intent", qid),
                gold_entity_ids=_str_list(raw_item, "gold_entity_ids", qid),
                gold_relationship_ids=_str_list(raw_item, "gold_relationship_ids", qid),
                gold_chunk_ids=_str_list(raw_item, "gold_chunk_ids", qid),
                gold_answer_substrings=_str_list(raw_item, "gold_answer_substrings", qid),
                notes=_optional_str(raw_item, "notes", qid),
            )
        )

    if not out:
        raise ValueError(f"gold set at {p} is empty")
    return out


def _optional_str(raw: dict, key: str, qid: str) -> Optional[str]:
    val = raw.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ValueError(f"gold[{qid}].{key} must be a string, got {type(val).__name__}")
    return val


def _str_list(raw: dict, key: str, qid: str) -> List[str]:
    val = raw.get(key, [])
    if val is None:
        return []
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise ValueError(f"gold[{qid}].{key} must be a list of strings")
    return list(val)


__all__ = ["GoldQuestion", "load_gold", "KNOWN_INTENTS"]
