from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import SCHEMA_VERSION


DatasetName = Literal["hotpotqa", "musique"]


@dataclass(frozen=True)
class ContextRecord:
    context_id: str
    title: str
    text: str
    source_index: int


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    context_id: str
    title: str
    text: str
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    source_id: str
    question: str
    standard_answers: tuple[str, ...]
    contexts: tuple[ContextRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class RuntimeSample:
    sample_id: str
    question: str
    contexts: tuple[ContextRecord, ...]


@dataclass(frozen=True)
class CQNode:
    node_id: str
    question: str
    answer_type: str
    depends_on: tuple[str, ...] = ()
    input_entities: tuple[str, ...] = ()
    output_variable: str = ""


@dataclass(frozen=True)
class CQEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_hints: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class CQDAG:
    dag_id: str
    sample_id: str
    root_question: str
    terminal_node_id: str
    nodes: tuple[CQNode, ...]
    edges: tuple[CQEdge, ...]
    generation: int = 0
    parent_dag_id: str | None = None
    schema_version: str = SCHEMA_VERSION


ConstraintType = Literal["seed", "relation", "type", "dependency", "answer_slot"]


@dataclass(frozen=True)
class ConstraintRecord:
    constraint_id: str
    node_id: str
    constraint_type: ConstraintType
    value: str
    required: bool = True


@dataclass(frozen=True)
class CoverageReport:
    dag_id: str
    covered_constraint_ids: tuple[str, ...]
    missing_constraint_ids: tuple[str, ...]
    supporting_triple_ids: tuple[str, ...]
    supporting_context_ids: tuple[str, ...]
    connected_path: bool

    @property
    def coverage(self) -> float:
        total = len(self.covered_constraint_ids) + len(self.missing_constraint_ids)
        return len(self.covered_constraint_ids) / total if total else 1.0


@dataclass(frozen=True)
class TripleRecord:
    triple_id: str
    sample_id: str
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    source_context_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCandidate:
    context_id: str
    triple_ids: tuple[str, ...]
    entity_labels: tuple[str, ...]
    covered_constraint_ids: tuple[str, ...]
    semantic_score: float
    token_count: int = 0


@dataclass(frozen=True)
class EvidenceSet:
    dag_id: str
    primary_context_ids: tuple[str, ...]
    supplemental_context_ids: tuple[str, ...]
    triple_ids: tuple[str, ...]
    covered_constraint_ids: tuple[str, ...]
    missing_constraint_ids: tuple[str, ...]
    connected_path: bool
    score: float

    @property
    def context_ids(self) -> tuple[str, ...]:
        return self.primary_context_ids + self.supplemental_context_ids


FailureType = Literal[
    "missing_seed", "missing_relation", "missing_answer_type", "disconnected_path", "insufficient_evidence",
]


@dataclass(frozen=True)
class RetrievalFailure:
    sample_id: str
    dag_id: str
    failure_types: tuple[FailureType, ...]
    missing_constraint_ids: tuple[str, ...]
    coverage: float
    connected_path: bool
    priority: float


@dataclass(frozen=True)
class AnswerRecord:
    sample_id: str
    dag_id: str
    answer: str
    answer_status: Literal["supported", "insufficient_evidence", "failed"]
    cited_triple_ids: tuple[str, ...] = ()
    cited_context_ids: tuple[str, ...] = ()
    error: str | None = None