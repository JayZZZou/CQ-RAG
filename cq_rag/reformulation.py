from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from typing import Sequence

from .dag import normalize_dag_payload
from .llm import JsonChatClient
from .models import CQDAG, ConstraintRecord, CoverageReport, FailureType, RetrievalFailure, SampleRecord
from .planning import PlannedDAG


REFORMULATION_SYSTEM_PROMPT = """Patch one CQ DAG after a retrieval failure.
Return a complete replacement DAG as one JSON object with terminal_node_id, nodes, and edges.
Use 1-4 nodes and preserve useful nodes. Change only what the structured failure report justifies.
Every non-terminal node must lead to the terminal node. The terminal must depend on every required branch.
Do not answer the question, invent evidence, or add unsupported entities."""


def classify_failure(
    sample_id: str,
    dag: CQDAG,
    constraints: Sequence[ConstraintRecord],
    report: CoverageReport,
) -> RetrievalFailure | None:
    if not report.missing_constraint_ids and report.connected_path:
        return None
    constraint_by_id = {item.constraint_id: item for item in constraints}
    missing_types = {
        constraint_by_id[constraint_id].constraint_type
        for constraint_id in report.missing_constraint_ids
        if constraint_id in constraint_by_id
    }
    failures: list[FailureType] = []
    if "seed" in missing_types:
        failures.append("missing_seed")
    if "relation" in missing_types or "dependency" in missing_types:
        failures.append("missing_relation")
    if "type" in missing_types or "answer_slot" in missing_types:
        failures.append("missing_answer_type")
    if not report.connected_path and report.supporting_triple_ids:
        failures.append("disconnected_path")
    if not report.supporting_triple_ids:
        failures.append("insufficient_evidence")
    coverage = report.coverage
    priority = (1.0 - coverage) + (0.5 if not report.connected_path else 0.0)
    return RetrievalFailure(
        sample_id,
        dag.dag_id,
        tuple(dict.fromkeys(failures)),
        report.missing_constraint_ids,
        coverage,
        report.connected_path,
        priority,
    )


def select_reformulations(
    failures: Sequence[RetrievalFailure], total_samples: int, max_rate: float,
) -> tuple[RetrievalFailure, ...]:
    if total_samples <= 0 or not failures or max_rate <= 0:
        return ()
    budget = min(len(failures), math.floor(total_samples * max_rate))
    return tuple(sorted(failures, key=lambda item: (-item.priority, item.sample_id))[:budget])


def reformulation_prompt(sample: SampleRecord, dag: CQDAG, failure: RetrievalFailure) -> str:
    dag_payload = {
        "terminal_node_id": dag.terminal_node_id,
        "nodes": [asdict(node) for node in dag.nodes],
        "edges": [asdict(edge) for edge in dag.edges],
    }
    failure_payload = {
        "failure_types": failure.failure_types,
        "coverage": failure.coverage,
        "connected_path": failure.connected_path,
    }
    return (
        f"sample_id: {sample.sample_id}\n"
        f"question: {sample.question}\n"
        f"current_dag: {json.dumps(dag_payload, ensure_ascii=False, sort_keys=True)}\n"
        f"retrieval_failure: {json.dumps(failure_payload, ensure_ascii=False, sort_keys=True)}"
    )


def reformulate_dag(
    sample: SampleRecord,
    dag: CQDAG,
    failure: RetrievalFailure,
    client: JsonChatClient,
    model: str,
) -> PlannedDAG:
    if dag.generation >= 1:
        return PlannedDAG(dag, True, "ReformulationLimit")
    try:
        payload = client.complete_json(model, REFORMULATION_SYSTEM_PROMPT, reformulation_prompt(sample, dag, failure))
        patched = normalize_dag_payload(payload, sample.sample_id, sample.question)
        patched = replace(patched, dag_id=f"{sample.sample_id}:DAG-1", generation=1, parent_dag_id=dag.dag_id)
        return PlannedDAG(patched, False)
    except Exception as error:
        return PlannedDAG(dag, True, type(error).__name__)