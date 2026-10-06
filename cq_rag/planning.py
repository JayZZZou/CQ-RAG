from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .dag import fallback_dag, normalize_dag_payload
from .llm import JsonChatClient
from .models import CQDAG, SampleRecord
from .progress import print_item_progress


DAG_SYSTEM_PROMPT = """You decompose one HotpotQA question into a minimal executable CQ DAG.
Return one JSON object only. Use 1-4 nodes. Every non-terminal node must lead to the terminal node.
Node fields: node_id, question, answer_type, depends_on, input_entities, output_variable.
Edge fields: edge_id, source_node_id, target_node_id, relation_hints, required.
Allowed answer_type values: Person, Location, Date, DateRange, Boolean, Number, Work, Organization, Entity.
Do not answer the question. Do not invent evidence. No node may merely repeat the root question.
For a bridge question, identify the bridge entity first, then resolve the requested property in the terminal node.
For a comparison or conjunction, create one evidence node per subject plus a terminal node that compares or combines them.
The terminal node must depend on every branch needed for the final answer; encode the same dependencies as edges."""

MUSIQUE_DAG_SYSTEM_PROMPT = DAG_SYSTEM_PROMPT.replace("HotpotQA", "MuSiQue")


@dataclass(frozen=True)
class PlannedDAG:
    dag: CQDAG
    used_fallback: bool
    error_type: str | None = None


def dag_prompt(sample: SampleRecord) -> str:
    return (
        "Create the CQ DAG for this question.\n"
        f"sample_id: {sample.sample_id}\n"
        f"question: {sample.question}\n"
        "Return keys: terminal_node_id, nodes, edges."
    )


def plan_dag(sample: SampleRecord, client: JsonChatClient, model: str) -> PlannedDAG:
    try:
        system_prompt = MUSIQUE_DAG_SYSTEM_PROMPT if sample.sample_id.startswith("musique:") else DAG_SYSTEM_PROMPT
        payload = client.complete_json(model, system_prompt, dag_prompt(sample))
        return PlannedDAG(normalize_dag_payload(payload, sample.sample_id, sample.question), False)
    except Exception as error:
        return PlannedDAG(fallback_dag(sample.sample_id, sample.question), True, type(error).__name__)


def plan_dags(
    samples: Sequence[SampleRecord],
    client: JsonChatClient,
    model: str,
) -> tuple[PlannedDAG, ...]:
    results: list[PlannedDAG] = []
    for completed, sample in enumerate(samples, start=1):
        results.append(plan_dag(sample, client, model))
        print_item_progress("DAG", completed, len(samples))
    return tuple(results)