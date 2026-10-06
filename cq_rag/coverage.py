from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Iterable, Sequence

from .models import CQDAG, ConstraintRecord, CoverageReport, TripleRecord


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def labels_match(left: str, right: str) -> bool:
    left_normalized = normalize_label(left)
    right_normalized = normalize_label(right)
    return bool(
        left_normalized
        and right_normalized
        and (
            left_normalized == right_normalized
            or left_normalized in right_normalized
            or right_normalized in left_normalized
        )
    )


def compile_constraints(dag: CQDAG) -> tuple[ConstraintRecord, ...]:
    constraints: list[ConstraintRecord] = []
    output_variables = {normalize_label(node.output_variable) for node in dag.nodes if node.output_variable}
    for node in dag.nodes:
        for index, entity in enumerate(node.input_entities, start=1):
            is_variable_reference = normalize_label(entity) in output_variables
            constraints.append(ConstraintRecord(
                f"{dag.dag_id}:{node.node_id}:seed:{index}", node.node_id, "seed", entity,
                required=not is_variable_reference,
            ))
        constraints.append(ConstraintRecord(
            f"{dag.dag_id}:{node.node_id}:type", node.node_id, "type", node.answer_type,
            required=node.node_id == dag.terminal_node_id,
        ))
        for dependency in node.depends_on:
            constraints.append(ConstraintRecord(
                f"{dag.dag_id}:{node.node_id}:dependency:{dependency}",
                node.node_id, "dependency", dependency,
            ))
    for edge in dag.edges:
        for index, relation in enumerate(edge.relation_hints, start=1):
            constraints.append(ConstraintRecord(
                f"{dag.dag_id}:{edge.edge_id}:relation:{index}",
                edge.target_node_id, "relation", relation, False,
            ))
    terminal = next(node for node in dag.nodes if node.node_id == dag.terminal_node_id)
    constraints.append(ConstraintRecord(
        f"{dag.dag_id}:{terminal.node_id}:answer-slot",
        terminal.node_id, "answer_slot", terminal.answer_type, False,
    ))
    return tuple(constraints)


def covered_constraints(
    constraints: Sequence[ConstraintRecord], triples: Sequence[TripleRecord]
) -> dict[str, tuple[str, ...]]:
    covered: dict[str, list[str]] = defaultdict(list)
    connected = _connected_triple_ids(triples)
    for constraint in constraints:
        for triple in triples:
            matches = False
            if constraint.constraint_type == "seed":
                matches = labels_match(constraint.value, triple.subject) or labels_match(constraint.value, triple.object)
            elif constraint.constraint_type == "relation":
                matches = labels_match(constraint.value, triple.relation)
            elif constraint.constraint_type in {"type", "answer_slot"}:
                matches = constraint.value == "Entity" or labels_match(constraint.value, triple.subject_type) or labels_match(constraint.value, triple.object_type)
            elif constraint.constraint_type == "dependency":
                matches = triple.triple_id in connected
            if matches:
                covered[constraint.constraint_id].append(triple.triple_id)
    return {constraint_id: tuple(dict.fromkeys(triple_ids)) for constraint_id, triple_ids in covered.items()}


def evaluate_coverage(
    dag: CQDAG,
    constraints: Sequence[ConstraintRecord],
    triples: Sequence[TripleRecord],
) -> CoverageReport:
    coverage = covered_constraints(constraints, triples)
    required_ids = {item.constraint_id for item in constraints if item.required}
    covered_ids = tuple(item.constraint_id for item in constraints if item.constraint_id in coverage)
    missing_ids = tuple(item.constraint_id for item in constraints if item.required and item.constraint_id not in coverage)
    supporting_triples = tuple(dict.fromkeys(
        triple_id for constraint_id in covered_ids for triple_id in coverage[constraint_id]
    ))
    triple_by_id = {triple.triple_id: triple for triple in triples}
    contexts = tuple(dict.fromkeys(
        context_id for triple_id in supporting_triples
        for context_id in triple_by_id[triple_id].source_context_ids
    ))
    return CoverageReport(
        dag.dag_id,
        covered_ids,
        missing_ids,
        supporting_triples,
        contexts,
        not missing_ids and required_ids <= set(covered_ids) and _has_connected_path(triples),
    )


def _connected_triple_ids(triples: Sequence[TripleRecord]) -> set[str]:
    entity_to_triples: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        entity_to_triples[normalize_label(triple.subject)].add(triple.triple_id)
        entity_to_triples[normalize_label(triple.object)].add(triple.triple_id)
    return {
        triple_id
        for triple_ids in entity_to_triples.values() if len(triple_ids) > 1
        for triple_id in triple_ids
    }


def _has_connected_path(triples: Sequence[TripleRecord]) -> bool:
    if len(triples) <= 1:
        return bool(triples)
    adjacency: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for triple in triples:
        subject = normalize_label(triple.subject)
        object_ = normalize_label(triple.object)
        nodes.update((subject, object_))
        adjacency[subject].add(object_)
        adjacency[object_].add(subject)
    start = next(iter(nodes))
    reached = {start}
    pending = deque([start])
    while pending:
        current = pending.popleft()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    return reached == nodes


def contexts_for_triples(triples: Iterable[TripleRecord]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(context_id for triple in triples for context_id in triple.source_context_ids))