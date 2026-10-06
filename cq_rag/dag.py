from __future__ import annotations

import re
from collections import deque
from typing import Any, Iterable

from .models import CQDAG, CQEdge, CQNode


class DAGContractError(ValueError):
    pass


ANSWER_TYPES = {"Person", "Location", "Date", "DateRange", "Boolean", "Number", "Work", "Organization", "Entity"}


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(text for item in value if (text := _text(item))))


def normalize_dag_payload(payload: dict[str, Any], sample_id: str, question: str) -> CQDAG:
    raw_nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    nodes: list[CQNode] = []
    for index, item in enumerate(raw_nodes[:4], start=1):
        if not isinstance(item, dict):
            continue
        node_id = _text(item.get("node_id")) or f"N{index}"
        node_question = _text(item.get("question"))
        if not node_question:
            continue
        answer_type = _text(item.get("answer_type"))
        nodes.append(CQNode(
            node_id=node_id,
            question=node_question,
            answer_type=answer_type if answer_type in ANSWER_TYPES else "Entity",
            depends_on=_strings(item.get("depends_on")),
            input_entities=_strings(item.get("input_entities")),
            output_variable=_text(item.get("output_variable")) or f"v{index}",
        ))
    if not nodes:
        return fallback_dag(sample_id, question)

    raw_edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
    edges: list[CQEdge] = []
    for index, item in enumerate(raw_edges, start=1):
        if not isinstance(item, dict):
            continue
        source = _text(item.get("source_node_id"))
        target = _text(item.get("target_node_id"))
        if source and target:
            edges.append(CQEdge(
                edge_id=_text(item.get("edge_id")) or f"E{index}",
                source_node_id=source,
                target_node_id=target,
                relation_hints=_strings(item.get("relation_hints")),
                required=item.get("required") is not False,
            ))
    terminal = _text(payload.get("terminal_node_id")) or nodes[-1].node_id
    dag = CQDAG(
        dag_id=f"{sample_id}:DAG-0",
        sample_id=sample_id,
        root_question=question,
        terminal_node_id=terminal,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )
    validate_dag(dag)
    return dag


def fallback_dag(sample_id: str, question: str) -> CQDAG:
    node = CQNode("N1", question, "Entity", output_variable="answer")
    return CQDAG(f"{sample_id}:DAG-0", sample_id, question, node.node_id, (node,), ())


def topological_order(dag: CQDAG) -> tuple[str, ...]:
    node_ids = {node.node_id for node in dag.nodes}
    incoming = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    dependencies = {(dependency, node.node_id) for node in dag.nodes for dependency in node.depends_on}
    dependencies.update((edge.source_node_id, edge.target_node_id) for edge in dag.edges)
    for source, target in dependencies:
        if source not in node_ids or target not in node_ids:
            raise DAGContractError(f"Unknown DAG dependency: {source!r} -> {target!r}")
        if source == target:
            raise DAGContractError(f"Self dependency is not allowed: {source!r}")
        outgoing[source].append(target)
        incoming[target] += 1
    ready = deque(sorted(node_id for node_id, count in incoming.items() if count == 0))
    ordered: list[str] = []
    while ready:
        node_id = ready.popleft()
        ordered.append(node_id)
        for target in sorted(outgoing[node_id]):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if len(ordered) != len(node_ids):
        raise DAGContractError("CQ DAG contains a cycle")
    return tuple(ordered)


def validate_dag(dag: CQDAG) -> None:
    if not 1 <= len(dag.nodes) <= 4:
        raise DAGContractError("CQ DAG must contain between 1 and 4 nodes")
    node_ids = [node.node_id for node in dag.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise DAGContractError("CQ DAG node IDs must be unique")
    if dag.terminal_node_id not in node_ids:
        raise DAGContractError("CQ DAG terminal node does not exist")
    edge_ids = [edge.edge_id for edge in dag.edges]
    if len(edge_ids) != len(set(edge_ids)):
        raise DAGContractError("CQ DAG edge IDs must be unique")
    order = topological_order(dag)
    reachable = _ancestors(dag.terminal_node_id, dag)
    if set(order) - reachable:
        raise DAGContractError("CQ DAG contains nodes that cannot reach the terminal node")
    variables = [node.output_variable for node in dag.nodes if node.output_variable]
    if len(variables) != len(set(variables)):
        raise DAGContractError("CQ DAG output variables must be unique")


def _ancestors(node_id: str, dag: CQDAG) -> set[str]:
    reverse: dict[str, set[str]] = {node.node_id: set(node.depends_on) for node in dag.nodes}
    for edge in dag.edges:
        reverse[edge.target_node_id].add(edge.source_node_id)
    reached: set[str] = set()
    pending = [node_id]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(reverse[current])
    return reached


def iter_nodes(dag: CQDAG) -> Iterable[CQNode]:
    by_id = {node.node_id: node for node in dag.nodes}
    return (by_id[node_id] for node_id in topological_order(dag))