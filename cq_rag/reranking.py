from __future__ import annotations

from typing import Sequence

from .models import ConstraintRecord, CQDAG, EvidenceCandidate, EvidenceSet


def rerank_evidence_sets(
    dag: CQDAG,
    constraints: Sequence[ConstraintRecord],
    candidates: Sequence[EvidenceCandidate],
    text_top_k: int = 10,
) -> EvidenceSet:
    if text_top_k <= 0:
        raise ValueError("text_top_k must be positive")
    required = {item.constraint_id for item in constraints if item.required}
    remaining = list(candidates)
    selected: list[EvidenceCandidate] = []
    covered: set[str] = set()
    entities: set[str] = set()
    score = 0.0

    while remaining and len(selected) < text_top_k and not required <= covered:
        ranked = sorted(
            remaining,
            key=lambda item: (
                -_marginal_score(item, required, covered, entities),
                item.context_id,
            ),
        )
        candidate = ranked[0]
        if not _advances_primary(candidate, required, covered, entities):
            break
        selected.append(candidate)
        remaining.remove(candidate)
        marginal = _marginal_score(candidate, required, covered, entities)
        score += marginal
        covered.update(candidate.covered_constraint_ids)
        entities.update(label.casefold() for label in candidate.entity_labels)

    primary = tuple(item.context_id for item in selected)
    supplemental_candidates = sorted(
        remaining,
        key=lambda item: (-item.semantic_score, item.token_count, item.context_id),
    )
    supplemental = tuple(
        item.context_id for item in supplemental_candidates[: max(0, text_top_k - len(primary))]
    )
    selected_ids = set(primary) | set(supplemental)
    final_candidates = [item for item in candidates if item.context_id in selected_ids]
    final_covered = tuple(dict.fromkeys(
        constraint_id for item in final_candidates for constraint_id in item.covered_constraint_ids
    ))
    triple_ids = tuple(dict.fromkeys(
        triple_id for item in final_candidates for triple_id in item.triple_ids
    ))
    missing = tuple(item.constraint_id for item in constraints if item.required and item.constraint_id not in final_covered)
    return EvidenceSet(
        dag.dag_id,
        primary,
        supplemental,
        triple_ids,
        final_covered,
        missing,
        not missing and _selected_entities_connect(selected),
        score,
    )


def _marginal_score(
    candidate: EvidenceCandidate,
    required: set[str],
    covered: set[str],
    selected_entities: set[str],
) -> float:
    candidate_constraints = set(candidate.covered_constraint_ids)
    new_required = len((candidate_constraints & required) - covered)
    new_optional = len(candidate_constraints - required - covered)
    candidate_entities = {label.casefold() for label in candidate.entity_labels}
    connectivity = 0.4 if selected_entities & candidate_entities else 0.0
    redundancy = len(candidate_constraints & covered) * 0.15
    return 2.0 * new_required + 0.4 * new_optional + connectivity + candidate.semantic_score - redundancy


def _advances_primary(
    candidate: EvidenceCandidate,
    required: set[str],
    covered: set[str],
    selected_entities: set[str],
) -> bool:
    if (set(candidate.covered_constraint_ids) & required) - covered:
        return True
    candidate_entities = {label.casefold() for label in candidate.entity_labels}
    return bool(selected_entities and candidate_entities & selected_entities)


def _selected_entities_connect(candidates: Sequence[EvidenceCandidate]) -> bool:
    if len(candidates) <= 1:
        return bool(candidates)
    groups = [{label.casefold() for label in item.entity_labels} for item in candidates if item.triple_ids]
    if not groups:
        return False
    reached = set(groups[0])
    pending = groups[1:]
    while pending:
        for index, group in enumerate(pending):
            if reached & group:
                reached.update(group)
                pending.pop(index)
                break
        else:
            return False
    return True