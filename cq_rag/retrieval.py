from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .coverage import covered_constraints
from .embeddings import TextEmbeddings
from .models import CQDAG, ConstraintRecord, EvidenceCandidate, RuntimeSample, TripleRecord


def build_evidence_candidates(
    sample: RuntimeSample,
    dag: CQDAG,
    constraints: Sequence[ConstraintRecord],
    triples: Sequence[TripleRecord],
    embeddings: TextEmbeddings,
    model: str,
    dimensions: int,
    triple_top_k: int = 8,
) -> tuple[EvidenceCandidate, ...]:
    local_triples = [triple for triple in triples if triple.sample_id == sample.sample_id]
    query_texts = [sample.question, *(node.question for node in dag.nodes)]
    triple_texts = [f"{item.subject} {item.relation} {item.object}" for item in local_triples]
    context_texts = [f"{item.title}: {item.text}" for item in sample.contexts]
    query_vectors = embeddings.embed_texts(query_texts, model, dimensions)
    triple_vectors = embeddings.embed_texts(triple_texts, model, dimensions)
    context_vectors = embeddings.embed_texts(context_texts, model, dimensions)
    triple_scores = np.max(triple_vectors @ query_vectors.T, axis=1) if local_triples else np.empty(0)
    context_scores = np.max(context_vectors @ query_vectors.T, axis=1) if sample.contexts else np.empty(0)
    ranked_triples = sorted(
        zip(local_triples, triple_scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].triple_id),
    )[:triple_top_k]
    triples_by_context: dict[str, list[tuple[TripleRecord, float]]] = defaultdict(list)
    for triple, score in ranked_triples:
        for context_id in triple.source_context_ids:
            triples_by_context[context_id].append((triple, float(score)))
    candidates: list[EvidenceCandidate] = []
    for index, context in enumerate(sample.contexts):
        linked = sorted(triples_by_context.get(context.context_id, ()), key=lambda item: (-item[1], item[0].triple_id))
        context_triples = tuple(item[0] for item in linked)
        coverage = covered_constraints(constraints, context_triples)
        entities = tuple(dict.fromkeys(
            label for triple in context_triples for label in (triple.subject, triple.object)
        ))
        triple_score = max((item[1] for item in linked), default=0.0)
        semantic_score = 0.6 * float(context_scores[index]) + 0.4 * triple_score
        candidates.append(EvidenceCandidate(
            context.context_id,
            tuple(triple.triple_id for triple in context_triples),
            entities,
            tuple(coverage),
            semantic_score,
            len(context.text.split()),
        ))
    return tuple(sorted(candidates, key=lambda item: (-item.semantic_score, item.context_id)))