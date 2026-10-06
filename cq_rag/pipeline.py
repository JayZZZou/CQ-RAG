from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from .artifacts import ArtifactStore
from .config import RunConfig
from .coverage import compile_constraints
from .embeddings import EmbeddingClient, TextEmbeddings
from .evaluation import evaluate_answers
from .kg import extract_triples
from .llm import JsonChatClient
from .models import AnswerRecord, CQDAG, CoverageReport, EvidenceSet, RetrievalFailure, SampleRecord, TripleRecord
from .planning import PlannedDAG, plan_dags
from .progress import print_item_progress
from .qa import answer_question
from .reformulation import classify_failure, reformulate_dag, select_reformulations
from .reranking import rerank_evidence_sets
from .retrieval import build_evidence_candidates
from .runtime import runtime_sample


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class PipelineResult:
    answers: tuple[AnswerRecord, ...]
    evaluation: dict[str, object]
    dag_fallback_count: int
    reformulation_count: int
    triple_count: int


def run_pipeline(
    samples: Sequence[SampleRecord],
    config: RunConfig,
    store: ArtifactStore,
    chat_client: JsonChatClient,
    embedding_client: EmbeddingClient,
) -> PipelineResult:
    planned = plan_dags(samples, chat_client, config.chat_model)
    dags = [item.dag for item in planned]
    dataset_dir = Path(config.dataset)
    store.write_jsonl(dataset_dir / "dags.jsonl", planned)
    store.update_manifest({"status": "dag_planned", "stage": "dags", "dag_count": len(dags)})

    triples_by_sample = dict(_serial_items(
        "KG",
        list(zip(samples, dags, strict=True)),
        lambda item: _safe_extract_triples(item[0], item[1], chat_client, config.chat_model),
    ))
    triples = tuple(
        triple for sample in samples for triple in triples_by_sample.get(sample.sample_id, ())
    )
    store.write_jsonl(dataset_dir / "triples.jsonl", triples)
    store.update_manifest({"status": "triples_extracted", "stage": "triples", "triple_count": len(triples)})

    embeddings = TextEmbeddings(embedding_client)
    dag_by_sample = {dag.sample_id: dag for dag in dags}

    def retrieve(sample: SampleRecord) -> tuple[EvidenceSet, RetrievalFailure | None]:
        dag = dag_by_sample[sample.sample_id]
        constraints = compile_constraints(dag)
        sample_triples = triples_by_sample.get(sample.sample_id, ())
        candidates = build_evidence_candidates(
            runtime_sample(sample), dag, constraints, sample_triples, embeddings,
            config.embedding_model, config.embedding_dimensions,
            config.triple_top_k,
        )
        evidence = rerank_evidence_sets(dag, constraints, candidates, config.text_top_k)
        report = CoverageReport(
            dag.dag_id,
            evidence.covered_constraint_ids,
            evidence.missing_constraint_ids,
            evidence.triple_ids,
            evidence.context_ids,
            evidence.connected_path,
        )
        failure = classify_failure(sample.sample_id, dag, constraints, report)
        return evidence, failure

    retrieval_by_sample = dict(_serial_items(
        "RETRIEVAL", list(samples), retrieve,
    ))
    evidence_by_sample = {
        sample_id: result[0] for sample_id, result in retrieval_by_sample.items()
    }
    failures = [result[1] for result in retrieval_by_sample.values() if result[1] is not None]

    selected_failures = select_reformulations(failures, len(samples), config.max_reformulation_rate)
    failure_by_sample = {failure.sample_id: failure for failure in selected_failures}
    reformulated = dict(_serial_items(
        "PATCH",
        [sample for sample in samples if sample.sample_id in failure_by_sample],
        lambda sample: reformulate_dag(
            sample, dag_by_sample[sample.sample_id], failure_by_sample[sample.sample_id],
            chat_client, config.chat_model,
        ),
    ))
    successful_patch_count = 0
    for sample in samples:
        patch = reformulated.get(sample.sample_id)
        if not isinstance(patch, PlannedDAG) or patch.used_fallback:
            continue
        successful_patch_count += 1
        dag = patch.dag
        dag_by_sample[sample.sample_id] = dag
        constraints = compile_constraints(dag)
        candidates = build_evidence_candidates(
            runtime_sample(sample), dag, constraints, triples_by_sample.get(sample.sample_id, ()), embeddings,
            config.embedding_model, config.embedding_dimensions,
            config.triple_top_k,
        )
        evidence_by_sample[sample.sample_id] = rerank_evidence_sets(
            dag, constraints, candidates, config.text_top_k,
        )
    store.write_jsonl(dataset_dir / "evidence_sets.jsonl", (
        evidence_by_sample[sample.sample_id] for sample in samples
    ))
    store.write_jsonl(dataset_dir / "reformulations.jsonl", reformulated.values())
    store.update_manifest({
        "status": "evidence_selected",
        "stage": "retrieval",
        "retrieval_failure_count": len(failures),
        "reformulation_count": successful_patch_count,
    })

    answers_by_sample = dict(_serial_items(
        "QA",
        list(samples),
        lambda sample: answer_question(
            runtime_sample(sample), dag_by_sample[sample.sample_id], evidence_by_sample[sample.sample_id],
            triples_by_sample.get(sample.sample_id, ()), chat_client, config.chat_model,
        ),
    ))
    answers = tuple(answers_by_sample[sample.sample_id] for sample in samples)
    evaluation = evaluate_answers(answers, samples)
    store.write_jsonl(dataset_dir / "answers.jsonl", answers)
    store.write_json(dataset_dir / "evaluation.json", evaluation)
    store.update_manifest({
        "status": "completed",
        "stage": "evaluation",
        "answer_count": len(answers),
        "exact_match": evaluation["exact_match"],
        "token_f1": evaluation["token_f1"],
    })
    return PipelineResult(
        answers,
        evaluation,
        sum(item.used_fallback for item in planned),
        successful_patch_count,
        len(triples),
    )


def _serial_items(
    stage: str,
    items: Sequence[T],
    operation: Callable[[T], R],
) -> tuple[tuple[str, R], ...]:
    results: list[tuple[str, R]] = []
    for completed, item in enumerate(items, start=1):
        results.append((_item_id(item), operation(item)))
        print_item_progress(stage, completed, len(items))
    return tuple(results)


def _item_id(item: object) -> str:
    if isinstance(item, tuple) and item and isinstance(item[0], SampleRecord):
        return item[0].sample_id
    if isinstance(item, SampleRecord):
        return item.sample_id
    raise TypeError(f"Cannot determine item ID for {type(item).__name__}")


def _safe_extract_triples(
    sample: SampleRecord, dag: CQDAG, client: JsonChatClient, model: str,
) -> tuple[TripleRecord, ...]:
    try:
        return extract_triples(runtime_sample(sample), dag, client, model)
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Triple extraction failed for %s (%s); continuing with text evidence",
            sample.sample_id, type(error).__name__,
        )
        return ()