from __future__ import annotations

import json
from typing import Any, Sequence

from .llm import JsonChatClient
from .models import AnswerRecord, CQDAG, EvidenceSet, RuntimeSample, TripleRecord
from .runtime import context_payload


QA_SYSTEM_PROMPT = """Answer one HotpotQA question only from the supplied evidence.
Return one JSON object with answer, answer_status, cited_triple_ids, and cited_context_ids.
The answer must be the shortest benchmark-style span: no explanation, restatement, or alternatives.
For Boolean questions output exactly yes or no. Use answer_status=insufficient_evidence when unsupported.
Cite only supplied IDs. Do not use outside knowledge."""


def qa_prompt(
    sample: RuntimeSample,
    dag: CQDAG,
    evidence: EvidenceSet,
    triples: Sequence[TripleRecord],
) -> str:
    context_by_id = {context.context_id: context for context in sample.contexts}
    triple_by_id = {triple.triple_id: triple for triple in triples}
    terminal = next(node for node in dag.nodes if node.node_id == dag.terminal_node_id)
    payload = {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "answer_type": terminal.answer_type,
        "triples": [
            {
                "triple_id": triple_id,
                "subject": triple_by_id[triple_id].subject,
                "relation": triple_by_id[triple_id].relation,
                "object": triple_by_id[triple_id].object,
                "source_context_ids": triple_by_id[triple_id].source_context_ids,
            }
            for triple_id in evidence.triple_ids if triple_id in triple_by_id
        ],
        "contexts": [context_payload(context_by_id[context_id]) for context_id in evidence.context_ids if context_id in context_by_id],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def validate_answer_payload(payload: dict[str, Any], sample_id: str, evidence: EvidenceSet) -> AnswerRecord:
    answer = str(payload.get("answer", "")).strip()
    raw_status = str(payload.get("answer_status", "")).strip().casefold()
    status = {
        "success": "supported",
        "confirmed": "supported",
        "sufficient_evidence": "supported",
        "insufficient": "insufficient_evidence",
        "unsupported": "insufficient_evidence",
    }.get(raw_status, raw_status)
    if status not in {"supported", "insufficient_evidence", "failed"}:
        status = "insufficient_evidence"
    raw_triples = payload.get("cited_triple_ids") if isinstance(payload.get("cited_triple_ids"), list) else []
    raw_contexts = payload.get("cited_context_ids") if isinstance(payload.get("cited_context_ids"), list) else []
    cited_triples = tuple(dict.fromkeys(str(item) for item in raw_triples if str(item) in evidence.triple_ids))
    cited_contexts = tuple(dict.fromkeys(str(item) for item in raw_contexts if str(item) in evidence.context_ids))
    if answer and (cited_triples or cited_contexts):
        status = "supported"
    elif status == "supported":
        status = "insufficient_evidence"
    return AnswerRecord(sample_id, evidence.dag_id, answer, status, cited_triples, cited_contexts)


def answer_question(
    sample: RuntimeSample,
    dag: CQDAG,
    evidence: EvidenceSet,
    triples: Sequence[TripleRecord],
    client: JsonChatClient,
    model: str,
) -> AnswerRecord:
    try:
        payload = client.complete_json(model, QA_SYSTEM_PROMPT, qa_prompt(sample, dag, evidence, triples))
        return validate_answer_payload(payload, sample.sample_id, evidence)
    except Exception as error:
        return AnswerRecord(sample.sample_id, dag.dag_id, "", "failed", error=type(error).__name__)