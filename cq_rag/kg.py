from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .llm import JsonChatClient
from .models import CQDAG, RuntimeSample, TripleRecord
from .runtime import runtime_payload


TRIPLE_SYSTEM_PROMPT = """Extract a high-recall set of explicit facts from one HotpotQA sample.
Return one JSON object: {"triples": [{"subject": ..., "subject_type": ..., "relation": ...,
"object": ..., "object_type": ..., "source_context_ids": [...]}]}.
Cover facts needed by every CQ DAG node and facts that connect evidence across hops.
Keep distinct facts separate. Use concrete entities or values, not placeholder variables.
Use only the supplied contexts, do not answer the question, and cite only exact supplied context IDs."""

MUSIQUE_TRIPLE_SYSTEM_PROMPT = TRIPLE_SYSTEM_PROMPT.replace("HotpotQA", "MuSiQue")


def triple_prompt(sample: RuntimeSample, dag: CQDAG) -> str:
    payload = runtime_payload(sample)
    payload["cq_nodes"] = [
        {
            "node_id": node.node_id,
            "question": node.question,
            "answer_type": node.answer_type,
            "input_entities": node.input_entities,
        }
        for node in dag.nodes
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def normalize_triple_payload(
    payload: dict[str, Any], sample: RuntimeSample,
) -> tuple[TripleRecord, ...]:
    valid_context_ids = {context.context_id for context in sample.contexts}
    raw_triples = payload.get("triples") if isinstance(payload.get("triples"), list) else []
    triples: list[TripleRecord] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for item in raw_triples:
        if not isinstance(item, dict):
            continue
        subject = _text(item.get("subject"))
        relation = _text(item.get("relation"))
        object_ = _text(item.get("object"))
        source_ids = tuple(dict.fromkeys(
            _text(value) for value in item.get("source_context_ids", [])
            if _text(value) in valid_context_ids
        )) if isinstance(item.get("source_context_ids"), list) else ()
        if not subject or not relation or not object_ or not source_ids:
            continue
        key = (_normalized(subject), _normalized(relation), _normalized(object_), source_ids)
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256(
            json.dumps((sample.sample_id, *key), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        triples.append(TripleRecord(
            f"{sample.sample_id}:T-{digest}",
            sample.sample_id,
            subject,
            _text(item.get("subject_type")) or "Entity",
            relation,
            object_,
            _text(item.get("object_type")) or "Entity",
            source_ids,
        ))
    return tuple(triples)


def extract_triples(
    sample: RuntimeSample, dag: CQDAG, client: JsonChatClient, model: str,
) -> tuple[TripleRecord, ...]:
    system_prompt = MUSIQUE_TRIPLE_SYSTEM_PROMPT if sample.sample_id.startswith("musique:") else TRIPLE_SYSTEM_PROMPT
    payload = client.complete_json(model, system_prompt, triple_prompt(sample, dag))
    return normalize_triple_payload(payload, sample)


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()