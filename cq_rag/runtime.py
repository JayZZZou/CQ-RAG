from __future__ import annotations

from typing import Any

from .models import ContextRecord, RuntimeSample, SampleRecord


def runtime_sample(sample: SampleRecord) -> RuntimeSample:
    return RuntimeSample(sample.sample_id, sample.question, sample.contexts)


def context_payload(context: ContextRecord) -> dict[str, str]:
    return {"context_id": context.context_id, "title": context.title, "text": context.text}


def runtime_payload(sample: RuntimeSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "contexts": [context_payload(context) for context in sample.contexts],
    }