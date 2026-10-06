from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator

from .models import ContextRecord, DatasetName, EvidenceRecord, SampleRecord


DATASET_ALIASES = {
    "hotopotqa": "hotpotqa",
    "musicqa": "musique",
    "musciqa": "musique",
}


class DatasetContractError(ValueError):
    pass


def normalize_dataset_name(value: str) -> DatasetName:
    name = DATASET_ALIASES.get(value.casefold(), value.casefold())
    if name == "hotpotqa":
        return "hotpotqa"
    if name == "musique":
        return "musique"
    raise DatasetContractError(f"Unsupported dataset {value!r}; use hotpotqa or musique")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_sample_id(dataset: DatasetName, source_id: str, record: dict[str, Any]) -> str:
    identity = source_id or json.dumps(record, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{dataset}:{digest}"


def _iter_array_values(stream: Any, initial: str, chunk_size: int, path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = initial
    position = 0
    while True:
        while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
            position += 1
        if position < len(buffer) and buffer[position] == "]":
            return
        try:
            value, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise DatasetContractError(f"Incomplete JSON array in {path}")
            buffer = buffer[position:] + chunk
            position = 0
            continue
        if not isinstance(value, dict):
            raise DatasetContractError(f"Every dataset item must be an object in {path}")
        yield value
        position = end


def iter_json_records(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    with path.open("r", encoding="utf-8") as stream:
        prefix = ""
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise DatasetContractError(f"Could not find a record array in {path}")
            prefix += chunk
            stripped = prefix.lstrip()
            if stripped.startswith("["):
                yield from _iter_array_values(stream, stripped[1:], chunk_size, path)
                return
            match = re.search(r'"data"\s*:\s*\[', stripped)
            if match:
                yield from _iter_array_values(stream, stripped[match.end():], chunk_size, path)
                return
            if stripped and not stripped.startswith("{"):
                raise DatasetContractError(f"Expected a JSON array or data wrapper in {path}")
            if len(prefix) > 1024 * 1024:
                raise DatasetContractError(f"Could not find top-level data array in {path}")


def normalize_hotpot(record: dict[str, Any], ordinal: int) -> SampleRecord:
    source_id = normalize_text(record.get("_id")) or str(ordinal)
    sample_id = stable_sample_id("hotpotqa", source_id, record)
    question = normalize_text(record.get("question"))
    answer = normalize_text(record.get("answer"))
    raw_contexts = record.get("context")
    raw_facts = record.get("supporting_facts")
    if not question or not answer or not isinstance(raw_contexts, list) or not isinstance(raw_facts, list):
        raise DatasetContractError(f"Incomplete HotpotQA record {source_id}")
    supporting: dict[str, set[int]] = {}
    for fact in raw_facts:
        if not isinstance(fact, list) or len(fact) != 2:
            raise DatasetContractError(f"Invalid supporting fact in {source_id}: {fact!r}")
        supporting.setdefault(normalize_text(fact[0]), set()).add(int(fact[1]))
    contexts: list[ContextRecord] = []
    evidence: list[EvidenceRecord] = []
    for context_index, item in enumerate(raw_contexts):
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], list):
            raise DatasetContractError(f"Invalid context in {source_id}: {item!r}")
        title = normalize_text(item[0])
        sentences = tuple(normalize_text(sentence) for sentence in item[1])
        indices = tuple(sorted(supporting.get(title, set())))
        if any(index < 0 or index >= len(sentences) for index in indices):
            raise DatasetContractError(f"Supporting sentence index out of range in {source_id}/{title}")
        context_id = f"{sample_id}:C-{context_index + 1:04d}"
        contexts.append(ContextRecord(context_id, title, " ".join(filter(None, sentences)), context_index))
        if indices:
            evidence.append(EvidenceRecord(
                f"{sample_id}:E-{len(evidence) + 1:04d}", context_id, title,
                " ".join(sentences[index] for index in indices), indices,
            ))
    if not evidence:
        raise DatasetContractError(f"HotpotQA record {source_id} has no resolved evidence")
    return SampleRecord(sample_id, source_id, question, (answer,), tuple(contexts), tuple(evidence))


def _answers(primary: Any, aliases: Any = None) -> tuple[str, ...]:
    values = [normalize_text(primary)]
    if isinstance(aliases, list):
        values.extend(normalize_text(alias) for alias in aliases)
    return tuple(dict.fromkeys(value for value in values if value))


def normalize_musique(record: dict[str, Any], ordinal: int) -> SampleRecord:
    source_id = normalize_text(record.get("id")) or str(ordinal)
    sample_id = stable_sample_id("musique", source_id, record)
    question = normalize_text(record.get("question"))
    answers = _answers(record.get("answer"), record.get("answer_aliases"))
    paragraphs = record.get("paragraphs")
    if record.get("answerable") is False:
        raise DatasetContractError(f"QA evaluation requires an answerable MuSiQue record: {source_id}")
    if not question or not answers or not isinstance(paragraphs, list):
        raise DatasetContractError(f"Incomplete MuSiQue record {source_id}")
    contexts: list[ContextRecord] = []
    evidence: list[EvidenceRecord] = []
    for fallback_index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            raise DatasetContractError(f"Invalid MuSiQue paragraph in {source_id}")
        source_index = int(paragraph.get("idx", fallback_index))
        title = normalize_text(paragraph.get("title"))
        text = normalize_text(paragraph.get("paragraph_text"))
        context_id = f"{sample_id}:C-{fallback_index + 1:04d}"
        contexts.append(ContextRecord(context_id, title, text, source_index))
        if paragraph.get("is_supporting") is True:
            evidence.append(EvidenceRecord(
                f"{sample_id}:E-{len(evidence) + 1:04d}", context_id, title, text, (source_index,),
            ))
    if not evidence:
        raise DatasetContractError(f"MuSiQue record {source_id} has no supporting paragraphs")
    return SampleRecord(sample_id, source_id, question, answers, tuple(contexts), tuple(evidence))


NORMALIZERS: dict[DatasetName, Callable[[dict[str, Any], int], SampleRecord]] = {
    "hotpotqa": normalize_hotpot,
    "musique": normalize_musique,
}


def iter_samples(path: Path, dataset: str = "hotpotqa") -> Iterator[SampleRecord]:
    canonical = normalize_dataset_name(dataset)
    for ordinal, record in enumerate(iter_json_records(path), start=1):
        yield NORMALIZERS[canonical](record, ordinal)