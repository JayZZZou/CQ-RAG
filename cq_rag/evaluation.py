from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .models import AnswerRecord, SampleRecord


@dataclass(frozen=True)
class AnswerScore:
    exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float


def normalize_answer(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"(?<=\d)[,.](?=\d)", "", text.lower())
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_answer(prediction: str, reference: str) -> AnswerScore:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    exact = float(predicted == expected)
    if not predicted or not expected:
        value = float(predicted == expected)
        return AnswerScore(exact, value, value, value)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    token_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return AnswerScore(exact, precision, recall, token_f1)


def best_answer_score(prediction: str, references: Iterable[str]) -> AnswerScore:
    scores = [score_answer(prediction, reference) for reference in references]
    if not scores:
        raise ValueError("At least one reference answer is required")
    return AnswerScore(*(max(getattr(score, field) for score in scores) for field in AnswerScore.__dataclass_fields__))


def evaluate_answers(answers: Sequence[AnswerRecord], samples: Sequence[SampleRecord]) -> dict[str, object]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    rows: list[dict[str, object]] = []
    for answer in answers:
        sample = sample_by_id[answer.sample_id]
        score = best_answer_score(answer.answer, sample.standard_answers)
        rows.append({"sample_id": answer.sample_id, "answer_status": answer.answer_status, **asdict(score)})
    count = len(rows)
    return {
        "sample_count": count,
        "exact_match": sum(float(row["exact_match"]) for row in rows) / count if count else 0.0,
        "token_f1": sum(float(row["token_f1"]) for row in rows) / count if count else 0.0,
        "per_answer": rows,
    }