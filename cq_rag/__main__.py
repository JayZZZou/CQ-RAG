from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .artifacts import ArtifactStore
from .config import RunConfig
from .datasets import iter_samples, normalize_dataset_name
from .llm import create_openai_client
from .pipeline import run_pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIR = ROOT_DIR / "datasets"
DEFAULT_OUTPUT = ROOT_DIR / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CQ-RAG: serial evaluation of ALL dataset records")
    parser.add_argument("--dataset", default="hotpotqa", help="hotpotqa or musique")
    parser.add_argument("--dataset-path", type=Path, help="Input JSON file; ALL records are used")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id")
    parser.add_argument("--text-top-k", type=int, default=10)
    parser.add_argument("--triple-top-k", type=int, default=8)
    parser.add_argument("--chat-model", default="gpt-4.1")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    dataset = normalize_dataset_name(args.dataset)
    return RunConfig(
        dataset_path=args.dataset_path or DEFAULT_DATASETS_DIR / f"{dataset}.json",
        output_dir=args.output_dir,
        dataset=dataset,
        text_top_k=args.text_top_k,
        triple_top_k=args.triple_top_k,
        chat_model=args.chat_model,
    )


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config: RunConfig, run_id: str | None) -> dict[str, object]:
    client = create_openai_client()
    samples = list(iter_samples(config.dataset_path, config.dataset))
    if not samples:
        raise ValueError("The dataset contains no records")
    result: dict[str, object] = {
        "dataset": config.dataset,
        "dataset_file": str(config.dataset_path.resolve()),
        "dataset_fingerprint": _fingerprint(config.dataset_path),
        "sample_scope": "ALL",
        "sample_count": len(samples),
        "text_top_k": config.text_top_k,
        "triple_top_k": config.triple_top_k,
        "chat_model": config.chat_model,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
    }
    store = ArtifactStore(config.output_dir / (run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    store.write_jsonl(Path(config.dataset) / "samples.jsonl", samples)
    store.update_manifest({**result, "status": "prepared", "stage": "samples"})
    pipeline = run_pipeline(samples, config, store, client, client)
    result.update({
        "answer_count": len(pipeline.answers),
        "dag_fallback_count": pipeline.dag_fallback_count,
        "reformulation_count": pipeline.reformulation_count,
        "triple_count": pipeline.triple_count,
        "exact_match": pipeline.evaluation["exact_match"],
        "token_f1": pipeline.evaluation["token_f1"],
        "run_dir": str(store.run_dir),
    })
    return result


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    print(json.dumps(run(config, args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()