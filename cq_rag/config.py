from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import DatasetName


@dataclass(frozen=True)
class RunConfig:
    dataset_path: Path
    output_dir: Path
    dataset: DatasetName = "hotpotqa"
    text_top_k: int = 10
    triple_top_k: int = 8
    dag_max_nodes: int = 4
    max_candidates_per_node: int = 12
    max_reformulation_rate: float = 0.25
    embedding_dimensions: int = 1024
    chat_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        if self.dataset not in {"hotpotqa", "musique"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        for name in ("text_top_k", "triple_top_k", "dag_max_nodes", "max_candidates_per_node", "embedding_dimensions"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dag_max_nodes > 4:
            raise ValueError("dag_max_nodes cannot exceed the validated limit of 4")
        if not 0 <= self.max_reformulation_rate <= 0.25:
            raise ValueError("max_reformulation_rate must be between 0 and 0.25")