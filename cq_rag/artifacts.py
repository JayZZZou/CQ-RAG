from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import SCHEMA_VERSION


def _serializable(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: Path, value: Any) -> Path:
        return self._atomic_write(
            self.run_dir / relative_path,
            json.dumps(_serializable(value), ensure_ascii=False, indent=2) + "\n",
        )

    def write_jsonl(self, relative_path: Path, rows: Iterable[Any]) -> Path:
        payload = "".join(json.dumps(_serializable(row), ensure_ascii=False) + "\n" for row in rows)
        return self._atomic_write(self.run_dir / relative_path, payload)

    def update_manifest(self, patch: dict[str, Any]) -> Path:
        target = self.run_dir / "manifest.json"
        manifest = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest.update(patch)
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        return self.write_json(Path("manifest.json"), manifest)

    @staticmethod
    def _atomic_write(target: Path, payload: str) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target