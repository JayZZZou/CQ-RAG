from __future__ import annotations


def print_item_progress(stage: str, completed: int, total: int, width: int = 24) -> None:
    ratio = completed / total if total else 1.0
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    digits = len(str(total))
    line = f"\r[{stage}] items={completed:>{digits}}/{total} [{bar}] {ratio * 100:5.1f}%"
    print(line, end="\n" if completed >= total else "", flush=True)