"""Model + pricing + paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("AGEVAL_ROOT", Path(__file__).resolve().parents[2]))
TASKS_PATH = ROOT / "tasks" / "tasks.yaml"
REPORTS = ROOT / "reports"

MODEL = os.getenv("AGEVAL_MODEL", "claude-opus-4-8")
MAX_STEPS = int(os.getenv("AGEVAL_MAX_STEPS", "8"))

# USD per 1M tokens (Claude Opus 4.8). Used for cost-per-task.
PRICE_PER_MTOK = {"input": 5.0, "output": 25.0}


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * PRICE_PER_MTOK["input"] + (output_tokens / 1e6) * PRICE_PER_MTOK["output"]
