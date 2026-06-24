"""Score a trajectory against a task: success, tool-call accuracy, step efficiency, cost.

Why these four (the things teams actually need to know about an agent):
  - success            did it get the right final answer?
  - tool F1            did it call the right tools (vs. reference trace)? — multiset overlap
  - step efficiency    optimal_steps / actual_steps, capped at 1.0 — did it dawdle?
  - cost_usd           token cost of the run
"""
from __future__ import annotations

import re
from collections import Counter

from .config import cost_usd


def _normalize(ans: str, answer_type: str) -> str:
    a = (ans or "").strip().lower().rstrip(".")
    if answer_type == "numeric":
        m = re.search(r"-?\d[\d,]*\.?\d*", a.replace(",", ""))
        return m.group(0) if m else a
    return a


def answer_correct(predicted: str, gold: str, answer_type: str) -> bool:
    p, g = _normalize(predicted, answer_type), _normalize(gold, answer_type)
    if answer_type == "numeric":
        try:
            return abs(float(p) - float(g)) < 1e-6
        except ValueError:
            return False
    return g in p or p == g  # tolerate "the manager is carol" vs "carol"


def tool_prf(used: list[str], expected: list[str]) -> dict:
    """Precision/recall/F1 on the multiset of tool names (order-independent)."""
    u, e = Counter(used), Counter(expected)
    overlap = sum((u & e).values())
    prec = overlap / max(sum(u.values()), 1)
    rec = overlap / max(sum(e.values()), 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def step_efficiency(n_steps: int, optimal_steps: int) -> float:
    if n_steps <= 0:
        return 0.0
    return min(1.0, optimal_steps / n_steps)


def score(traj, task: dict) -> dict:
    success = (traj.error is None) and answer_correct(
        traj.final_answer, str(task["gold"]), task["answer_type"])
    prf = tool_prf(traj.tools_used, task["expected_tools"])
    return {
        "task_id": task["id"],
        "success": bool(success),
        "predicted": traj.final_answer,
        "gold": str(task["gold"]),
        "tool_precision": round(prf["precision"], 3),
        "tool_recall": round(prf["recall"], 3),
        "tool_f1": round(prf["f1"], 3),
        "n_steps": traj.n_steps,
        "optimal_steps": task["optimal_steps"],
        "step_efficiency": round(step_efficiency(traj.n_steps, task["optimal_steps"]), 3),
        "input_tokens": traj.input_tokens,
        "output_tokens": traj.output_tokens,
        "cost_usd": round(cost_usd(traj.input_tokens, traj.output_tokens), 6),
        "hit_step_limit": traj.hit_step_limit,
        "error": traj.error,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    mean = lambda k: sum(r[k] for r in rows) / n
    return {
        "n_tasks": n,
        "success_rate": round(mean("success"), 4),
        "mean_tool_f1": round(mean("tool_f1"), 4),
        "mean_step_efficiency": round(mean("step_efficiency"), 4),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "mean_cost_per_task_usd": round(mean("cost_usd"), 6),
        "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
    }
