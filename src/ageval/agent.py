"""Agents under test. Both produce a `Trajectory` the scorer grades identically.

- ClaudeAgent  : a real tool-calling agent (Anthropic manual loop). Needs ANTHROPIC_API_KEY.
- HeuristicAgent: a key-free reference agent that routes the benchmark task families to
                  the same tools — so the harness, tests, and demo run offline.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .config import MAX_STEPS, MODEL
from .tools import CATALOG, anthropic_tools, run_tool

SYSTEM = (
    "You are a precise assistant with access to tools. Use the tools to look up facts and "
    "compute results — do not guess. Take the fewest steps necessary. When you have the "
    "answer, reply with ONLY the final answer (a number or short phrase), no preamble."
)


@dataclass
class Step:
    tool: str
    args: dict
    result: object


@dataclass
class Trajectory:
    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    hit_step_limit: bool = False

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]


# -----------------------------------------------------------------------------
# Claude agent — manual tool-use loop
# -----------------------------------------------------------------------------
class ClaudeAgent:
    name = "claude"

    def __init__(self, model: str = MODEL, max_steps: int = MAX_STEPS):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_steps = max_steps
        self.tools = anthropic_tools()

    def run(self, task_id: str, prompt: str) -> Trajectory:
        traj = Trajectory(task_id=task_id)
        messages = [{"role": "user", "content": prompt}]
        try:
            for _ in range(self.max_steps):
                resp = self.client.messages.create(
                    model=self.model, max_tokens=1024, system=SYSTEM,
                    tools=self.tools, messages=messages,
                )
                traj.input_tokens += resp.usage.input_tokens
                traj.output_tokens += resp.usage.output_tokens

                if resp.stop_reason != "tool_use":
                    traj.final_answer = next(
                        (b.text for b in resp.content if b.type == "text"), "").strip()
                    return traj

                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for b in resp.content:
                    if b.type == "tool_use":
                        out = run_tool(b.name, b.input)
                        traj.steps.append(Step(b.name, dict(b.input), out))
                        results.append({"type": "tool_result", "tool_use_id": b.id,
                                        "content": str(out)})
                messages.append({"role": "user", "content": results})
            traj.hit_step_limit = True
        except Exception as e:  # noqa: BLE001
            traj.error = f"{type(e).__name__}: {e}"
        return traj


# -----------------------------------------------------------------------------
# Heuristic agent — key-free reference baseline over the same tools
# -----------------------------------------------------------------------------
_ITEM = "|".join(sorted(CATALOG, key=len, reverse=True))


class HeuristicAgent:
    name = "heuristic"

    def run(self, task_id: str, prompt: str) -> Trajectory:
        t = Trajectory(task_id=task_id)
        p = prompt.lower()

        # multi-hop: department of the manager of X
        m = re.search(r"manager of (\w+)", p)
        if "department" in p and m:
            mgr = self._call(t, "employee_lookup", {"name": m.group(1)})
            if isinstance(mgr, dict) and mgr.get("manager"):
                rec = self._call(t, "employee_lookup", {"name": mgr["manager"]})
                t.final_answer = rec.get("dept", "") if isinstance(rec, dict) else ""
            return t

        # who is X's manager
        m = re.search(r"(\w+)'s manager|manager of (\w+)", p)
        if "manager" in p and m:
            name = m.group(1) or m.group(2)
            rec = self._call(t, "employee_lookup", {"name": name})
            t.final_answer = (rec.get("manager") or "none") if isinstance(rec, dict) else ""
            return t

        # salary / department of X
        m = re.search(r"(salary|department) of (\w+)", p)
        if m:
            rec = self._call(t, "employee_lookup", {"name": m.group(2)})
            key = "salary" if m.group(1) == "salary" else "dept"
            t.final_answer = str(rec.get(key, "")) if isinstance(rec, dict) else ""
            return t

        # stock sufficiency: enough stock for N X
        m = re.search(r"(\d+)\s+(" + _ITEM + r")s?", p)
        if "stock" in p or "enough" in p:
            if m:
                rec = self._call(t, "catalog_lookup", {"item": m.group(2)})
                qty = int(m.group(1))
                t.final_answer = "yes" if isinstance(rec, dict) and rec["stock"] >= qty else "no"
            return t

        # catalog arithmetic: total cost of N1 X and N2 Y ...
        pairs = re.findall(r"(\d+)\s+(" + _ITEM + r")s?", p)
        if pairs and ("total" in p or "cost" in p or "price" in p):
            terms = []
            for qty, item in pairs:
                rec = self._call(t, "catalog_lookup", {"item": item})
                if isinstance(rec, dict):
                    terms.append(f"{qty}*{rec['price']}")
            if terms:
                total = self._call(t, "calculator", {"expression": "+".join(terms)})
                t.final_answer = _fmt_num(total)
            return t

        # fact lookup: founded / launched / moved / spun off -> year
        if any(k in p for k in ("founded", "launched", "moved", "spun", "year")):
            hits = self._call(t, "search", {"query": prompt})
            yr = re.search(r"\b(19|20)\d{2}\b", " ".join(hits) if isinstance(hits, list) else "")
            t.final_answer = yr.group(0) if yr else (hits[0] if isinstance(hits, list) else "")
            return t

        t.error = "no matching strategy"
        return t

    def _call(self, traj: Trajectory, tool: str, args: dict):
        out = run_tool(tool, args)
        traj.steps.append(Step(tool, args, out))
        return out


def _fmt_num(x) -> str:
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(round(x, 2)) if isinstance(x, float) else str(x)


# -----------------------------------------------------------------------------
# Naive agent — a deliberately mediocre baseline, to show the harness discriminates
# -----------------------------------------------------------------------------
class NaiveAgent:
    """Common failure modes: guesses arithmetic instead of using the calculator,
    truncates multi-hop reasoning, and makes redundant tool calls. The harness should
    score it well below the competent agent."""

    name = "naive"

    def run(self, task_id: str, prompt: str) -> Trajectory:
        t = Trajectory(task_id=task_id)
        p = prompt.lower()

        # arithmetic: looks up one item, then GUESSES the total (no calculator) -> wrong
        pairs = re.findall(r"(\d+)\s+(" + _ITEM + r")s?", p)
        if pairs and ("total" in p or "cost" in p or "price" in p):
            rec = self._call(t, "catalog_lookup", {"item": pairs[0][1]})
            price = rec["price"] if isinstance(rec, dict) else 0
            t.final_answer = _fmt_num(price * int(pairs[0][0]))  # ignores the other items
            return t

        # multi-hop: stops after the first hop -> returns the wrong person's department
        m = re.search(r"manager of (\w+)", p)
        if "department" in p and m:
            rec = self._call(t, "employee_lookup", {"name": m.group(1)})
            t.final_answer = rec.get("dept", "") if isinstance(rec, dict) else ""  # not the manager's
            return t

        # simple lookups: correct, but with a redundant duplicate call (hurts efficiency)
        m = re.search(r"(\w+)'s manager|manager of (\w+)", p)
        if "manager" in p and m:
            name = m.group(1) or m.group(2)
            self._call(t, "employee_lookup", {"name": name})
            rec = self._call(t, "employee_lookup", {"name": name})  # redundant
            t.final_answer = (rec.get("manager") or "none") if isinstance(rec, dict) else ""
            return t

        m = re.search(r"(salary|department) of (\w+)", p)
        if m:
            rec = self._call(t, "employee_lookup", {"name": m.group(2)})
            t.final_answer = str(rec.get("salary" if m.group(1) == "salary" else "dept", "")) \
                if isinstance(rec, dict) else ""
            return t

        if "stock" in p or "enough" in p:
            mm = re.search(r"(\d+)\s+(" + _ITEM + r")s?", p)
            if mm:
                rec = self._call(t, "catalog_lookup", {"item": mm.group(2)})
                t.final_answer = "yes" if isinstance(rec, dict) and rec["stock"] >= int(mm.group(1)) else "no"
            return t

        # facts: answers from "memory" without searching -> often wrong
        t.final_answer = "2000"
        return t

    def _call(self, traj: Trajectory, tool: str, args: dict):
        out = run_tool(tool, args)
        traj.steps.append(Step(tool, args, out))
        return out
