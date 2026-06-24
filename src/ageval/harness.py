"""Run an agent over the benchmark, score every task, aggregate, and write a report.

    python -m ageval.harness --agent heuristic            # offline, no key
    python -m ageval.harness --agent claude               # real eval (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import yaml

from .agent import ClaudeAgent, HeuristicAgent, NaiveAgent
from .config import REPORTS, TASKS_PATH
from .scorer import aggregate, score

AGENTS = {"heuristic": HeuristicAgent, "naive": NaiveAgent, "claude": ClaudeAgent}


def run(agent_name: str, tasks_path=TASKS_PATH, limit: int = 0) -> dict:
    tasks = yaml.safe_load(open(tasks_path))["tasks"]
    if limit:
        tasks = tasks[:limit]
    agent = AGENTS[agent_name]()

    rows, trajectories = [], []
    for task in tasks:
        traj = agent.run(task["id"], task["prompt"])
        rows.append(score(traj, task))
        trajectories.append({"task_id": task["id"], "prompt": task["prompt"],
                             "steps": [asdict(s) for s in traj.steps],
                             "final_answer": traj.final_answer})

    summary = aggregate(rows)
    report = {"agent": agent_name, "summary": summary, "rows": rows, "trajectories": trajectories}

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"report_{agent_name}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _print(report: dict) -> None:
    s = report["summary"]
    print(f"\nagent: {report['agent']}   tasks: {s['n_tasks']}")
    print("-" * 76)
    print(f"{'task':10} {'ok':>3} {'toolF1':>7} {'eff':>5} {'steps':>6} {'cost$':>9}  pred / gold")
    for r in report["rows"]:
        ok = "✅" if r["success"] else "❌"
        print(f"{r['task_id']:10} {ok:>3} {r['tool_f1']:>7.2f} {r['step_efficiency']:>5.2f} "
              f"{r['n_steps']:>3}/{r['optimal_steps']:<2} {r['cost_usd']:>9.5f}  "
              f"{r['predicted'][:18]!r} / {r['gold']}")
    print("-" * 76)
    print(f"success_rate={s['success_rate']:.3f}  mean_tool_f1={s['mean_tool_f1']:.3f}  "
          f"mean_step_eff={s['mean_step_efficiency']:.3f}  "
          f"cost/task=${s['mean_cost_per_task_usd']:.5f}  total=${s['total_cost_usd']:.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="run the agent eval harness")
    ap.add_argument("--agent", choices=list(AGENTS), default="heuristic")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    _print(run(args.agent, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
