"""Streamlit trajectory viewer: per-task pass/fail + the exact tool calls the agent made.

Run the harness first (`ageval --agent heuristic`) to produce reports/report_*.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPORTS = Path("reports")

st.set_page_config(page_title="Agent Eval Harness", layout="wide")
st.title("🧪 Agent Eval Harness")
st.caption("Did the agent actually complete the task? Success · tool-call accuracy · "
           "step efficiency · cost.")

reports = sorted(REPORTS.glob("report_*.json"))
if not reports:
    st.warning("No reports. Run `ageval --agent heuristic` (or `naive`/`claude`) first.")
    st.stop()

choice = st.selectbox("agent report", [p.name for p in reports])
report = json.loads((REPORTS / choice).read_text())
s = report["summary"]

c = st.columns(4)
c[0].metric("success rate", f"{s['success_rate']*100:.0f}%")
c[1].metric("mean tool F1", f"{s['mean_tool_f1']:.2f}")
c[2].metric("mean step efficiency", f"{s['mean_step_efficiency']:.2f}")
c[3].metric("cost / task", f"${s['mean_cost_per_task_usd']:.5f}")

st.subheader("Per-task scores")
st.dataframe([{
    "task": r["task_id"], "ok": "✅" if r["success"] else "❌",
    "tool_f1": r["tool_f1"], "step_eff": r["step_efficiency"],
    "steps": f"{r['n_steps']}/{r['optimal_steps']}",
    "predicted": r["predicted"], "gold": r["gold"], "cost$": r["cost_usd"],
} for r in report["rows"]], use_container_width=True, hide_index=True)

st.subheader("Trajectory inspector")
traj_by_id = {t["task_id"]: t for t in report["trajectories"]}
tid = st.selectbox("task", list(traj_by_id))
traj = traj_by_id[tid]
st.write(f"**Prompt:** {traj['prompt']}")
st.write(f"**Final answer:** `{traj['final_answer']}`")
for i, step in enumerate(traj["steps"], 1):
    st.markdown(f"**Step {i} — `{step['tool']}`**")
    st.json({"args": step["args"], "result": step["result"]})
if not traj["steps"]:
    st.info("No tool calls — the agent answered without using tools.")
