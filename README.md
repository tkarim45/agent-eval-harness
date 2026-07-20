# 🧪 Agent Eval Harness: *does my agent actually complete the task?*

> Everyone ships tool-using LLM agents; almost no one can **measure** them. This harness
> runs a tool-calling agent over a benchmark of multi-step tasks with **known-correct
> outcomes and reference tool traces**, and scores **task success, tool-call accuracy,
> step efficiency, and cost per task**, the four things a team actually needs to know
> before trusting an agent in production.

A final-answer-only metric hides *how* the agent got there. This harness grades the whole
**trajectory**: every tool call, the step count vs. optimal, and the token cost, so a
"correct" answer reached by 8 flailing steps and $0.40 scores differently from the same
answer in 3 steps and $0.05.

---

## What it measures

| Metric | Question it answers |
|---|---|
| **Success rate** | did the agent produce the correct final answer? (numeric-tolerant / text-substring) |
| **Tool-call accuracy** | did it call the right tools?, precision/recall/**F1** on the multiset vs. the reference trace |
| **Step efficiency** | `optimal_steps / actual_steps`, capped at 1.0, did it dawdle or loop? |
| **Cost per task** | token cost in USD (per-model pricing) |

---

## It discriminates good agents from bad ones

Three agents share one scorer. Run them against the same 12-task benchmark:

| Agent | success | tool F1 | step eff | cost/task |
|---|---|---|---|---|
| **`heuristic`** (competent reference) | **1.00** | **1.00** | **1.00** | $0.00 |
| **`naive`** (guesses arithmetic, truncates multi-hop, redundant calls) | **0.58** | **0.59** | **0.75** | $0.00 |
| **`claude`** (real tool-calling agent) | *run it* |, |, | *measured* |

The `naive` agent is deliberately flawed, that the harness scores it *well below* the
competent one is the point: **a good eval separates agents, it doesn't just rubber-stamp them.**

```
$ ageval --agent naive
task        ok  toolF1   eff  steps     cost$  pred / gold
cost_01      ❌    0.50  1.00   1/3    0.00000  '37.5' / 75.5      # guessed: ignored the 2nd item
hop_01       ✅    0.67  1.00   1/2    0.00000  'Engineering'      # truncated hop, right by luck
mgr_01       ✅    0.67  0.50   2/1    0.00000  'carol'            # redundant lookup -> eff 0.5
fact_01      ❌    0.00  0.00   0/1    0.00000  '2000' / 1993      # answered from memory, no search
...
success_rate=0.583  mean_tool_f1=0.589  mean_step_eff=0.750
```

---

## Architecture

```
tasks/tasks.yaml          agent (one of: heuristic · naive · claude)
  prompt                       │  manual tool-use loop, records the trajectory
  gold answer                  ▼
  expected tool trace     ┌──────────────┐    tools.py: calculator · catalog_lookup
  optimal steps  ───────► │  Trajectory  │◄── employee_lookup · search  (deterministic world)
                          │ steps+tokens │
                          └──────┬───────┘
                                 │  scorer.py: success · tool-F1 · step-eff · cost
                                 ▼
            reports/report_<agent>.json  ──►  CLI table · Streamlit trajectory viewer
```

The **`claude`** agent is a real Anthropic tool-calling loop (`claude-opus-4-8`); the
**`heuristic`** and **`naive`** agents are key-free so the harness, tests, and demo run
fully offline.

---

## Quickstart

> Uses the conda **`personal`** env (per environment conventions, never `base`).

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY -m pip install -e ".[all]"

# offline, no API key
ageval --agent heuristic        # competent reference agent
ageval --agent naive            # flawed baseline (watch the scores drop)
$PY -m streamlit run app/viewer.py   # per-task pass/fail + trajectory inspector

# real eval, needs a key
export ANTHROPIC_API_KEY=sk-ant-...
ageval --agent claude
```

Each run writes `reports/report_<agent>.json` (summary + per-task scores + full trajectories).

---

## The benchmark

`tasks/tasks.yaml`, 12 tasks across families that exercise multi-step tool use:

- **catalog arithmetic**, "total cost of 3 widgets and 2 gadgets?" (lookup × N → calculator)
- **directory lookup**, "who is Alice's manager?"
- **multi-hop**, "what department is the manager of Alice in?" (lookup → lookup)
- **stock sufficiency**, "enough stock for 10 cogs?"
- **fact retrieval**, "what year was Acme founded?" (search)

Each task declares `gold`, `expected_tools` (the optimal multiset), and `optimal_steps`.
Add your own tasks/tools to evaluate a real agent on your domain.

---

## Repo layout

```
agent-eval-harness/
├── src/ageval/
│   ├── tools.py     deterministic tools + a tiny in-memory world (catalog, directory, facts)
│   ├── agent.py     ClaudeAgent (real loop) · HeuristicAgent · NaiveAgent → Trajectory
│   ├── scorer.py    success · tool-F1 · step-efficiency · cost
│   ├── harness.py   run over the benchmark, aggregate, write report  (CLI: `ageval`)
│   └── config.py    model, pricing, paths
├── tasks/tasks.yaml the benchmark
├── app/viewer.py    Streamlit per-task + trajectory viewer
├── tests/           scorer + discrimination tests (key-free)
└── pyproject.toml · Dockerfile · Makefile · .github/workflows/ci.yml
```

---

## Résumé framing

> *Built an agent-evaluation harness measuring task success, tool-call accuracy (trace F1),
> step efficiency, and cost-per-task for tool-using LLM agents (Anthropic Claude); benchmark
> of multi-step tasks with reference traces, and a Streamlit trajectory inspector, the
> harness separates a competent agent (100% success) from a flawed one (58%) on the same suite.*

## License
MIT (`LICENSE`).
