PY  ?= ~/miniconda3/envs/personal/bin/python
PIP ?= ~/miniconda3/envs/personal/bin/pip

.PHONY: install eval naive claude compare viewer test

install:
	$(PIP) install -e ".[all]"

eval:        # offline reference agent
	$(PY) -m ageval.harness --agent heuristic

naive:       # deliberately-flawed baseline (shows the harness discriminates)
	$(PY) -m ageval.harness --agent naive

claude:      # real eval — needs ANTHROPIC_API_KEY
	$(PY) -m ageval.harness --agent claude

compare: eval naive

viewer:
	$(PY) -m streamlit run app/viewer.py

test:
	$(PY) -m pytest -q
