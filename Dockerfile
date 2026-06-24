FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[claude]"

COPY tasks ./tasks
# Runs the offline heuristic agent by default; pass ANTHROPIC_API_KEY + override the
# command for the live Claude eval: docker run -e ANTHROPIC_API_KEY=... img ageval --agent claude
ENTRYPOINT ["ageval"]
CMD ["--agent", "heuristic"]
