FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY carryover/ ./carryover/
COPY nanobot/ ./nanobot/
COPY scripts/ ./scripts/

# Results go here (mount a volume to persist)
RUN mkdir -p /app/results

# API key via env var — pass at runtime:
#   docker run -e NANOBOT_PROVIDERS__OPENAI__API_KEY=sk-... carryover-bench
#   docker run -e NANOBOT_PROVIDERS__ANTHROPIC__API_KEY=sk-ant-... carryover-bench
#
# Model override (optional):
#   docker run -e NANOBOT_AGENTS__DEFAULTS__MODEL=gpt-4o carryover-bench

ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/run_benchmark.py", "--clean-only"]
