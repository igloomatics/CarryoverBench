# CarryoverBench

**CarryoverBench: Benchmarking Carryover Attacks on LLM Agents**

Accepted to **EMNLP 2026**.

CarryoverBench evaluates prompt injections that survive the session in which
they are introduced. A planted payload can later reappear through a scheduled
heartbeat or long-term memory and act during a fresh task.

## What is included

The benchmark contains 12 personal-assistant scenarios and 101 attack tasks:

`email_management`, `e_shopping`, `code_review_commit`, `travel_booking`,
`finance_tracking`, `social_media`, `health_management`, `banking_transfer`,
`literature_review`, `food_delivery`, `price_tracking`, and
`resume_submission`.

It measures three channels:

- **Instant**: same-turn injection and action.
- **Autonomous**: persistence through `HEARTBEAT.md`.
- **Latent**: persistence through `MEMORY.md`.

Autonomous and Latent trials use five fresh follow-up trigger rounds (`N=5`),
reporting Plant, 1-Trig through 5-Trig, ASR, and task utility.

The repository also includes the three defenses evaluated in the paper:

- **D1 Principles**: operational safety principles in the system prompt.
- **D2 Review**: explicit authority to review and edit persistent state.
- **D3 Spotlighting**: an untrusted-data envelope around persistent-state reads.

## Quick start

Install the Python dependencies and configure a local `api.env` file with your
model endpoint. `api.env` is ignored by Git.

Run the clean utility baseline:

```bash
python scripts/run_benign_task.py
```

Run the main benchmark with five trigger rounds:

```bash
python scripts/run_benchmark.py --rounds 5 --model <model-name>
```

Run a defense condition:

```bash
python scripts/run_defense.py --defense system_prompt --rounds 5 --model <model-name>
```

Use `--defense skill` or `--defense spotlighting` for the corresponding
defense implementation. Results are written to the local `results/` directory,
which is intentionally excluded from the public repository.

## Repository layout

- `carryover/`: benchmark definitions, attack payloads, task environments, and verifiers.
- `nanobot/`: the agent harness used by the benchmark.
- `scripts/`: benchmark runners, defense evaluation, aggregation, and statistical analysis.
- `tests/`: lightweight regression tests for tool validation and shared verifier logic.

## Citation

If you use CarryoverBench, please cite the EMNLP 2026 paper:

```bibtex
@inproceedings{carryoverbench2026,
  title     = {CarryoverBench: Benchmarking Carryover Attacks on LLM Agents},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```

The benchmark is intended for controlled research environments. Do not run
the task payloads against real accounts, credentials, or production systems.
