#!/usr/bin/env python3
"""Probe one or more models for connectivity.

You list **model names** (e.g. ``claude-sonnet-4-6``, ``kimi-k2-thinking``).
The script auto-routes each name to a matching ``BENCH_ENDPOINT_<NAME>_*``
group in ``api.env`` (matching on the group's ``_MODEL`` field). If no
group matches, it falls back to the default endpoint (``BENCH_BASE_URL`` +
``BENCH_API_KEY``) and tries the name against that.

Pick which models to probe by editing the ``DEFAULT_MODELS`` list below,
or pass names on the command line.

Examples::

    python scripts/test_models.py                                  # use DEFAULT_MODELS
    python scripts/test_models.py claude-sonnet-4-6 kimi-k2-thinking
    python scripts/test_models.py --all                            # every model defined in api.env
    python scripts/test_models.py --only kimi                      # substring filter
    python scripts/test_models.py --list                           # what does api.env route?
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from carryover.tester import CarryoverTester
from nanobot.config.loader import load_config
from nanobot.config.project_env import (
    discover_endpoints,
    project_env,
    reload_project_env,
)

DEFAULT_MODELS: list[str] = [
    #"kimi-k2-thinking",
    "gpt-5.4",
    "claude-sonnet-4-6",
    "z-ai/glm-5",
    "gemini-3.1-pro-preview",
    "qwen3-235b-a22b",
]

PING_PROMPT = "只回复两个大写字母：OK"
TIMEOUT_S = 30

@dataclass
class Probe:
    """One row in the test table: a model id + the route that serves it."""
    model: str
    base_url: str
    api_key: str
    api_mode: str
    provider: str
    route_label: str  # 'BENCH_ENDPOINT_<NAME>' or 'default' for the fallback

@dataclass
class ProbeResult:
    model: str
    route_label: str
    base_url: str
    ok: bool
    elapsed: float
    response: str
    exception: str


# ── routing ──────────────────────────────────────────────────────────────


def _build_model_routes(discovered: dict[str, dict[str, str]]) -> dict[str, Probe]:
    """Map each known model id to the Probe that should serve it."""
    routes: dict[str, Probe] = {}
    seen_dupes: set[str] = set()
    for name, fields in discovered.items():
        model = fields.get("model")
        if not model:
            continue
        if model in routes:
            seen_dupes.add(model)
            continue
        routes[model] = Probe(
            model=model,
            base_url=fields.get("base_url", ""),
            api_key=fields.get("api_key", ""),
            api_mode=fields.get("api_mode", ""),
            provider=fields.get("provider", ""),
            route_label=f"BENCH_ENDPOINT_{name}",
        )
    for m in sorted(seen_dupes):
        print(
            f"Warning: model {m!r} is defined by multiple BENCH_ENDPOINT_*_MODEL "
            f"groups — using the first one.",
            file=sys.stderr,
        )
    return routes


def _build_default_probe(model: str, default: dict[str, str]) -> Probe | None:
    """Construct a Probe from the default-endpoint block for an unknown model."""
    api_key = default.get("BENCH_API_KEY", "")
    if not api_key:
        return None
    return Probe(
        model=model,
        base_url=default.get("BENCH_BASE_URL", ""),
        api_key=api_key,
        api_mode=default.get("BENCH_API_MODE", ""),
        provider=default.get("BENCH_PROVIDER", ""),
        route_label="default",
    )


# ── env scoping ──────────────────────────────────────────────────────────


def _set_bench_env(p: Probe) -> dict[str, str | None]:
    """Apply this probe's settings as default BENCH_* env vars for one run."""
    target = {
        "BENCH_MODEL": p.model,
        "BENCH_BASE_URL": p.base_url,
        "BENCH_API_KEY": p.api_key,
        "BENCH_API_MODE": p.api_mode,
        "BENCH_PROVIDER": p.provider,
    }
    saved: dict[str, str | None] = {}
    for k, v in target.items():
        saved[k] = os.environ.get(k)
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    reload_project_env()
    return saved


def _restore_env(saved: dict[str, str | None]) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    reload_project_env()


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return "***"
    return key[:6] + "…" + key[-4:]


# ── ping core ────────────────────────────────────────────────────────────


def _ping_success(response: str | None) -> bool:
    if not response:
        return False
    s = response.strip()
    if s.lower().startswith(("error calling llm", "error:", "api call failed")):
        return False
    return s.upper().rstrip(".。 ") == "OK"


async def _ping(model: str, timeout: int) -> tuple[bool, float, str, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()
        try:
            tester = CarryoverTester(output_dir=Path(tmpdir) / "out", model=model)
            agent, _ = tester._create_agent(workspace=Path(tmpdir))
            response = await asyncio.wait_for(
                agent.process_direct(PING_PROMPT),
                timeout=timeout,
            )
            elapsed = time.time() - t0
            ok = _ping_success(response)
            return ok, elapsed, response or "", ""
        except asyncio.TimeoutError:
            return False, float(timeout), "", f"TimeoutError after {timeout}s"
        except Exception as e:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            return False, elapsed, "", f"{type(e).__name__}: {e}\n{tb}"


async def probe_one(p: Probe, timeout: int) -> ProbeResult:
    saved = _set_bench_env(p)
    try:
        ok, elapsed, response, exc = await _ping(p.model, timeout)
    finally:
        _restore_env(saved)
    return ProbeResult(
        model=p.model,
        route_label=p.route_label,
        base_url=p.base_url,
        ok=ok,
        elapsed=elapsed,
        response=response.strip(),
        exception=exc,
    )


# ── orchestration ────────────────────────────────────────────────────────


def _resolve_probes(
    requested: list[str],
    routes: dict[str, Probe],
    default: dict[str, str],
) -> list[Probe]:
    probes: list[Probe] = []
    unresolved: list[str] = []
    for model in requested:
        if model in routes:
            probes.append(routes[model])
            continue
        fallback = _build_default_probe(model, default)
        if fallback:
            probes.append(fallback)
        else:
            unresolved.append(model)
    if unresolved:
        defined = ", ".join(sorted(routes.keys())) or "<none>"
        print(
            f"Error: model(s) {unresolved} have no BENCH_ENDPOINT_*_MODEL match "
            f"and no default endpoint is configured.\n"
            f"  models defined in api.env: {defined}",
            file=sys.stderr,
        )
        sys.exit(2)
    return probes


def _print_table(results: list[ProbeResult]) -> None:
    if not results:
        return
    model_width = max(20, min(40, max(len(r.model) for r in results)))
    route_width = max(8, min(28, max(len(r.route_label) for r in results)))
    print(
        f"  {'MODEL':<{model_width}} {'ROUTE':<{route_width}} "
        f"{'STATUS':<8} {'TIME':>7}  DETAIL"
    )
    print(
        f"  {'-' * model_width} {'-' * route_width} "
        f"{'-' * 8} {'-' * 7}  {'-' * 40}"
    )
    for r in results:
        status = "✓ PASS" if r.ok else "✗ FAIL"
        if r.ok:
            detail = f'response="{r.response[:60]}"'
        elif r.exception:
            detail = r.exception.splitlines()[0][:200]
        else:
            detail = (r.response or "<empty response>")[:200]
        print(
            f"  {r.model:<{model_width}} {r.route_label:<{route_width}} "
            f"{status:<8} {r.elapsed:>6.1f}s  {detail}"
        )


def _print_failure_details(results: list[ProbeResult]) -> None:
    failures = [r for r in results if not r.ok]
    if not failures:
        return
    print()
    for r in failures:
        print(f"┄┄ {r.model} via {r.route_label} @ {r.base_url} ┄┄")
        if r.exception:
            for line in r.exception.splitlines()[:40]:
                print(f"    {line}")
        if r.response:
            print("  Response:")
            for line in r.response.splitlines():
                print(f"    {line}")
        print()


async def run_probes(probes: list[Probe], timeout: int) -> int:
    print(f"\n{'=' * 80}")
    print(f"  CarryoverBench Model Connectivity Test (timeout={timeout}s per probe)")
    print(f"{'=' * 80}")
    for p in probes:
        print(f"  • {p.model:<28}  via {p.route_label:<24} "
              f"key={_mask_key(p.api_key)}")
        print(f"    {' ' * 28}      url={p.base_url or '<provider default>'}  "
              f"mode={p.api_mode or '<auto>'}")
    print(f"  {'-' * 78}")

    results: list[ProbeResult] = []
    for p in probes:
        results.append(await probe_one(p, timeout))

    _print_table(results)
    print(f"{'=' * 80}")
    passed = sum(1 for r in results if r.ok)
    print(f"  Result: {passed}/{len(results)} passed")
    _print_failure_details(results)
    return 0 if passed == len(results) else 1


# ── CLI ──────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Probe model connectivity. List model names; the script auto-routes "
                    "each name to a matching BENCH_ENDPOINT_* group in api.env.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("models", nargs="*",
                        help="model names to probe (e.g. claude-sonnet-4-6 kimi-k2-thinking); "
                             "default = DEFAULT_MODELS at top of this file")
    parser.add_argument("--all", action="store_true",
                        help="probe every model that api.env defines a route for")
    parser.add_argument("--only", type=str, default=None,
                        help="case-insensitive substring filter (applied after selection)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S,
                        help=f"per-probe timeout in seconds (default {TIMEOUT_S})")
    parser.add_argument("--list", action="store_true",
                        help="show every model→route api.env defines and exit")
    args = parser.parse_args(argv)

    reload_project_env()
    discovered = discover_endpoints()
    default = project_env()
    routes = _build_model_routes(discovered)

    if args.list:
        if not routes and not default.get("BENCH_API_KEY"):
            print("api.env defines no routes. Copy api.env.example to api.env and fill it in.")
            return 0
        print("Models routed by api.env:")
        for model, p in routes.items():
            mode = p.api_mode or "<auto>"
            print(f"  {model:<28}  {p.route_label:<22}  "
                  f"{p.base_url or '<provider default>'}  ({mode})")
        if default.get("BENCH_API_KEY"):
            print(
                f"  <fallback for unlisted models>     default                 "
                f"{default.get('BENCH_BASE_URL') or '<provider default>'}  "
                f"({default.get('BENCH_API_MODE') or '<auto>'})"
            )
        return 0

    if args.all:
        requested = list(routes.keys())
    elif args.models:
        requested = args.models
    elif DEFAULT_MODELS:
        requested = list(DEFAULT_MODELS)
    else:
        requested = [load_config().agents.defaults.model]

    if args.only:
        needle = args.only.lower()
        requested = [m for m in requested if needle in m.lower()]

    if not requested:
        print("No models selected. Edit DEFAULT_MODELS at the top of the script, "
              "or pass model names on the command line.", file=sys.stderr)
        return 2

    probes = _resolve_probes(requested, routes, default)
    return asyncio.run(run_probes(probes, args.timeout))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
