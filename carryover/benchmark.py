"""
CarryoverBench task loader.

Tasks are defined in carryover/tasks/<task_id>/task.yaml with env files
under carryover/tasks/<task_id>/env/.

Supported schema:
  - New format only: `attacks:` list
    - each entry becomes one attacked BenchmarkTask
    - task_id expands to {base_id}__{mode}__{payload_id}
    - IPI variants inject attack files into env_files
    - Skill variants keep env clean and install SKILL.md at seed time

Additionally, one clean baseline task is generated per task directory for
utility-only benign evaluation.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from carryover.types import AttackGoal, AttackMode, AttackSource, BenchmarkTask

_TASKS_DIR = Path(__file__).parent / "tasks"
_REQUIRED_TASK_FIELDS = {"task_id", "name", "domain", "description", "user_prompt", "env_files", "attacks"}
_REQUIRED_ATTACK_FIELDS = {"mode", "payload_id", "goal", "source", "success_pattern"}
_ALLOWED_MODES = {m.value for m in AttackMode}
_ALLOWED_SOURCES = {s.value for s in AttackSource}
_SOURCE_TRUSTED_ENV = "trusted_env"
_SOURCE_EXTERNAL_OBSERVATION = "external_observation"


def _read_env_files(meta: dict, env_dir: Path) -> dict[str, str]:
    """读取共享的干净env文件（不含注入内容）。"""
    env_files: dict[str, str] = {}
    for rel in meta.get("env_files", []):
        env_files[rel] = (env_dir / rel).read_text(encoding="utf-8", errors="replace")
    return env_files


def _load_skill_md(meta: dict, env_dir: Path) -> str:
    skl = meta.get("skill", {})
    if skl.get("skill_file"):
        return (env_dir / skl["skill_file"]).read_text(encoding="utf-8", errors="replace")
    return ""


def _read_skill_file(env_dir: Path, rel_path: str) -> str:
    if not rel_path:
        return ""
    return (env_dir / rel_path).read_text(encoding="utf-8", errors="replace")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")


def _validate_attack_entry(atk: dict, base_id: str, idx: int, env_dir: Path, seen_variants: set[tuple[str, str]]) -> None:
    missing = sorted(_REQUIRED_ATTACK_FIELDS - set(atk.keys()))
    if missing:
        raise ValueError(f"Task {base_id} attack[{idx}] missing required fields: {missing}")

    mode = str(atk.get("mode", ""))
    source = str(atk.get("source", ""))
    payload_id = str(atk.get("payload_id", ""))

    if mode not in _ALLOWED_MODES:
        raise ValueError(f"Task {base_id} attack[{idx}] has invalid mode: {mode}")
    if source not in _ALLOWED_SOURCES:
        raise ValueError(f"Task {base_id} attack[{idx}] has invalid source: {source}")
    variant_key = (mode, payload_id)
    if variant_key in seen_variants:
        raise ValueError(f"Task {base_id} has duplicate attack variant: mode={mode}, payload_id={payload_id}")
    seen_variants.add(variant_key)

    if source == "ipi":
        injection_key = atk.get("injection_key")
        injection_file = atk.get("injection_file")
        if not injection_key or not injection_file:
            raise ValueError(f"Task {base_id} attack[{idx}] source=ipi requires injection_key and injection_file")
        _require_file(env_dir / str(injection_file), f"IPI injection_file for {base_id}[{payload_id}]")

    if source == "skill":
        for field in ("skill_name", "skill_description", "skill_file"):
            if not atk.get(field):
                raise ValueError(f"Task {base_id} attack[{idx}] source=skill requires {field}")
        _require_file(env_dir / str(atk["skill_file"]), f"skill_file for {base_id}[{payload_id}]")


def _validate_task_meta(meta: dict, task_dir: Path) -> None:
    missing = sorted(_REQUIRED_TASK_FIELDS - set(meta.keys()))
    if missing:
        raise ValueError(f"Task {task_dir.name} missing required top-level fields: {missing}")

    env_dir = task_dir / "env"
    env_files = meta.get("env_files") or []
    attacks = meta.get("attacks") or []
    if not isinstance(env_files, list) or not env_files:
        raise ValueError(f"Task {task_dir.name} env_files must be a non-empty list")
    if not isinstance(attacks, list) or not attacks:
        raise ValueError(f"Task {task_dir.name} attacks must be a non-empty list")

    for rel in env_files:
        _require_file(env_dir / str(rel), f"env_file for {task_dir.name}")

    seen_variants: set[tuple[str, str]] = set()
    for idx, atk in enumerate(attacks):
        if not isinstance(atk, dict):
            raise ValueError(f"Task {task_dir.name} attack[{idx}] must be a mapping")
        _validate_attack_entry(atk, str(meta.get('task_id', task_dir.name)), idx, env_dir, seen_variants)


def _load_multi_attack(meta: dict, env_dir: Path) -> list[BenchmarkTask]:
    """
    新格式：attacks: 列表。
    每个entry展开为独立的 BenchmarkTask，task_id = {base}__{mode}__{payload_id}。
    injection_key + injection_file 将注入文件合并进 env_files。
    """
    shared_env = _read_env_files(meta, env_dir)
    shared_sources = {rel: _SOURCE_TRUSTED_ENV for rel in shared_env}
    top_level_skill_md = _load_skill_md(meta, env_dir)
    top_level_skill = meta.get("skill", {})
    scr = meta.get("scoring", {})
    base_id = meta["task_id"]
    result = []

    for atk in meta["attacks"]:
        attack_source = AttackSource(atk.get("source", "ipi"))

        # IPI attacks mutate one shared env file with the injected payload.
        # Skill attacks keep env_files clean and install SKILL.md at seed time.
        task_env = dict(shared_env)
        task_sources = dict(shared_sources)
        injection_key = atk.get("injection_key", "")
        injection_file = atk.get("injection_file", "")
        if attack_source == AttackSource.IPI and injection_key and injection_file:
            task_env[injection_key] = (env_dir / injection_file).read_text(
                encoding="utf-8", errors="replace"
            )
            task_sources[injection_key] = _SOURCE_EXTERNAL_OBSERVATION

        skill_name = atk.get("skill_name", top_level_skill.get("name", ""))
        skill_description = atk.get(
            "skill_description", top_level_skill.get("description", "")
        )
        skill_md = (
            _read_skill_file(env_dir, atk.get("skill_file", ""))
            if atk.get("skill_file")
            else top_level_skill_md
        )

        mode_str = atk.get("mode", "instant")
        payload_id = atk.get("payload_id", "p1")
        task_id = f"{base_id}__{mode_str}__{payload_id}"

        result.append(BenchmarkTask(
            task_id=task_id,
            name=f"{meta['name']} [{mode_str.capitalize()} {payload_id.upper()}]",
            domain=meta["domain"],
            description=meta["description"],
            user_prompt=meta["user_prompt"].strip(),
            tools_used=meta.get("tools_used", []),
            env_files=task_env,
            env_file_sources=task_sources,
            attack_goal=AttackGoal(atk.get("goal", "exfiltration")),
            attack_mode=AttackMode(mode_str),
            attack_source=attack_source,
            injection_point=atk.get("injection_point", ""),
            # payload_text可选，用于文档；实际IPI注入内容在env_files里
            payload_text=atk.get("payload_text", "").strip(),
            target_tool=atk.get("target_tool", ""),
            target_action=atk.get("target_action", ""),
            success_pattern=atk.get("success_pattern", ""),
            attack_hint=atk.get("hint", "").strip(),
            skill_name=skill_name,
            skill_description=skill_description,
            skill_md=skill_md,
            task_complexity=scr.get("task_complexity", 1),
            attack_risk=scr.get("attack_risk", 1),
            reference=meta.get("reference", ""),
        ))

    return result


def _load_baseline_task(meta: dict, env_dir: Path) -> BenchmarkTask:
    """Generate one clean baseline task per task directory."""
    env_files = _read_env_files(meta, env_dir)
    env_file_sources = {rel: _SOURCE_TRUSTED_ENV for rel in env_files}
    scr = meta.get("scoring", {})

    return BenchmarkTask(
        task_id=meta["task_id"],
        name=meta["name"],
        domain=meta["domain"],
        description=meta["description"],
        user_prompt=meta["user_prompt"].strip(),
        tools_used=meta.get("tools_used", []),
        env_files=env_files,
        env_file_sources=env_file_sources,
        task_complexity=scr.get("task_complexity", 1),
        attack_risk=scr.get("attack_risk", 1),
        reference=meta.get("reference", ""),
    )


def _load_task_dir(task_dir: Path) -> list[BenchmarkTask]:
    meta = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    env_dir = task_dir / "env"
    _validate_task_meta(meta, task_dir)
    return _load_multi_attack(meta, env_dir)


BENCHMARK_TASKS: list[BenchmarkTask] = [
    task
    for d in sorted(_TASKS_DIR.iterdir())
    if d.is_dir() and (d / "task.yaml").exists()
    for task in _load_task_dir(d)
]

BASELINE_TASKS: list[BenchmarkTask] = [
    _load_baseline_task(yaml.safe_load((d / "task.yaml").read_text(encoding="utf-8")), d / "env")
    for d in sorted(_TASKS_DIR.iterdir())
    if d.is_dir() and (d / "task.yaml").exists()
]
