#!/usr/bin/env python3
"""
multi_agent_research.py — Multi-Agent Orchestration Layer
===========================================================

Upgrades autoresearch.py from a linear PRAR loop (Level 3 Collaborator)
to a Level 5 Autonomous Problem Solver by layering eight specialised
agent roles on top of the existing primitives.

Agent team
----------
  1. Memory Agent       — Reads results.tsv; produces rich lineage memory
  2. Lead Researcher    — Manager; enforces ReCAP; makes ratchet decision
  3. Hypothesis Agent   — Proposes ONE hypothesis; plateau/pivot-aware
  4. Code Agent         — Surgical snippet edit (delegates to autoresearch)
  5. Auditor Agent      — Validates edit (embedded in Code Agent retry loop)
  6. Validator Agent    — Runs train.py; extracts accuracy + memory
  7. Belief Agent       — Distils forbidden patterns into program.md
  8. Plot Agent         — Regenerates progress.png

Governance additions (beyond the original PRAR loop)
------------------------------------------------------
  - Tournament Logic    — Track discard-ratio per lineage; force PIVOT > 80%
  - Budget Manager      — Halt loop after BUDGET_STALL_LIMIT non-improving iters
  - ReCAP Enforcement   — program.md re-read at iteration start AND injected
                          into every LLM call via existing autoresearch helpers
  - Surgical Integrity  — Full rewrites are a CRITICAL FAILURE; only
                          snippet-based patching allowed (enforced by Auditor)

Usage
-----
    uv run multi_agent_research.py
"""

import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Import ALL primitives from the existing orchestrator ───────────────────────
# We orchestrate; we do NOT rewrite existing logic.
from autoresearch import (
    MAX_CODE_RETRIES,
    PLATEAU_THRESHOLD,
    PLATEAU_WINDOW,
    PREPARE_FILE,
    PROGRAM_FILE,
    RESULTS_FILE,
    TRAIN_FILE,
    act_with_self_correction,
    append_result,
    check_plateau,
    distill_beliefs,
    git_commit,
    git_log_results,
    git_revert,
    lead_researcher,
    read_file,
    read_results,
    regenerate_plot,
    run_training,
    summarize_results,
    write_file,
)

# ── Multi-Agent Governance Constants ──────────────────────────────────────────
TOURNAMENT_WINDOW             = 5     # minimum runs per lineage before ratio check
TOURNAMENT_DISCARD_THRESHOLD  = 0.80  # >80 % discard → PIVOT required
BUDGET_STALL_LIMIT            = 20    # consecutive non-improving training runs → halt
MAX_ITERATIONS                = 200   # absolute iteration cap

# ── Lineage taxonomy (keyword → family) ───────────────────────────────────────
_LINEAGE_MAP: dict[str, list[str]] = {
    "optimizer":      ["learning_rate", "adamw", "betas", "weight_decay",
                       "grad_clip", "gradient_clip", "warmup"],
    "architecture":   ["n_layers", "d_model", "d_ff", "n_heads"],
    "regularization": ["dropout", "label_smoothing"],
    "schedule":       ["cosine", "lr_schedule", "warmup_steps", "one_cycle"],
    "batch":          ["batch_size"],
    "mixed_precision":["autocast", "amp", "mixed precision"],
}


def _classify_lineage(hypothesis: str) -> str:
    h = hypothesis.lower()
    for lineage, keywords in _LINEAGE_MAP.items():
        if any(kw in h for kw in keywords):
            return lineage
    return "other"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Memory Agent
# Reads results.tsv and builds structured experiment memory.
# Output is consumed by the Lead Researcher and Hypothesis Agent.
# ═══════════════════════════════════════════════════════════════════════════════

def memory_agent(results: list[dict]) -> dict:
    """
    ReCAP: re-reads every row of results.tsv to build an up-to-date picture.

    Returns
    -------
    dict with keys:
        best_accuracy      float — highest committed/baseline accuracy
        best_run           dict  — row that achieved best_accuracy
        stall_count        int   — consecutive non-improving training runs
        lineage_stats      dict  — per-lineage committed/reverted counts (all time)
        discard_ratios     dict  — per-lineage ratio over the last 30 runs
        pivot_needed       list  — lineages whose discard_ratio > threshold
        forbidden_patterns dict  — hypothesis prefix → repeat count (≥3 reverts)
        total_committed    int
        total_reverted     int
        total_runs         int
        summary_text       str   — human-readable summary (for LLM prompts)
    """
    committed = [r for r in results if r["status"] == "COMMITTED"]
    reverted  = [r for r in results if r["status"] == "REVERTED"]
    ran       = [r for r in results if r["status"] in ("COMMITTED", "REVERTED")]

    all_acc = [
        float(r["accuracy"])
        for r in results
        if r.get("status") in ("COMMITTED", "BASELINE") and r.get("accuracy")
    ]
    best_accuracy = max(all_acc) if all_acc else 0.0
    best_run = (
        max(
            (r for r in results
             if r.get("accuracy") and r.get("status") in ("COMMITTED", "BASELINE")),
            key=lambda r: float(r["accuracy"]),
        )
        if all_acc else None
    )

    # Stall counter — consecutive non-improving training runs from the end
    stall_count = 0
    for r in reversed(ran):
        if r["status"] == "COMMITTED":
            break
        stall_count += 1

    # All-time lineage stats
    lineage_stats: dict = defaultdict(lambda: {"committed": 0, "reverted": 0})
    for r in ran:
        lin = _classify_lineage(r["hypothesis"])
        lineage_stats[lin][r["status"].lower()] += 1

    # Recent discard ratios (last 30 runs only — recency matters)
    recent_by_lineage: dict[str, list[str]] = defaultdict(list)
    for r in ran[-30:]:
        lin = _classify_lineage(r["hypothesis"])
        recent_by_lineage[lin].append(r["status"])

    discard_ratios: dict[str, float] = {}
    for lin, statuses in recent_by_lineage.items():
        if len(statuses) >= TOURNAMENT_WINDOW:
            discard_ratios[lin] = statuses.count("REVERTED") / len(statuses)

    pivot_needed = [
        lin for lin, ratio in discard_ratios.items()
        if ratio > TOURNAMENT_DISCARD_THRESHOLD
    ]

    # Forbidden patterns: hypothesis prefix tried ≥3 times, always reverted
    hyp_counts: Counter = Counter()
    hyp_statuses: dict[str, list[str]] = {}
    for r in ran:
        key = r["hypothesis"][:50].strip()
        hyp_counts[key] += 1
        hyp_statuses.setdefault(key, []).append(r["status"])

    forbidden_patterns = {
        h: c
        for h, c in hyp_counts.items()
        if c >= 3 and all(s == "REVERTED" for s in hyp_statuses[h])
    }

    return {
        "best_accuracy":      best_accuracy,
        "best_run":           best_run,
        "stall_count":        stall_count,
        "lineage_stats":      dict(lineage_stats),
        "discard_ratios":     discard_ratios,
        "pivot_needed":       pivot_needed,
        "forbidden_patterns": forbidden_patterns,
        "total_committed":    len(committed),
        "total_reverted":     len(reverted),
        "total_runs":         len(ran),
        "summary_text":       summarize_results(results),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Hypothesis Agent
# Wraps lead_researcher() with plateau/pivot/stall awareness.
# Never proposes from a lineage that Memory Agent flagged for pivot.
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_agent(
    iteration: int,
    program_goal: str,
    memory: dict,
    current_train_py: str,
    code_failure_context: str = "",
) -> dict:
    """
    Proposes ONE hypothesis.  Injects pivot + stall signals into the prompt
    so that the LLM knows which lineages are exhausted.
    """
    extra_context = ""

    if memory["pivot_needed"]:
        exhausted = ", ".join(memory["pivot_needed"])
        extra_context += (
            f"\n\n## ⚡ TOURNAMENT PIVOT REQUIRED\n"
            f"Lineage(s) with >80 % discard ratio — DO NOT propose from: {exhausted}\n"
            f"Mandatory: shift to a completely different unexplored lineage."
        )

    if memory["stall_count"] >= 10:
        extra_context += (
            f"\n\n## 🚨 STALL WARNING: {memory['stall_count']} consecutive non-improving runs\n"
            f"Simple hyperparameter tuning has stalled. Propose a structurally "
            f"different change — e.g. cosine LR decay, mixed precision, parameter-group "
            f"LR, gradient accumulation, or CLS token pooling."
        )

    enriched_summary = memory["summary_text"] + extra_context

    return lead_researcher(
        iteration=iteration,
        program_goal=program_goal,
        results_summary=enriched_summary,
        current_train_py=current_train_py,
        code_failure_context=code_failure_context,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4 + 5 — Code Agent + Auditor Agent
# Thin wrapper: delegates entirely to act_with_self_correction().
# Surgical edits only — full rewrites → CRITICAL FAILURE (caught by Auditor).
# ═══════════════════════════════════════════════════════════════════════════════

def code_and_audit_agents(
    hypothesis: dict,
    current_train_py: str,
    program_goal: str,
    prepare_content: str,
) -> tuple[str, bool, str]:
    """
    Code Agent applies surgical old_snippet → new_snippet edit.
    Auditor validates (up to MAX_CODE_RETRIES self-correcting retries).
    Returns (modified_train_py, approved, reason).
    """
    return act_with_self_correction(
        hypothesis, current_train_py, program_goal, prepare_content
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — Validator Agent
# Executes train.py subprocess; extracts val_accuracy + peak_memory_mb.
# ═══════════════════════════════════════════════════════════════════════════════

def validator_agent(cwd: str) -> tuple[float, float, str]:
    """Pure delegation to run_training()."""
    return run_training(cwd)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 7 — Belief Agent
# Distils forbidden patterns into program.md.
# Also injects tournament pivot notes when lineages are exhausted.
# ═══════════════════════════════════════════════════════════════════════════════

def belief_agent(results: list[dict], memory: dict) -> None:
    """
    1. Calls distill_beliefs() — writes FORBIDDEN block to program.md.
    2. If tournament pivot is needed, writes/updates a TOURNAMENT PIVOT NOTE
       section so the Hypothesis Agent has a persistent reminder.
    """
    # Step 1: standard distillation
    distill_beliefs(results)

    # Step 2: tournament pivot annotation
    if not memory["pivot_needed"]:
        return

    program  = read_file(PROGRAM_FILE)
    marker   = "## TOURNAMENT PIVOT NOTE"
    lines    = [f"\n\n{marker}", "> Auto-generated by Belief Agent.\n"]
    for lin in memory["pivot_needed"]:
        ratio = memory["discard_ratios"].get(lin, 0.0)
        lines.append(f"- ⛔ Lineage '{lin}' exhausted (discard ratio {ratio:.0%})")
    block = "\n".join(lines) + "\n"

    if marker in program:
        start = program.index(marker) - 2   # include the blank line before
        program = program[:start] + block
    else:
        program = program + block

    write_file(PROGRAM_FILE, program)
    print(f"  🧠 Belief Agent: pivot note written for: {', '.join(memory['pivot_needed'])}")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 8 — Plot Agent
# Regenerates progress.png after every completed training run.
# ═══════════════════════════════════════════════════════════════════════════════

def plot_agent() -> None:
    """Pure delegation to regenerate_plot()."""
    regenerate_plot()


# ═══════════════════════════════════════════════════════════════════════════════
# LEAD RESEARCHER — Ratchet Decision (called twice per iteration)
# First call: reads program.md (ReCAP).
# Second call: decides commit vs revert.
# ═══════════════════════════════════════════════════════════════════════════════

def lead_researcher_decision(
    val_accuracy: float,
    old_best: float,
    iteration: int,
    hypothesis: dict,
    peak_memory: float,
) -> tuple[bool, str]:
    """
    Git Ratchet: commit on genuine improvement, revert otherwise.
    Returns (improved, commit_hash).
    """
    if val_accuracy > old_best:
        delta = val_accuracy - old_best
        print(f"  🎉  IMPROVEMENT +{delta:.4f} — committing (ratchet locked)")
        commit_hash = git_commit(iteration, hypothesis.get("hypothesis", "?"), val_accuracy)
        return True, commit_hash

    delta = val_accuracy - old_best
    print(f"  ⬇️   No gain ({delta:+.4f}) — reverting via git checkout")
    git_revert()
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# BUDGET MANAGER
# Halts the loop when too many consecutive runs show no improvement.
# ═══════════════════════════════════════════════════════════════════════════════

def budget_manager_check(stall_count: int, best_accuracy: float) -> bool:
    """Returns True → continue loop.  False → halt to conserve API budget."""
    if stall_count >= BUDGET_STALL_LIMIT:
        print(f"\n{'='*68}")
        print(f"⚠️  BUDGET MANAGER: {stall_count} consecutive non-improving iterations.")
        print(f"   Best accuracy so far: {best_accuracy:.4f}")
        print(f"   Shutting down to conserve API budget.")
        print(f"   Inspect results.tsv and program.md, then restart if needed.")
        print(f"{'='*68}")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Multi-Agent Orchestration Loop
# ═══════════════════════════════════════════════════════════════════════════════

def _print_agent_banner(agent_name: str) -> None:
    print(f"\n{'·'*68}")
    print(f"  🤖  {agent_name}")
    print(f"{'·'*68}")


def main() -> None:
    print("=" * 68)
    print("🤖  AutoResearch — Multi-Agent Orchestration System")
    print("=" * 68)
    print("  Agents: Memory | Hypothesis | Code | Auditor | Validator")
    print("          Belief | Plot | Lead Researcher (Manager)")
    print("=" * 68)

    # ── Validate environment ───────────────────────────────────────────────────
    for f in [PROGRAM_FILE, TRAIN_FILE, PREPARE_FILE]:
        if not Path(f).exists():
            print(f"❌  Missing: {f}  — run `uv run prepare.py` first.")
            sys.exit(1)
    if not Path("data").exists():
        print("❌  data/ missing — run `uv run prepare.py` first.")
        sys.exit(1)

    cwd = str(Path(__file__).parent.resolve())

    # ── Baseline (skip if already recorded) ───────────────────────────────────
    existing = read_results()
    existing_baseline = next(
        (r for r in existing if r.get("status") == "BASELINE"), None
    )

    if existing_baseline:
        all_acc = [
            float(r["accuracy"])
            for r in existing
            if r.get("status") in ("BASELINE", "COMMITTED") and r.get("accuracy")
        ]
        best_accuracy = max(all_acc) if all_acc else float(existing_baseline["accuracy"])
        print(f"\n📊  Resuming — best acc={best_accuracy:.4f}  (skipping baseline re-run)")
    else:
        _print_agent_banner("Validator Agent — Baseline Run")
        baseline_acc, baseline_mem, _ = validator_agent(cwd)
        print(f"    Baseline → acc={baseline_acc:.4f}  mem={baseline_mem:.1f}MB")
        if baseline_acc == 0.0:
            print("❌  Baseline failed.  Fix train.py / prepare.py first.")
            sys.exit(1)
        best_accuracy = baseline_acc
        append_result(0, "Baseline", baseline_acc, "BASELINE", "",
                      baseline_mem, "Initial run")
        subprocess.run(["git", "add", "-A"], capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", f"baseline: acc={baseline_acc:.4f}"],
            capture_output=True,
        )
        print(f"    ✅  Committed baseline={best_accuracy:.4f}\n")

    iteration = max(
        (int(r.get("iteration", 0))
         for r in existing
         if str(r.get("iteration", "")).isdigit()),
        default=0,
    )
    code_failure_context = ""
    session_start = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ══════════════════════════════════════════════════════════════════════════
    # MULTI-AGENT RESEARCH LOOP
    # ══════════════════════════════════════════════════════════════════════════
    print("🔁  Starting Multi-Agent Research Loop  (Ctrl-C to stop)\n")

    while iteration < MAX_ITERATIONS:
        iteration += 1
        bar = "═" * 68
        print(f"\n{bar}")
        print(
            f"  ITERATION {iteration:03d}  |  Best: {best_accuracy:.4f}"
            f"  |  Session: {session_start}"
        )
        print(bar)

        # ReCAP: re-read program.md at the very start of every iteration
        program_goal     = read_file(PROGRAM_FILE)
        current_train_py = read_file(TRAIN_FILE)
        prepare_content  = read_file(PREPARE_FILE)
        results          = read_results()

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1 — Memory Agent
        # ──────────────────────────────────────────────────────────────────────
        _print_agent_banner("Agent 1 — Memory Agent")
        memory = memory_agent(results)

        print(
            f"  📚 Best: {memory['best_accuracy']:.4f}  |  "
            f"Stall: {memory['stall_count']}  |  "
            f"Runs: {memory['total_runs']}  |  "
            f"Commits: {memory['total_committed']}"
        )

        if memory["discard_ratios"]:
            print("  📊 Lineage discard ratios (recent 30 runs):")
            for lin, ratio in sorted(memory["discard_ratios"].items(),
                                     key=lambda x: -x[1]):
                flag = "⛔" if ratio > TOURNAMENT_DISCARD_THRESHOLD else "  "
                print(f"     {flag} {lin:<20s}: {ratio:.0%}")

        if memory["pivot_needed"]:
            print(
                f"  ⚡ TOURNAMENT PIVOT required for: "
                f"{', '.join(memory['pivot_needed'])}"
            )

        if memory["forbidden_patterns"]:
            print(
                f"  🚫 {len(memory['forbidden_patterns'])} forbidden pattern(s) "
                f"(≥3 reverts each)"
            )

        # ──────────────────────────────────────────────────────────────────────
        # Budget Manager — check before spending any more API tokens
        # ──────────────────────────────────────────────────────────────────────
        if not budget_manager_check(memory["stall_count"], memory["best_accuracy"]):
            break

        # ──────────────────────────────────────────────────────────────────────
        # Plateau check (PRAR guard, unchanged)
        # ──────────────────────────────────────────────────────────────────────
        if check_plateau(results):
            print(
                f"\n🏁  PLATEAU — accuracy hasn't moved >{PLATEAU_THRESHOLD*100:.1f}% "
                f"over last {PLATEAU_WINDOW} training runs."
            )
            print(f"    Best: {best_accuracy:.4f}")
            break

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2 — Lead Researcher (ReCAP phase)
        # ──────────────────────────────────────────────────────────────────────
        _print_agent_banner("Agent 2 — Lead Researcher (ReCAP)")
        print("  🎯 PRIMARY GOAL re-read from program.md: maximize val_accuracy ≤300s")
        print(f"  📋 {len(results)} prior experiments visible to all agents")

        # ──────────────────────────────────────────────────────────────────────
        # STEP 3 — Hypothesis Agent
        # ──────────────────────────────────────────────────────────────────────
        _print_agent_banner("Agent 3 — Hypothesis Agent")
        hypothesis = hypothesis_agent(
            iteration=iteration,
            program_goal=program_goal,
            memory=memory,
            current_train_py=current_train_py,
            code_failure_context=code_failure_context,
        )
        print(f"  💡 {hypothesis.get('hypothesis', '?')}")
        print(
            f"  🔧 {hypothesis.get('old_snippet', '?')[:55].strip()!r} → "
            f"{hypothesis.get('new_snippet', '?')[:55].strip()!r}"
        )
        print(
            f"  📐 risk={hypothesis.get('risk_level','?')}  "
            f"Δ≈{hypothesis.get('expected_delta','?')}"
        )

        hyp_text = hypothesis.get("hypothesis", "?")
        if hyp_text.startswith("JSON"):
            # LLM returned malformed output; skip this iteration
            code_failure_context = "Previous JSON parse failure — propose a simpler, concrete change."
            append_result(
                iteration, hyp_text, 0.0, "CODE_FAILED", "", 0.0,
                "JSON parse failure from Hypothesis Agent"
            )
            git_log_results()
            continue

        # ──────────────────────────────────────────────────────────────────────
        # STEP 4 + 5 — Code Agent + Auditor Agent
        # ──────────────────────────────────────────────────────────────────────
        _print_agent_banner("Agent 4 — Code Agent  |  Agent 5 — Auditor Agent")
        modified_train_py, approved, final_reason = code_and_audit_agents(
            hypothesis, current_train_py, program_goal, prepare_content
        )

        if not approved:
            print(
                f"  ❌  All {MAX_CODE_RETRIES} Code+Audit attempts failed: "
                f"{final_reason[:120]}"
            )
            code_failure_context = (
                f"Hypothesis '{hyp_text}' failed after {MAX_CODE_RETRIES} attempts. "
                f"Reason: {final_reason[:200]}"
            )
            append_result(
                iteration, hyp_text, 0.0, "CODE_FAILED", "", 0.0,
                final_reason[:100]
            )
            git_log_results()
            continue

        print("  ✅  Auditor approved — writing modified train.py")
        code_failure_context = ""
        write_file(TRAIN_FILE, modified_train_py)

        # ──────────────────────────────────────────────────────────────────────
        # STEP 6 — Validator Agent
        # ──────────────────────────────────────────────────────────────────────
        _print_agent_banner("Agent 6 — Validator Agent")
        val_accuracy, peak_memory, _ = validator_agent(cwd)
        print(f"  val_accuracy : {val_accuracy:.4f}  (current best={best_accuracy:.4f})")
        print(f"  peak_memory  : {peak_memory:.1f} MB")

        # ──────────────────────────────────────────────────────────────────────
        # STEP 7 — Lead Researcher Decision (Git Ratchet)
        # ──────────────────────────────────────────────────────────────────────
        _print_agent_banner("Agent 2 — Lead Researcher Decision (Git Ratchet)")
        old_best = best_accuracy
        improved, commit_hash = lead_researcher_decision(
            val_accuracy, old_best, iteration, hypothesis, peak_memory
        )

        if improved:
            delta = val_accuracy - old_best
            best_accuracy = val_accuracy
            append_result(
                iteration, hyp_text, val_accuracy,
                "COMMITTED", commit_hash, peak_memory,
                f"Δ=+{delta:.4f} | "
                f"{hypothesis.get('old_snippet','')[:40]}→"
                f"{hypothesis.get('new_snippet','')[:40]}",
            )
            print(f"  ✅  {commit_hash}  new best={best_accuracy:.4f}")
        else:
            delta = val_accuracy - old_best
            append_result(
                iteration, hyp_text, val_accuracy,
                "REVERTED", "", peak_memory,
                f"Δ={delta:.4f}",
            )
            print("  ↩️   Reverted to last best train.py")

        git_log_results()

        # ──────────────────────────────────────────────────────────────────────
        # STEP 8 — Belief Agent
        # ──────────────────────────────────────────────────────────────────────
        _print_agent_banner("Agent 7 — Belief Agent")
        updated_results = read_results()
        belief_agent(updated_results, memory)

        # ──────────────────────────────────────────────────────────────────────
        # STEP 9 — Plot Agent
        # ──────────────────────────────────────────────────────────────────────
        _print_agent_banner("Agent 8 — Plot Agent")
        plot_agent()
        print("  📊  progress.png updated")

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    plot_agent()   # ensure final chart is up to date

    print("\n" + "=" * 68)
    print("📋  MULTI-AGENT AUTORESEARCH COMPLETE")
    print("=" * 68)

    final_results = read_results()
    committed = [r for r in final_results if r["status"] == "COMMITTED"]
    ran       = [r for r in final_results if r["status"] in ("COMMITTED", "REVERTED")]

    print(f"  Experiments run    : {len(ran)}")
    print(f"  Successful commits : {len(committed)}")
    print(f"  Best accuracy      : {best_accuracy:.4f}")

    if committed:
        print("\n  Top improvements:")
        top = sorted(committed, key=lambda r: float(r["accuracy"]), reverse=True)[:5]
        for r in top:
            print(f"    [{r.get('commit_hash','?')}] acc={r['accuracy']} — {r['hypothesis'][:60]}")

    # Final memory snapshot
    final_memory = memory_agent(final_results)
    if final_memory["discard_ratios"]:
        print("\n  Lineage discard ratios (all-time):")
        for lin, ratio in sorted(
            final_memory["discard_ratios"].items(), key=lambda x: -x[1]
        ):
            print(f"    {lin:<20s}: {ratio:.0%}")

    print("\n  See results.tsv and progress.png for the full experiment log.")
    print("=" * 68)


if __name__ == "__main__":
    main()
