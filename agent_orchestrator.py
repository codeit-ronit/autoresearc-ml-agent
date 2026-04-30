#!/usr/bin/env python3
"""
AutoResearch Agent Team Orchestrator  — v2 (Senior Scientist Edition)
=======================================================================
8-agent PRAR loop built on top of autoresearch.py.
Does NOT rewrite existing logic — orchestrates it.

Agent Roster
------------
1. Memory Agent        — reads results.tsv → rich structured memory + causal beliefs
2. Lead Researcher     — ReCAP manager, checks budget, coordinates team
3. Hypothesis Agent    — Senior ML Research Scientist: pattern analysis → causal
                         diagnosis → high-impact hypothesis (mandatory pivot after 3
                         consecutive non-improving runs)
4. Code Agent          — surgical old_snippet → new_snippet edit
5. Auditor Agent       — validates the edit (no cheating, no constraint violations)
6. Validator Agent     — runs train.py (300s limit), extracts metrics
7. Belief Agent        — Research Memory Distillation: BELIEF/EVIDENCE/CONFIDENCE
                         structured knowledge → updates program.md
8. Plot Agent          — regenerates progress.png 

v2 upgrades
-----------
* Senior ML Research Scientist prompt — forces causal diagnosis before any hypothesis
* Mandatory structural pivot after PIVOT_THRESHOLD consecutive non-improving runs
* Research Memory Distillation — BELIEF/EVIDENCE/CONFIDENCE format, not summaries
* Reliable JSON parsing — max_tokens=8192, no adaptive thinking for hypothesis calls
* Tournament Logic — detects lineages with >80% discard rate; orders pivots
* Budget Management — shuts down after MAX_NO_IMPROVEMENT_RUNS flat runs
"""

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Import all infrastructure from autoresearch.py ─────────────────────────────
# We orchestrate; we do NOT rewrite.
from autoresearch import (
    MODEL,
    MAX_CODE_RETRIES,
    PLATEAU_WINDOW,
    PLATEAU_THRESHOLD,
    PROGRAM_FILE,
    TRAIN_FILE,
    PREPARE_FILE,
    RESULTS_FILE,
    act_with_self_correction,
    append_result,
    check_plateau,
    client,
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

# ── Team-level configuration ────────────────────────────────────────────────────
MAX_NO_IMPROVEMENT_RUNS  = 20    # budget: stop if no gain for this many training runs
PIVOT_THRESHOLD          = 5     # mandatory structural pivot after this many flat runs
DISCARD_RATIO_THRESHOLD  = 0.80  # tournament: pivot if lineage >80% discard over 5 runs
LINEAGE_WINDOW           = 5     # minimum runs before tournament pruning kicks in

# Keywords used to detect hyperparameter lineage from hypothesis text
LINEAGE_KEYWORDS = [
    "n_layers", "d_ff", "n_heads", "batch_size", "dropout",
    "learning_rate", "lr=", "warmup", "betas", "weight_decay",
    "label_smooth", "cosine", "gradient_clip", "scheduler",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def agent_banner(name: str, role: str = "") -> None:
    """Print a prominent banner when an agent takes over."""
    width = 68
    print(f"\n{'━' * width}")
    print(f"  🤖  AGENT: {name.upper()}")
    if role:
        print(f"  📋  Role : {role}")
    print(f"{'━' * width}")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — MEMORY AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def memory_agent() -> dict:
    """
    Reads results.tsv and builds a rich memory snapshot consumed by all
    other agents. Returns a plain dict (no LLM call needed here).

    Keys:
        summary            str   — formatted experiment history for prompt injection
        best_accuracy      float — highest accuracy seen so far
        forbidden_patterns list  — [{pattern, tries}] tried 3+ times, never committed
        failed_lineages    list  — [{lineage, discard_ratio}] tournament losers
        no_improve_count   int   — consecutive training runs with no improvement
        recent_trend       str   — "improving" | "plateau" | "declining" | "cold_start"
        results            list  — raw rows from results.tsv
    """
    agent_banner(
        "Memory Agent",
        "Reads results.tsv → structured memory (best runs, forbidden patterns, trends)",
    )

    results = read_results()
    if not results:
        print("  [Memory] No experiments yet — cold start")
        return {
            "summary": "No experiments yet. Cold start.",
            "best_accuracy": 0.0,
            "forbidden_patterns": [],
            "failed_lineages": [],
            "no_improve_count": 0,
            "recent_trend": "cold_start",
            "results": [],
        }

    # ── Basic statistics ───────────────────────────────────────────────────────
    committed = [r for r in results if r["status"] == "COMMITTED"]
    ran       = [r for r in results if r["status"] in ("COMMITTED", "REVERTED", "BASELINE")]
    all_acc   = [float(r["accuracy"]) for r in ran if r.get("accuracy")]
    best_accuracy = max(all_acc) if all_acc else 0.0

    # ── Consecutive non-improving training runs ────────────────────────────────
    no_improve_count = 0
    for r in reversed(ran):
        if r["status"] == "COMMITTED":
            break
        no_improve_count += 1

    # ── Consecutive CODE_FAILED (Code Agent couldn't apply the edit) ───────────
    code_failed = [r for r in results if r["status"] == "CODE_FAILED"]
    consecutive_code_fails = 0
    for r in reversed(results):
        if r["status"] == "CODE_FAILED":
            consecutive_code_fails += 1
        else:
            break

    # ── Forbidden patterns (tried 3+ times, never committed) ──────────────────
    hyp_counts: Counter = Counter()
    for r in ran:
        hyp_counts[r["hypothesis"][:50].strip()] += 1
    committed_hyps = {r["hypothesis"][:50].strip() for r in committed}
    forbidden_patterns = [
        {"pattern": h, "tries": c}
        for h, c in hyp_counts.items()
        if c >= 3 and h not in committed_hyps
    ]

    # ── Recent trend (last 5 training runs) ───────────────────────────────────
    recent_ran  = ran[-5:] if len(ran) >= 5 else ran
    recent_accs = [float(r["accuracy"]) for r in recent_ran]
    if len(recent_accs) >= 2:
        delta = recent_accs[-1] - recent_accs[0]
        recent_trend = "improving" if delta > 0.005 else ("declining" if delta < -0.005 else "plateau")
    else:
        recent_trend = "insufficient_data"

    # ── Tournament: detect failed lineages (last 20 runs) ─────────────────────
    lineage_map: dict[str, list[str]] = {}
    for r in ran[-20:]:
        hyp = r["hypothesis"].lower()
        for kw in LINEAGE_KEYWORDS:
            if kw in hyp:
                lineage_map.setdefault(kw, []).append(r["status"])
                break

    failed_lineages = [
        {"lineage": lg, "discard_ratio": round(statuses.count("REVERTED") / len(statuses), 2)}
        for lg, statuses in lineage_map.items()
        if len(statuses) >= LINEAGE_WINDOW
        and statuses.count("REVERTED") / len(statuses) > DISCARD_RATIO_THRESHOLD
    ]

    # ── Print snapshot ─────────────────────────────────────────────────────────
    print(f"  [Memory] Experiments total  : {len(results)}")
    print(f"  [Memory] Best accuracy      : {best_accuracy:.4f}")
    print(f"  [Memory] No-improve streak  : {no_improve_count} training runs")
    print(f"  [Memory] Code-fail streak   : {consecutive_code_fails} consecutive")
    print(f"  [Memory] Recent trend       : {recent_trend}")
    print(f"  [Memory] Forbidden patterns : {len(forbidden_patterns)}")
    if failed_lineages:
        losers = [f"{l['lineage']} ({l['discard_ratio']:.0%})" for l in failed_lineages]
        print(f"  [Memory] Tournament losers  : {', '.join(losers)}")

    return {
        "summary": summarize_results(results),
        "best_accuracy": best_accuracy,
        "forbidden_patterns": forbidden_patterns,
        "failed_lineages": failed_lineages,
        "no_improve_count": no_improve_count,
        "consecutive_code_fails": consecutive_code_fails,
        "recent_trend": recent_trend,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — HYPOTHESIS AGENT  (Senior ML Research Scientist)
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_agent(
    iteration: int,
    program_goal: str,
    memory: dict,
    current_train_py: str,
    code_failure_context: str = "",
) -> dict:
    """
    Senior ML Research Scientist.

    Prompt instructs the model to reason internally (DO NOT OUTPUT) then emit
    exactly 3 lines:
        HYPOTHESIS: <sentence>
        REASON:     <1-2 lines>
        CHANGE_TYPE: architecture | optimization | training

    No JSON → no parse errors. Internal chain-of-thought stays intact.
    When plateau ≥ PIVOT_THRESHOLD → structural change is mandated.
    Returns a hypothesis dict; old_snippet/new_snippet are left empty so
    the Code Agent's LLM path resolves them from the hypothesis text.
    """
    no_improve   = memory["no_improve_count"]
    code_fails   = memory.get("consecutive_code_fails", 0)
    best_acc     = memory["best_accuracy"]
    recent_trend = memory["recent_trend"]
    results_all  = memory.get("results", [])

    # ── Count distinct optimization strategies tried in recent history ─────────
    STRATEGY_KEYWORDS = {
        "lr":             ["learning_rate", "lr", "1e-", "2e-", "5e-"],
        "batch":          ["batch_size", "batch size"],
        "regularization": ["weight_decay", "dropout", "label_smoothing"],
        "gradient":       ["grad_clip", "gradient clip"],
        "optimizer":      ["optimizer", "adam", "sgd", "betas", "momentum"],
        "warmup":         ["warmup", "warm_up"],
        "capacity":       ["d_model", "d_ff", "n_heads", "hidden"],
    }
    strategies_tried: set[str] = set()
    recent_ran = [r for r in results_all if r["status"] in ("COMMITTED", "REVERTED")][-15:]
    for r in recent_ran:
        h = r["hypothesis"].lower()
        for strat, kws in STRATEGY_KEYWORDS.items():
            if any(k in h for k in kws):
                strategies_tried.add(strat)

    # ── Determine mode ─────────────────────────────────────────────────────────
    # Architecture only if: 5+ flat valid runs AND 3+ optimization strategies tried
    arch_allowed  = (no_improve >= PIVOT_THRESHOLD) and (len(strategies_tried) >= 3)
    needs_simple  = code_fails >= 2          # code failures → single-line constants only
    is_stabilize  = recent_trend == "declining" and best_acc >= 0.82
    is_refinement = best_acc >= 0.82 and not arch_allowed

    if needs_simple:
        mode_label = "🔧 SIMPLICITY MANDATE"
    elif is_stabilize:
        mode_label = "🔁 STABILIZATION MODE"
    elif arch_allowed:
        mode_label = "⚠️  STRUCTURAL PIVOT ALLOWED"
    elif is_refinement:
        mode_label = "🎯 REFINEMENT MODE (acc>0.82)"
    else:
        mode_label = "🔬 EXPLORATION"

    agent_banner(
        "Hypothesis Agent  [Senior ML Research Scientist]",
        f"Internal causal reasoning → 3-line output  | {mode_label}",
    )

    # ── Build strategy section injected into prompt ────────────────────────────
    pruned = [l["lineage"] for l in memory["failed_lineages"]]

    if needs_simple:
        strategy_section = f"""
⚠️  SIMPLICITY MANDATE: The last {code_fails} hypotheses all failed at the CODE LEVEL.
Your hypotheses required too many lines of new code to implement reliably.

You MUST propose a SINGLE-LINE change to an EXISTING constant:
- ONE numeric constant: LR, BATCH_SIZE, WEIGHT_DECAY, N_HEADS, D_MODEL, N_LAYERS, GRAD_CLIP, WARMUP_STEPS
- ONE tuple value: betas=(X, Y)
- ONE existing flag already in the code

FORBIDDEN: new layers, new classes, new imports, BiLSTM, CNN, attention pooling,
any change requiring more than 2 lines of new code.
"""

    elif is_stabilize:
        strategy_section = f"""
🔁  STABILIZATION MODE: Recent runs show DECLINING accuracy (trend={recent_trend}).
Current best is {best_acc:.4f} but recent experiments are making things worse.

PRIORITY: Recover stability before exploring new directions.
- Propose a SMALL, LOW-RISK tweak to optimizer settings or regularization
- Focus on changes that are UNLIKELY to hurt (e.g., slightly adjust LR, WD, or GRAD_CLIP)
- Do NOT propose architecture changes — they increase variance and destabilize training
- Do NOT try anything that has already been tried and failed

Goal: stabilize near {best_acc:.4f} then inch upward with small refinements.
"""

    elif arch_allowed:
        strategy_section = f"""
⚠️  STRUCTURAL PIVOT: {no_improve} consecutive flat runs with {len(strategies_tried)} optimization
strategies already tried ({', '.join(sorted(strategies_tried))}).

Architecture change is now justified — but keep it SIMPLE and IMPLEMENTABLE:
- Prefer single-module swaps (e.g., change pooling method, add one layer type)
- Changes must be implementable in ≤5 lines via str.replace
- Avoid full model rewrites — surgical edits only
- Pruned lineages to avoid: {pruned if pruned else 'none'}
"""

    elif is_refinement:
        untried_opts = [s for s in STRATEGY_KEYWORDS if s not in strategies_tried]
        strategy_section = f"""
🎯  REFINEMENT MODE: Best accuracy is already {best_acc:.4f} (>0.82).
At this level, small precise changes beat risky architectural experiments.

Strategies NOT yet tried: {untried_opts if untried_opts else 'most basics covered'}
Strategies tried so far: {sorted(strategies_tried)}

PRIORITY ORDER:
1. Refine optimizer hyperparameters (betas, LR schedule shape, weight decay ratio)
2. Tune regularization carefully (dropout, label smoothing at 0.05)
3. Adjust training dynamics (warmup length, gradient clipping threshold)
4. Only last resort: small capacity change (single constant like D_FF or N_HEADS)

Do NOT: propose architecture rewrites, new modules, or high-risk changes.
Prefer: changes with expected Δ of +0.001 to +0.005 — small, safe, cumulative.
"""

    else:
        strategy_section = f"""
🔬  EXPLORATION: Early stage — no strong constraints yet.
Strategies tried so far: {sorted(strategies_tried) if strategies_tried else 'none yet'}
Try the most promising unexplored direction from the history.
"""

    failure_section = ""
    if code_failure_context:
        failure_section = f"\nPREVIOUS CODE FAILURE — avoid similar changes:\n{code_failure_context}\n"

    prompt = f"""You are a senior ML researcher in an automated research loop.
Best accuracy so far: {best_acc:.4f} | No-improve streak: {no_improve} | Trend: {recent_trend}

========================
INTERNAL THINKING (DO NOT OUTPUT)
========================
Think step by step internally:
1. What is the current bottleneck? (underfitting / overfitting / optimization / compute)
2. What category of change has the highest expected gain vs risk?
3. Is this change already in the failed history? If yes, pick something else.
4. Is this implementable as a single str.replace on existing code? If not, simplify.
{strategy_section}
{failure_section}
IMPORTANT: Do NOT output this reasoning. Keep it internal.

========================
CONTEXT
========================
Goal: {program_goal[:400]}

Experiment history:
{memory['summary']}

Current train.py:
```python
{current_train_py}
```

========================
OUTPUT — EXACTLY 3 LINES, NOTHING ELSE
========================

HYPOTHESIS: <one sentence: exact parameter/change + causal reason>
REASON: <1-2 lines: mechanism — WHY this fixes the bottleneck>
CHANGE_TYPE: <architecture | optimization | training>

Rules: no JSON, no markdown, no extra text. Only these 3 lines.
Refinement example (preferred when acc>0.82):
HYPOTHESIS: Reduce LEARNING_RATE from 1e-4 to 7e-5 to allow finer convergence near the current optimum
REASON: At 0.8234 accuracy the loss landscape is nearly flat; smaller steps prevent overshooting minima
CHANGE_TYPE: optimization""".strip()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=256,   # 3 lines only — small, fast, reliable
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        print(f"  [Hypothesis] ⚠️  API error: {exc}")
        return {"hypothesis": "API error", "old_snippet": "", "new_snippet": "",
                "risk_level": "high", "expected_delta": "0"}

    raw = ""
    for block in response.content:
        if block.type == "text":
            raw = block.text.strip()
            break

    if not raw:
        print("  [Hypothesis] ⚠️  Empty response from API")
        return {"hypothesis": "empty response", "old_snippet": "", "new_snippet": "",
                "risk_level": "high", "expected_delta": "0"}

    # ── Parse 3-line text format (robust, case-insensitive) ───────────────────
    hypothesis_text = ""
    reason_text     = ""
    change_type     = "optimization"

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        lo = line.lower()
        if lo.startswith("hypothesis:"):
            hypothesis_text = line.split(":", 1)[1].strip()
        elif lo.startswith("reason:"):
            reason_text = line.split(":", 1)[1].strip()
        elif lo.startswith("change_type:"):
            change_type = line.split(":", 1)[1].strip().lower()

    # Fallback: if parsing failed, use the raw text as hypothesis
    if not hypothesis_text:
        hypothesis_text = raw[:200]
        print(f"  [Hypothesis] ⚠️  Could not parse 3-line format, using raw:\n  {raw[:150]}")
    else:
        print(f"\n  [Hypothesis] 💡 {hypothesis_text[:100]}")
        print(f"  [Hypothesis] 🔍 {reason_text[:100]}")
        print(f"  [Hypothesis] 🔧 Change type: {change_type}")

    return {
        "hypothesis":     hypothesis_text,
        "reason":         reason_text,
        "change_type":    change_type,
        "old_snippet":    "",   # Code Agent resolves from hypothesis text
        "new_snippet":    "",
        "risk_level":     "medium",
        "expected_delta": "+?%",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — LEAD RESEARCHER (Manager)
# ═══════════════════════════════════════════════════════════════════════════════

def lead_researcher_agent(
    iteration: int,
    memory: dict,
    code_failure_context: str = "",
) -> tuple[dict, str, str]:
    """
    ReCAP-aware manager. Every call re-reads program.md to prevent plan drift.
    Checks budget, logs team state, then delegates to Hypothesis Agent.

    Returns:
        hypothesis       dict  — the proposed change
        program_goal     str   — fresh program.md content
        current_train_py str   — fresh train.py content
    """
    agent_banner(
        "Lead Researcher",
        "ReCAP manager — re-reads program.md, checks budget, coordinates team",
    )

    # ReCAP: re-read program.md every single iteration
    program_goal     = read_file(PROGRAM_FILE)
    current_train_py = read_file(TRAIN_FILE)

    print(f"  [Lead] ReCAP: loaded program.md ({len(program_goal)} chars)")
    print(f"  [Lead] Memory trend        : {memory['recent_trend']}")
    print(f"  [Lead] No-improve streak   : {memory['no_improve_count']} / {MAX_NO_IMPROVEMENT_RUNS}")
    print(f"  [Lead] Tournament losers   : {[l['lineage'] for l in memory['failed_lineages']]}")

    if memory["no_improve_count"] >= MAX_NO_IMPROVEMENT_RUNS * 0.75:
        print(
            f"\n  ⚠️  [Lead] Budget warning — "
            f"{memory['no_improve_count']} flat runs. "
            f"Shutdown at {MAX_NO_IMPROVEMENT_RUNS}."
        )

    # Delegate hypothesis proposal to Hypothesis Agent
    hypothesis = hypothesis_agent(
        iteration=iteration,
        program_goal=program_goal,
        memory=memory,
        current_train_py=current_train_py,
        code_failure_context=code_failure_context,
    )

    return hypothesis, program_goal, current_train_py


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — CODE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def code_agent_run(
    hypothesis: dict,
    program_goal: str,
    current_train_py: str,
) -> tuple[str, bool, str]:
    """
    Applies surgical old_snippet → new_snippet edit.
    Wraps the existing act_with_self_correction() (which internally runs the
    syntax gate and Auditor loop with up to MAX_CODE_RETRIES retries).

    Returns:
        modified_train_py  str   — edited file content (unchanged if failed)
        approved           bool  — True if edit passed audit
        reason             str   — human-readable outcome
    """
    agent_banner(
        "Code Agent",
        "Surgical edit — str.replace(old_snippet, new_snippet). Full-file rewrite = FAILURE.",
    )

    prepare_content = read_file(PREPARE_FILE)

    modified, approved, reason = act_with_self_correction(
        hypothesis=hypothesis,
        current_train_py=current_train_py,
        program_goal=program_goal,
        prepare_content=prepare_content,
    )

    status = "✅  EDIT APPROVED" if approved else f"❌  EDIT FAILED ({reason[:80]})"
    print(f"  [Code] {status}")
    return modified, approved, reason


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — AUDITOR AGENT (logging wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

def auditor_agent_log(approved: bool, reason: str) -> None:
    """
    The Auditor runs inside act_with_self_correction() above.
    This function prints the consolidated decision with a named banner so the
    user can see which agent made the call.
    """
    agent_banner(
        "Auditor Agent",
        "Confirms: no cheating, evaluate() intact, MAX_TRAIN_TIME=300 unchanged",
    )
    verdict = "✅  APPROVED" if approved else "❌  REJECTED"
    print(f"  [Auditor] Decision : {verdict}")
    print(f"  [Auditor] Reason   : {reason[:120]}")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — VALIDATOR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def validator_agent(cwd: str) -> tuple[float, float]:
    """
    Runs train.py with the 300-second hard budget and extracts metrics.
    Wraps the existing run_training() function.

    Returns:
        val_accuracy   float
        peak_memory_mb float
    """
    agent_banner(
        "Validator Agent",
        "Executes train.py (300 s wall-clock limit) → val_accuracy + peak_memory_mb",
    )

    val_accuracy, peak_memory, _ = run_training(cwd)

    print(f"  [Validator] val_accuracy  : {val_accuracy:.4f}")
    print(f"  [Validator] peak_memory   : {peak_memory:.1f} MB")
    if val_accuracy == 0.0:
        print("  [Validator] ⚠️  Zero accuracy — training may have crashed")

    return val_accuracy, peak_memory


# ═══════════════════════════════════════════════════════════════════════════════
# LEAD RESEARCHER — COMMIT / REVERT DECISION
# ═══════════════════════════════════════════════════════════════════════════════

def lead_researcher_decide(
    iteration: int,
    hypothesis: dict,
    val_accuracy: float,
    peak_memory: float,
    best_accuracy: float,
) -> tuple[float, str]:
    """
    Git ratchet: commit only on genuine improvement; revert otherwise.
    Returns (new_best_accuracy, status_str).
    """
    agent_banner(
        "Lead Researcher — Decision",
        "Git ratchet: commit on improvement | revert on no gain",
    )

    if val_accuracy > best_accuracy:
        delta = val_accuracy - best_accuracy
        print(f"  🎉  IMPROVEMENT +{delta:.4f} — committing (ratchet locked)")
        commit_hash = git_commit(iteration, hypothesis.get("hypothesis", "?"), val_accuracy)
        append_result(
            iteration,
            hypothesis.get("hypothesis", "?"),
            val_accuracy,
            "COMMITTED",
            commit_hash,
            peak_memory,
            f"Δ=+{delta:.4f} | {hypothesis.get('old_snippet','')[:40]}"
            f"→{hypothesis.get('new_snippet','')[:40]}",
        )
        print(f"  ✅  {commit_hash}  |  New best: {val_accuracy:.4f}")
        return val_accuracy, "COMMITTED"

    delta = val_accuracy - best_accuracy
    print(f"  ⬇️   No gain ({delta:+.4f}) — reverting via git checkout")
    git_revert()
    append_result(
        iteration,
        hypothesis.get("hypothesis", "?"),
        val_accuracy,
        "REVERTED",
        "",
        peak_memory,
        f"Δ={delta:.4f}",
    )
    print("  ↩️   Reverted to last best train.py")
    return best_accuracy, "REVERTED"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 7 — BELIEF AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def belief_agent() -> None:
    """
    Research Memory Distillation Agent.

    Extracts CAUSAL patterns in BELIEF/EVIDENCE/CONFIDENCE format and writes:
      - FORBIDDEN_PATTERNS  — what to never try again (with WHY)
      - PROMISING_DIRECTIONS — what the data suggests to try next
      - CAUSAL_BELIEFS       — mechanistic understanding

    Runs rule-based distiller first (3+ failed = FORBIDDEN), then LLM
    deep-distillation when ≥5 training runs exist.
    """
    agent_banner(
        "Belief Agent  [Research Memory Distillation]",
        "BELIEF/EVIDENCE/CONFIDENCE → FORBIDDEN_PATTERNS + PROMISING_DIRECTIONS",
    )

    results = read_results()
    distill_beliefs(results)   # rule-based: 3+ failed → FORBIDDEN

    ran = [r for r in results if r["status"] in ("COMMITTED", "REVERTED")]
    if len(ran) < 5:
        print(f"  [Belief] {len(ran)} training runs — skipping LLM distillation (need ≥5)")
        return

    # ── Build structured experiment log for LLM distillation ──────────────────
    last_50 = ran[-50:]
    committed = [r for r in last_50 if r["status"] == "COMMITTED"]
    reverted  = [r for r in last_50 if r["status"] == "REVERTED"]

    best_so_far = 0.0
    log_lines = []
    for r in last_50:
        acc = float(r["accuracy"])
        delta = acc - best_so_far
        marker = "✅ COMMITTED" if r["status"] == "COMMITTED" else f"❌ REVERTED (Δ={delta:+.4f})"
        log_lines.append(
            f"iter={r['iteration']} acc={r['accuracy']} [{marker}] | {r['hypothesis'][:70]}"
        )
        if r["status"] == "COMMITTED":
            best_so_far = acc

    experiment_log = "\n".join(log_lines)
    program = read_file(PROGRAM_FILE)

    prompt = f"""
You are a Research Memory Distillation Agent.
Convert experiment history into REUSABLE CAUSAL KNOWLEDGE — not summaries.

========================
EXPERIMENT LOG ({len(last_50)} runs, {len(committed)} committed, {len(reverted)} reverted)
========================
{experiment_log}

========================
EXISTING BELIEFS (do NOT repeat)
========================
{program[-2000:]}

========================
TASK: Extract CAUSAL beliefs with evidence and mechanism
========================

Output ONLY valid JSON — no markdown, no preamble:
{{
  "causal_beliefs": [
    {{
      "belief": "<causal pattern in one sentence>",
      "evidence": "<which iterations and how>",
      "confidence": "low|medium|high",
      "mechanism": "<WHY this happens mechanistically>"
    }}
  ],
  "forbidden_patterns": [
    {{
      "pattern": "<what not to try>",
      "reason": "<causal explanation>"
    }}
  ],
  "promising_directions": [
    {{
      "direction": "<what to try>",
      "rationale": "<why the data suggests this>"
    }}
  ],
  "model_state_diagnosis": "<underfitting|overfitting|compute_bound|capacity_saturated|optimization_stuck>",
  "next_best_strategy": "<single most important action>"
}}
""".strip()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = ""
        for block in response.content:
            if block.type == "text":
                raw = block.text.strip()
                break

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            print("  [Belief] ⚠️  No JSON in distillation response")
            return

        data = json.loads(match.group())
        beliefs       = data.get("causal_beliefs", [])
        forbidden     = data.get("forbidden_patterns", [])
        promising     = data.get("promising_directions", [])
        diagnosis     = data.get("model_state_diagnosis", "")
        next_strategy = data.get("next_best_strategy", "")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        block_text = f"\n\n## DISTILLED BELIEFS — {ts}\n"
        block_text += f"> Model state: **{diagnosis}**\n\n"

        if beliefs:
            block_text += "### Causal Beliefs\n"
            for b in beliefs[:4]:
                conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(b.get("confidence", ""), "⚪")
                block_text += (
                    f"- {conf_icon} **{b['belief']}**  \n"
                    f"  *Evidence*: {b.get('evidence', '')}  \n"
                    f"  *Mechanism*: {b.get('mechanism', '')}\n"
                )
        if forbidden:
            block_text += "\n### FORBIDDEN_PATTERNS\n"
            for f_item in forbidden[:5]:
                block_text += f"- ❌ `{f_item['pattern']}` — {f_item.get('reason', '')}\n"
        if promising:
            block_text += "\n### PROMISING_DIRECTIONS\n"
            for p in promising[:3]:
                block_text += f"- 🔬 **{p['direction']}** — {p.get('rationale', '')}\n"
        if next_strategy:
            block_text += f"\n**→ NEXT BEST STRATEGY**: {next_strategy}\n"

        current_program = read_file(PROGRAM_FILE)
        if "DISTILLED BELIEFS" not in current_program[-800:]:
            write_file(PROGRAM_FILE, current_program + block_text)
            print(f"  [Belief] program.md updated — {len(beliefs)} beliefs, "
                  f"{len(forbidden)} forbidden, {len(promising)} promising")
        else:
            print("  [Belief] Recent distillation already in program.md — skipping write")

        print(f"  [Belief] Diagnosis    : {diagnosis}")
        print(f"  [Belief] Next strategy: {next_strategy}")

    except json.JSONDecodeError as e:
        print(f"  [Belief] JSON parse error: {e}")
    except Exception as exc:
        print(f"  [Belief] Distillation error: {exc}")

    print("  [Belief] Distillation complete")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 8 — PLOT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_agent() -> None:
    """Regenerates progress.png by calling the existing plot_progress.py."""
    agent_banner("Plot Agent", "Regenerates progress.png via plot_progress.py")
    regenerate_plot()
    print("  [Plot] progress.png updated")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MULTI-AGENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("🔬  AutoResearch Agent Team  |  8-Agent PRAR Orchestrator  v2")
    print("=" * 68)
    print("  Agents: Memory → Lead → [Senior ML Scientist] Hypothesis")
    print("          → Code → Audit → Validate → Decide → Belief → Plot")
    print(f"  Pivot  : structural change mandated after {PIVOT_THRESHOLD} flat runs")
    print(f"  Budget : shutdown after {MAX_NO_IMPROVEMENT_RUNS} consecutive non-improving runs")
    print(f"  Tournament: lineage pruned if discard ratio > {DISCARD_RATIO_THRESHOLD:.0%}")
    print("=" * 68)

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    for f in [PROGRAM_FILE, TRAIN_FILE, PREPARE_FILE]:
        if not Path(f).exists():
            print(f"❌  Missing: {f}  — run `uv run prepare.py` first.")
            sys.exit(1)
    if not Path("data").exists():
        print("❌  data/ missing — run `uv run prepare.py` first.")
        sys.exit(1)

    cwd = str(Path(__file__).parent.resolve())

    # ── Baseline ───────────────────────────────────────────────────────────────
    existing          = read_results()
    existing_baseline = next((r for r in existing if r.get("status") == "BASELINE"), None)

    if existing_baseline:
        all_acc      = [float(r["accuracy"]) for r in existing
                        if r.get("status") in ("BASELINE", "COMMITTED") and r.get("accuracy")]
        best_accuracy = max(all_acc) if all_acc else float(existing_baseline["accuracy"])
        print(f"\n📊  Resuming — best acc={best_accuracy:.4f}  (skipping baseline re-run)")
    else:
        print("\n📊  Running baseline train.py...")
        baseline_acc, baseline_mem, _ = run_training(cwd)
        print(f"    Baseline → acc={baseline_acc:.4f}  mem={baseline_mem:.1f} MB")
        if baseline_acc == 0.0:
            print("❌  Baseline failed. Fix train.py / prepare.py first.")
            sys.exit(1)
        best_accuracy = baseline_acc
        append_result(0, "Baseline", baseline_acc, "BASELINE", "", baseline_mem, "Initial run")
        subprocess.run(["git", "add", "-A"], capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", f"baseline: acc={baseline_acc:.4f}"],
            capture_output=True,
        )
        print(f"    ✅  Committed baseline={best_accuracy:.4f}\n")

    iteration = max(
        (int(r.get("iteration", 0)) for r in existing if str(r.get("iteration", "")).isdigit()),
        default=0,
    )
    code_failure_context = ""
    session_code_fails   = 0   # tracks CODE_FAILED only within this session
    session_start        = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Research Loop ──────────────────────────────────────────────────────────
    print("\n🔁  Starting Multi-Agent Research Loop  (Ctrl-C to stop)\n")

    while True:
        iteration += 1
        print(f"\n{'═' * 68}")
        print(
            f"  ITERATION {iteration}  |  Best: {best_accuracy:.4f}"
            f"  |  Session: {session_start}"
        )
        print(f"{'═' * 68}")

        # ── AGENT 1: Memory Agent ──────────────────────────────────────────────
        memory = memory_agent()

        # ── Budget shutdown ────────────────────────────────────────────────────
        if memory["no_improve_count"] >= MAX_NO_IMPROVEMENT_RUNS:
            print(f"\n{'=' * 68}")
            print(
                f"🛑  BUDGET SHUTDOWN: {memory['no_improve_count']} consecutive"
                f" non-improving training runs (limit={MAX_NO_IMPROVEMENT_RUNS})."
            )
            print(f"    Best accuracy achieved: {best_accuracy:.4f}")
            print("    Lead Researcher has escalated — halting to save API budget.")
            print(f"{'=' * 68}")
            break

        # ── Code-failure budget (session-local safety net) ────────────────────
        code_fail_cap = 8
        if session_code_fails >= code_fail_cap:
            print(f"\n{'=' * 68}")
            print(
                f"🛑  CODE-FAIL SHUTDOWN: {session_code_fails} consecutive"
                f" CODE_FAILED iterations this session (cap={code_fail_cap})."
            )
            print("    Code Agent unable to apply edits. Check autoresearch.py / train.py.")
            print(f"    Best accuracy achieved: {best_accuracy:.4f}")
            print(f"{'=' * 68}")
            break

        # ── Plateau check (existing logic) ────────────────────────────────────
        if check_plateau(memory["results"]):
            print(
                f"\n🏁  PLATEAU — accuracy span < {PLATEAU_THRESHOLD * 100:.1f}%"
                f" over last {PLATEAU_WINDOW} training runs."
            )
            print(f"    Best: {best_accuracy:.4f}")
            break

        # ── AGENTS 2 + 3: Lead Researcher + Hypothesis Agent ──────────────────
        hypothesis, program_goal, current_train_py = lead_researcher_agent(
            iteration=iteration,
            memory=memory,
            code_failure_context=code_failure_context,
        )

        if hypothesis.get("hypothesis") in ("JSON parse failed", "JSON decode failed"):
            print("  ❌  Hypothesis generation failed (JSON parse error) — skipping iteration")
            code_failure_context = "Hypothesis generation failed (JSON parse error)"
            continue

        # ── AGENT 4: Code Agent ────────────────────────────────────────────────
        modified_train_py, approved, final_reason = code_agent_run(
            hypothesis=hypothesis,
            program_goal=program_goal,
            current_train_py=current_train_py,
        )

        # ── AGENT 5: Auditor Agent (logging) ──────────────────────────────────
        auditor_agent_log(approved, final_reason)

        if not approved:
            session_code_fails += 1
            code_failure_context = (
                f"Hypothesis '{hypothesis.get('hypothesis', '?')}' failed after "
                f"{MAX_CODE_RETRIES} attempts. Reason: {final_reason[:200]}"
            )
            append_result(
                iteration, hypothesis.get("hypothesis", "?"), 0.0,
                "CODE_FAILED", "", 0.0, final_reason[:100],
            )
            git_log_results()
            continue

        session_code_fails = 0   # reset on any successful code application
        code_failure_context = ""
        write_file(TRAIN_FILE, modified_train_py)

        # ── AGENT 6: Validator Agent ───────────────────────────────────────────
        val_accuracy, peak_memory = validator_agent(cwd)

        # ── Lead Researcher Decision (Ratchet) ────────────────────────────────
        best_accuracy, _status = lead_researcher_decide(
            iteration=iteration,
            hypothesis=hypothesis,
            val_accuracy=val_accuracy,
            peak_memory=peak_memory,
            best_accuracy=best_accuracy,
        )

        # ── AGENT 7: Belief Agent ──────────────────────────────────────────────
        git_log_results()
        belief_agent()

        # ── AGENT 8: Plot Agent ────────────────────────────────────────────────
        plot_agent()

    # ── Final summary ──────────────────────────────────────────────────────────
    plot_agent()
    print("\n" + "=" * 68)
    print("📋  AGENT TEAM SESSION COMPLETE")
    print("=" * 68)
    final_results = read_results()
    committed_all = [r for r in final_results if r["status"] == "COMMITTED"]
    ran_all       = [r for r in final_results if r["status"] in ("COMMITTED", "REVERTED")]
    print(f"  Training runs      : {len(ran_all)}")
    print(f"  Successful commits : {len(committed_all)}")
    print(f"  Best accuracy      : {best_accuracy:.4f}")
    if committed_all:
        print("\n  Top improvements:")
        top = sorted(committed_all, key=lambda r: float(r["accuracy"]), reverse=True)[:5]
        for r in top:
            print(f"    [{r['commit_hash']}] acc={r['accuracy']} — {r['hypothesis'][:60]}")
    print("\n  See results.tsv and progress.png for the full experiment log.")
    print("=" * 68)


if __name__ == "__main__":
    main()
