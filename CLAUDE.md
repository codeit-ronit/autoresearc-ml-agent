# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AutoResearch** is an autonomous ML research system that implements the Karpathy Loop to automatically optimize neural network hyperparameters for SST-2 sentiment classification. It uses Claude as a team of AI agents to iteratively run experiments within a **strict 300-second training budget per run**, targeting maximum validation accuracy.

## Setup & Running

**Prerequisites**: `ANTHROPIC_API_KEY` in a `.env` file (loaded via python-dotenv).

**One-time data preparation** (downloads SST-2, trains BPE tokenizer, pre-tokenizes all splits):
```bash
uv run prepare.py
```

**Run the research loop**:
```bash
uv run autoresearch.py
```

**Manually test a modified train.py**:
```bash
uv run train.py
```

**Visualize experiment progress**:
```bash
uv run plot_progress.py
```

There is no test suite or linter configuration. Validation is done via syntax checking (`ast.parse`), LLM audit, and accuracy measurement from training runs.

## Architecture

The system implements a **PRAR Loop** (Perceive → Reason → Act → Reflect):

```
PERCEIVE: Read program.md + results.tsv + train.py
    ↓
REASON: Lead Researcher (Claude) proposes 1 hypothesis
    ↓
ACT: Code Agent applies surgical edit (old_snippet → new_snippet via str.replace)
     + 2-second syntax gate
     + Auditor validates (up to 3 self-correcting retries)
    ↓
EXECUTE: Run train.py with 300s hard limit
    ↓
REFLECT: If accuracy improves → git commit (ratchet-lock)
         If not → git reset --hard (revert)
         If pattern fails 3+ times → auto-distill into program.md as FORBIDDEN
```

### Key Files

| File | Role | Mutable by agents? |
|------|------|--------------------|
| `autoresearch.py` | Main orchestrator — defines all agent roles and the PRAR loop | No |
| `train.py` | Training script — the **only** file agents are allowed to modify | Yes |
| `prepare.py` | Immutable harness: data loading + locked `evaluate()` function | No |
| `program.md` | Director's brief: goal, constraints, research directions, auto-updated beliefs | Auto-updated |
| `results.tsv` | TSV log of all experiments (iteration, hypothesis, accuracy, status, memory) | Auto-appended |

### `autoresearch.py` Agent Roles

- `lead_researcher()` — Proposes a single hypothesis given current state
- `code_agent()` — Translates hypothesis into a surgical `old_snippet → new_snippet` edit
- `audit_code_agent()` — Validates the edit doesn't violate constraints; rejection feeds back to code_agent for retry
- `run_training()` — Executes `train.py` subprocess with 300s timeout, parses output
- `distill_beliefs()` — Auto-updates `program.md` when a pattern is tried 3+ times with no improvement

### `train.py` Contract

- Imports `get_dataloaders`, `evaluate`, `VOCAB_SIZE`, `MAX_SEQ_LEN`, `PAD_TOKEN_ID` from `prepare.py`
- **Must** end with these two print lines (load-bearing for result parsing):
  ```
  val_accuracy: X.XXXX
  peak_memory_mb: X.XX
  ```
- `MAX_TRAIN_TIME = 300` must never be changed
- `evaluate()` must never be reimported or reimplemented — use only from `prepare.py`

### `program.md` Structure

Acts as persistent memory between iterations. Key sections:
- `PRIMARY_GOAL` — Maximize `val_accuracy` in 300s
- `HARD_CONSTRAINTS` — What agents must never do
- `RESEARCH_DIRECTIONS` — Suggested hyperparameter ranges
- `FORBIDDEN` patterns — Auto-populated when 3+ identical attempts all regress
- `PRIORITY_RESEARCH_PATHS` — Ranked list of next hypotheses to try

Every LLM call re-injects `program.md` to prevent plan drift across long-running loops.

## Hard Constraints (Never Violate)

1. **Never modify `prepare.py`**
2. **Never modify or replicate `evaluate()`**
3. **Never change `MAX_TRAIN_TIME = 300`**
4. **One hypothesis change per iteration** — isolates causality
5. **Check `results.tsv` before proposing** — do not repeat failed experiments
6. **Never touch the final two `print` lines** in `train.py`

## Current State

- Best accuracy: **0.8222** (iter 23) with `AdamW betas=(0.9, 0.95)`
- Forbidden patterns include: `N_LAYERS=3`, `D_FF=512`, `N_HEADS=8`, `BATCH_SIZE=64`, `LR>2e-4`, `DROPOUT=0.2`
- Believed insight: model capacity is saturated for the 300s budget — focus on optimizer throughput and LR scheduling
