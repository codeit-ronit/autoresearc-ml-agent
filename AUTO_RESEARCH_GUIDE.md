# AutoResearch: A Framework for Autonomous Hyperparameter Optimization

> A minimal, principled system for wrapping any existing ML training script with an AI-driven research loop that explores, learns, and improves — automatically.

---

## What Is This?

AutoResearch is an autonomous ML research framework that applies the **Karpathy Loop** (run → measure → hypothesize → edit → repeat) using a team of Claude LLM agents. It wraps your existing training script and iteratively proposes, implements, evaluates, and commits changes — all without human intervention.

The key insight: **treat hyperparameter search as a research process**, not a grid search. Use an LLM that reasons causally about *why* something might work, checks experiment history, avoids dead ends, and gradually walks toward better performance.

---

## Files to Share

| File | Purpose | Required? |
|------|---------|-----------|
| `autoresearch.py` | Main orchestrator — all agent logic + PRAR loop | ✅ Core |
| `agent_orchestrator.py` | 8-agent extended version with pivot logic + belief distillation | ✅ Core |
| `program.md` | Director's brief — goal, constraints, beliefs, forbidden paths | ✅ Core |
| `AUTO_RESEARCH_GUIDE.md` | This guide | ✅ Share this |
| `results.tsv` | Experiment log (auto-appended each run) | Optional (shows real output) |
| `train.py` | Your target training script (the only file agents modify) | Adapt for your codebase |
| `prepare.py` | Immutable harness with locked `evaluate()` | Adapt for your codebase |

**Minimal set to adapt**: `autoresearch.py` + `program.md` + your own `train.py`-equivalent.

---

## Core Design Principles

### 1. The PRAR Loop

Every iteration follows four phases:

```
PERCEIVE  → Read program.md + results.tsv + train.py (fresh every iteration)
    ↓
REASON    → Lead Researcher proposes ONE hypothesis with old_snippet + new_snippet
    ↓
ACT       → Code Agent applies surgical edit → syntax check → Auditor validates
    ↓
REFLECT   → Run training → if accuracy improves: git commit (ratchet)
                         → if not: git checkout HEAD -- train.py (revert)
```

Re-reading `program.md` at the top of every iteration is the **ReCAP pattern** — it prevents plan drift across long sessions. The LLM always knows the goal, constraints, and current beliefs.

---

### 2. Surgical Editing (Not Full Rewrites)

The biggest failure mode in LLM code modification is **context decay**: when you ask an LLM to rewrite a whole file, it silently truncates, hallucinates, or drifts. We solve this with surgical edits.

**How it works:**
- The Lead Researcher outputs an `old_snippet` (exact verbatim lines to replace) and `new_snippet` (the replacement).
- The Code Agent uses `str.replace(old_snippet, new_snippet, 1)` — one exact substitution.
- If the snippet isn't found verbatim, the edit fails safely (no partial corruption).

```python
# Fast path — Lead Researcher already supplied snippets
if old_snip and old_snip in current_train_py:
    modified = current_train_py.replace(old_snip, new_snip, 1)
    return modified
```

**Why this works:**
- Immune to truncation — you're only moving a small delta
- Reproducible — the same snippet always produces the same edit
- Auditable — you can diff exactly what changed in one line

**For your codebase:** Replace `train.py` with whatever your training script is. The only requirement is that it prints two parseable lines at the end:
```
val_accuracy: X.XXXX
peak_memory_mb: X.XX
```
(Adapt the regex parsing in `run_training()` to your metric names.)

---

### 3. Self-Correcting Audit Loop

Code Agent mistakes are caught and fed back — never silently swallowed.

```
Code Agent → edit
     ↓
Syntax Check (ast.parse) — 2 seconds, zero LLM cost
     ↓
Auditor Agent (LLM) — validates no constraint violations
     ↓
If rejected → rejection reason fed back to Code Agent → retry (up to 3x)
```

The key is the **feedback loop**: the rejection reason is explicitly injected into the next Code Agent call. The agent is told *exactly* what went wrong and must fix that specific issue.

```python
rejection_feedback = (
    f"AUDIT REJECTED on attempt {attempt}: {audit_reason}\n"
    "Fix the exact issue described above in your next old_snippet/new_snippet."
)
```

This turns a dumb retry into an informed correction. In practice it resolves ~90% of initial failures.

**What the Auditor checks:**
1. Evaluation logic is not reimplemented or replaced
2. Hard time limit (`MAX_TRAIN_TIME = 300`) is unchanged
3. Required output `print` lines are present and intact
4. No obvious syntax errors
5. Training loop still checks the wall-clock time limit

---

### 4. The Ratchet (Git-Based State Machine)

Accuracy can only go up. The git history is your ratchet:

```python
if val_accuracy > best_accuracy:
    git_commit(...)          # lock in the improvement
    best_accuracy = val_accuracy
else:
    git_checkout("HEAD", train_file)   # revert to last best
```

**Why git instead of just saving the file?**
- The entire experiment history is in git log
- You can always `git log --oneline` to see every improvement
- `git reset --hard` is one command to get back to any state
- Avoids any "which version was best?" confusion

**For your codebase:** Initialize a git repo in your project directory. The orchestrator commits on improvement automatically.

---

### 5. Plateau Detection

Don't run forever. Stop when you've stopped learning.

```python
PLATEAU_WINDOW = 5       # look at last N actual training runs
PLATEAU_THRESHOLD = 0.001  # < 0.1% gain = plateau

def check_plateau(results):
    ran = [r for r in results if r["status"] in ("COMMITTED", "REVERTED", "BASELINE")]
    if len(ran) < PLATEAU_WINDOW:
        return False
    recent = [float(r["accuracy"]) for r in ran[-PLATEAU_WINDOW:]]
    return (max(recent) - min(recent)) < PLATEAU_THRESHOLD
```

**Important:** Plateau check only looks at actual training runs (COMMITTED/REVERTED/BASELINE) — it ignores AUDIT_FAILED and CODE_FAILED entries. Don't let failed code attempts mask a real plateau.

**Tuning for your task:**
- Increase `PLATEAU_WINDOW` if your metric is noisy
- Decrease `PLATEAU_THRESHOLD` if you want to keep pushing harder
- In `agent_orchestrator.py`, `MAX_NO_IMPROVEMENT_RUNS = 20` is a harder budget cap

---

### 6. Not Repeating Mistakes: The FORBIDDEN List

The system builds a **cumulative forbidden list** automatically. Two mechanisms:

**A. Dynamic summarizer** (in-context, every iteration):
```python
# In summarize_results():
forbidden = [(h, c) for h, c in hyp_counts.items()
             if c >= 3 and h not in committed_hyps]
# → injected into Lead Researcher prompt as "⛔ FORBIDDEN — DO NOT RETRY"
```

**B. Distill Beliefs** (writes to `program.md` permanently):
```python
def distill_beliefs(results):
    # Find hypotheses tried 3+ times that were never committed
    # → Writes an AUTO-DISTILLED FORBIDDEN CHANGES block to program.md
```

Since `program.md` is re-read at every iteration (ReCAP), permanently written forbidden patterns persist across sessions, restarts, and context resets. **The LLM can never "forget" a failed direction.**

**How hypotheses are fingerprinted:**
- First 50 characters of the hypothesis string become the key
- So "Change LEARNING_RATE from 1e-3 to 3e-4" and "Change LEARNING_RATE from 1e-3 to 5e-4" are treated as *different* (which is correct — they're different experiments)
- But "Change LEARNING_RATE from 1e-3 to 3e-4 to improve convergence" repeated verbatim is caught

---

### 7. Intelligent Hypothesis Generation

The Lead Researcher is not a random sampler. It reasons causally:

```
Prompt structure:
1. ReCAP (program.md goal, constraints, beliefs) — always re-injected
2. Experiment history (last 5 runs + best result + forbidden list)
3. Current train.py (so it can propose verbatim snippets)
4. Code failure context (if previous Code Agent attempt failed)

Output (JSON):
{
  "hypothesis": "one sentence — what changes and why",
  "rationale": "causal chain — why this should improve accuracy",
  "old_snippet": "verbatim lines from train.py to replace",
  "new_snippet": "the replacement",
  "risk_level": "low | medium | high",
  "expected_delta": "+0.5%"
}
```

Key design decisions:
- **One hypothesis per iteration** — isolates causality. If you change LR and batch size together and accuracy improves, you don't know which helped.
- **Mandatory causal rationale** — forces the LLM to reason, not guess. "Because N_LAYERS=3 failed 3x, I conclude the model is capacity-saturated in 300s, therefore reducing to 1 layer gives 2× more optimizer steps."
- **Extended thinking** (`"thinking": {"type": "adaptive"}`) on the Lead Researcher call gives noticeably better hypothesis quality for complex decisions.

---

### 8. Swarming the Search Space (Without Grid Search)

The system explores the hyperparameter space intelligently by maintaining **lineage tracking** in `agent_orchestrator.py`:

```python
LINEAGE_KEYWORDS = [
    "n_layers", "d_ff", "n_heads", "batch_size", "dropout",
    "learning_rate", "lr=", "warmup", "betas", "weight_decay",
    "label_smooth", "cosine", "gradient_clip", "scheduler",
]
```

**Tournament Logic:** If a keyword lineage (e.g., all `dropout` experiments) has >80% discard rate over 5 runs, that lineage is flagged for pivot. The Hypothesis Agent is forced to try a different direction.

**Pivot Threshold:** After `PIVOT_THRESHOLD = 5` consecutive non-improving runs, the agent is forced to make a structural change — not just tweak the same parameter.

**What this achieves in practice:**
- Early iterations: broad exploration (architecture, LR, regularization)
- Middle iterations: focus on what's working (optimizer, schedule)
- Late iterations: fine-tuning within proven directions
- Plateau: auto-stop or forced pivot to unexplored territory

The `program.md` `PRIORITY_RESEARCH_PATHS` section lets you seed the search with expert knowledge — the LLM uses this as a ranked list of what to try next.

---

### 9. Belief Distillation: Persistent Causal Memory

The `distill_beliefs()` function and the Belief Agent (in `agent_orchestrator.py`) maintain a **structured knowledge base** that grows with each session:

```markdown
## DISTILLED BELIEFS
- 🟢 BELIEF: Constant post-warmup LR dominates cosine decay
  EVIDENCE: iter4 (0.8119), iter11 (-0.008 vs best). Both failed.
  MECHANISM: Cosine decay wastes compute at near-zero LR when steps are scarce.
  CONFIDENCE: HIGH (2 independent trials)

## FORBIDDEN PATTERNS
- ❌ Cosine LR decay — 2x tested, always reverts
- ❌ Label smoothing — 2x tested, always reverts  
- ❌ N_LAYERS=3 — 3x tested, always reverts
```

This is not just a log — it's a **causal model** of what the system has learned. Each belief has:
- The observable fact (what happened)
- The proposed mechanism (why it happened)
- Confidence level (how many independent trials confirmed it)

The Belief Agent writes structured BELIEF/EVIDENCE/CONFIDENCE blocks, not vague summaries. This prevents the next session's Lead Researcher from "re-discovering" dead ends.

---

## Adapting for Your Codebase

### Minimal Adaptation Checklist

1. **Your training script** (`train.py` equivalent):
   - Must be runnable as `python train.py` (or `uv run train.py`)
   - Must print exactly two parseable lines at the end:
     ```
     your_metric: X.XXXX
     peak_memory_mb: X.XX
     ```
   - Must have a hard time limit constant (e.g., `MAX_TRAIN_TIME = 300`)

2. **Your `program.md`**:
   - Set `PRIMARY_GOAL` to your metric (e.g., `val_loss`, `test_accuracy`, `f1_score`)
   - Set `HARD_CONSTRAINTS` — what must never be changed (evaluation logic, time limit)
   - Fill in `RESEARCH_DIRECTIONS` with your domain knowledge of promising hyperparameters
   - Leave `FORBIDDEN` and `ESTABLISHED BELIEFS` empty — the system fills these in

3. **`autoresearch.py` changes needed**:
   - `TRAIN_FILE = "train.py"` → your script name
   - `run_training()` regex → match your metric name
   - Comparison: `if val_accuracy > best_accuracy` → `if metric > best_metric` (or `<` for loss)
   - `audit_code_agent()` prompt → update the 6 audit rules to match your constraints

4. **`.env` file**:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

### What You Do NOT Need

- A `prepare.py` / data harness — that's specific to our SST-2 setup
- BPE tokenizer pre-training — that's domain-specific
- `uv` package manager — regular `python` works fine, just change the subprocess call

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    program.md                        │
│   (goal · constraints · beliefs · forbidden paths)  │
│             Re-read EVERY iteration (ReCAP)          │
└─────────────────────┬───────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │   PERCEIVE              │
         │   read results.tsv      │
         │   read train.py         │
         │   check plateau         │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │   REASON                │
         │   Lead Researcher       │
         │   → one hypothesis      │
         │   → old_snippet         │
         │   → new_snippet         │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │   ACT                   │
         │   Code Agent            │◄─── rejection_feedback (loop)
         │   → str.replace edit    │
         │   → ast.parse gate      │
         │   → Auditor validates   │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │   EXECUTE               │
         │   subprocess train.py   │
         │   parse metric output   │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │   REFLECT               │
         │   improved? → git commit│
         │   regressed? → git revert│
         │   distill_beliefs()     │
         │   check plateau         │
         └─────────────────────────┘
```

---

## Key Prompts

### Lead Researcher Prompt Structure

```
# ReCAP — PRIMARY RESEARCH GOAL (re-injected every iteration)
{program.md content}

# Your Role: Lead Researcher (Iteration N)
Analyse the experiment history and propose exactly ONE hypothesis.

## Experiment History
{last 5 runs, best result, forbidden list}

## Current train.py
{full file content}

## Instructions
1. Do NOT repeat any hypothesis already attempted.
2. Propose ONE measurable change with exact values.
3. Prefer SIMPLE changes (single constant like LR, DROPOUT, N_LAYERS).
4. Never modify evaluation logic or time limit.

## Output — return ONLY valid JSON:
{
  "hypothesis": "...",
  "rationale": "...",
  "old_snippet": "verbatim from train.py",
  "new_snippet": "the replacement",
  "risk_level": "low|medium|high",
  "expected_delta": "+0.5%"
}
```

### Code Agent Prompt Structure (Marker-Based, Not JSON)

```
<<<OLD>>>
<exact verbatim lines from train.py — character-for-character copy>
<<<NEW>>>
<replacement lines — valid Python, same indentation>
<<<END>>>
```

**Why markers instead of JSON for code?** JSON requires escaping backslashes, quotes, and indentation. Code snippets embedded in JSON are fragile. Custom markers (`<<<OLD>>>`, `<<<NEW>>>`, `<<<END>>>`) are immune to escaping issues.

### Auditor Prompt Structure

```
## Reject if ANY of these fail:
1. Does NOT reimport/replicate evaluate() from harness
2. Hard time limit constant unchanged
3. Final two print lines intact (metric + memory)
4. No syntax errors
5. Training loop still checks wall-clock time

Return JSON: { "approved": true/false, "reason": "...", "issues": [] }
```

---

## Common Failure Modes and Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| LLM hallucinates `old_snippet` | "old_snippet not found verbatim" | Ensure train.py is in full in the prompt; use the fast-path where LR itself proposes snippets |
| Code Agent repeats same mistake | Audit rejects 3x, CODE_FAILED | Feed rejection reason explicitly in `rejection_feedback` string |
| Plateau too early | Stops after 5 runs | Increase `PLATEAU_WINDOW` or `PLATEAU_THRESHOLD` |
| LLM forgets forbidden paths | Re-proposes old ideas | Ensure `distill_beliefs()` runs after every iteration; check `program.md` is being re-read |
| git conflicts after revert | Merge error on `git checkout` | Use `git checkout HEAD -- train.py` (single file), not `git reset --hard` |
| Metric not parsed | `val_accuracy: 0.0000` in log | Check your `print(f"your_metric: {val:.4f}")` format and update the regex in `run_training()` |

---

## What Makes This Different from Optuna / Ray Tune

| Feature | Optuna / Ray Tune | AutoResearch |
|---------|------------------|--------------|
| Search strategy | Statistical (TPE, random) | Causal reasoning (LLM) |
| Handles code changes | No (config only) | Yes (any code edit) |
| Learns from failures | Partially (prunes) | Explicitly (FORBIDDEN list + beliefs) |
| Explains decisions | No | Yes (rationale in every hypothesis) |
| Architecture changes | No | Yes (add layers, change activations, etc.) |
| Persistent memory | Session-only | `program.md` + git history |
| Constraint enforcement | Via config bounds | Auditor Agent validates code |

The tradeoff: slower per-iteration (LLM call overhead ~5–15s) but explores a much richer space than any grid or bayesian optimizer. Best suited for problems where you've exhausted obvious hyperparameter tuning and want to explore structural code changes.

---

## Results from SST-2 Run

| Iteration | Change | Accuracy | Status |
|-----------|--------|----------|--------|
| 0 | Baseline | 0.8050 | BASELINE |
| 11 | WEIGHT_DECAY = 0.1 | 0.8165 | COMMITTED |
| 23 | AdamW betas=(0.9, 0.95) | 0.8222 | **BEST** |
| ... | N_LAYERS=3 (×3 tries) | 0.8050–0.8108 | REVERTED × 3 → FORBIDDEN |
| ... | Cosine LR (×2 tries) | regressed | REVERTED × 2 → FORBIDDEN |

After 30+ iterations, the system correctly identified: the model is data/signal-bottlenecked (not capacity-bottlenecked), and the only productive axis was optimizer quality (betas, weight decay, grad clip, batch size chain).

---

## Quick Start for a New Project

```bash
# 1. Copy autoresearch.py and agent_orchestrator.py to your project
# 2. Write your program.md (goal + constraints + research directions)
# 3. Ensure your train.py prints "your_metric: X.XXXX" at the end
# 4. Initialize git
git init && git add -A && git commit -m "initial"

# 5. Set API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 6. Run
python autoresearch.py        # basic 4-agent version
# OR
python agent_orchestrator.py  # full 8-agent version with pivot logic
```

That's it. The loop runs until plateau is detected or you hit `Ctrl-C`.
