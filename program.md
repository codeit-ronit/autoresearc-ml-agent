# AutoResearch Research Program
> This file is the "Human Command" — the Director's brief for the AI research team.
> The Lead Researcher reads this at the start of EVERY iteration (ReCAP pattern).

---

## PRIMARY GOAL
**Maximize validation accuracy on SST-2 binary sentiment classification within exactly 5 minutes (300 seconds) of compute per experiment.**

- **Primary metric**: `val_accuracy` on the SST-2 validation set (higher = better)
- **Secondary metric**: `peak_memory_mb` (lower = more hardware-efficient)

---

## HARD CONSTRAINTS (Never Violate)

1. **NEVER modify `prepare.py`** — It is the immutable harness. Touching it is cheating.
2. **NEVER change the evaluation logic** — `evaluate()` from `prepare.py` is the single source of truth.
3. **NEVER modify `MAX_TRAIN_TIME = 300`** — Each run must be exactly 5 minutes.
4. **ONE change per iteration** — Do not combine multiple hypotheses. Isolate each change.
5. **Do not repeat failed experiments** — Check `results.tsv` before proposing a hypothesis.

---

## DECISION CRITERIA

| Result | Action |
|--------|--------|
| `val_accuracy` improved vs best | `git commit` — this is the new frontier |
| `val_accuracy` same or lower | `git reset --hard HEAD` — revert train.py |

---

## RESEARCH DIRECTIONS TO EXPLORE

### Architecture
- Increase model depth (N_LAYERS: 2 → 3 or 4)
- Widen the model (D_MODEL: 128 → 256)
- Increase feedforward dimension (D_FF: 256 → 512)
- Adjust attention heads (N_HEADS: 4 → 8)
- Add a CLS token for classification instead of mean pooling
- Try learned positional embeddings vs sinusoidal

### Optimization
- Learning rate schedule: cosine decay, warmup + cosine, one-cycle
- AdamW hyperparameters: betas=(0.9, 0.95), eps=1e-8
- Weight decay tuning (0.01 → 0.1)
- Gradient clipping threshold (1.0 → 0.5)
- Batch size (32 → 64 → 128)

### Regularization
- Dropout tuning (0.1 → 0.2 → 0.3)
- Label smoothing (0.0 → 0.1)
- Layer dropout / stochastic depth

### Training Dynamics
- Gradient accumulation (effective larger batch)
- Mixed precision training (torch.autocast)
- Learning rate warmup steps (200 → 500)

---

## STOPPING CRITERION
Stop the loop when `val_accuracy` improves by **less than 0.1% over 5 consecutive runs** (plateau detected).

---

## ANTI-PATTERNS TO AVOID
- Do not make the model so large it overfits in 300 seconds
- Do not change `BATCH_SIZE` without accounting for learning rate scaling
- Do not propose identical changes to ones already in `results.tsv`
- Do not import from `prepare.py` beyond `get_dataloaders`, `evaluate`, `VOCAB_SIZE`, `MAX_SEQ_LEN`, `PAD_TOKEN_ID`

---

## ESTABLISHED RESEARCH BELIEFS
> Auto-distilled from experiment failures. **Search paths marked FORBIDDEN must NOT be re-attempted.**

### FORBIDDEN — Capacity Trap (3+ failures, always degrades accuracy in 300s window)
- ❌ `N_LAYERS = 3` — Tried iters 6, 12, 18: scored 0.8050, 0.8108, 0.8050. Deeper model underfits within 300s.
- ❌ `D_FF = 512` — Tried iters 8, 20: scored 0.8062, 0.8073. Wider FFN overfits / converges slower.
- ❌ `N_HEADS = 8` — Tried iter 15: scored 0.8050. More heads degrades with same D_MODEL=128.
- ❌ `BATCH_SIZE = 64` — Tried iters 10, 17: scored 0.8085, 0.7993. Larger batch fewer optimizer steps → worse.
- ❌ `LEARNING_RATE > 2e-4` — Tried iters 9 (3e-4→0.8050), 21 (3e-4→0.8016). Higher LR overshoots.
- ❌ `DROPOUT = 0.2` with current config — Tried iters 7, 13, 19: 0.8119, 0.8131, 0.8085. Marginal at best.

### COMMITTED — Proven improvements
- ✅ `WEIGHT_DECAY = 0.1` (from 0.01) → 0.8165 at iter 11.
- ✅ `AdamW betas=(0.9, 0.95)` (from default (0.9, 0.999)) → 0.8222 at iter 23. **Current best.**

### BELIEF 1 — Model Capacity is Saturated for 300s Budget
All capacity-increase attempts (N_LAYERS, D_FF, N_HEADS, D_MODEL) degrade accuracy.
The 300s budget is too short for larger models to converge. **Capacity scaling is CLOSED.**
Shift focus to: Optimizer Steps (throughput) and Learning Quality (LR schedule, regularization).

### BELIEF 2 — Dropout > 0.1 is Detrimental for SST-2
Tried 3 times (iters 7, 13, 19). Scores: 0.8119, 0.8131, 0.8085. All below best.
The model does NOT benefit from more dropout on SST-2. **DROPOUT increase is FORBIDDEN.**

### PRIORITY RESEARCH PATHS — ordered by expected impact
1. **🎯 SIMPLIFICATION PASS (try first)**: `N_LAYERS = 1` — Because 2-layer model is at the capacity ceiling, reducing to 1 layer gives ~50% more optimizer steps in 300s. A sharper minima may be reachable. Hypothesis: "1-layer model + current WD+betas stack → more steps → better convergence."
   - Implementation: change `N_LAYERS = 2` → `N_LAYERS = 1`

2. **Parameter-group LR (Muon-style)**: Give embedding/bias params LOWER lr, weight matrices HIGHER lr. Rationale: large weight matrices benefit from aggressive updates; embeddings need stability.
   - Implementation: split optimizer into two param groups — `{"params": embedding_params, "lr": LR*0.1}` and `{"params": weight_params, "lr": LR}`

3. **Cosine LR schedule**: After linear warmup, decay to 0 with cosine. Never tested due to old truncation bug (now fixed with surgical edit). This is the single highest-expected-value unexplored change.
   - Implementation: replace `get_lr()` to return `lr * 0.5 * (1 + cos(pi * (step-warmup) / (max_steps-warmup)))` for step > warmup_steps

4. **Lower LR**: try 1e-4 — slower but more stable convergence at this model size within 300s.
5. **Gradient clipping**: reduce from 1.0 to 0.5 — stabilizes training with betas=(0.9, 0.95).
6. **BATCH_SIZE = 16**: more gradient updates per second, especially helpful for 1-layer model.

### CAUSAL REASONING FRAMEWORK
When proposing a hypothesis, explain the causal chain:
> "Because [observed failure/success], I conclude [causal mechanism], therefore changing [X→Y] should [expected effect on accuracy via specific mechanism]."

The fundamental constraint is: **Accuracy = f(Model Capacity × Optimizer Steps × Learning Quality)**
- Capacity increases FAILED → shift to Optimizer Steps or Learning Quality
- In 300s: 1-layer model ≈ 3× more steps/epoch; 2-layer ≈ 2× more steps/epoch than 3-layer


## BELIEF AGENT UPDATE (2026-03-26 16:06)
### Distilled Forbidden Patterns:
- ❌ Label smoothing consistently hurts: tested twice, both reverted
- ❌ Cosine decay tested twice post-fix, still reverts—likely overtunes decay rate
### New Insights:
- ✅ Regularization techniques (label smoothing, cosine decay, lower LR) all degrade accuracy, suggesting the 1-layer model is capacity-starved not overfitting—optimization should maximize effective learning per step rather than regularize. The two COMMITTED changes (grad clip 0.5, batch 16) both increased update frequency or stability without reducing effective capacity, confirming the bottleneck is learning quality per step not generalization.

**Next Strategy Recommendation**: Try parameter-group LR (Muon-style: embeddings at LR*0.1, weight matrices at LR*1.0) since it targets learning quality per step—the only axis with proven gains—without adding any regularization that has repeatedly failed.


## BELIEF AGENT UPDATE (2026-03-26 16:17)
### Distilled Forbidden Patterns:
- ❌ Reducing model capacity (N_LAYERS 2→1) consistently degrades accuracy
- ❌ Re-testing previously reverted hypotheses yields same or worse results
### New Insights:
- ✅ The only COMMITTED changes (grad_clip 0.5, batch_size 16, LR halving to match batch) all improved learning quality per step without reducing capacity or adding regularization—confirming the model is capacity-starved and the sole productive axis is maximizing effective gradient signal per update, not controlling overfitting.

**Next Strategy Recommendation**: Try differential learning rates (lower LR for embeddings, higher for attention/MLP) to maximize per-step learning quality—the only axis with proven gains—without any regularization.


## BELIEF AGENT UPDATE (2026-03-26 16:25)
### Distilled Forbidden Patterns:
- ❌ Reducing warmup steps below 200 always degrades accuracy
- ❌ Re-applying failed hypotheses with minor tweaks still fails
### New Insights:
- ✅ LR halving (iter19) only worked AFTER batch halving (iter14)—hyperparameter changes succeed only when coupled with their prerequisite context, suggesting sequential co-adaptation matters more than isolated tuning.

**Next Strategy Recommendation**: Try increasing N_LAYERS from 2 to 3 (or widening D_MODEL) to directly address the capacity bottleneck, since all capacity-neutral optimization gains are plateauing around 0.823.


## BELIEF AGENT UPDATE (2026-03-26 16:44)
### Distilled Forbidden Patterns:
- ❌ Reducing WEIGHT_DECAY below 0.1 degrades accuracy
- ❌ Cosine LR decay consistently underperforms constant post-warmup LR
### New Insights:
- ✅ All three committed improvements (iter7→14→19) form a causal chain: tighter grad_clip enabled smaller batch, which enabled LR halving—suggesting future gains require building on this exact chain rather than independent exploration.

**Next Strategy Recommendation**: Increase model capacity by widening D_MODEL (e.g., 128→192) since the optimization axis is exhausted at 0.823 and all regularization/schedule changes fail against the capacity ceiling.


## BELIEF AGENT UPDATE (2026-03-26 16:52)
### Distilled Forbidden Patterns:
- ❌ Increasing D_MODEL from 128 to 192 fails to improve accuracy
- ❌ Label smoothing consistently hurts regardless of training context
### New Insights:
- ✅ Capacity increases (both wider D_MODEL iter34 and shallower N_LAYERS iter10) degrade accuracy, suggesting the model is not capacity-bottlenecked but rather data/token-bottlenecked—optimization gains plateau because the training signal is exhausted, not because the model is too small.

**Next Strategy Recommendation**: Increase effective training data via augmentation, longer sequences, or curriculum strategies rather than model capacity or hyperparameter tuning, since both capacity expansion and optimization refinement have hit the same ~0.823 ceiling.


## DISTILLED BELIEFS — 2026-03-26 17:00
> Model state: **capacity_saturated**

### Causal Beliefs
- 🟢 **Hyperparameter changes succeed only when their prerequisites are already in place, forming causal chains rather than independent improvements.**  
  *Evidence*: LR halving (iter19, +0.0012) succeeded only after batch halving (iter14, +0.0034), while LR halving alone (iter6, 0.8131) failed against the iter7 baseline. Grad clip tightening (iter7) preceded and enabled batch halving.  
  *Mechanism*: Smaller batches increase gradient noise; tighter grad clipping (iter7) pre-stabilized gradients, making the model robust to that noise. The halved batch doubled optimizer steps, and only then did halving LR become beneficial because the step-count/LR ratio was restored to an optimal regime. Each change reshapes the loss landscape traversal in ways that unlock the next change.
- 🟢 **This task is data/signal-bottlenecked, not capacity-bottlenecked: both increasing and decreasing model capacity degrade accuracy.**  
  *Evidence*: Reducing N_LAYERS 2→1 (iter10, -0.0172; iter3, 0.8165 pre-baseline). Increasing D_MODEL 128→192 (iter34, -0.0080). Neither direction helps.  
  *Mechanism*: The 2-layer 128-dim model already has sufficient representational capacity for the training signal available. Making it larger overfits or wastes compute on redundant parameters; making it smaller underfits. The ceiling at ~0.823 reflects exhaustion of learnable signal from the data, not architectural limitation.
- 🟢 **Constant post-warmup LR strictly dominates cosine decay in this training regime.**  
  *Evidence*: Cosine decay failed twice: iter4 (0.8119, pre-baseline) and iter11 (-0.0080 vs committed baseline). Both attempts were in different baseline contexts.  
  *Mechanism*: With limited training steps (constrained compute budget), cosine decay spends too many steps at very low LR where the model barely updates, effectively wasting compute. Constant LR maintains learning pressure throughout, which matters more when total steps are scarce.
- 🟢 **Label smoothing consistently degrades accuracy in this setting regardless of training context.**  
  *Evidence*: iter5 (0.8131, pre-baseline) and iter12 (-0.0126 vs committed baseline). Tested at two different baseline performance levels, both failed.  
  *Mechanism*: The model is not overconfident—it's data-limited. Label smoothing reduces the effective signal per example by softening targets, which is harmful when the bottleneck is insufficient training signal rather than overconfident predictions.

### FORBIDDEN_PATTERNS
- ❌ `Applying cosine LR decay (to zero or near-zero) in place of constant post-warmup LR` — Wastes scarce compute budget on near-zero LR steps; tested twice in different contexts, consistently underperforms constant LR.
- ❌ `Adding label smoothing (0.1 or similar)` — Reduces effective training signal per example in a data-bottlenecked regime; failed in two distinct baseline contexts.
- ❌ `Reducing warmup steps below 200` — Introduces early training instability with small batch + low LR; monotonically worsens with larger reductions.
- ❌ `Reducing weight decay below 0.1` — Removes necessary regularization in data-limited setting; WD/LR ratio logic doesn't apply when regularization is the dominant role.
- ❌ `Changing model capacity in either direction (fewer/more layers, wider/narrower)` — Current 2-layer 128-dim architecture is well-matched to available data signal; changes in either direction degrade accuracy.

### PROMISING_DIRECTIONS
- 🔬 **Data augmentation strategies: token-level dropout, synonym replacement, or sequence-level perturbations** — The model is data/signal-bottlenecked (not capacity-bottlenecked). Augmentation increases effective training signal without changing architecture. All optimization and capacity axes are exhausted at ~0.823.
- 🔬 **Increasing maximum sequence length to capture more context per example** — If sequences are being truncated, extending them provides more training signal per example, directly addressing the data bottleneck without architectural changes.
- 🔬 **Multi-head attention modifications: increasing number of heads while keeping D_MODEL fixed at 128** — This changes how capacity is utilized (more diverse attention patterns) without changing total capacity, potentially extracting more signal from existing data.

**→ NEXT BEST STRATEGY**: Implement embedding-level data augmentation (e.g., token dropout with probability 0.1 or embedding mixup) to increase effective training signal, since the model architecture and all hyperparameter axes have converged at the ~0.823 data-signal ceiling.
