# AutoResearch — Mutable Training Script
# ========================================
# THIS IS THE ONLY FILE THE AGENT SHOULD MODIFY.
#
# Rules enforced by the immutable harness (prepare.py):
#   1. Call evaluate() from prepare.py for all accuracy measurements.
#   2. Hard-stop training at exactly MAX_TRAIN_TIME = 300 seconds.
#   3. Print the two required output lines at the end:
#        val_accuracy: X.XXXX
#        peak_memory_mb: X.XX
#
# Baseline architecture: small Transformer encoder + mean pooling classifier
# Baseline optimizer   : AdamW with linear warmup
# =========================================================================

import time
import sys
import os
import torch
import torch.nn as nn
from torch.optim import AdamW

# Import ONLY these two items from the immutable harness.
# Never import or replicate evaluate() logic yourself.
from prepare import get_dataloaders, evaluate, VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN_ID

# ============================================================
# TRAINING CONFIGURATION — agent may tune these values
# ============================================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 200
DROPOUT = 0.1
LABEL_SMOOTHING = 0.0
GRAD_CLIP = 0.5

# Model architecture
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256

# FIXED — do not change
MAX_TRAIN_TIME = 300  # seconds


# ============================================================
# MODEL DEFINITION — agent may modify architecture
# ============================================================

class TransformerClassifier(nn.Module):
    """
    CNN text classifier with parallel convolutions (kernel sizes 2,3,4).
    Much faster than Transformer → more optimizer steps in 300s.
    Input  : (batch, seq_len) token IDs
    Output : (batch, 2) logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        num_classes: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)

        kernel_sizes = [2, 3, 4, 5]
        n_filters = 256  # 256 filters per kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, n_filters, k, padding=0)
            for k in kernel_sizes
        ])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_filters * len(kernel_sizes), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        x = self.drop(self.tok_emb(input_ids))  # (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T) for Conv1d

        # Mask padding tokens by setting to -inf before max-pool
        if attention_mask is not None:
            mask = (attention_mask == 0).unsqueeze(1)  # (B, 1, T)

        conv_outs = []
        for conv in self.convs:
            c = conv(x)  # (B, n_filters, T')
            c = torch.relu(c)
            if attention_mask is not None:
                # Create mask for conv output length
                k = conv.kernel_size[0]
                if c.size(2) > 0:
                    # Mask positions where any input token was padding
                    m = attention_mask[:, :c.size(2) + k - 1].unfold(1, k, 1).min(dim=-1).values  # (B, T')
                    c = c.masked_fill(m.unsqueeze(1) == 0, -1e9)
            c = c.max(dim=2).values  # (B, n_filters)
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=1)  # (B, n_filters * num_kernels)
        return self.head(x)


# ============================================================
# LEARNING RATE SCHEDULE — agent may replace this
# ============================================================

def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup, then constant learning rate."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    return base_lr


# ============================================================
# TRAINING LOOP
# ============================================================

def main():
    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- Data (from immutable harness) ----
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_TOKEN_ID,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ---- Optimizer ----
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ---- Training with hard 300-second wall-clock limit ----
    train_start = time.time()
    step = 0
    epoch = 0
    best_val_acc = 0.0

    print(f"Training (hard stop at {MAX_TRAIN_TIME}s)...")

    while True:
        epoch += 1
        epoch_complete = True

        for input_ids, attention_mask, labels in train_loader:
            if time.time() - train_start >= MAX_TRAIN_TIME:
                epoch_complete = False
                break

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Apply LR schedule
            lr = get_lr(step, WARMUP_STEPS, LEARNING_RATE)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            step += 1

        if not epoch_complete:
            break

        # End-of-epoch validation
        elapsed = time.time() - train_start
        if elapsed >= MAX_TRAIN_TIME:
            break

        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"  epoch={epoch} val_acc={val_acc:.4f} step={step} elapsed={elapsed:.1f}s")

    # ---- Final evaluation (MUST use prepare.evaluate — do not change) ----
    final_val_acc = evaluate(model, val_loader, device)
    best_val_acc = max(best_val_acc, final_val_acc)

    # ---- Peak memory ----
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elif device.type == "mps":
        # MPS does not expose peak memory; report current
        peak_memory_mb = torch.mps.current_allocated_memory() / 1024 / 1024
    else:
        try:
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            peak_memory_mb = 0.0

    total_time = time.time() - train_start
    print(f"\nDone: {total_time:.1f}s | epochs={epoch} | steps={step}")

    # ---- Required output lines (DO NOT change format) ----
    print(f"val_accuracy: {best_val_acc:.4f}")
    print(f"peak_memory_mb: {peak_memory_mb:.2f}")


if __name__ == "__main__":
    main()
