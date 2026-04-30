"""
prepare.py - IMMUTABLE HARNESS
===============================
DO NOT MODIFY AFTER FIRST RUN.

This file:
1. Downloads SST-2 dataset from HuggingFace
2. Trains a custom BPE tokenizer on the training corpus
3. Pre-tokenizes all data splits and saves as binary .npz files
4. Defines the FIXED evaluate() function used by all training runs

The evaluation metric is locked here to prevent the agent from "cheating"
by changing how accuracy is computed to show fake improvements.
"""

import sys
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
TRAIN_DATA_PATH = DATA_DIR / "train.npz"
VAL_DATA_PATH = DATA_DIR / "val.npz"

VOCAB_SIZE = 8192      # BPE vocabulary size
MAX_SEQ_LEN = 128      # Maximum token sequence length
PAD_TOKEN_ID = 0       # Padding token index


def prepare_data():
    """
    One-time setup: download SST-2, train tokenizer, pre-tokenize data.
    Call this once before starting the research loop.
    """
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("prepare.py — One-time data preparation")
    print("=" * 60)

    print("\n[1/3] Loading SST-2 dataset from HuggingFace...")
    dataset = load_dataset("glue", "sst2")
    print(f"      Train: {len(dataset['train'])} examples")
    print(f"      Validation: {len(dataset['validation'])} examples")

    # Train a custom BPE tokenizer on the training corpus
    print("\n[2/3] Training BPE tokenizer (vocab_size={})...".format(VOCAB_SIZE))
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
        min_frequency=2,
    )

    train_texts = dataset["train"]["sentence"]
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"      Tokenizer saved to {TOKENIZER_PATH}")

    # Pre-tokenize and save all splits as numpy arrays
    print("\n[3/3] Pre-tokenizing and saving data splits...")

    def tokenize_and_save(split_name: str, save_path: Path):
        data = dataset[split_name]
        texts = data["sentence"]
        labels = data["label"]

        input_ids_list = []
        attention_masks_list = []

        for text in texts:
            encoding = tokenizer.encode(text)
            ids = encoding.ids[:MAX_SEQ_LEN]
            pad_len = MAX_SEQ_LEN - len(ids)
            padded_ids = ids + [PAD_TOKEN_ID] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            input_ids_list.append(padded_ids)
            attention_masks_list.append(mask)

        np.savez(
            save_path,
            input_ids=np.array(input_ids_list, dtype=np.int32),
            attention_mask=np.array(attention_masks_list, dtype=np.int32),
            labels=np.array(labels, dtype=np.int32),
        )
        print(
            f"      Saved '{split_name}' ({len(labels)} examples) → {save_path}"
        )

    tokenize_and_save("train", TRAIN_DATA_PATH)
    tokenize_and_save("validation", VAL_DATA_PATH)

    print("\n✅ Data preparation complete. DO NOT RUN prepare.py AGAIN.")
    print("   DO NOT MODIFY prepare.py — it is the immutable harness.")


def get_dataloaders(batch_size: int = 32):
    """
    Load pre-tokenized SST-2 data and return PyTorch DataLoaders.
    Called by train.py at the start of each training run.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if not TRAIN_DATA_PATH.exists() or not VAL_DATA_PATH.exists():
        print("ERROR: Data files not found. Run `uv run prepare.py` first.")
        sys.exit(1)

    train_data = np.load(TRAIN_DATA_PATH)
    val_data = np.load(VAL_DATA_PATH)

    train_dataset = TensorDataset(
        torch.tensor(train_data["input_ids"], dtype=torch.long),
        torch.tensor(train_data["attention_mask"], dtype=torch.long),
        torch.tensor(train_data["labels"], dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_data["input_ids"], dtype=torch.long),
        torch.tensor(val_data["attention_mask"], dtype=torch.long),
        torch.tensor(val_data["labels"], dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


# ============================================================
# FIXED EVALUATION FUNCTION — DO NOT MODIFY
# This function is the single source of truth for accuracy.
# ============================================================

def evaluate(model, val_loader, device) -> float:
    """
    FIXED evaluation function. Returns validation accuracy on SST-2.

    This function is LOCKED. The agent must NOT modify it, import it
    differently, or create alternative evaluation functions. All
    accuracy numbers reported in results.tsv come exclusively from here.
    """
    import torch

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    prepare_data()

    # Sanity-check the saved files
    print("\n--- Verification ---")
    train_data = np.load(TRAIN_DATA_PATH)
    val_data = np.load(VAL_DATA_PATH)
    print(f"Train input_ids shape : {train_data['input_ids'].shape}")
    print(f"Val   input_ids shape : {val_data['input_ids'].shape}")
    print(f"Train label counts    : {np.bincount(train_data['labels'])}")
    print(f"Val   label counts    : {np.bincount(val_data['labels'])}")
    print("\nprepare.py ✅ — Immutable harness ready.")
