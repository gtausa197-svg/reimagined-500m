"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD — Training Script for 500M Model                  ║
║                                                                        ║
║  Usage:                                                                ║
║      python train_nord_500m.py                                         ║
║                                                                        ║
║  Hardware requirements:                                                ║
║      - NVIDIA L40 (48GB) or 4× RTX 3090 (24GB each)                   ║
║      - ~20-30GB VRAM for training with batch_size=8, seq_len=512       ║
║                                                                        ║
║  Changes from 144M:                                                    ║
║      - d_model: 512 → 1024                                            ║
║      - n_layers: 6 → 12                                               ║
║      - d_ff: 1024 → 4096                                              ║
║      - n_heads: 8 → 16                                                ║
║      - batch_size: 4 → 2 (compensated with grad_accum=4)             ║
║      - lr: 5e-4 → 3e-4 (lower for stability at scale)                ║
║      - warmup: 500 → 1000 steps                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Requirements:
    pip install torch transformers lmdb tqdm
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from nord_core_500m import NordConfig, NordModel


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

class NordTokenizer:
    def __init__(self, cfg: NordConfig):
        from transformers import AutoTokenizer

        print(f"  [*] Loading Llama-3.2 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_id, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_len = cfg.max_seq_len
        self.vocab_size = self.tokenizer.vocab_size
        if cfg.vocab_size < self.vocab_size:
            cfg.vocab_size = self.vocab_size

        print(f"  [✓] Tokenizer ready (vocab={self.vocab_size:,})")

    def encode(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text, return_tensors="pt",
            max_length=self.max_len, truncation=True, padding="max_length",
        )
        return enc.input_ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id


# ─────────────────────────────────────────────────────────────────────────────
# LMDB DATASET
# ─────────────────────────────────────────────────────────────────────────────

class LMDBDataset(Dataset):
    def __init__(self, db_path: str, max_seq_len: int):
        import lmdb
        self.db_path = db_path
        self.max_seq_len = max_seq_len
        self._env = None

        env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            raw = txn.get(b"__len__")
            self.length = struct.unpack("<Q", raw)[0]
        env.close()
        print(f"  [✓] LMDB: {self.length:,} samples")

    def _get_env(self):
        if self._env is None:
            import lmdb
            self._env = lmdb.open(
                self.db_path, readonly=True, lock=False,
                readahead=True, meminit=False, max_readers=64,
            )
        return self._env

    def __len__(self): return self.length

    def __getitem__(self, idx):
        env = self._get_env()
        with env.begin(write=False) as txn:
            raw = txn.get(f"sample_{idx:010d}".encode())
        ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
        S = self.max_seq_len
        return ids[:S] if ids.shape[0] >= S else F.pad(ids, (0, S - ids.shape[0]))


def build_lmdb(jsonl_path: str, db_path: str, tokenizer: NordTokenizer,
               max_seq_len: int, map_size_gb: float = 50.0):
    import lmdb
    from tqdm import tqdm

    print(f"\n  [*] Building LMDB database...")
    print(f"      Source:  {jsonl_path}")
    print(f"      Target:  {db_path}")

    print(f"  [*] Counting lines...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    print(f"      Found: {n_lines:,} lines")

    env = lmdb.open(db_path, map_size=int(map_size_gb * (1024 ** 3)))
    count = 0
    total_tokens = 0

    txn = env.begin(write=True)
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=n_lines, desc="  Tokenizing", unit=" doc"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = obj.get("text") or obj.get("content") or obj.get("passage", "")
                if len(text) < 30:
                    continue

                ids = tokenizer.encode(text).squeeze(0)
                non_pad = (ids != tokenizer.pad_id).sum().item()
                if non_pad < 10:
                    continue

                txn.put(f"sample_{count:010d}".encode(),
                        ids.to(torch.int32).numpy().tobytes())
                count += 1
                total_tokens += non_pad

                if count % 50_000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    print(f"      ... {count:,} samples, {total_tokens/1e6:.1f}M tokens")

        txn.put(b"__len__", struct.pack("<Q", count))
        txn.put(b"__total_tokens__", struct.pack("<Q", total_tokens))
        txn.commit()
    except BaseException:
        txn.abort()
        raise

    env.close()

    db_size = sum(f.stat().st_size for f in Path(db_path).rglob("*") if f.is_file())
    print(f"\n  [✓] LMDB ready!")
    print(f"      Samples:  {count:,}")
    print(f"      Tokens:   {total_tokens:,} ({total_tokens/1e6:.1f}M)")
    print(f"      On disk:  {db_size / (1024**3):.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# LR SCHEDULE — Fixed LR after warmup (proven effective for Nord)
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: NordConfig) -> float:
    """Warmup then fixed LR — no cosine decay (learned from 144M training)."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr  # fixed after warmup


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, save_dir: str, keep_last: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, model, optimizer, scaler, step, loss, cfg):
        path = self.save_dir / f"nord_500m_step_{step:07d}.pt"
        torch.save({
            "step": step, "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": {k: v for k, v in cfg.__dict__.items()
                       if not k.startswith("_") and k != "dtype"},
        }, path)

        latest = self.save_dir / "nord_500m_latest.pt"
        if latest.exists():
            latest.unlink()
        shutil.copy2(path, latest)

        ckpts = sorted(self.save_dir.glob("nord_500m_step_*.pt"), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:max(0, len(ckpts) - self.keep_last)]:
            old.unlink()

        print(f"  [💾] Saved: {path.name} (loss={loss:.4f})")

    def load(self, model, optimizer, scaler, device) -> int:
        latest = self.save_dir / "nord_500m_latest.pt"
        if not latest.exists():
            ckpts = sorted(self.save_dir.glob("nord_500m_step_*.pt"))
            latest = ckpts[-1] if ckpts else None
        if latest is None:
            return 0

        print(f"  [*] Resuming from: {latest.name}")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt["step"]
        print(f"  [✓] Resumed at step {step:,} (loss={ckpt.get('loss', '?')})")
        return step

    def save_final(self, model, cfg):
        path = self.save_dir / "nord_500m_final.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {k: v for k, v in cfg.__dict__.items()
                       if not k.startswith("_") and k != "dtype"},
        }, path)
        print(f"  [⭐] Final model: {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset_path: str, model_dir: str):
    # ── Config — 500M ──
    cfg = NordConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        # 500M architecture
        d_model=1024,
        n_heads=16,
        n_layers=12,
        d_ff=4096,
        n_clusters=128,
        # Temporal
        T=8,
        T_slow=2,
        persistent_mem=False,
        max_seq_len=512,
        # Training — adjusted for 500M
        batch_size=8,
        grad_accum=4,
        lr=4e-4,           # lower lr for larger model
        warmup_steps=1000,  # longer warmup
        max_steps=200_000,
        save_every=1000,
        log_every=10,
    )

    print()
    print("═" * 60)
    print("  PROJECT NORD v3.5 — 500M SNN Model Training")
    print("═" * 60)
    print(f"  GPU:            {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "  CPU mode")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  VRAM:           {vram:.1f} GB")
    print(f"  Model:          d={cfg.d_model}, layers={cfg.n_layers}, ff={cfg.d_ff}, heads={cfg.n_heads}")
    print(f"  Clusters:       {cfg.n_clusters}")
    print(f"  Effective batch: {cfg.batch_size} × {cfg.grad_accum} = {cfg.batch_size * cfg.grad_accum}")
    print(f"  LR:             {cfg.lr} (fixed after {cfg.warmup_steps} warmup steps)")
    print(f"  Max steps:      {cfg.max_steps:,}")
    print(f"  Dataset:        {dataset_path}")
    print(f"  Model dir:      {model_dir}")
    print()

    # ── Tokenizer ──
    tokenizer = NordTokenizer(cfg)

    # ── LMDB ──
    db_path = str(Path(dataset_path).with_suffix("")) + "_lmdb"
    if not Path(db_path).exists():
        build_lmdb(dataset_path, db_path, tokenizer, cfg.max_seq_len)

    dataset = LMDBDataset(db_path, cfg.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True,
    )

    # ── Model ──
    print(f"\n  [*] Building 500M model...")
    model = NordModel(cfg).to(cfg.device)
    print(f"  [✓] {model.count_params()}")

    # VRAM estimate
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"  [*] Model VRAM: {allocated:.2f} GB")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.dtype == torch.float16))

    # ── Checkpoints ──
    ckpt_mgr = CheckpointManager(model_dir)
    start_step = ckpt_mgr.load(model, optimizer, scaler, cfg.device)

    # ── Training loop ──
    model.train()
    data_iter = iter(dataloader)
    running_loss = 0.0
    tokens_seen = 0
    t_start = time.time()

    print(f"\n  {'─' * 55}")
    print(f"  Starting from step {start_step:,}  |  {len(dataset):,} samples")
    print(f"  Ctrl+C = stop (model will be saved!)")
    print(f"  {'─' * 55}\n")

    try:
        for step in range(start_step, cfg.max_steps):
            accum_loss = 0.0
            stats = {}

            for _ in range(cfg.grad_accum):
                try:
                    input_ids = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    input_ids = next(data_iter)

                input_ids = input_ids.to(cfg.device, non_blocking=True)

                with autocast(device_type="cuda", dtype=torch.float16,
                              enabled=(cfg.dtype == torch.float16)):
                    logits, stats = model(input_ids)

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()

                    loss = F.cross_entropy(
                        shift_logits.reshape(-1, cfg.vocab_size),
                        shift_labels.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    ) / cfg.grad_accum

                scaler.scale(loss).backward()
                accum_loss += loss.item()
                tokens_seen += input_ids.numel()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # LR schedule (fixed after warmup)
            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            running_loss += accum_loss

            if step % cfg.log_every == 0 and step > start_step:
                avg = running_loss / cfg.log_every
                elapsed = time.time() - t_start
                tps = tokens_seen / elapsed / 1000 if elapsed > 0 else 0
                sp = stats.get("sparsity", 0)

                # VRAM monitoring
                vram_used = ""
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.memory_allocated() / (1024**3)
                    vram_used = f" | VRAM {vram_gb:.1f}G"

                print(
                    f"  step {step:>7,} │ "
                    f"loss {avg:.4f} │ "
                    f"lr {lr:.1e} │ "
                    f"grad {grad_norm:.1f} │ "
                    f"sparsity {sp:.0%} │ "
                    f"{tps:.1f}k tok/s"
                    f"{vram_used}"
                )
                running_loss = 0.0

            if step > 0 and step % cfg.save_every == 0:
                ckpt_mgr.save(model, optimizer, scaler, step, accum_loss, cfg)

    except KeyboardInterrupt:
        print(f"\n\n  [⏸] Stopped at step {step:,}")
        ckpt_mgr.save(model, optimizer, scaler, step, accum_loss, cfg)
        print(f"  To resume — just run the script again.")

    ckpt_mgr.save_final(model, cfg)

    print(f"\n  {'═' * 55}")
    print(f"  Training complete!")
    print(f"  Model saved in: {model_dir}")
    print(f"  {'═' * 55}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PROJECT NORD 500M — SNN Training")
    print("=" * 60)

    default_data = "train_data.jsonl"
    print(f"\n  Dataset path? (JSONL file)")
    print(f"  (Enter = {default_data})")
    data_input = input("  Dataset: ").strip()
    dataset_path = data_input if data_input else default_data

    if not Path(dataset_path).exists():
        print(f"\n  [✗] File not found: {dataset_path}")
        sys.exit(1)

    default_model = "nord_500m_model"
    print(f"\n  Model save directory?")
    print(f"  (Enter = {default_model})")
    model_input = input("  Model dir: ").strip()
    model_dir = model_input if model_input else default_model

    train(dataset_path, model_dir)


if __name__ == "__main__":
    main()
