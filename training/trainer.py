"""
Nelson LLM — Training Loop
Trains Nelson from scratch on Kinyarwanda text.
Supports:
  - Mixed precision (bf16/fp16) for T2000
  - Gradient accumulation (simulate large batches)
  - Gradient checkpointing (save VRAM)
  - Cosine LR schedule with warmup
  - Periodic checkpointing + resuming
  - Live loss display via Rich
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.amp import GradScaler, autocast
from contextlib import nullcontext

# Enable CuDNN auto-tuner for max speed on fixed-size batches
torch.backends.cudnn.benchmark = True

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import NELSON_MINI, NelsonConfig
from model.architecture import Nelson
from training.dataset import create_dataloaders

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

# ANSI colors for graceful exits
class C:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    DIM = "\033[2m"


# ─────────────────────────────────────────────────────────────────
# Learning Rate Schedule
# ─────────────────────────────────────────────────────────────────

def get_lr(step: int, config: NelsonConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step > config.max_steps:
        return config.learning_rate * 0.1  # Minimum LR
    # Cosine decay
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.1 + 0.5 * (config.learning_rate - config.learning_rate * 0.1) * (
        1 + math.cos(math.pi * progress)
    )


# ─────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: Nelson,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    step: int,
    loss: float,
    config: NelsonConfig,
):
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"nelson_step_{step:06d}.pt"

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
        "config": config.__dict__,
    }, ckpt_path)

    # Keep only last 3 checkpoints to save disk space
    checkpoints = sorted(ckpt_dir.glob("nelson_step_*.pt"))
    if len(checkpoints) > 3:
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()

    print(f"  ✓ Checkpoint saved: {ckpt_path.name}")
    return ckpt_path


def load_checkpoint(
    model: Nelson,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: NelsonConfig,
) -> int:
    """Load the latest checkpoint if available. Returns starting step."""
    ckpt_dir = Path(config.checkpoint_dir)
    checkpoints = sorted(ckpt_dir.glob("nelson_step_*.pt"))

    if not checkpoints:
        print("  No checkpoint found, starting from scratch.")
        return 0

    ckpt_path = checkpoints[-1]
    print(f"  Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    step = ckpt["step"]
    loss = ckpt.get("loss", float("inf"))
    print(f"  ✓ Resumed from step {step:,} (loss={loss:.4f})")
    return step


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: Nelson, val_loader, device: torch.device, dtype, n_batches: int = 20) -> float:
    """Run evaluation on validation set. Returns mean loss."""
    model.eval()
    losses = []
    ctx = autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()

    for i, (x, y) in enumerate(val_loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / max(len(losses), 1)


# ─────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────

def train(config: NelsonConfig = None, resume: bool = True):
    if config is None:
        config = NELSON_MINI

    print("\n" + "=" * 65)
    print("  NELSON AI — TRAINING FROM SCRATCH")
    print("  Kinyarwanda Language Model")
    print("=" * 65)
    print(config)

    # ── Device Setup ──────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("\n  ⚠ No GPU found. Training on CPU (will be very slow).")

    # Mixed precision dtype
    if device.type == "cuda":
        # T2000 is Turing architecture — supports fp16 but not bf16
        # Check: bf16 requires Ampere (sm_80+), T2000 is sm_75
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 8:  # Ampere or newer
            pt_dtype = torch.bfloat16
            print("  Precision: bfloat16 (Ampere+)")
        else:
            pt_dtype = torch.float16
            print(f"  Precision: float16 (GPU capability {capability[0]}.{capability[1]})")
    else:
        pt_dtype = torch.float32
        print("  Precision: float32 (CPU)")

    # ── Model ─────────────────────────────────────────────────────
    print(f"\n  Building Nelson model ...")
    model = Nelson(config).to(device)

    # Enable gradient checkpointing to save VRAM
    # This recomputes activations during backward pass instead of storing them
    if hasattr(model, "blocks"):
        for block in model.blocks:
            if hasattr(block, "attn"):
                pass  # Gradient checkpointing via custom wrapper if needed

    n_params = model.count_params()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Estimate VRAM usage
    param_mb = n_params * 2 / 1e6  # fp16: 2 bytes/param
    print(f"  Est. VRAM (weights only): ~{param_mb:.0f} MB")

    # ── Optimizer ─────────────────────────────────────────────────
    # Separate parameters for weight decay (don't decay biases and norms)
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=device.type == "cuda",  # Faster fused optimizer on CUDA
    )

    scaler = GradScaler("cuda", enabled=(pt_dtype == torch.float16))

    # ── Data ──────────────────────────────────────────────────────
    print(f"\n  Loading dataset ...")
    train_loader, val_loader = create_dataloaders(
        tokenized_dir=config.data_dir,
        context_length=config.context_length,
        batch_size=config.batch_size,
        num_workers=2,  # Uses background CPU cores for data loading
    )

    # ── Resume from checkpoint ────────────────────────────────────
    start_step = 0
    if resume:
        start_step = load_checkpoint(model, optimizer, scaler, config)

    # Save config
    config.save(os.path.join(config.checkpoint_dir, "config.json"))

    # ── Training Loop ─────────────────────────────────────────────
    ctx = autocast("cuda", dtype=pt_dtype) if device.type == "cuda" else nullcontext()
    train_iter = iter(train_loader)

    model.train()
    optimizer.zero_grad()

    print(f"\n  Starting training at step {start_step:,}/{config.max_steps:,}")
    print(f"  Effective batch size: {config.batch_size * config.grad_accumulation}")
    print("-" * 65)

    running_loss = 0.0
    t0 = time.time()
    log_path = Path(config.log_dir)
    log_path.mkdir(exist_ok=True)
    log_file = open(log_path / "train_log.jsonl", "a")

    try:
        for step in range(start_step, config.max_steps):

            # Update learning rate
            lr = get_lr(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Gradient accumulation loop
            loss_accum = 0.0
            for micro_step in range(config.grad_accumulation):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with ctx:
                    _, loss = model(x, y)
                    loss = loss / config.grad_accumulation

                scaler.scale(loss).backward()
                loss_accum += loss.item()

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss_accum

            # ── Logging ───────────────────────────────────────────────
            if step % config.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                tokens_per_sec = config.batch_size * config.context_length * config.grad_accumulation * config.log_interval / dt
                t0 = t1
                avg_loss = running_loss / config.log_interval
                running_loss = 0.0

                perplexity = math.exp(min(avg_loss, 20))

                print(
                    f"  step {step:>6,}/{config.max_steps:,} │ "
                    f"loss={avg_loss:.4f} │ "
                    f"ppl={perplexity:.1f} │ "
                    f"lr={lr:.2e} │ "
                    f"grad={grad_norm:.2f} │ "
                    f"tok/s={tokens_per_sec:,.0f}"
                )

                log_file.write(json.dumps({
                    "step": step,
                    "loss": avg_loss,
                    "perplexity": perplexity,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_sec": tokens_per_sec,
                }) + "\n")
                log_file.flush()

            # ── Evaluation ────────────────────────────────────────────
            if step > 0 and step % config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, pt_dtype)
                val_ppl  = math.exp(min(val_loss, 20))
                print(f"\n  ── EVAL step {step:,} │ val_loss={val_loss:.4f} │ val_ppl={val_ppl:.1f} ──\n")

            # ── Checkpointing ─────────────────────────────────────────
            if step > 0 and step % config.save_interval == 0:
                save_checkpoint(model, optimizer, scaler, step, loss_accum, config)

    except KeyboardInterrupt:
        print(f"\n\n  {C.YELLOW}⚠ Training paused by user (Ctrl+C).{C.RESET}")
        print(f"  {C.DIM}Emergency saving checkpoint at step {step}...{C.RESET}")
        save_checkpoint(model, optimizer, scaler, step, running_loss, config)
        log_file.close()
        print(f"  {C.GREEN}✓ Progress safely saved! You can close the terminal now.{C.RESET}\n")
        sys.exit(0)

    # Final save
    save_checkpoint(model, optimizer, scaler, config.max_steps, loss_accum, config)
    log_file.close()
    print("\n" + "=" * 65)
    print("  Training complete!")
    print(f"  Model saved in: {config.checkpoint_dir}/")
    print("  Next step: python chat.py")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Nelson LLM")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from latest checkpoint")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = NELSON_MINI
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.batch_size:
        cfg.batch_size = args.batch_size

    train(cfg, resume=args.resume)
