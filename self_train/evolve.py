"""
Nelson AI — Self-Evolution Engine
Nelson improves itself automatically by:

  1. Learning from every conversation (fine-tunes on chat logs)
  2. Scraping fresh Kinyarwanda content from the web nightly
  3. Continuing pre-training on newly discovered text
  4. Saving an updated checkpoint after each evolution cycle

Run manually:
    python self_train/evolve.py

Or trigger from chat.py automatically after N conversations.
"""

import os
import sys
import json
import math
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import NelsonConfig
from model.architecture import Nelson
from tokenizers import Tokenizer
from self_train.memory import ConversationMemory
from tools.web_tools import scrape_news_articles

TOKENIZER_PATH   = Path("tokenizer/nelson_tokenizer/tokenizer.json")
CHECKPOINT_DIR   = Path("checkpoints")
EVOLUTION_LOG    = Path("memory/evolution_log.jsonl")
NEW_DATA_DIR     = Path("data/evolution")
NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def load_latest_checkpoint() -> tuple[Nelson, NelsonConfig]:
    # Prioritize evolved checkpoints over base training steps
    evolved = sorted(CHECKPOINT_DIR.glob("nelson_evolved_*.pt"))
    if evolved:
        ckpt_path = evolved[-1]
    else:
        step_ckpts = sorted(CHECKPOINT_DIR.glob("nelson_step_*.pt"))
        if not step_ckpts:
            raise FileNotFoundError("No checkpoint found. Train Nelson first.")
        ckpt_path = step_ckpts[-1]
        
    print(f"  Loading: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = NelsonConfig(**ckpt["config"])
    model = Nelson(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config, ckpt_path


def text_to_tokens(text: str, tokenizer: Tokenizer) -> list[int]:
    tokenizer.no_padding()
    enc = tokenizer.encode(text, add_special_tokens=False)
    return enc.ids


def tokens_to_tensor(token_ids: list[int], context_length: int) -> list[torch.Tensor]:
    """Split flat token list into (input, target) chunks."""
    chunks = []
    for i in range(0, len(token_ids) - context_length - 1, context_length):
        chunk = token_ids[i : i + context_length + 1]
        if len(chunk) < context_length + 1:
            break
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        chunks.append((x, y))
    return chunks


# ─────────────────────────────────────────────────────────────────
# Step 1 — Collect new Kinyarwanda data from the web
# ─────────────────────────────────────────────────────────────────

def collect_web_data(tokenizer: Tokenizer) -> int:
    """Scrape fresh Kinyarwanda content and add to evolution data pool."""
    print("\n  [1/3] Collecting fresh Kinyarwanda content from the web ...")
    articles = scrape_news_articles(max_per_source=8)

    if not articles:
        print("  ⚠ No articles scraped (check internet connection).")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = NEW_DATA_DIR / f"web_{timestamp}.txt"
    total_chars = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for article in articles:
            text = article["text"].strip()
            if len(text) > 100:
                f.write(text + "\n\n")
                total_chars += len(text)

    print(f"  ✓ Scraped {len(articles)} articles → {total_chars:,} chars → {out_file.name}")
    return total_chars


# ─────────────────────────────────────────────────────────────────
# Step 2 — Format conversation logs into training data
# ─────────────────────────────────────────────────────────────────

def collect_conversation_data(memory: ConversationMemory) -> list[str]:
    """Load saved conversations as training texts."""
    print("\n  [2/3] Loading conversation training data ...")
    examples = memory.get_training_examples(min_turns=2)
    print(f"  ✓ Found {len(examples)} conversation training examples")
    return [e["text"] for e in examples]


# ─────────────────────────────────────────────────────────────────
# Step 3 — Fine-tune on collected data
# ─────────────────────────────────────────────────────────────────

def run_evolution_training(
    model: Nelson,
    config: NelsonConfig,
    texts: list[str],
    tokenizer: Tokenizer,
    device: torch.device,
    n_steps: int = 500,
    lr: float = 1e-5,
) -> dict:
    """
    Run a mini fine-tuning session on new data.
    Uses a much lower LR than pre-training to avoid catastrophic forgetting.
    """
    print(f"\n  [3/3] Running evolution fine-tuning ({n_steps} steps, lr={lr:.0e}) ...")

    if not texts:
        print("  ⚠ No training texts provided. Skipping fine-tune.")
        return {"steps": 0, "final_loss": None}

    # Also include any evolution text files
    for txt_file in sorted(NEW_DATA_DIR.glob("*.txt")):
        try:
            content = txt_file.read_text(encoding="utf-8", errors="ignore")
            texts.append(content)
        except Exception:
            pass

    # Tokenize all texts into training chunks
    print("  Tokenizing evolution data ...")
    all_chunks = []
    for text in texts:
        token_ids = text_to_tokens(text, tokenizer)
        chunks = tokens_to_tensor(token_ids, config.context_length)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("  ⚠ Not enough data for training chunks. Need longer texts.")
        return {"steps": 0, "final_loss": None}

    print(f"  ✓ {len(all_chunks)} training chunks prepared")

    # Setup
    model = model.to(device)
    model.train()

    # Lower LR + weight decay for fine-tuning (avoid catastrophic forgetting)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # Detect precision
    capability = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
    pt_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    scaler = GradScaler(enabled=(pt_dtype == torch.float16 and device.type == "cuda"))
    ctx = autocast(device_type="cuda", dtype=pt_dtype) if device.type == "cuda" else nullcontext()

    # Shuffle chunks
    import random
    random.shuffle(all_chunks)
    chunk_iter = iter(all_chunks * (n_steps // len(all_chunks) + 2))  # Repeat if needed

    losses = []
    for step in range(n_steps):
        x, y = next(chunk_iter)
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        if step % 50 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            ppl = math.exp(min(avg, 20))
            print(f"    step {step:>4}/{n_steps} │ loss={avg:.4f} │ ppl={ppl:.1f}")

    final_loss = sum(losses[-20:]) / len(losses[-20:]) if losses else None
    print(f"  ✓ Fine-tuning done. Final loss: {final_loss:.4f}")
    return {"steps": n_steps, "final_loss": final_loss, "n_chunks": len(all_chunks)}


# ─────────────────────────────────────────────────────────────────
# Save evolved checkpoint
# ─────────────────────────────────────────────────────────────────

def save_evolved_checkpoint(model: Nelson, config: NelsonConfig, evolution_num: int):
    ckpt_path = CHECKPOINT_DIR / f"nelson_evolved_{evolution_num:03d}.pt"
    torch.save({
        "step":              f"evolved_{evolution_num}",
        "model_state_dict":  model.state_dict(),
        "config":            config.__dict__,
        "evolved_at":        datetime.now().isoformat(),
        "evolution_num":     evolution_num,
    }, ckpt_path)
    print(f"  ✓ Evolved checkpoint saved: {ckpt_path.name}")
    return ckpt_path


# ─────────────────────────────────────────────────────────────────
# Main Evolution Orchestrator
# ─────────────────────────────────────────────────────────────────

def evolve(n_steps: int = 300, collect_web: bool = True):
    """
    Full evolution cycle:
    1. Scrape fresh web data
    2. Load conversation logs
    3. Fine-tune on all new data
    4. Save updated checkpoint
    """
    print("\n" + "=" * 65)
    print("  ⚡ NELSON SELF-EVOLUTION CYCLE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    start_time = time.time()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load tokenizer
    if not TOKENIZER_PATH.exists():
        print("  ✗ Tokenizer not found. Run tokenizer/train_tokenizer.py first.")
        return
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # Load latest model
    try:
        model, config, base_ckpt = load_latest_checkpoint()
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return

    # Count current evolution number
    evolved = sorted(CHECKPOINT_DIR.glob("nelson_evolved_*.pt"))
    evolution_num = len(evolved) + 1

    # Collect data
    memory = ConversationMemory()

    if collect_web:
        collect_web_data(tokenizer)

    conv_texts = collect_conversation_data(memory)

    # Gather all available text for training
    all_texts = conv_texts.copy()

    # Run fine-tuning
    stats = run_evolution_training(
        model, config, all_texts, tokenizer, device, n_steps=n_steps
    )

    # Save evolved model
    save_evolved_checkpoint(model, config, evolution_num)

    # Log evolution event
    elapsed = time.time() - start_time
    log_entry = {
        "evolution_num":    evolution_num,
        "timestamp":        datetime.now().isoformat(),
        "base_checkpoint":  str(base_ckpt.name),
        "n_steps":          stats.get("steps", 0),
        "final_loss":       stats.get("final_loss"),
        "n_conv_examples":  len(conv_texts),
        "elapsed_seconds":  elapsed,
        "device":           str(device),
    }
    EVOLUTION_LOG.parent.mkdir(exist_ok=True)
    with open(EVOLUTION_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    memory.mark_evolution()

    print("\n" + "=" * 65)
    print(f"  ✅ Evolution #{evolution_num} complete in {elapsed:.0f}s")
    print(f"  Base: {base_ckpt.name}")
    print(f"  New : nelson_evolved_{evolution_num:03d}.pt")
    print("=" * 65)
    return evolution_num


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Nelson self-evolution cycle")
    parser.add_argument("--steps",     type=int,  default=300,  help="Fine-tuning steps")
    parser.add_argument("--no-web",    action="store_true",     help="Skip web scraping")
    args = parser.parse_args()

    evolve(n_steps=args.steps, collect_web=not args.no_web)
