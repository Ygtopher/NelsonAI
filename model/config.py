"""
Nelson LLM — Model Configuration
Optimized for NVIDIA Quadro T2000 (4GB VRAM)
Languages: Kinyarwanda (primary) + English + French (trilingual)
"""

from dataclasses import dataclass, field
from typing import Optional
import json, os


@dataclass
class NelsonConfig:
    # ── Architecture ──────────────────────────────────────────────
    vocab_size: int = 32_000       # Trilingual BPE: Kinyarwanda + English + French
    context_length: int = 512      # Max tokens per sequence
    d_model: int = 512             # Embedding / hidden dimension
    n_layers: int = 6              # Number of transformer blocks
    n_heads: int = 8               # Number of attention heads
    n_kv_heads: int = 4            # For Grouped Query Attention (saves VRAM)
    d_ffn: int = 1408              # FFN inner dim (SwiGLU: 2/3 * 4 * d_model, rounded)
    dropout: float = 0.1           # Dropout rate

    # ── Language settings ─────────────────────────────────────────
    languages: list = None         # ['rw', 'en', 'fr']
    primary_language: str = 'rw'   # Default response language
    bias: bool = False             # No bias in Linear layers (cleaner, faster)
    tie_embeddings: bool = True    # Share input/output embedding weights

    # ── RoPE Positional Encoding ──────────────────────────────────
    rope_theta: float = 10_000.0   # RoPE base frequency

    # ── Training ──────────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 50_000
    batch_size: int = 4            # Per-GPU batch (small for 4GB VRAM)
    grad_accumulation: int = 16    # Effective batch = 4 * 16 = 64
    eval_interval: int = 500
    save_interval: int = 100
    log_interval: int = 10

    # ── Mixed Precision ───────────────────────────────────────────
    dtype: str = "bfloat16"        # Use bf16 for Ampere+, else "float16"

    # ── Paths ─────────────────────────────────────────────────────
    data_dir: str = "data/tokenized"
    checkpoint_dir: str = "checkpoints"
    tokenizer_path: str = "tokenizer/nelson_tokenizer"
    log_dir: str = "logs"

    # ── Special Tokens ────────────────────────────────────────────
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    user_token_id: int = 3
    nelson_token_id: int = 4

    # ── Language tokens ──────────────────────────────────────────
    lang_rw_token_id: int = 7      # <|lang_rw|> switch to Kinyarwanda
    lang_en_token_id: int = 8      # <|lang_en|> switch to English
    lang_fr_token_id: int = 9      # <|lang_fr|> switch to French

    # ── Model name ────────────────────────────────────────────────
    model_name: str = "nelson-nano"
    version: str = "0.2.0"

    def __post_init__(self):
        if self.languages is None:
            self.languages = ['rw', 'en', 'fr']
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """Estimate total parameter count."""
        embed = self.vocab_size * self.d_model
        attn = self.n_layers * (
            self.d_model * (self.n_heads + 2 * self.n_kv_heads) * self.head_dim  # Q, K, V
            + self.d_model * self.d_model  # Output projection
        )
        ffn = self.n_layers * 3 * self.d_model * self.d_ffn  # gate, up, down
        norm = self.n_layers * 2 * self.d_model + self.d_model  # RMSNorms
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return embed + attn + ffn + norm + lm_head

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "NelsonConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self):
        return (
            f"NelsonConfig(\n"
            f"  model      = {self.model_name} v{self.version}\n"
            f"  params     = ~{self.n_params / 1e6:.1f}M\n"
            f"  layers     = {self.n_layers}\n"
            f"  d_model    = {self.d_model}\n"
            f"  n_heads    = {self.n_heads} (kv={self.n_kv_heads})\n"
            f"  d_ffn      = {self.d_ffn}\n"
            f"  ctx_len    = {self.context_length}\n"
            f"  vocab_size = {self.vocab_size}\n"
            f"  languages  = {self.languages}\n"
            f")"
        )


# Pre-defined configs
NELSON_NANO = NelsonConfig()  # ~35M params — trilingual, default for T2000

NELSON_MICRO = NelsonConfig(  # ~12M params — very tight VRAM / fast experiments
    vocab_size=16_000,
    d_model=256,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    d_ffn=704,
    context_length=256,
    model_name="nelson-micro",
)

NELSON_MINI = NelsonConfig(  # ~140M params — GPT-2 Small equivalent, fits perfectly in 4GB VRAM
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    d_ffn=2048,
    context_length=512,
    model_name="nelson-mini",
)

NELSON_SMALL = NelsonConfig( # ~300M params — Maxing out 4GB VRAM for serious training
    d_model=1024,
    n_layers=18,
    n_heads=16,
    n_kv_heads=4,
    d_ffn=2816,
    context_length=512,
    model_name="nelson-small",
)


if __name__ == "__main__":
    cfg = NELSON_NANO
    print(cfg)
    print(f"\n  Estimated VRAM (fp16, batch=4): ~{cfg.n_params * 4 * 4 / 1e9:.2f} GB")
