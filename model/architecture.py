"""
Nelson LLM — Transformer Architecture
Implements a GPT-style decoder-only transformer with:
  - Rotary Position Embeddings (RoPE)
  - RMSNorm (faster than LayerNorm)
  - SwiGLU Feed-Forward Network
  - Grouped Query Attention (GQA) to save VRAM
  - Flash Attention 2 (if available)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try importing Flash Attention for speed
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import NelsonConfig


# ─────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ─────────────────────────────────────────────────────────────────
# Rotary Position Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10_000.0, device=None):
    """Precompute the cosine/sine frequencies for RoPE."""
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)                          # (seq_len, head_dim/2)
    cos = freqs.cos()[None, None, :, :]                    # (1, 1, seq_len, head_dim/2)
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query or key tensor."""
    # x:   (batch, n_heads, seq_len, head_dim)
    # cos: (1, 1, seq_len, head_dim/2)  — expand to full head_dim
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    # Repeat cos/sin across last dim to match full head_dim
    cos = torch.cat([cos, cos], dim=-1)  # (1, 1, T, head_dim)
    sin = torch.cat([sin, sin], dim=-1)  # (1, 1, T, head_dim)
    return x * cos + rotated * sin


# ─────────────────────────────────────────────────────────────────
# Grouped Query Attention
# ─────────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    Uses fewer K/V heads than Q heads → saves VRAM during inference.
    """

    def __init__(self, config: NelsonConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # times to repeat K/V

        # Projections (no bias for cleaner gradients)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=config.bias)

        self.dropout_p = config.dropout if self.training else 0.0
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, cos[:, :, :T, :], sin[:, :, :T, :])
        k = apply_rope(k, cos[:, :, :T, :], sin[:, :, :T, :])

        # Repeat K/V heads to match Q heads (GQA expansion)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention — use PyTorch's built-in scaled_dot_product_attention
        # (automatically uses Flash Attention if available + causal mask support)
        dropout_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,          # we use is_causal instead
            dropout_p=dropout_p,
            is_causal=True,          # causal/autoregressive masking
        )

        # Reshape and project back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)


# ─────────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network — used in LLaMA, PaLM.
    Better than vanilla GELU FFN.
    """

    def __init__(self, config: NelsonConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ffn, bias=config.bias)
        self.up_proj   = nn.Linear(config.d_model, config.d_ffn, bias=config.bias)
        self.down_proj = nn.Linear(config.d_ffn, config.d_model, bias=config.bias)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Single decoder-only transformer block with pre-norm."""

    def __init__(self, config: NelsonConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn  = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn   = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm + residual
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────
# Nelson — Full Language Model
# ─────────────────────────────────────────────────────────────────

class Nelson(nn.Module):
    """
    Nelson: A Kinyarwanda Language Model.
    GPT-style causal transformer with modern improvements.
    """

    def __init__(self, config: NelsonConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final norm and language model head
        self.norm_final = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding and lm_head weights
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(
            config.head_dim, config.context_length * 2, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Scale output projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            targets:   (batch, seq_len) target token IDs for loss computation
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar cross-entropy loss if targets provided
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context length {self.config.context_length}"
        )

        # Token embeddings
        x = self.emb_dropout(self.token_embedding(input_ids))

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)

        # Final normalization
        x = self.norm_final(x)

        # LM head (compute logits)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        else:
            # Inference: only compute logits for the last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature + top-p + top-k sampling.
        """
        self.eval()
        B, T = input_ids.shape

        for _ in range(max_new_tokens):
            # Crop context if too long
            ctx = input_ids if T <= self.config.context_length else input_ids[:, -self.config.context_length:]

            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature   # (B, vocab_size)

            # Top-K filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append and check EOS
            input_ids = torch.cat([input_ids, next_token], dim=1)
            T += 1

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from model.config import NELSON_MINI

    cfg = NELSON_MINI
    model = Nelson(cfg)
    total = model.count_params()
    print(f"\nNelson model initialized!")
    print(f"  Total parameters : {total:,} ({total/1e6:.1f}M)")
    print(f"  Config           : {cfg}")

    # Quick forward pass test
    dummy_ids = torch.randint(0, cfg.vocab_size, (2, 64))
    dummy_targets = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(dummy_ids, dummy_targets)
    print(f"\nForward pass OK!")
    print(f"  logits shape : {logits.shape}")
    print(f"  loss         : {loss.item():.4f}")
