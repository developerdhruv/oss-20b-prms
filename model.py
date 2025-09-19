from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Optional Flash-Attention-2 / 3 -----------------------------------------
_flash_available = False
try:
    # flash_attn >= 2.x API
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
    _flash_available = True
except Exception:
    _flash_attn_func = None
    _flash_available = False


# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------

@dataclass
class RopeScalingConfig:
    # Simple scalar stretch for YaRN-like extension (pragmatic approximation)
    factor: float = 32.0


@dataclass
class ModelConfig:
    # Core dims
    vocab_size: int = 201_088
    hidden_size: int = 2880
    num_hidden_layers: int = 24
    head_dim: int = 64

    # Attention (GQA)
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    attention_bias: bool = True
    attention_dropout: float = 0.0
    dropout: float = 0.0

    # Patterns / positions
    max_position_embeddings: int = 131_072
    sliding_window: int = 128
    layer_types: Optional[List[Literal["sliding_attention", "full_attention"]]] = None

    # MoE
    num_local_experts: int = 32
    experts_per_token: int = 4
    router_aux_loss_coef: float = 0.02  # conservative; 0.01–0.1 common

    # MLP inside each expert (SwiGLU uses 2*FF)
    intermediate_size: int = 2880
    swiglu_clip: float = 7.0

    # RoPE / YaRN
    rope_theta: float = 150_000.0
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)

    # Sink (null-attention bias)
    enable_sink_logit: bool = True
    sink_logit_init: float = 4.0  # positive → allows "attend to nothing"

    # Norms / init / tying
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    eos_token_id: Optional[int] = None

    def __post_init__(self):
        if self.layer_types is None:
            # Default: alternate full <-> sliding attention
            self.layer_types = [
                "full_attention" if i % 2 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        assert len(self.layer_types) == self.num_hidden_layers
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim > 0

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


# ------------------------------------------------------------------------------------
# Layers
# ------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x

    # FSDP meta materialization support
    def reset_parameters(self):
        with torch.no_grad():
            self.weight.fill_(1.0)


def swiglu(x: torch.Tensor, clip: Optional[float] = None) -> torch.Tensor:
    up, gate = x.chunk(2, dim=-1)
    if clip is not None:
        up = up.clamp(-clip, clip)
        gate = gate.clamp(-clip, clip)
    return F.silu(gate) * up


class RotaryEmbedding(nn.Module):
    """
    Standard RoPE with a simple YaRN-style stretch: positions/factor.
    Buffers must be (re)materialized on nonzero ranks during FSDP meta build.
    """
    def __init__(self, head_dim: int, rope_theta: float, scale_cfg: RopeScalingConfig):
        super().__init__()
        self.head_dim = int(head_dim)
        self.theta = float(rope_theta)
        self.factor = float(scale_cfg.factor)

        # Placeholders; real buffers (CPU) in reset_parameters()
        self.register_buffer("inv_freq_base", torch.empty(0), persistent=False)
        self._seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

        # Immediately build once so rank0 (non-meta) has valid buffers
        self.reset_parameters()

    def reset_parameters(self):
        # Build base frequencies on CPU; FSDP will move as needed.
        device = torch.device("cpu")
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        self.register_buffer("inv_freq_base", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, dtype=torch.float32, device=device), persistent=False)

    def _update_cache(self, seqlen: int, device, dtype):
        if (seqlen <= self._seq_len_cached and
            self.cos_cached.device == device and
            self.cos_cached.dtype == dtype):
            return
        pos = torch.arange(seqlen, device=device, dtype=torch.float32) / self.factor
        freqs = torch.einsum("s,f->sf", pos, self.inv_freq_base.to(device=device))
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        # interleave
        cos = torch.stack([cos, cos], dim=-1).reshape(seqlen, -1)
        sin = torch.stack([sin, sin], dim=-1).reshape(seqlen, -1)
        self.cos_cached = cos
        self.sin_cached = sin
        self._seq_len_cached = seqlen

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B*H, T, Dh)
        BxH, T, Dh = x.shape
        self._update_cache(int(positions.max().item()) + 1, x.device, x.dtype)
        cos = self.cos_cached[positions]  # (B*H, T, Dh)
        sin = self.sin_cached[positions]
        x1, x2 = x[..., ::2], x[..., 1::2]
        xr1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
        xr2 = x1 * sin[..., ::2] + x2 * cos[..., ::2]
        out = torch.empty_like(x)
        out[..., ::2], out[..., 1::2] = xr1, xr2
        return out


class MultiheadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.hidden_size
        self.n_head = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.dh = int(cfg.head_dim)
        self.group_size = int(cfg.group_size)
        self.scale = 1.0 / math.sqrt(self.dh)
        self.drop_attn = nn.Dropout(cfg.attention_dropout)
        self.drop_resid = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.dh, cfg.rope_theta, cfg.rope_scaling)

        self.q = nn.Linear(H, self.n_head * self.dh, bias=cfg.attention_bias)
        self.k = nn.Linear(H, self.n_kv * self.dh, bias=cfg.attention_bias)
        self.v = nn.Linear(H, self.n_kv * self.dh, bias=cfg.attention_bias)
        self.o = nn.Linear(self.n_head * self.dh, H, bias=True)

        # Learned "null" attention logit per head (attention sink column)
        self.use_sink = bool(cfg.enable_sink_logit)
        if self.use_sink:
            self.sink_logit = nn.Parameter(torch.full((self.n_head,), float(cfg.sink_logit_init)))
        else:
            # create a dummy buffer so reset_parameters() can always reference it safely
            self.register_buffer("sink_logit", torch.empty(0), persistent=False)

        # store init settings for reset_parameters()
        self.init_std = float(cfg.initializer_range)
        self.sink_init = float(cfg.sink_logit_init)

        # init once for rank0
        self.reset_parameters()

    def reset_parameters(self):
        # Robust to attributes missing during meta materialization
        init_std = getattr(self, "init_std", 0.02)
        with torch.no_grad():
            for m in (self.q, self.k, self.v, self.o):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
                if getattr(m, "bias", None) is not None:
                    m.bias.zero_()
            if getattr(self, "use_sink", False) and hasattr(self, "sink_logit") and self.sink_logit.numel() > 0:
                self.sink_logit.fill_(getattr(self, "sink_init", 4.0))
        # also reset RoPE buffers so nonzero ranks get real tensors
        if hasattr(self, "rope") and hasattr(self.rope, "reset_parameters"):
            self.rope.reset_parameters()

    def _kv_expand(self, kv: torch.Tensor) -> torch.Tensor:
        # (B, T, n_kv*Dh) -> (B, H, T, Dh)
        B, T, _ = kv.shape
        kv = kv.view(B, T, self.n_kv, self.dh)
        kv = kv.unsqueeze(3).expand(B, T, self.n_kv, self.group_size, self.dh)
        kv = kv.reshape(B, T, self.n_head, self.dh).transpose(1, 2).contiguous()
        return kv

    @staticmethod
    def _build_local_mask(T: int, device, win: int) -> torch.Tensor:
        # (T,T) True=keep
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        dist = i[:, None] - j[None, :]
        return (dist >= 0) & (dist < win)

    def forward(
        self,
        x: torch.Tensor,               # (B,T,H)
        positions: torch.Tensor,       # (B,T)
        causal_mask: torch.Tensor,     # (T,T) bool True=keep
        is_sliding_layer: bool,
        sliding_window: int,
    ) -> torch.Tensor:
        B, T, H = x.shape

        q = self.q(x).view(B, T, self.n_head, self.dh).transpose(1, 2).contiguous()  # (B,H,T,Dh)
        k = self._kv_expand(self.k(x))  # (B,H,T,Dh)
        v = self._kv_expand(self.v(x))  # (B,H,T,Dh)

        # RoPE on q,k
        pos_rep = positions.repeat_interleave(self.n_head, 0)  # (B*H,T)
        q = self.rope.apply(q.view(B * self.n_head, T, self.dh), pos_rep).view(B, self.n_head, T, self.dh)
        k = self.rope.apply(k.view(B * self.n_head, T, self.dh), pos_rep).view(B, self.n_head, T, self.dh)

        # Fast path: use FlashAttention only for pure causal full-attn w/o sink column
        use_flash = _flash_available and (not is_sliding_layer) and (not getattr(self, "use_sink", False))
        if use_flash:
            # Flash-Attn expects (B,T,H,D)
            qf = q.transpose(1, 2)  # (B,T,H,D)
            kf = k.transpose(1, 2)
            vf = v.transpose(1, 2)
            out = _flash_attn_func(qf, kf, vf, causal=True)  # (B,T,H,D)
            out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.dh)
            out = self.o(out)
            return self.drop_resid(out)

        # Manual path (supports sliding + sink)
        att = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale  # (B,H,T,S=T)

        # Base causal
        mask = causal_mask  # (T,T) True=keep
        # Sliding window (only limit the source side)
        if is_sliding_layer:
            local = self._build_local_mask(T, x.device, sliding_window)
            mask = mask & local  # (T,T)

        # Apply mask
        att = att.masked_fill(~mask.view(1, 1, T, T), float("-inf"))

        # Append sink column (null attention) if enabled
        if getattr(self, "use_sink", False) and hasattr(self, "sink_logit") and self.sink_logit.numel() > 0:
            sink_col = self.sink_logit.view(1, self.n_head, 1, 1).expand(B, -1, T, 1)
            att = torch.cat([att, sink_col], dim=-1)  # (B,H,T,T+1)

        p = F.softmax(att, dim=-1)
        if getattr(self, "use_sink", False) and hasattr(self, "sink_logit") and self.sink_logit.numel() > 0:
            p = p[..., :-1]  # drop sink prob (mass = "attend to nothing")
        p = self.drop_attn(p)

        y = torch.einsum("bhts,bhsd->bhtd", p, v).contiguous()
        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.dh)
        y = self.o(y)
        return self.drop_resid(y)


class MoE(nn.Module):
    """
    A fast, vectorized MoE layer using einsum.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.hidden_size
        E = cfg.num_local_experts
        FF = cfg.intermediate_size
        self.E = int(E)
        self.K = int(cfg.experts_per_token)
        self.clip = float(cfg.swiglu_clip)
        self.router_aux_loss_coef = float(cfg.router_aux_loss_coef)

        # Expert parameters: W_in (H -> 2*FF), W_out (FF -> H)
        self.W_in = nn.Parameter(torch.empty(E, H, 2 * FF))
        self.b_in = nn.Parameter(torch.zeros(E, 2 * FF))
        self.W_out = nn.Parameter(torch.empty(E, FF, H))
        self.b_out = nn.Parameter(torch.zeros(E, H))

        # Router
        self.router = nn.Linear(H, E, bias=True)

        # store init std for reset
        self.init_std = float(cfg.initializer_range)

        # init once for rank0
        self.reset_parameters()

    def reset_parameters(self):
        init_std = getattr(self, "init_std", 0.02)
        with torch.no_grad():
            nn.init.normal_(self.W_in,  mean=0.0, std=init_std)
            nn.init.normal_(self.W_out, mean=0.0, std=init_std)
            self.b_in.zero_(); self.b_out.zero_()
            nn.init.normal_(self.router.weight, mean=0.0, std=init_std)
            if self.router.bias is not None:
                self.router.bias.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B,T,H) -> (S,H) where S=B*T
        B, T, H = x.shape
        S = B * T
        x_flat = x.view(S, H)

        # Route tokens to experts
        logits = self.router(x_flat)  # (S, E)

        # Top-K routing
        topk_weights, topk_indices = torch.topk(logits, self.K, dim=-1)  # (S, K), (S, K)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Aux (Switch-style) loss for load balancing
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        importance = probs.mean(dim=0)          # (E,)
        load = F.one_hot(topk_indices.argmax(dim=-1), num_classes=self.E).float().mean(dim=0)
        aux_loss = self.router_aux_loss_coef * self.E * (importance * load).sum()

        # Create a one-hot mask for selected experts for each token and top-k choice
        # (S, K) -> (S, K, E)
        expert_mask = F.one_hot(topk_indices, num_classes=self.E)

        # Combine the mask with the weights
        # (S, K, E) * (S, K, 1) -> (S, K, E)
        gating_weights = expert_mask * topk_weights.unsqueeze(-1)

        # Sum weights over K choices to get the final weight for each expert for each token
        # (S, K, E) -> (S, E)
        final_expert_weights = gating_weights.sum(dim=1)

        # --- Vectorized Expert Computation ---
        # 1. Apply all experts' W_in to all tokens
        #    'sh,ehd->sed': (S,H) @ (E,H,2FF) -> (S,E,2FF)
        expert_inputs = torch.einsum('sh,ehd->sed', x_flat, self.W_in) + self.b_in
        
        # 2. Apply SwiGLU activation
        #    swiglu halves the last dimension
        expert_outputs = swiglu(expert_inputs, clip=self.clip) # (S,E,FF)

        # 3. Apply all experts' W_out
        #    'sef,efh->seh': (S,E,FF) @ (E,FF,H) -> (S,E,H)
        expert_outputs = torch.einsum('sef,efh->seh', expert_outputs, self.W_out) + self.b_out

        # 4. Weight the expert outputs by the router weights and sum
        #    'seh,se->sh': (S,E,H) * (S,E) -> (S,H)
        weighted_output = torch.einsum('seh,se->sh', expert_outputs, final_expert_weights)

        # Reshape back to (B, T, H)
        out = weighted_output.view(B, T, H)

        return out, {"router_aux_loss": aux_loss}


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = MultiheadSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.moe = MoE(cfg)

    def reset_parameters(self):
        # Submodules handle their own init; keep for FSDP completeness
        if hasattr(self.norm1, "reset_parameters"): self.norm1.reset_parameters()
        if hasattr(self.attn,  "reset_parameters"): self.attn.reset_parameters()
        if hasattr(self.norm2, "reset_parameters"): self.norm2.reset_parameters()
        if hasattr(self.moe,   "reset_parameters"): self.moe.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        causal_mask: torch.Tensor,
        is_sliding_layer: bool,
        sliding_window: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        a = self.attn(self.norm1(x), positions, causal_mask, is_sliding_layer, sliding_window)
        x = x + a
        m, aux = self.moe(self.norm2(x))
        x = x + m
        return x, aux


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        H = cfg.hidden_size
        self.embed = nn.Embedding(cfg.vocab_size, H)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm_f = RMSNorm(H, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)

        # store init std for reset
        self.init_std = float(cfg.initializer_range)

        # optional tying
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        # init once for rank0
        self.reset_parameters()

    @staticmethod
    def build_causal_mask(T: int, device, dtype=torch.bool) -> torch.Tensor:
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        return (j[None, :] <= i[:, None]).to(dtype)

    def reset_parameters(self):
        init_std = getattr(self, "init_std", 0.02)
        with torch.no_grad():
            nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)
            if self.lm_head.weight is not self.embed.weight:
                nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_std)
        # Cascade to blocks + final norm
        for blk in getattr(self, "layers", []):
            if hasattr(blk, "reset_parameters"):
                blk.reset_parameters()
        if hasattr(self.norm_f, "reset_parameters"):
            self.norm_f.reset_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,                  # (B,T)
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed(input_ids)
        x = self.drop(x)
        positions = torch.arange(T, device=device).view(1, T).expand(B, T)
        causal_mask = self.build_causal_mask(T, device)  # (T,T) bool

        aux_losses: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            is_sliding = (self.config.layer_types[i] == "sliding_attention")
            x, aux = layer(x, positions, causal_mask, is_sliding, self.config.sliding_window)
            if aux and "router_aux_loss" in aux:
                aux_losses.append(aux["router_aux_loss"])

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        aux_out: Dict[str, torch.Tensor] = {}
        if labels is not None:
            # next-token loss
            logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = labels[:, 1:].contiguous().view(-1)
            nll = F.cross_entropy(logits_flat, targets, ignore_index=-100)
            if aux_losses:
                aux_total = torch.stack(aux_losses).mean()
                nll = nll + self.config.router_aux_loss_coef * aux_total
                aux_out["router_aux_loss"] = aux_total.detach()
            loss = nll
        return logits, {"loss": loss, **aux_out}


# ------------------------------------------------------------------------------------
# Quick param sanity check for the 20B config
# ------------------------------------------------------------------------------------

def gpt_oss_20b_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=201_088,
        hidden_size=2880,
        num_hidden_layers=24,
        head_dim=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        attention_bias=True,
        attention_dropout=0.0,
        dropout=0.0,
        max_position_embeddings=131_072,
        sliding_window=128,
        num_local_experts=32,
        experts_per_token=4,
        router_aux_loss_coef=0.02,
        intermediate_size=2880,
        swiglu_clip=7.0,
        rope_theta=150_000.0,
        enable_sink_logit=True,   # sink-bias enabled (flash kept on full-attn layers)
        sink_logit_init=4.0,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False,
        eos_token_id=None,
    )


if __name__ == "__main__":
    cfg = gpt_oss_20b_config()
    model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params/1e9:.3f} B")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable:        {trainable/1e9:.3f} B")
