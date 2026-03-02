"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PROJECT NORD — Core Engine  v3.5                       ║
║          Spiking Neural Network LLM with Associative Memory Manifold       ║
║                                                                            ║
║  v3.5 — 500M Parameter Scale-Up:                                           ║
║    • d_model: 512 → 1024                                                   ║
║    • n_layers: 6 → 12                                                      ║
║    • d_ff: 1024 → 4096                                                     ║
║    • n_heads: 8 → 16                                                       ║
║    • n_clusters: 64 → 128                                                  ║
║    • ~416M parameters                                                      ║
║                                                                            ║
║  All 7 bottleneck fixes from v3 preserved:                                 ║
║    1. Multi-Scale Temporal: T_fast + T_slow + persistent membrane state    ║
║    2. LeakyClamp: keeps small negatives (parametric floor, not hard ReLU)  ║
║    3. Adaptive Cascade: learnable per-cluster gain + soft neighbor weights  ║
║    4. Reward-Modulated STDP: LM loss guides plasticity direction           ║
║    5. Sparse Resonance: top-K co-firing instead of full O(S²)             ║
║    6. Temporal Smoothing Readout: EMA on membrane for long dependencies    ║
║    7. Fused ops: no per-block GPU sync, sparse spike buffers              ║
║                                                                            ║
║  Target HW: NVIDIA L40 (48 GB VRAM) / 4× RTX 3090 (24 GB each)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# §0  CONFIGURATION — 500M SCALE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NordConfig:
    # Tokenizer
    tokenizer_id: str = "meta-llama/Llama-3.2-1B"

    # ═══ SCALED DIMENSIONS (500M) ═══
    vocab_size:    int = 128_256
    d_model:       int = 1024          # was 512
    n_heads:       int = 16            # was 8
    n_layers:      int = 12            # was 6
    d_ff:          int = 4096          # was 1024 (4× d_model, same ratio as GPT-2)
    max_seq_len:   int = 1024

    # ═══ FIX #1: Multi-Scale Temporal ═══
    T:             int = 8
    T_slow:        int = 2
    persistent_mem: bool = True

    # LIF Neuron Dynamics
    tau_mem:       float = 0.9
    tau_syn:       float = 0.50
    v_threshold:   float = 0.25
    v_reset:       float = -0.1
    refractory_t:  int   = 2
    threshold_lr:  float = 0.01

    # ═══ FIX #3: Adaptive Cascade (scaled clusters) ═══
    n_clusters:    int = 128           # was 64 — more clusters for wider model
    cascade_radius: int = 3
    cascade_gain:  float = 0.8

    # ═══ FIX #4: Reward-Modulated STDP ═══
    stdp_a_plus:   float = 0.005
    stdp_a_minus:  float = 0.005
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    stdp_w_max:    float = 1.0
    stdp_w_min:    float = -0.3
    stdp_reward_scale: float = 1.0

    # ═══ FIX #5: Sparse Resonance ═══
    resonance_top_k: int = 64

    # ═══ FIX #2: LeakyClamp ═══
    clamp_floor:   float = -0.1

    # Surrogate Gradient
    surrogate_alpha: float = 4.0

    # ═══ TRAINING (adjusted for 500M) ═══
    batch_size:    int   = 2           # smaller batch — more VRAM for params
    grad_accum:    int   = 16          # compensate with more accumulation
    lr:            float = 3e-4        # lower lr for larger model stability
    min_lr:        float = 1e-5
    weight_decay:  float = 0.01
    warmup_steps:  int   = 1000        # longer warmup for 500M
    max_steps:     int   = 200_000
    save_every:    int   = 1000
    log_every:     int   = 10
    max_grad_norm: float = 1.0

    # Hardware
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    @property
    def T_total(self) -> int:
        """Total effective timesteps (fast + slow)."""
        return self.T + self.T_slow


# ─────────────────────────────────────────────────────────────────────────────
# §1  SURROGATE GRADIENT — ATan
# ─────────────────────────────────────────────────────────────────────────────

class ATanSurrogate(torch.autograd.Function):
    alpha = 2.0

    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: Tensor) -> Tensor:
        ctx.save_for_backward(membrane, threshold)
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        membrane, threshold = ctx.saved_tensors
        orig_dtype = membrane.dtype
        x = (membrane.float() - threshold.float())
        grad = ATanSurrogate.alpha / (
            2.0 * math.pi * (1.0 + (ATanSurrogate.alpha * x) ** 2))
        grad_v = (grad_output.float() * grad).to(orig_dtype)
        return grad_v, -grad_v


def spike_fn(v: Tensor, th: Tensor, alpha: float = 2.0) -> Tensor:
    ATanSurrogate.alpha = alpha
    return ATanSurrogate.apply(v, th)


# ─────────────────────────────────────────────────────────────────────────────
# §2  ASSOCIATIVE LIF NEURON (v3 — Adaptive Cascade + Persistent State)
# ─────────────────────────────────────────────────────────────────────────────

class AssociativeLIF(nn.Module):
    def __init__(self, d: int, cfg: NordConfig, persistent: bool = False):
        super().__init__()
        self.cfg = cfg
        self.d = d
        self.persistent = persistent

        self.threshold = nn.Parameter(torch.full((d,), cfg.v_threshold))
        self.beta_mem_raw = nn.Parameter(torch.tensor(
            math.log(cfg.tau_mem / (1 - cfg.tau_mem + 1e-6))))
        self.beta_syn_raw = nn.Parameter(torch.tensor(
            math.log(cfg.tau_syn / (1 - cfg.tau_syn + 1e-6))))

        nc = cfg.n_clusters
        cluster_ids = torch.arange(d) % nc
        self.register_buffer("cluster_ids", cluster_ids)

        r = cfg.cascade_radius
        idx = torch.arange(nc)
        init_weights = torch.zeros(nc, nc)
        for offset in range(-r, r + 1):
            if offset != 0:
                dist_weight = 1.0 - abs(offset) / (r + 1)
                init_weights[idx, (idx + offset) % nc] = dist_weight
        self.neighbor_weights = nn.Parameter(init_weights)
        self.cluster_gain = nn.Parameter(torch.full((nc,), cfg.cascade_gain))

        if persistent:
            self.register_buffer("_v_mem_state", torch.zeros(1, d))
            self.register_buffer("_i_syn_state", torch.zeros(1, d))

    @property
    def beta_mem(self) -> Tensor:
        return torch.sigmoid(self.beta_mem_raw)

    @property
    def beta_syn(self) -> Tensor:
        return torch.sigmoid(self.beta_syn_raw)

    def _cascade_amplify(self, spikes: Tensor) -> Tensor:
        B, D = spikes.shape
        nc = self.cfg.n_clusters
        cid = self.cluster_ids.unsqueeze(0).expand(B, -1)

        cluster_fire = torch.zeros(B, nc, device=spikes.device, dtype=spikes.dtype)
        cluster_fire.scatter_add_(1, cid, spikes)
        cluster_fire = cluster_fire / max(D // nc, 1)

        W = torch.sigmoid(self.neighbor_weights)
        neighbor_signal = (W.to(cluster_fire.dtype) @ cluster_fire.T).T

        gain = self.cluster_gain.to(cluster_fire.dtype)
        neighbor_signal = neighbor_signal * gain.unsqueeze(0)

        return neighbor_signal.gather(1, cid)

    def reset_state(self):
        if self.persistent:
            self._v_mem_state.zero_()
            self._i_syn_state.zero_()

    def forward(self, current_in: Tensor) -> Tuple[Tensor, Tensor]:
        T, B, D = current_in.shape
        device = current_in.device
        dtype = current_in.dtype
        beta_m = self.beta_mem
        beta_s = self.beta_syn

        if self.persistent and self._v_mem_state.shape[0] == B:
            v_mem = self._v_mem_state.clone()
            i_syn = self._i_syn_state.clone()
        else:
            v_mem = torch.zeros(B, D, device=device, dtype=dtype)
            i_syn = torch.zeros(B, D, device=device, dtype=dtype)
            if self.persistent:
                self._v_mem_state = torch.zeros(B, D, device=device, dtype=dtype)
                self._i_syn_state = torch.zeros(B, D, device=device, dtype=dtype)

        refrac_counter = torch.zeros(B, D, device=device, dtype=torch.int32)

        spikes_out = []
        v_trace = []

        for t in range(T):
            i_syn = beta_s * i_syn + current_in[t]

            refractory_mask = (refrac_counter > 0)
            v_mem = torch.where(
                refractory_mask,
                torch.full_like(v_mem, self.cfg.v_reset),
                beta_m * v_mem + (1.0 - beta_m) * i_syn,
            )

            s = spike_fn(v_mem, self.threshold, self.cfg.surrogate_alpha)

            if s.sum() > 0:
                cascade = self._cascade_amplify(s)
                i_syn = i_syn + cascade

            v_mem = v_mem - s * self.threshold.detach()
            refrac_counter = torch.where(
                s.bool(),
                torch.full_like(refrac_counter, self.cfg.refractory_t),
                (refrac_counter - 1).clamp(min=0),
            )

            spikes_out.append(s)
            v_trace.append(v_mem)

        if self.persistent:
            self._v_mem_state = v_mem.detach()
            self._i_syn_state = i_syn.detach()

        return torch.stack(spikes_out), torch.stack(v_trace)


# ─────────────────────────────────────────────────────────────────────────────
# §3  TEMPORAL ENCODER (v3 — Multi-Scale)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSpikeEncoder(nn.Module):
    def __init__(self, cfg: NordConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        T = cfg.T
        T_slow = cfg.T_slow

        self.embed = nn.Embedding(cfg.vocab_size, D)
        nn.init.kaiming_uniform_(self.embed.weight, a=math.sqrt(5))

        self.temporal_proj = nn.Linear(D, D, bias=False)
        self.drive_scale = nn.Parameter(torch.tensor(15.0))

        self.fast_basis = nn.Parameter(torch.randn(T, D) * 0.02)
        self.slow_basis = nn.Parameter(torch.randn(T_slow, D) * 0.02)
        self.slow_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, token_ids: Tensor) -> Tensor:
        B, S = token_ids.shape
        D = self.cfg.d_model

        x = self.temporal_proj(self.embed(token_ids))
        x = x.reshape(B * S, D)

        fast_gates = torch.sigmoid(self.fast_basis)
        fast = fast_gates.unsqueeze(1) * x.unsqueeze(0) * self.drive_scale

        slow_gates = torch.sigmoid(self.slow_basis)
        slow = slow_gates.unsqueeze(1) * x.unsqueeze(0) * self.slow_scale

        return torch.cat([fast, slow], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# §4  SPIKING SYNAPTIC RESONANCE (v3 — Sparse Top-K)
# ─────────────────────────────────────────────────────────────────────────────

class SpikingSynapticResonance(nn.Module):
    def __init__(self, cfg: NordConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.top_k = cfg.resonance_top_k
        D = cfg.d_model

        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)

        self.lif_q = AssociativeLIF(D, cfg)
        self.lif_k = AssociativeLIF(D, cfg)

        self.resonance_temp = nn.Parameter(
            torch.tensor(1.0 / math.sqrt(self.d_head)))

    def forward(self, x_spikes: Tensor) -> Tensor:
        T_total, B, S, D = x_spikes.shape
        H, Dh = self.n_heads, self.d_head

        x_flat = x_spikes.reshape(T_total * B * S, D)
        q_current = self.W_q(x_flat).reshape(T_total, B * S, D)
        k_current = self.W_k(x_flat).reshape(T_total, B * S, D)
        v_raw     = self.W_v(x_flat).reshape(T_total, B, S, D)

        q_spikes, _ = self.lif_q(q_current)
        k_spikes, _ = self.lif_k(k_current)

        q_spikes = q_spikes.reshape(T_total, B, S, H, Dh)
        k_spikes = k_spikes.reshape(T_total, B, S, H, Dh)

        q_flat = q_spikes.permute(1, 3, 2, 0, 4).reshape(B, H, S, T_total * Dh)
        k_flat = k_spikes.permute(1, 3, 2, 0, 4).reshape(B, H, S, T_total * Dh)

        resonance = torch.matmul(q_flat, k_flat.transpose(-2, -1))
        resonance = resonance * self.resonance_temp

        causal_mask = torch.triu(
            torch.ones(S, S, device=x_spikes.device, dtype=torch.bool), diagonal=1
        )
        resonance.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        K = min(self.top_k, S)
        if K < S:
            top_vals, top_idx = torch.topk(resonance, K, dim=-1)
            sparse_res = torch.full_like(resonance, float("-inf"))
            sparse_res.scatter_(-1, top_idx, top_vals)
            resonance = sparse_res

        attn = F.softmax(resonance.float(), dim=-1).to(resonance.dtype)

        v_mean = v_raw.mean(dim=0)
        v_heads = v_mean.reshape(B, S, H, Dh).permute(0, 2, 1, 3)
        context = torch.matmul(attn, v_heads)
        context = context.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.W_o(context)

        return out.unsqueeze(0).expand(T_total, -1, -1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# §5  NORD BLOCK (v3 — LeakyClamp + LayerScale)
# ─────────────────────────────────────────────────────────────────────────────

class SpikingFeedForward(nn.Module):
    def __init__(self, cfg: NordConfig):
        super().__init__()
        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.lif1 = AssociativeLIF(cfg.d_ff, cfg)
        self.lif2 = AssociativeLIF(cfg.d_model, cfg)

    def forward(self, x: Tensor) -> Tensor:
        T, B, S, D = x.shape
        h = self.up(x.reshape(T * B * S, D)).reshape(T, B * S, -1)
        h, _ = self.lif1(h)
        h = h.reshape(T, B, S, -1)
        h = self.down(h.reshape(T * B * S, -1)).reshape(T, B * S, D)
        h, _ = self.lif2(h)
        return h.reshape(T, B, S, D)


class LeakyClamp(nn.Module):
    def __init__(self, d: int, floor_init: float = -0.1, leak_init: float = 0.1):
        super().__init__()
        self.floor = nn.Parameter(torch.full((d,), floor_init))
        self.leak_raw = nn.Parameter(torch.full((d,), math.log(leak_init / (1 - leak_init + 1e-6))))

    @property
    def leak(self) -> Tensor:
        return torch.sigmoid(self.leak_raw)

    def forward(self, x: Tensor) -> Tensor:
        neg_part = (self.leak * x).clamp(min=self.floor)
        return torch.where(x >= 0, x, neg_part)


class NordBlock(nn.Module):
    def __init__(self, cfg: NordConfig, layer_idx: int = 0):
        super().__init__()
        D = cfg.d_model
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.resonance = SpikingSynapticResonance(cfg)
        self.ffn = SpikingFeedForward(cfg)

        init_scale = 0.1 / max(cfg.n_layers, 1)
        self.gamma_attn = nn.Parameter(torch.full((D,), init_scale))
        self.gamma_ffn  = nn.Parameter(torch.full((D,), init_scale))

        self.clamp = LeakyClamp(D, floor_init=cfg.clamp_floor)

    @staticmethod
    def _safe_norm(norm_layer: nn.LayerNorm, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            norm_layer.normalized_shape,
            norm_layer.weight.float() if norm_layer.weight is not None else None,
            norm_layer.bias.float() if norm_layer.bias is not None else None,
            norm_layer.eps,
        ).to(orig_dtype)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self._safe_norm(self.norm1, x)
        x = x + self.gamma_attn * self.resonance(x_norm)

        x_norm = self._safe_norm(self.norm2, x)
        x = x + self.gamma_ffn * self.ffn(x_norm)

        x = self.clamp(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# §6  STDP ENGINE (v3 — Reward-Modulated)
# ─────────────────────────────────────────────────────────────────────────────

class STDPEngine:
    def __init__(self, cfg: NordConfig):
        self.cfg = cfg
        self.a_plus  = cfg.stdp_a_plus
        self.a_minus = cfg.stdp_a_minus
        self.tau_plus  = cfg.stdp_tau_plus
        self.tau_minus = cfg.stdp_tau_minus
        self.w_max = cfg.stdp_w_max
        self.w_min = cfg.stdp_w_min
        self.reward_scale = cfg.stdp_reward_scale

        self._loss_ema: float = 10.0
        self._ema_decay: float = 0.99

    def update_reward(self, current_loss: float):
        self._loss_ema = self._ema_decay * self._loss_ema + (1 - self._ema_decay) * current_loss

    def _compute_reward(self, current_loss: float) -> float:
        delta = self._loss_ema - current_loss
        return float(torch.sigmoid(torch.tensor(delta * self.reward_scale)).item())

    @torch.no_grad()
    def compute_stdp_update(self, pre_spikes: Tensor, post_spikes: Tensor) -> Tensor:
        T = pre_spikes.shape[0]
        device = pre_spikes.device
        trace_pre  = torch.zeros_like(pre_spikes[0])
        trace_post = torch.zeros_like(post_spikes[0])
        decay_plus  = math.exp(-1.0 / self.tau_plus)
        decay_minus = math.exp(-1.0 / self.tau_minus)

        dW = torch.zeros(
            post_spikes.shape[1], pre_spikes.shape[1],
            device=device, dtype=pre_spikes.dtype)

        for t in range(T):
            trace_pre  = trace_pre * decay_plus  + pre_spikes[t]
            trace_post = trace_post * decay_minus + post_spikes[t]
            if post_spikes[t].any():
                dW += self.a_plus * torch.outer(post_spikes[t], trace_pre)
            if pre_spikes[t].any():
                dW -= self.a_minus * torch.outer(trace_post, pre_spikes[t])
        return dW

    @torch.no_grad()
    def apply_to_layer(self, layer: nn.Linear, pre_spikes: Tensor,
                       post_spikes: Tensor, current_loss: Optional[float] = None):
        if pre_spikes.dim() == 3:
            pre_spikes = pre_spikes.mean(dim=1)
        if post_spikes.dim() == 3:
            post_spikes = post_spikes.mean(dim=1)

        dW = self.compute_stdp_update(pre_spikes, post_spikes)

        if current_loss is not None:
            reward = self._compute_reward(current_loss)
            dW = dW * (2.0 * reward - 1.0)
            self.update_reward(current_loss)

        out_dim, in_dim = layer.weight.shape
        dW = dW[:out_dim, :in_dim]
        layer.weight.data = (layer.weight.data + dW).clamp(self.w_min, self.w_max)


# ─────────────────────────────────────────────────────────────────────────────
# §7  NORD MODEL (v3.5 — 500M Scale)
# ─────────────────────────────────────────────────────────────────────────────

class NordModel(nn.Module):
    def __init__(self, cfg: NordConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = TemporalSpikeEncoder(cfg)

        self.input_lif = AssociativeLIF(
            cfg.d_model, cfg, persistent=cfg.persistent_mem)

        self.blocks = nn.ModuleList([
            NordBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])

        self.readout_lif = AssociativeLIF(
            cfg.d_model, cfg, persistent=cfg.persistent_mem)

        self.readout_ema_raw = nn.Parameter(torch.tensor(1.4))

        self.readout_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.stdp = STDPEngine(cfg)
        self._stdp_cache: Dict[str, Tensor] = {}
        self._last_loss: Optional[float] = None

    @property
    def readout_ema_decay(self) -> Tensor:
        return torch.sigmoid(self.readout_ema_raw)

    def reset_state(self):
        self.input_lif.reset_state()
        self.readout_lif.reset_state()

    def forward(
        self,
        token_ids: Tensor,
        enable_stdp: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, S = token_ids.shape
        T_total = self.cfg.T_total
        D = self.cfg.d_model

        current = self.encoder(token_ids)
        spikes, _ = self.input_lif(current)
        spikes = spikes.reshape(T_total, B, S, D)

        _rates = [spikes.detach().mean()]

        if enable_stdp:
            self._stdp_cache["input"] = spikes.detach()

        x = spikes
        for i, block in enumerate(self.blocks):
            prev = x.detach() if enable_stdp else None
            x = block(x)
            _rates.append(x.detach().mean())

            if enable_stdp and prev is not None:
                self._stdp_cache[f"block_{i}_pre"] = prev
                self._stdp_cache[f"block_{i}_post"] = x.detach()

        x_flat = x.reshape(T_total, B * S, D)
        readout_spikes, v_membrane = self.readout_lif(x_flat)

        alpha = self.readout_ema_decay
        ema = torch.zeros(B * S, D, device=x.device, dtype=v_membrane.dtype)
        for t in range(T_total):
            ema = alpha * ema + (1 - alpha) * v_membrane[t]
        v_smooth = ema.reshape(B, S, D)

        s_mean = readout_spikes.mean(dim=0).reshape(B, S, D)
        readout = v_smooth + s_mean

        x_norm = F.layer_norm(
            readout.float(),
            self.readout_norm.normalized_shape,
            self.readout_norm.weight.float() if self.readout_norm.weight is not None else None,
            self.readout_norm.bias.float() if self.readout_norm.bias is not None else None,
            self.readout_norm.eps,
        ).to(readout.dtype)
        logits = self.lm_head(x_norm)

        stats = {}
        stats["encoder_spike_rate"] = _rates[0].item()
        for i in range(self.cfg.n_layers):
            stats[f"block_{i}_spike_rate"] = _rates[i + 1].item()
        out_rate = readout_spikes.detach().mean().item()
        stats["output_spike_rate"] = out_rate
        stats["sparsity"] = 1.0 - out_rate

        return logits, stats

    @torch.no_grad()
    def stdp_update(self, current_loss: Optional[float] = None):
        loss_val = current_loss or self._last_loss
        for i, block in enumerate(self.blocks):
            pre_key  = f"block_{i}_pre"
            post_key = f"block_{i}_post"
            if pre_key in self._stdp_cache and post_key in self._stdp_cache:
                pre  = self._stdp_cache[pre_key]
                post = self._stdp_cache[post_key]
                T_dim = pre.shape[0]
                pre_flat  = pre.reshape(T_dim, -1, self.cfg.d_model).mean(dim=1)
                post_flat = post.reshape(T_dim, -1, self.cfg.d_model).mean(dim=1)
                self.stdp.apply_to_layer(
                    block.resonance.W_v, pre_flat, post_flat,
                    current_loss=loss_val,
                )
        self._stdp_cache.clear()

    def set_last_loss(self, loss: float):
        self._last_loss = loss

    def count_params(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total: {total/1e6:.1f}M | Trainable: {train/1e6:.1f}M"


# ─────────────────────────────────────────────────────────────────────────────
# §8  UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def estimate_vram(cfg: NordConfig) -> str:
    param_bytes = (
        cfg.vocab_size * cfg.d_model
        + cfg.n_layers * (
            4 * cfg.d_model * cfg.d_model
            + 2 * cfg.d_model * cfg.d_ff
            + 6 * cfg.d_model
            + cfg.n_clusters * cfg.n_clusters
        )
        + cfg.vocab_size * cfg.d_model
    ) * (2 if cfg.dtype == torch.float16 else 4)

    act_bytes = cfg.T_total * 1 * cfg.max_seq_len * cfg.d_model * cfg.n_layers * 2 * 2
    total_gb = (param_bytes + act_bytes) / (1024 ** 3)
    return (
        f"Parameters:   ~{param_bytes / 1e6:.0f} MB\n"
        f"Activations:  ~{act_bytes / 1e6:.0f} MB  (B=1, S={cfg.max_seq_len})\n"
        f"Total Est:    ~{total_gb:.2f} GB  (target: 48 GB L40 / 24 GB 3090)"
    )
