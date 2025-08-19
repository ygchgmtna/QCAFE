"""
QAFv2 — readability-focused refactor with stronger encapsulation
----------------------------------------------------------------
Goals:
- keep the public interface stable: QuantumAttention(in_embed=..., config=QAFConfig(...))
- preserve core behavior (two-ring / shifted-cross / anneal) while making code easier to read
- centralize masking & anneal scaling, reduce repetition, and add docstrings

Notes:
- Circuits remain function-level qnodes (stable with PennyLane), but are called through
  small private wrappers on the module to make call sites clean.
- All gate-parameter shapes and semantics match the original file (QAFv2). See bottom for a
  sanity-check comment on expected shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# =============================
# Config & constants
# =============================
qubits: int = 4  # number of qubits for the quantum attention core


@dataclass
class QAFConfig:
    """Feature switches (same semantics as QAFv2).

    - use_two_ring:  CRX ring + (IsingXX ring | fallback CNOT ring)
    - use_shifted_cross: add the shifted cross-links in score circuit
    - use_anneal: scale entangling params and value gates by an anneal alpha
    """
    use_two_ring: bool = True
    use_shifted_cross: bool = True
    use_anneal: bool = True


# Devices
_dev_score = qml.device("default.qubit", wires=qubits * 2, shots=None)
_dev_value = qml.device("default.qubit", wires=qubits, shots=None)


# =============================
# Helpers & primitives
# =============================

def _to_tensor_preserve_grad(x, *, device=None) -> torch.Tensor:
    """Convert qnode return to a torch.Tensor while preserving grad when possible."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return torch.empty(0, device=device, dtype=torch.get_default_dtype())
        if all(torch.is_tensor(e) for e in x):
            return torch.stack([e.to(device=device) for e in x])
        return torch.as_tensor(x, device=device, dtype=torch.get_default_dtype())
    return torch.as_tensor(x, device=device, dtype=torch.get_default_dtype())


def _apply_rot_layer(rot_params: torch.Tensor, *, offset: int = 0, n: int = qubits, start: int = 0) -> None:
    """Apply an RZ-RX-RZ layer.

    rot_params is organized as [layer, qubit, gate] where each layer has 3 parameters per qubit.
    Use start to offset into the correct layer.
    """
    for i in range(n):
        idx = start + 3 * i
        w = offset + i
        qml.RZ(rot_params[idx], wires=w)
        qml.RX(rot_params[idx + 1], wires=w)
        qml.RZ(rot_params[idx + 2], wires=w)


def _two_ring_block(rot_params: torch.Tensor, crx_params: torch.Tensor, *, offset: int = 0) -> None:
    """Two-ring variational block used in Q/K/V branches.

    Structure (backward-compatible with the original 3n/6n layout):
      - If depth == 1 (3n params): apply one rotation *before* entanglers
      - If depth >= 2 (>=6n params): apply (depth-1) rotation layers *before*, and 1 rotation layer *after*
      - Entanglers: CRX ring + optional IsingXX ring (or CNOT fallback)
    """
    n = qubits
    depth = len(rot_params) // (3 * n)

    # --- pre-rotation layers ---
    pre_layers = max(0, depth - 1) if depth >= 2 else depth
    for layer in range(pre_layers):
        _apply_rot_layer(rot_params, offset=offset, n=n, start=3 * n * layer)

    # --- entanglers ---
    # CRX ring (first n angles)
    for i in range(n):
        a = offset + i
        b = offset + (i + 1) % n
        qml.CRX(crx_params[i], wires=[a, b])

    # IsingXX ring if provided, else fallback to CNOT ring
    if len(crx_params) >= 2 * n:
        for i in range(n):
            a = offset + i
            b = offset + (i + 1) % n
            qml.IsingXX(crx_params[n + i], wires=[a, b])
    else:
        for i in range(n):
            a = offset + i
            b = offset + (i + 1) % n
            qml.CNOT(wires=[a, b])

    # --- post-rotation layer (only one), if depth >= 2 ---
    if depth >= 2:
        _apply_rot_layer(rot_params, offset=offset, n=n, start=3 * n * (depth - 1))

# =============================
# Circuits (qnodes)
# =============================
@qml.qnode(_dev_score, interface="torch", diff_method="backprop")
def _score_circuit(
    xq: torch.Tensor,
    xk: torch.Tensor,
    weights_q_rot: torch.Tensor,
    weights_q_crx: torch.Tensor,
    weights_k_rot: torch.Tensor,
    weights_k_crx: torch.Tensor,
    weights_cross_rot: torch.Tensor,
    weights_cross_crx: torch.Tensor,
    weights_cross_crx2: torch.Tensor,
) -> List[torch.Tensor]:
    # Q branch
    qml.AngleEmbedding(xq, wires=list(range(qubits)))
    _two_ring_block(weights_q_rot, weights_q_crx)

    # K branch
    qml.AngleEmbedding(xk, wires=list(range(qubits, 2 * qubits)))
    _two_ring_block(weights_k_rot, weights_k_crx, offset=qubits)

    # Cross CRX: direct pair (i <-> i + n)
    for i in range(qubits):
        qml.CRX(weights_cross_crx[i], wires=[i, i + qubits])
        qml.CRX(weights_cross_crx[i + qubits], wires=[i + qubits, i])

    # Shifted cross (i -> (i+1)%n + n) both directions
    for i in range(qubits):
        qml.CRX(weights_cross_crx2[i], wires=[i, (i + 1) % qubits + qubits])
    for i in range(qubits):
        qml.CRX(weights_cross_crx2[i + qubits], wires=[i + qubits, (i + 1) % qubits])

    # Residual cross CNOTs (bidirectional)
    for i in range(qubits):
        qml.CNOT(wires=[i, i + qubits])
        qml.CNOT(wires=[i + qubits, i])

    # Second rotation on Q (always just one layer)
    for i in range(qubits):
        idx = 3 * i
        qml.RZ(weights_cross_rot[idx], wires=i)
        qml.RX(weights_cross_rot[idx + 1], wires=i)
        qml.RZ(weights_cross_rot[idx + 2], wires=i)

    # Measure Z and X on Q wires
    return [qml.expval(qml.PauliZ(w)) for w in range(qubits)] + [
        qml.expval(qml.PauliX(w)) for w in range(qubits)
    ]


@qml.qnode(_dev_value, interface="torch", diff_method="backprop")
def _value_circuit(
    xv: torch.Tensor,
    score_amp: torch.Tensor,
    weights_v_rot: torch.Tensor,
    weights_v_crx: torch.Tensor,
    gates: torch.Tensor,
) -> List[torch.Tensor]:
    qml.AngleEmbedding(xv, wires=list(range(qubits)))
    _two_ring_block(weights_v_rot, weights_v_crx)

    # score-controlled RX on value wires
    for i in range(qubits):
        qml.RX(torch.tanh(score_amp[i]) * gates[i], wires=i)

    # CNOT ring (kept as in v2)
    for i in range(qubits):
        qml.CNOT(wires=[i, (i + 1) % qubits])

    return [qml.expval(qml.PauliZ(w)) for w in range(qubits)] + [
        qml.expval(qml.PauliX(w)) for w in range(qubits)
    ]


# =============================
# Module: QuantumAttention
# =============================
class QuantumAttention(nn.Module):
    def __init__(self, in_embed: int = 96, config: QAFConfig | None = None) -> None:
        super().__init__()
        self.qubits = qubits
        self.config = config or QAFConfig()
        embed_dim = self.qubits

        # classical preprocess
        self.pre_text = nn.Linear(in_embed, embed_dim)
        self.pre_image = nn.Linear(in_embed, embed_dim)

        # rotation params (layers * 3 * qubits for Q/K/V; 3 * qubits for cross-rot)
        self.weights_q_rot = nn.Parameter(torch.empty(self.qubits * 6).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_k_rot = nn.Parameter(torch.empty(self.qubits * 6).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_v_rot = nn.Parameter(torch.empty(self.qubits * 6).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_cross_rot = nn.Parameter(torch.empty(self.qubits * 3).uniform_(-np.pi / 2, np.pi / 2))

        # CRX params: first n for CRX ring, next n for IsingXX (if enabled)
        self.weights_q_crx = nn.Parameter(torch.randn(self.qubits * 2) * 0.1)
        self.weights_k_crx = nn.Parameter(torch.randn(self.qubits * 2) * 0.1)
        self.weights_v_crx = nn.Parameter(torch.randn(self.qubits * 2) * 0.1)

        # cross entanglers (direct & shifted)
        self.weights_cross_crx = nn.Parameter(torch.randn(self.qubits * 2) * 0.1)
        self.weights_cross_crx2 = nn.Parameter(torch.randn(self.qubits * 2) * 0.1)

        # value gates
        self.score_gates = nn.Parameter(torch.ones(self.qubits) * np.pi)

        # anneal factor (buffer, non-trainable)
        self.register_buffer("anneal", torch.tensor(1.0))

        # feedforward head (kept, minor naming cleanup)
        self.norm = nn.LayerNorm(24)
        mlp_hidden_dim = 4 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(24, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(0.5)

        self.linear = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.ReLU())

    # -------- public API --------
    def set_anneal(self, alpha: float) -> None:
        """Set anneal factor alpha in [0, 1]."""
        self.anneal.fill_(float(alpha))

    def forward(self, x_text: torch.Tensor, x_image: torch.Tensor) -> torch.Tensor:
        xq = self.pre_text(x_text)
        xk = self.pre_image(x_image)

        # Compute effective (masked & annealed) parameters
        alpha = self._alpha()
        wq_crx, wk_crx, wv_crx, wc_crx, wc_crx2, gates = self._effective_params(alpha)

        outputs: List[torch.Tensor] = []
        for i in range(xq.shape[0]):
            # Score circuit
            score_out = _score_circuit(
                xq[i],
                xk[i],
                self.weights_q_rot,
                wq_crx,
                self.weights_k_rot,
                wk_crx,
                self.weights_cross_rot,
                wc_crx,
                wc_crx2,
            )
            score_1 = _to_tensor_preserve_grad(score_out, device=xq.device)

            # Value circuit
            amp_1 = torch.sqrt(score_1[: self.qubits] ** 2 + score_1[self.qubits :] ** 2)
            value_out = _value_circuit(xk[i], amp_1, self.weights_v_rot, wv_crx, gates)
            value_1 = _to_tensor_preserve_grad(value_out, device=xq.device)

            outputs.append(value_1)

        return torch.stack(outputs).float()

    # -------- private helpers --------
    def _alpha(self) -> torch.Tensor:
        """Return the effective anneal scalar (device-correct), 1.0 if disabled."""
        if self.config.use_anneal:
            return self.anneal
        return torch.tensor(1.0, device=self.anneal.device)

    def _mask_two_ring(self, params: torch.Tensor) -> torch.Tensor:
        """When two-ring is disabled, zero the second-ring angles while preserving shape."""
        if self.config.use_two_ring:
            return params
        p = params.clone()
        if p.numel() >= 2 * self.qubits:
            p[self.qubits :] = 0.0
        return p

    def _mask_shifted(self, params: torch.Tensor) -> torch.Tensor:
        """Disable shifted cross-links by zeroing their parameters."""
        return params if self.config.use_shifted_cross else torch.zeros_like(params)

    def _effective_params(
        self, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine masking and anneal scaling for all entangling parameters & gates."""
        wq_crx_eff = self._mask_two_ring(self.weights_q_crx) * alpha
        wk_crx_eff = self._mask_two_ring(self.weights_k_crx) * alpha
        wv_crx_eff = self._mask_two_ring(self.weights_v_crx) * alpha
        wc_crx_eff = self.weights_cross_crx * alpha
        wc_crx2_eff = self._mask_shifted(self.weights_cross_crx2) * alpha
        gates_eff = self.score_gates * alpha
        return wq_crx_eff, wk_crx_eff, wv_crx_eff, wc_crx_eff, wc_crx2_eff, gates_eff


# =============================
# Shape sanity check (comment only)
# -----------------------------
# - weights_*_rot: depth * 3 * n for Q/K/V; cross-rot: 3 * n
# - weights_*_crx: 2 * n (first n for CRX ring, next n for IsingXX or masked)
# - weights_cross_crx / _crx2: 2 * n (forward + backward)
# - score_gates: n
# Return of circuits: 2 * n expvals (Z on n wires + X on n wires)
