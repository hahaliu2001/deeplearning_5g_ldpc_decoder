"""Table-driven neural layered min-sum LDPC decoder.

This model follows the memory layout of ``c_lib/fixedpoint_ldpc_decoder.c``:
it keeps one variable-node LLR array (``LQ``) and one check-to-variable
message buffer (``Lr``) only for the non-null base-graph entries.  It avoids
the dense ``(checks, variables, batch)`` tensors used by the older prototype.

The base-graph columns, shifts, row offsets, ``Zc`` and ``iLS`` are read from
the compiled C library, so the Python model uses the same ``ldpc_tables.c``
tables and ``gen_ldpc_parameters`` logic as the C decoder.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


BG1_NUM_ROWS = 46
BG1_NUM_COLS = 68
BG1_MAX_NON_NULL_ELEMENTS = 316
BG2_NUM_ROWS = 42
BG2_NUM_COLS = 52
BG2_MAX_NON_NULL_ELEMENTS = 197


class LdpcInConfig(ctypes.Structure):
    _fields_ = [
        ("bgn", ctypes.c_uint16),
        ("A", ctypes.c_uint16),
        ("crctype", ctypes.c_uint16),
        ("E", ctypes.c_uint16),
        ("rv", ctypes.c_uint16),
        ("Qm", ctypes.c_uint16),
        ("HARQ_on", ctypes.c_uint16),
    ]


class LdpcConfig(ctypes.Structure):
    _fields_ = [
        ("bgn", ctypes.c_uint16),
        ("A", ctypes.c_uint16),
        ("crctype", ctypes.c_uint16),
        ("E", ctypes.c_uint16),
        ("rv", ctypes.c_uint16),
        ("Qm", ctypes.c_uint16),
        ("HARQ_on", ctypes.c_uint16),
        ("B", ctypes.c_uint16),
        ("Zc", ctypes.c_uint16),
        ("iLS", ctypes.c_uint16),
        ("K", ctypes.c_uint16),
        ("F", ctypes.c_uint16),
        ("N", ctypes.c_uint16),
        ("k0", ctypes.c_uint16),
        ("Vij_array", ctypes.POINTER(ctypes.c_uint16)),
        ("HBG_array", ctypes.POINTER(ctypes.c_uint16)),
        ("HBG_offset_array", ctypes.POINTER(ctypes.c_uint16)),
    ]


@dataclass(frozen=True)
class LdpcTableConfig:
    bgn: int
    A: int
    crctype: int
    E: int
    rv: int
    Qm: int
    HARQ_on: int
    B: int
    Zc: int
    iLS: int
    K: int
    F: int
    N: int
    k0: int
    num_rows: int
    num_cols: int
    num_edges: int
    hbg_columns: Tuple[int, ...]
    vij: Tuple[int, ...]
    row_offsets: Tuple[int, ...]


def _default_lib_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "c_lib", "libnrLDPC.so")


def _load_c_library(lib_path: Optional[str] = None) -> ctypes.CDLL:
    path = lib_path or _default_lib_path()
    lib = ctypes.CDLL(path)
    lib.gen_ldpc_parameters.argtypes = [
        ctypes.POINTER(LdpcConfig),
        ctypes.POINTER(LdpcInConfig),
    ]
    lib.gen_ldpc_parameters.restype = ctypes.c_int
    return lib


def load_ldpc_table_config(
    bgn: int,
    A: int,
    crctype: int,
    E: int,
    rv: int,
    Qm: int,
    HARQ_on: int = 0,
    lib_path: Optional[str] = None,
) -> LdpcTableConfig:
    """Load LDPC parameters and base-graph tables through the C library."""
    lib = _load_c_library(lib_path)
    in_config = LdpcInConfig(bgn, A, crctype, E, rv, Qm, HARQ_on)
    config = LdpcConfig()
    status = lib.gen_ldpc_parameters(ctypes.byref(config), ctypes.byref(in_config))
    if status != 0:
        raise ValueError("gen_ldpc_parameters failed for the supplied LDPC config")

    if config.bgn == 1:
        num_rows = BG1_NUM_ROWS
        num_cols = BG1_NUM_COLS
        num_edges = BG1_MAX_NON_NULL_ELEMENTS
    elif config.bgn == 2:
        num_rows = BG2_NUM_ROWS
        num_cols = BG2_NUM_COLS
        num_edges = BG2_MAX_NON_NULL_ELEMENTS
    else:
        raise ValueError("bgn must be 1 or 2")

    hbg_columns = tuple(int(config.HBG_array[i]) for i in range(num_edges))
    vij = tuple(int(config.Vij_array[i]) for i in range(num_edges))
    row_offsets = tuple(int(config.HBG_offset_array[i]) for i in range(num_rows + 1))

    return LdpcTableConfig(
        bgn=int(config.bgn),
        A=int(config.A),
        crctype=int(config.crctype),
        E=int(config.E),
        rv=int(config.rv),
        Qm=int(config.Qm),
        HARQ_on=int(config.HARQ_on),
        B=int(config.B),
        Zc=int(config.Zc),
        iLS=int(config.iLS),
        K=int(config.K),
        F=int(config.F),
        N=int(config.N),
        k0=int(config.k0),
        num_rows=num_rows,
        num_cols=num_cols,
        num_edges=num_edges,
        hbg_columns=hbg_columns,
        vij=vij,
        row_offsets=row_offsets,
    )


def _inverse_sigmoid(value: float) -> torch.Tensor:
    v = torch.tensor(float(value)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(v / (1.0 - v))


class CTableNeuralLayeredMinSumDecoder(nn.Module):
    """Memory-conscious neural layered min-sum decoder.

    Args:
        table_config: LDPC table/config object from ``load_ldpc_table_config``.
        num_iterations: number of layered decoding iterations.
        learnable_scaling: train normalized min-sum alpha per iteration/HBG row.
        learnable_beta: train offset beta per iteration/HBG row.
        init_alpha: initial alpha after sigmoid mapping.
        init_beta: initial beta after sigmoid mapping.
        clip_llr: optional LLR clipping value. ``None`` disables clipping.

    Input LLR shape is ``[batch, N + 2*Zc]`` or ``[N + 2*Zc]``, matching the
    C decoder's ``LQ`` layout.  The returned final LLR has the same length.
    """

    def __init__(
        self,
        table_config: LdpcTableConfig,
        num_iterations: int = 8,
        learnable_scaling: bool = True,
        learnable_beta: bool = True,
        init_alpha: float = 0.75,
        init_beta: float = 0.0,
        clip_llr: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.config = table_config
        self.num_iterations = int(num_iterations)
        self.clip_llr = clip_llr
        self.n = table_config.N + 2 * table_config.Zc
        self.B = table_config.B
        self.num_rows = table_config.num_rows
        self.num_edges = table_config.num_edges
        self.Zc = table_config.Zc

        raw_alpha = _inverse_sigmoid(init_alpha).repeat(self.num_iterations, self.num_rows)
        raw_beta = _inverse_sigmoid(init_beta).repeat(self.num_iterations, self.num_rows)
        if learnable_scaling:
            self.raw_alpha = nn.Parameter(raw_alpha)
        else:
            self.register_buffer("raw_alpha", raw_alpha)
        if learnable_beta:
            self.raw_beta = nn.Parameter(raw_beta)
        else:
            self.register_buffer("raw_beta", raw_beta)

        self._row_degrees: List[int] = []
        self._row_columns: List[torch.Tensor] = []
        self._row_flat_columns: List[torch.Tensor] = []

        zx = torch.arange(self.Zc, dtype=torch.long).unsqueeze(1)
        for row in range(self.num_rows):
            start = table_config.row_offsets[row]
            stop = table_config.row_offsets[row + 1]
            degree = stop - start
            hbg_cols = torch.tensor(table_config.hbg_columns[start:stop], dtype=torch.long)
            shifts = torch.tensor(table_config.vij[start:stop], dtype=torch.long)
            lifted_cols = hbg_cols.unsqueeze(0) * self.Zc + ((shifts.unsqueeze(0) + zx) % self.Zc)

            self._row_degrees.append(degree)
            self.register_buffer(f"row_{row}_columns", lifted_cols)
            self.register_buffer(f"row_{row}_flat_columns", lifted_cols.reshape(-1))
            self._row_columns.append(getattr(self, f"row_{row}_columns"))
            self._row_flat_columns.append(getattr(self, f"row_{row}_flat_columns"))

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha).clamp(min=1e-6, max=1.0)

    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_beta).clamp(min=0.0, max=1.0 - 1e-6)

    def forward(
        self,
        llr: torch.Tensor,
        return_llr: bool = False,
        return_iteration_llrs: bool = False,
    ):
        single = False
        if llr.dim() == 1:
            llr = llr.unsqueeze(0)
            single = True
        if llr.dim() != 2:
            raise ValueError("llr must have shape [n] or [batch, n]")
        if llr.shape[1] != self.n:
            raise ValueError(f"expected llr length {self.n}, got {llr.shape[1]}")

        batch_size = llr.shape[0]
        LQ = llr.clone()
        if self.clip_llr is not None:
            LQ = LQ.clamp(-self.clip_llr, self.clip_llr)

        Lr_rows = [
            llr.new_zeros((batch_size, self.Zc, degree))
            for degree in self._row_degrees
        ]
        alphas = self.alpha()
        betas = self.beta()
        iteration_llrs = [] if return_iteration_llrs else None

        for iteration in range(self.num_iterations):
            for row in range(self.num_rows):
                degree = self._row_degrees[row]
                flat_cols = self._row_flat_columns[row].to(device=llr.device)
                old_lr = Lr_rows[row]

                rel_lq = LQ[:, flat_cols].reshape(batch_size, self.Zc, degree) - old_lr
                signs = torch.where(rel_lq < 0, -torch.ones_like(rel_lq), torch.ones_like(rel_lq))
                abs_rel = rel_lq.abs()
                min1, min1_idx = torch.min(abs_rel, dim=2)

                min1_mask = torch.nn.functional.one_hot(min1_idx, num_classes=degree).to(torch.bool)
                second_source = abs_rel.masked_fill(min1_mask, float("inf"))
                min2, _ = torch.min(second_source, dim=2)

                sign_prod = torch.prod(signs, dim=2, keepdim=True)
                use_min = torch.where(min1_mask, min2.unsqueeze(2), min1.unsqueeze(2))
                magnitude = torch.clamp(use_min - betas[iteration, row], min=0.0)
                magnitude = alphas[iteration, row] * magnitude
                new_lr = sign_prod * signs * magnitude

                new_lq_values = rel_lq + new_lr
                if self.clip_llr is not None:
                    new_lq_values = new_lq_values.clamp(-self.clip_llr, self.clip_llr)

                next_LQ = LQ.clone()
                next_LQ[:, flat_cols] = new_lq_values.reshape(batch_size, -1)
                LQ = next_LQ
                Lr_rows[row] = new_lr
            if iteration_llrs is not None:
                iteration_llrs.append(LQ.clone())

        final_llr = LQ
        bits = (final_llr < 0).to(torch.int64)
        if single:
            bits = bits.squeeze(0)
            final_llr = final_llr.squeeze(0)
            if iteration_llrs is not None:
                iteration_llrs = [iter_llr.squeeze(0) for iter_llr in iteration_llrs]
        if return_llr and return_iteration_llrs:
            return bits, final_llr, iteration_llrs
        if return_llr:
            return bits, final_llr
        if return_iteration_llrs:
            return bits, iteration_llrs
        return bits

    def trainable_parameters(self):
        return [(name, tuple(param.shape)) for name, param in self.named_parameters() if param.requires_grad]

    def memory_summary(self, batch_size: int, dtype_bytes: int = 4) -> Dict[str, float]:
        lq_values = batch_size * self.n
        lr_values = batch_size * self.num_edges * self.Zc
        return {
            "LQ_MB": lq_values * dtype_bytes / 1024 / 1024,
            "Lr_MB": lr_values * dtype_bytes / 1024 / 1024,
            "total_message_MB": (lq_values + lr_values) * dtype_bytes / 1024 / 1024,
        }


__all__ = [
    "CTableNeuralLayeredMinSumDecoder",
    "LdpcTableConfig",
    "load_ldpc_table_config",
]
