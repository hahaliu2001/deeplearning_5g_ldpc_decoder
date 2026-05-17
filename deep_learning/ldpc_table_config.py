"""LDPC table/config loading helpers for deep-learning decoder models."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional, Tuple


BG1_NUM_ROWS = 46
BG1_NUM_COLS = 68
BG1_MAX_NON_NULL_ELEMENTS = 316
BG2_NUM_ROWS = 42
BG2_NUM_COLS = 52
BG2_MAX_NON_NULL_ELEMENTS = 197


class LdpcInConfig(ctypes.Structure):
    """ctypes mirror of the public C LDPC input configuration."""

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
    """ctypes mirror of the derived C LDPC decoder configuration."""

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
    """Sparse LDPC table metadata used by the PyTorch decoder."""

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
    """Return the repository-local default path to the LDPC C shared library."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "c_lib", "libnrLDPC.so")


def _load_c_library(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """Load the C LDPC shared library and register gen_ldpc_parameters()."""
    path = lib_path or _default_lib_path()
    lib = ctypes.CDLL(path)
    lib.gen_ldpc_parameters.argtypes = [
        ctypes.POINTER(LdpcConfig),
        ctypes.POINTER(LdpcInConfig),
    ]
    lib.gen_ldpc_parameters.restype = ctypes.c_int
    return lib


def _base_graph_dimensions(bgn: int) -> Tuple[int, int, int]:
    """Return row count, column count, and non-null edge count for a base graph."""
    if bgn == 1:
        return BG1_NUM_ROWS, BG1_NUM_COLS, BG1_MAX_NON_NULL_ELEMENTS
    if bgn == 2:
        return BG2_NUM_ROWS, BG2_NUM_COLS, BG2_MAX_NON_NULL_ELEMENTS
    raise ValueError("bgn must be 1 or 2")


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
    """Load derived LDPC parameters and sparse base-graph tables from C."""
    lib = _load_c_library(lib_path)
    in_config = LdpcInConfig(bgn, A, crctype, E, rv, Qm, HARQ_on)
    config = LdpcConfig()
    status = lib.gen_ldpc_parameters(ctypes.byref(config), ctypes.byref(in_config))
    if status != 0:
        raise ValueError("gen_ldpc_parameters failed for the supplied LDPC config")

    num_rows, num_cols, num_edges = _base_graph_dimensions(config.bgn)
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


__all__ = [
    "LdpcConfig",
    "LdpcInConfig",
    "LdpcTableConfig",
    "load_ldpc_table_config",
]
