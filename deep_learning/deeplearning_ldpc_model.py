"""Table-driven neural layered min-sum LDPC decoder.

The model follows the memory layout used by the C fixed-point decoder: one
variable-node LLR array (``LQ``) plus one check-to-variable message buffer
(``Lr``) for each non-null base-graph row. This avoids dense
``(checks, variables, batch)`` tensors and keeps training memory much lower.

Base-graph columns, circulant shifts, row offsets, ``Zc`` and ``iLS`` are
provided by ``ldpc_table_config.py``. That keeps the PyTorch model aligned with
``ldpc_tables.c`` and the same ``gen_ldpc_parameters`` logic used by the C
decoder.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from deep_learning.ldpc_table_config import LdpcTableConfig, load_ldpc_table_config

def _inverse_sigmoid(value: float) -> torch.Tensor:
    """Map a bounded value in (0, 1) back to its raw trainable logit."""
    v = torch.tensor(float(value)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(v / (1.0 - v))

def _inverse_bounded_sigmoid(value: float, min_value: float, max_value: float) -> torch.Tensor:
    """Map a value in (min_value, max_value) back to an unconstrained logit."""
    if not min_value < float(value) < max_value:
        raise ValueError(
            f"bounded sigmoid value must satisfy {min_value} < value < {max_value}; got {value}"
        )
    normalized = (float(value) - min_value) / (max_value - min_value)
    return _inverse_sigmoid(normalized)

def _inverse_bounded_sigmoid_tensor(
    values: torch.Tensor,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    """Vectorized version of _inverse_bounded_sigmoid()."""
    if bool(torch.any(values <= min_value).item()) or bool(torch.any(values >= max_value).item()):
        raise ValueError(
            f"bounded sigmoid values must satisfy {min_value} < value < {max_value}"
        )
    normalized = (values - min_value) / (max_value - min_value)
    normalized = normalized.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(normalized / (1.0 - normalized))

def _row_degrees(table_config: LdpcTableConfig) -> List[int]:
    """Return the number of non-null base-graph entries in each HBG row."""
    return [
        table_config.row_offsets[row + 1] - table_config.row_offsets[row]
        for row in range(table_config.num_rows)
    ]

def _degree_alpha_indices(row_degrees: Sequence[int]) -> Tuple[Tuple[int, ...], torch.Tensor]:
    """Return sorted unique degrees and each row's index into that list.

    Degree mode stores one alpha per unique row degree. This lookup maps each
    HBG row to the correct alpha entry.
    """
    unique_degrees = tuple(sorted(set(row_degrees)))
    degree_to_index = {degree: idx for idx, degree in enumerate(unique_degrees)}
    indices = torch.tensor(
        [degree_to_index[degree] for degree in row_degrees],
        dtype=torch.long,
    )
    return unique_degrees, indices


class CTableNeuralLayeredMinSumDecoder(nn.Module):
    """Memory-conscious neural layered min-sum decoder.

    Args:
        table_config: LDPC table/config object from ``load_ldpc_table_config``.
        num_iterations: number of layered decoding iterations.
        learnable_scaling: train normalized min-sum alpha values.
        learnable_beta: train offset beta per iteration/HBG row.
        init_alpha: initial alpha after sigmoid mapping. It may be a scalar for
            any mode, or an explicit table with shape ``[1]`` for ``single``,
            ``[num_iterations, num_rows]`` for ``checkpoint``, or
            ``[num_unique_row_degrees]`` for ``degree``.
        init_beta: initial beta after sigmoid mapping.
        alpha_mode: ``"single"`` uses one alpha for all iterations and HBG rows.
            ``"checkpoint"`` matches the C neural decoder table layout and uses
            one alpha for each iteration and HBG row. ``"degree"`` uses one
            alpha for each exact HBG row degree, independent of iteration and
            row index.
        alpha_min: minimum normalized min-sum alpha after sigmoid mapping.
        alpha_max: maximum normalized min-sum alpha after sigmoid mapping.
        init_alpha_degree_values: optional alpha initial values for
            ``alpha_mode="degree"``. Provide either one value per unique row
            degree or a 1-based exact-degree table where degree ``d`` reads
            entry ``d - 1``.
        clip_llr: optional LLR clipping value. ``None`` disables clipping.
        detach_check_update_inputs: detach layered decoder state before each
            check-node update. This truncates backpropagation through previous
            decoder states while still training alpha/beta for the current
            update.

    Input LLR shape is ``[batch, N + 2*Zc]`` or ``[N + 2*Zc]``, matching the
    C decoder's ``LQ`` layout.  The returned final LLR has the same length.
    """

    def __init__(
        self,
        table_config: LdpcTableConfig,
        num_iterations: int = 8,
        learnable_scaling: bool = True,
        learnable_beta: bool = True,
        init_alpha: Union[float, Sequence[Sequence[float]], torch.Tensor] = 0.75,
        init_beta: float = 0.0,
        alpha_mode: str = "single",
        alpha_min: float = 0.5,
        alpha_max: float = 1.0 - 1e-6,
        init_alpha_degree_values: Optional[Sequence[float]] = None,
        clip_llr: Optional[float] = None,
        detach_check_update_inputs: bool = False,
    ) -> None:
        """Initialize the unrolled neural layered min-sum decoder.

        The constructor converts the sparse base-graph table into lifted column
        index buffers, creates trainable or fixed alpha/beta tensors, and stores
        the C-compatible decoder dimensions used by ``forward()``.
        """
        super().__init__()
        if alpha_mode not in {"single", "checkpoint", "degree"}:
            raise ValueError("alpha_mode must be 'single', 'checkpoint', or 'degree'")
        if not 0.0 < alpha_min < alpha_max < 1.0:
            raise ValueError("alpha_min and alpha_max must satisfy 0 < alpha_min < alpha_max < 1")

        self.config = table_config
        self.num_iterations = int(num_iterations)
        self.clip_llr = clip_llr
        self.detach_check_update_inputs = bool(detach_check_update_inputs)
        self.alpha_mode = alpha_mode
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.n = table_config.N + 2 * table_config.Zc
        self.B = table_config.B
        self.num_rows = table_config.num_rows
        self.num_edges = table_config.num_edges
        self.Zc = table_config.Zc

        row_degrees = _row_degrees(table_config)
        unique_degrees, degree_alpha_indices = _degree_alpha_indices(row_degrees)
        self.unique_row_degrees = unique_degrees
        # In degree mode, each HBG row indexes the alpha value associated with
        # its exact row degree.
        self.register_buffer("row_alpha_degree_indices", degree_alpha_indices)

        raw_alpha = self._make_raw_alpha(init_alpha, init_alpha_degree_values)
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
            degree = row_degrees[row]
            hbg_cols = torch.tensor(table_config.hbg_columns[start:stop], dtype=torch.long)
            shifts = torch.tensor(table_config.vij[start:stop], dtype=torch.long)
            # Expand each non-null base-graph entry into its Zc lifted variable
            # indices. The resulting shape is [Zc, row_degree].
            lifted_cols = hbg_cols.unsqueeze(0) * self.Zc + ((shifts.unsqueeze(0) + zx) % self.Zc)

            self._row_degrees.append(degree)
            self.register_buffer(f"row_{row}_columns", lifted_cols)
            self.register_buffer(f"row_{row}_flat_columns", lifted_cols.reshape(-1))
            self._row_columns.append(getattr(self, f"row_{row}_columns"))
            self._row_flat_columns.append(getattr(self, f"row_{row}_flat_columns"))

    def _make_raw_alpha(
        self,
        init_alpha: Union[float, Sequence[Sequence[float]], torch.Tensor],
        init_alpha_degree_values: Optional[Sequence[float]],
    ) -> torch.Tensor:
        """Create unconstrained alpha logits for the selected sharing mode.

        The learnable parameter is stored before sigmoid mapping. ``alpha()``
        converts it back into the bounded interval [alpha_min, alpha_max].
        """
        if self.alpha_mode == "single":
            if init_alpha_degree_values is not None:
                raise ValueError("init_alpha_degree_values is only valid for alpha_mode='degree'")
            if not isinstance(init_alpha, (int, float)):
                return self._make_table_raw_alpha(init_alpha, (1,), "single")
            return _inverse_bounded_sigmoid(
                float(init_alpha),
                self.alpha_min,
                self.alpha_max,
            ).reshape(1)

        if self.alpha_mode == "checkpoint":
            if init_alpha_degree_values is not None:
                raise ValueError("init_alpha_degree_values is only valid for alpha_mode='degree'")
            expected_shape = (self.num_iterations, self.num_rows)
            if isinstance(init_alpha, (int, float)):
                return _inverse_bounded_sigmoid(
                    float(init_alpha),
                    self.alpha_min,
                    self.alpha_max,
                ).expand(expected_shape).clone()
            return self._make_table_raw_alpha(init_alpha, expected_shape, "checkpoint")

        num_unique_degrees = len(self.unique_row_degrees)
        if init_alpha_degree_values is not None:
            init_alpha_degree_tensor = torch.as_tensor(
                init_alpha_degree_values,
                dtype=torch.float32,
            )
            if init_alpha_degree_tensor.dim() != 1:
                raise ValueError("init_alpha_degree_values must be a one-dimensional sequence")
            if init_alpha_degree_tensor.numel() == num_unique_degrees:
                selected_alpha_values = init_alpha_degree_tensor
            elif init_alpha_degree_tensor.numel() >= max(self.unique_row_degrees):
                selected_alpha_values = torch.stack(
                    [
                        init_alpha_degree_tensor[degree - 1]
                        for degree in self.unique_row_degrees
                    ]
                )
            else:
                raise ValueError(
                    "init_alpha_degree_values must contain either one value per "
                    f"unique row degree {self.unique_row_degrees} or a 1-based "
                    f"exact-degree table with at least {max(self.unique_row_degrees)} "
                    f"entries; got {init_alpha_degree_tensor.numel()}"
                )
            return _inverse_bounded_sigmoid_tensor(
                selected_alpha_values,
                self.alpha_min,
                self.alpha_max,
            )

        if isinstance(init_alpha, (int, float)):
            return _inverse_bounded_sigmoid(
                float(init_alpha),
                self.alpha_min,
                self.alpha_max,
            ).repeat(num_unique_degrees)

        return self._make_table_raw_alpha(
            init_alpha,
            (num_unique_degrees,),
            "degree",
        )

    def alpha_parameter_shape(self) -> Tuple[int, ...]:
        """Return the raw alpha shape required by the selected mode."""
        return tuple(self.raw_alpha.shape)

    def _make_table_raw_alpha(
        self,
        init_alpha: Union[Sequence[float], Sequence[Sequence[float]], torch.Tensor],
        expected_shape: Tuple[int, ...],
        alpha_mode: str,
    ) -> torch.Tensor:
        """Validate an explicit alpha table and map it to raw logits."""
        init_alpha_tensor = torch.as_tensor(init_alpha, dtype=torch.float32)
        if tuple(init_alpha_tensor.shape) != expected_shape:
            raise ValueError(
                "init_alpha must be a scalar or have shape "
                f"{expected_shape} when alpha_mode='{alpha_mode}'; "
                f"got {tuple(init_alpha_tensor.shape)}"
            )
        return _inverse_bounded_sigmoid_tensor(
            init_alpha_tensor,
            self.alpha_min,
            self.alpha_max,
        )

    def alpha(self) -> torch.Tensor:
        """Return bounded alpha values in the shape used by ``alpha_mode``."""
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)
        return alpha.clamp(min=self.alpha_min, max=self.alpha_max)

    def _alpha_for_row(
        self,
        alphas: torch.Tensor,
        iteration: int,
        row: int,
    ) -> torch.Tensor:
        """Return alpha broadcastable to a row message tensor.

        Row messages use shape ``[batch, Zc, row_degree]``. Single and degree
        modes return a scalar; checkpoint mode returns the scalar for the
        current ``[iteration, HBG row]`` entry, matching
        ``alpha_list[iter * num_rows + HBG_row]`` in the C decoder.
        """
        if self.alpha_mode == "single":
            return alphas[0]
        if self.alpha_mode == "checkpoint":
            return alphas[iteration, row]

        degree_idx = self.row_alpha_degree_indices[row].to(device=alphas.device)
        return alphas[degree_idx]

    def beta(self) -> torch.Tensor:
        """Return beta expanded to shape [iterations, HBG rows]."""
        return torch.sigmoid(self.raw_beta).clamp(min=0.0, max=1.0 - 1e-6)

    def forward(
        self,
        llr: torch.Tensor,
        return_llr: bool = False,
        return_iteration_llrs: bool = False,
    ):
        """Decode one batch of LLRs with neural layered min-sum updates.

        Args:
            llr: input LLR tensor with shape ``[n]`` or ``[batch, n]``.
            return_llr: also return the final soft LLR values.
            return_iteration_llrs: also return soft LLRs after each iteration.

        Returns:
            Hard bits by default. Optional soft outputs follow the flags above.
        """
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
        # Cache bounded alpha/beta values once per forward pass. Gradients still
        # flow back through raw_alpha/raw_beta because these tensors are derived
        # from trainable parameters.
        alphas = self.alpha()
        betas = self.beta()
        iteration_llrs = [] if return_iteration_llrs else None

        for iteration in range(self.num_iterations):
            for row in range(self.num_rows):
                degree = self._row_degrees[row]
                flat_cols = self._row_flat_columns[row].to(device=llr.device)
                old_lr = Lr_rows[row]

                # Remove the previous check-to-variable message before forming
                # the extrinsic input for this HBG row.
                rel_lq = LQ[:, flat_cols].reshape(batch_size, self.Zc, degree) - old_lr
                update_rel_lq = rel_lq.detach() if self.detach_check_update_inputs else rel_lq
                signs = torch.where(
                    update_rel_lq < 0,
                    -torch.ones_like(update_rel_lq),
                    torch.ones_like(update_rel_lq),
                )
                abs_rel = update_rel_lq.abs()
                min1, min1_idx = torch.min(abs_rel, dim=2)

                # Standard min-sum check update: each edge receives the second
                # minimum when that edge supplied the first minimum.
                min1_mask = torch.nn.functional.one_hot(min1_idx, num_classes=degree).to(torch.bool)
                second_source = abs_rel.masked_fill(min1_mask, float("inf"))
                min2, _ = torch.min(second_source, dim=2)

                sign_prod = torch.prod(signs, dim=2, keepdim=True)
                use_min = torch.where(min1_mask, min2.unsqueeze(2), min1.unsqueeze(2))
                magnitude = torch.clamp(use_min - betas[iteration, row], min=0.0)
                magnitude = self._alpha_for_row(alphas, iteration, row) * magnitude
                new_lr = sign_prod * signs * magnitude

                new_lq_values = update_rel_lq + new_lr
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
        """Return trainable parameter names and shapes for quick inspection."""
        return [(name, tuple(param.shape)) for name, param in self.named_parameters() if param.requires_grad]

    def memory_summary(self, batch_size: int, dtype_bytes: int = 4) -> Dict[str, float]:
        """Estimate the main decoder message memory footprint in megabytes."""
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
