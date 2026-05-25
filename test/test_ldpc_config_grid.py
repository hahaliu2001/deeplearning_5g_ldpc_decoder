# -*- coding: utf-8 -*-
"""Tests for LDPC simulation configuration-grid validation."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ldpc_c_sim_common import make_config_grid


def test_make_config_grid_filters_unsupported_product_pairs():
    with pytest.warns(UserWarning, match="BG2 supports B <= 3840"):
        config_grid = make_config_grid(
            Qm_list=[1],
            rv_list=[0],
            B_list=[3840, 4096],
            coderate_list=[1 / 2],
            bgn_list=[1, 2],
            crctype_list=[1],
        )

    assert config_grid == [
        (1, 0, 3840, 1 / 2, 1, 1),
        (1, 0, 3840, 1 / 2, 2, 1),
        (1, 0, 4096, 1 / 2, 1, 1),
    ]


def test_make_config_grid_raises_when_every_config_is_unsupported():
    with pytest.warns(UserWarning, match="Qm must be one of"):
        with pytest.raises(
            ValueError,
            match="make_config_grid did not produce any supported configs",
        ):
            make_config_grid(
                Qm_list=[10],
                rv_list=[0],
                B_list=[256],
                coderate_list=[1 / 2],
                bgn_list=[1],
                crctype_list=[1],
            )


def test_make_config_grid_strict_mode_raises_on_first_unsupported_config():
    with pytest.raises(ValueError, match="unsupported LDPC config"):
        make_config_grid(
            Qm_list=[1],
            rv_list=[0],
            B_list=[4096],
            coderate_list=[1 / 2],
            bgn_list=[2],
            crctype_list=[1],
            strict=True,
        )
