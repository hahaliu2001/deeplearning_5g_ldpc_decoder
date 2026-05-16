# Xilinx FPGA LDPC Degree Min-Sum Decoder

This folder contains a synthesizable SystemVerilog implementation derived from
`c_lib/degree_fixedpoint_ldpc_decoder.c`.

## Files

- `rtl/ldpc_degree_decoder.sv` - layered degree-dependent fixed-point min-sum
  decoder core.
- `rtl/ldpc_tables_rom.sv` - Verilog ROM functions for BG1/BG2 HBG column,
  row-offset, and `Vij` tables. This is generated from the C table file and does
  not require Xilinx `.coe` memory initialization files.
- `tools/gen_ldpc_table_rom.py` - regenerates `rtl/ldpc_tables_rom.sv` from
  `c_lib/ldpc_tables.c`.

## Interface

The core accepts one signed 16-bit LLR per cycle while `llr_ready` is asserted.
Provide `N + 2*Zc` LLR values in the same order as the C decoder's `LLr`/HARQ
memory input. After `max_iterations`, the core streams packed hard bits
MSB-first on `hardbit_data`, matching the C packing loop:

```c
hardbits[i/8] = (hardbits[i/8] << 1) | (LQ[i] < 0 ? 1 : 0);
```

Use `bgn = 0` for BG1 and `bgn = 1` for BG2. The `i_ls`, `Zc`, `B`, `N`,
`beta`, `max_llr`, and `min_llr` inputs correspond to the C `ldpc_config_t` and
`ldpc_decoder_cfg_t` fields.

## Algorithm Mapping

The hardware FSM follows the same loop nesting as the C implementation:

1. Load `LQ` from input LLRs.
2. Clear `Lr`.
3. For each iteration, base-graph row, and lifted row `zx`:
   - read connected columns from the table ROM,
   - compute `rel_Lq = LQ[column] - Lr`,
   - find sign, minimum, and second minimum,
   - select alpha by row degree:
     - degree <= 4: `26623`
     - degree 5..8: `24575`
     - degree > 8: `22527`
   - update `Lr` and saturating `LQ`.
4. Output hard bits after the configured iteration count.

The C decoder performs CRC early termination after each iteration. This first
FPGA core emits hard bits after `max_iterations`; add a CRC checker around the
output stream if early termination is required in hardware.

## Xilinx Notes

Vivado supports these `.sv` sources directly. Add `rtl/ldpc_degree_decoder.sv`
to the project; it includes `rtl/ldpc_tables_rom.sv`. Set the include directory
to `FPGA/rtl` if your project does not resolve local includes automatically.

The design uses inferred memories for `LQ` and `Lr`. For production throughput,
the next natural optimization is to split the memories into banks and process
multiple lifted checks in parallel.
