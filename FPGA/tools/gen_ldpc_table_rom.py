#!/usr/bin/env python3
"""Generate Verilog ROM functions for the 5G LDPC base-graph tables."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "c_lib" / "ldpc_tables.c"
OUT = ROOT / "FPGA" / "rtl" / "ldpc_tables_rom.sv"


def extract_array(text: str, name: str) -> list[int]:
    match = re.search(rf"{re.escape(name)}[^\n=]*=\s*\{{(.*?)\}};", text, re.S)
    if not match:
        raise RuntimeError(f"array {name} not found")

    body = match.group(1)
    body = body.replace("BG1_MAX_NON_NULL_ELEMENTS", "316")
    body = body.replace("BG2_MAX_NON_NULL_ELEMENTS", "197")
    return [int(token) for token in re.findall(r"\d+", body)]


def extract_matrix(text: str, name: str, rows: int) -> list[list[int]]:
    match = re.search(rf"{re.escape(name)}[^\n=]*=\s*\{{(.*?)\n\}};", text, re.S)
    if not match:
        raise RuntimeError(f"matrix {name} not found")

    body = match.group(1)
    row_texts = re.findall(r"\{([^{}]*)\}", body)
    if len(row_texts) != rows:
        raise RuntimeError(f"matrix {name} has {len(row_texts)} rows, expected {rows}")
    return [[int(token) for token in re.findall(r"\d+", row)] for row in row_texts]


def emit_case_function(name: str, width: int, index_width: int, values: list[int]) -> str:
    lines = [
        f"function automatic logic [{width - 1}:0] {name}(input logic [{index_width - 1}:0] idx);",
        "    begin",
        "        unique case (idx)",
    ]
    for idx, value in enumerate(values):
        lines.append(f"            {index_width}'d{idx}: {name} = {width}'d{value};")
    lines.extend(
        [
            f"            default: {name} = '0;",
            "        endcase",
            "    end",
            "endfunction",
            "",
        ]
    )
    return "\n".join(lines)


def emit_vij_function(bg1: list[list[int]], bg2: list[list[int]]) -> str:
    lines = [
        "function automatic logic [8:0] ldpc_vij(",
        "    input logic bgn,",
        "    input logic [2:0] ils,",
        "    input logic [8:0] idx",
        ");",
        "    begin",
        "        unique case ({bgn, ils})",
    ]

    for ils, row in enumerate(bg1):
        lines.append(f"            4'b0{ils:03b}: begin")
        lines.append("                unique case (idx)")
        for idx, value in enumerate(row):
            lines.append(f"                    9'd{idx}: ldpc_vij = 9'd{value};")
        lines.append("                    default: ldpc_vij = '0;")
        lines.append("                endcase")
        lines.append("            end")

    for ils, row in enumerate(bg2):
        lines.append(f"            4'b1{ils:03b}: begin")
        lines.append("                unique case (idx)")
        for idx, value in enumerate(row):
            lines.append(f"                    9'd{idx}: ldpc_vij = 9'd{value};")
        lines.append("                    default: ldpc_vij = '0;")
        lines.append("                endcase")
        lines.append("            end")

    lines.extend(
        [
            "            default: ldpc_vij = '0;",
            "        endcase",
            "    end",
            "endfunction",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    text = SRC.read_text()
    bg1_offsets = extract_array(text, "BG1_HBG_offset_array")
    bg2_offsets = extract_array(text, "BG2_HBG_offset_array")
    bg1_cols = extract_array(text, "BG1_HBG_array")
    bg2_cols = extract_array(text, "BG2_HBG_array")
    bg1_vij = extract_matrix(text, "BG1_HBG_Vij_array", 8)
    bg2_vij = extract_matrix(text, "BG2_HBG_Vij_array", 8)

    output = [
        "`ifndef LDPC_TABLES_ROM_SV",
        "`define LDPC_TABLES_ROM_SV",
        "",
        "// Generated from c_lib/ldpc_tables.c by FPGA/tools/gen_ldpc_table_rom.py.",
        "// bgn input convention: 0 = BG1, 1 = BG2.",
        "",
        emit_case_function("ldpc_bg1_row_offset", 9, 6, bg1_offsets),
        emit_case_function("ldpc_bg2_row_offset", 8, 6, bg2_offsets),
        emit_case_function("ldpc_bg1_hbg_col", 7, 9, bg1_cols),
        emit_case_function("ldpc_bg2_hbg_col", 6, 8, bg2_cols),
        emit_vij_function(bg1_vij, bg2_vij),
        "`endif",
        "",
    ]
    OUT.write_text("\n".join(output))


if __name__ == "__main__":
    main()
