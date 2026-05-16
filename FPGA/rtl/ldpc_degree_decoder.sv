`timescale 1ns/1ps

`include "ldpc_tables_rom.sv"

module ldpc_degree_decoder #(
    parameter int LLR_WIDTH = 16,
    parameter int MAX_ZC = 384,
    parameter int BG1_NUM_ROWS = 46,
    parameter int BG2_NUM_ROWS = 42,
    parameter int BG1_NUM_COLS = 68,
    parameter int BG1_NONZERO = 316,
    parameter int BG2_NONZERO = 197,
    parameter int MAX_DEGREE = 19,
    parameter int LQ_DEPTH = BG1_NUM_COLS * MAX_ZC,
    parameter int LR_DEPTH = BG1_NONZERO * MAX_ZC
) (
    input  logic clk,
    input  logic rst,

    input  logic start,
    input  logic bgn,              // 0 = BG1, 1 = BG2
    input  logic [2:0] i_ls,
    input  logic [8:0] zc,
    input  logic [15:0] b_bits,
    input  logic [15:0] n_bits,
    input  logic [7:0] max_iterations,
    input  logic signed [15:0] beta,
    input  logic signed [15:0] max_llr,
    input  logic signed [15:0] min_llr,

    input  logic llr_valid,
    output logic llr_ready,
    input  logic signed [LLR_WIDTH-1:0] llr_data,

    output logic hardbit_valid,
    output logic [7:0] hardbit_data,
    input  logic hardbit_ready,

    output logic busy,
    output logic done
);
    localparam logic signed [15:0] ALPHA_LE4  = 16'sd26623;
    localparam logic signed [15:0] ALPHA_5_8  = 16'sd24575;
    localparam logic signed [15:0] ALPHA_GT8  = 16'sd22527;

    typedef enum logic [3:0] {
        S_IDLE,
        S_LOAD_LLR,
        S_CLEAR_LR,
        S_ROW_INIT,
        S_ZX_INIT,
        S_READ_EDGE,
        S_PREPARE_UPDATE,
        S_WRITE_EDGE,
        S_NEXT_CHECK,
        S_OUTPUT_INIT,
        S_OUTPUT_BYTE,
        S_DONE
    } state_t;

    state_t state;

    (* ram_style = "block" *) logic signed [LLR_WIDTH-1:0] lq_mem [0:LQ_DEPTH-1];
    (* ram_style = "block" *) logic signed [LLR_WIDTH-1:0] lr_mem [0:LR_DEPTH-1];

    logic [15:0] load_count;
    logic [16:0] load_total;
    logic [16:0] clear_count;
    logic [7:0] iter_count;
    logic [5:0] row;
    logic [8:0] zx;
    logic [4:0] edge;
    logic [8:0] row_offset;
    logic [8:0] next_row_offset;
    logic [4:0] degree;
    logic [16:0] row_lr_base;
    logic [16:0] lr_addr_list [0:MAX_DEGREE-1];
    logic [15:0] column_list [0:MAX_DEGREE-1];
    logic signed [LLR_WIDTH-1:0] rel_lq [0:MAX_DEGREE-1];
    logic signed [LLR_WIDTH-1:0] min1;
    logic signed [LLR_WIDTH-1:0] min2;
    logic [4:0] min1_idx;
    logic sign_neg;
    logic signed [15:0] alpha;
    logic signed [LLR_WIDTH-1:0] new_min1;
    logic signed [LLR_WIDTH-1:0] new_min2;
    logic [15:0] hardbit_index;
    logic [2:0] hardbit_bit;
    logic [7:0] hardbit_shift;
    logic output_last;

    function automatic logic [8:0] row_offset_for(input logic bg, input logic [5:0] row_idx);
        begin
            row_offset_for = bg ? {1'b0, ldpc_bg2_row_offset(row_idx)}
                                : ldpc_bg1_row_offset(row_idx);
        end
    endfunction

    function automatic logic [6:0] hbg_col_for(input logic bg, input logic [8:0] idx);
        begin
            hbg_col_for = bg ? {1'b0, ldpc_bg2_hbg_col(idx[7:0])}
                             : ldpc_bg1_hbg_col(idx);
        end
    endfunction

    function automatic logic [5:0] num_rows_for(input logic bg);
        begin
            num_rows_for = bg ? 6'd42 : 6'd46;
        end
    endfunction

    function automatic logic [16:0] lr_count_for(input logic bg, input logic [8:0] zc_value);
        begin
            lr_count_for = (bg ? 17'd197 : 17'd316) * zc_value;
        end
    endfunction

    function automatic logic [8:0] add_mod_zc(
        input logic [8:0] shift,
        input logic [8:0] zx_value,
        input logic [8:0] zc_value
    );
        logic [9:0] value;
        int m;
        begin
            value = shift + zx_value;
            for (m = 0; m < MAX_ZC; m = m + 1) begin
                if ((zc_value != 9'd0) && (value >= zc_value)) begin
                    value = value - zc_value;
                end
            end
            add_mod_zc = value[8:0];
        end
    endfunction

    function automatic logic signed [15:0] abs16(input logic signed [15:0] value);
        begin
            abs16 = value[15] ? -value : value;
        end
    endfunction

    function automatic logic signed [15:0] saturating_add(
        input logic signed [15:0] a,
        input logic signed [15:0] b,
        input logic signed [15:0] hi,
        input logic signed [15:0] lo
    );
        logic signed [16:0] sum;
        begin
            sum = a + b;
            if (sum > hi) begin
                saturating_add = hi;
            end else if (sum < lo) begin
                saturating_add = lo;
            end else begin
                saturating_add = sum[15:0];
            end
        end
    endfunction

    function automatic logic signed [15:0] scale_min(
        input logic signed [15:0] value,
        input logic signed [15:0] offset,
        input logic signed [15:0] scale
    );
        logic signed [15:0] adjusted;
        logic signed [31:0] product;
        begin
            adjusted = (value > offset) ? (value - offset) : 16'sd0;
            product = scale * adjusted;
            scale_min = (product + 32'sd16384) >>> 15;
        end
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            llr_ready <= 1'b0;
            hardbit_valid <= 1'b0;
            hardbit_data <= 8'h00;
            output_last <= 1'b0;
        end else begin
            done <= 1'b0;
            hardbit_valid <= 1'b0;
            llr_ready <= 1'b0;

            unique case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        load_count <= 16'd0;
                        load_total <= n_bits + ({8'd0, zc} << 1);
                        state <= S_LOAD_LLR;
                    end
                end

                S_LOAD_LLR: begin
                    llr_ready <= 1'b1;
                    if (llr_valid) begin
                        lq_mem[load_count] <= llr_data;
                        load_count <= load_count + 16'd1;
                        if (load_count + 16'd1 >= load_total) begin
                            clear_count <= 17'd0;
                            state <= S_CLEAR_LR;
                        end
                    end
                end

                S_CLEAR_LR: begin
                    lr_mem[clear_count] <= '0;
                    clear_count <= clear_count + 17'd1;
                    if (clear_count + 17'd1 >= lr_count_for(bgn, zc)) begin
                        iter_count <= 8'd0;
                        row <= 6'd0;
                        state <= S_ROW_INIT;
                    end
                end

                S_ROW_INIT: begin
                    row_offset <= row_offset_for(bgn, row);
                    next_row_offset <= row_offset_for(bgn, row + 6'd1);
                    degree <= row_offset_for(bgn, row + 6'd1) - row_offset_for(bgn, row);
                    row_lr_base <= row_offset_for(bgn, row) * zc;
                    alpha <= ((row_offset_for(bgn, row + 6'd1) - row_offset_for(bgn, row)) <= 5'd4) ? ALPHA_LE4 :
                             ((row_offset_for(bgn, row + 6'd1) - row_offset_for(bgn, row)) <= 5'd8) ? ALPHA_5_8 :
                                                                                                      ALPHA_GT8;
                    zx <= 9'd0;
                    state <= S_ZX_INIT;
                end

                S_ZX_INIT: begin
                    edge <= 5'd0;
                    min1 <= max_llr;
                    min2 <= max_llr;
                    min1_idx <= 5'd0;
                    sign_neg <= 1'b0;
                    state <= S_READ_EDGE;
                end

                S_READ_EDGE: begin
                    logic [8:0] table_idx;
                    logic [6:0] hbg_col;
                    logic [8:0] shift;
                    logic [8:0] zidx;
                    logic [15:0] column;
                    logic [16:0] lr_addr;
                    logic signed [15:0] rel;
                    logic signed [15:0] abs_rel;

                    table_idx = row_offset + edge;
                    hbg_col = hbg_col_for(bgn, table_idx);
                    shift = ldpc_vij(bgn, i_ls, table_idx);
                    zidx = add_mod_zc(shift, zx, zc);
                    column = (hbg_col * zc) + zidx;
                    lr_addr = row_lr_base + (zx * degree) + edge;
                    rel = lq_mem[column] - lr_mem[lr_addr];
                    abs_rel = abs16(rel);

                    column_list[edge] <= column;
                    lr_addr_list[edge] <= lr_addr;
                    rel_lq[edge] <= rel;
                    sign_neg <= sign_neg ^ rel[15];

                    if (abs_rel < min1) begin
                        min2 <= min1;
                        min1 <= abs_rel;
                        min1_idx <= edge;
                    end else if (abs_rel < min2) begin
                        min2 <= abs_rel;
                    end

                    edge <= edge + 5'd1;
                    if (edge + 5'd1 >= degree) begin
                        state <= S_PREPARE_UPDATE;
                    end
                end

                S_PREPARE_UPDATE: begin
                    logic signed [15:0] clip_min1;
                    logic signed [15:0] clip_min2;
                    clip_min1 = (min1 > max_llr) ? max_llr : min1;
                    clip_min2 = (min2 > max_llr) ? max_llr : min2;
                    new_min1 <= scale_min(clip_min1, beta, alpha);
                    new_min2 <= scale_min(clip_min2, beta, alpha);
                    edge <= 5'd0;
                    state <= S_WRITE_EDGE;
                end

                S_WRITE_EDGE: begin
                    logic signed [15:0] new_val;
                    logic signed [15:0] sel_lr;
                    logic sel_neg;
                    new_val = (edge == min1_idx) ? new_min2 : new_min1;
                    sel_neg = sign_neg ^ rel_lq[edge][15];
                    sel_lr = sel_neg ? -new_val : new_val;

                    lr_mem[lr_addr_list[edge]] <= sel_lr;
                    lq_mem[column_list[edge]] <= saturating_add(rel_lq[edge], sel_lr, max_llr, min_llr);

                    edge <= edge + 5'd1;
                    if (edge + 5'd1 >= degree) begin
                        state <= S_NEXT_CHECK;
                    end
                end

                S_NEXT_CHECK: begin
                    if (zx + 9'd1 < zc) begin
                        zx <= zx + 9'd1;
                        state <= S_ZX_INIT;
                    end else if (row + 6'd1 < num_rows_for(bgn)) begin
                        row <= row + 6'd1;
                        state <= S_ROW_INIT;
                    end else if (iter_count + 8'd1 < max_iterations) begin
                        iter_count <= iter_count + 8'd1;
                        row <= 6'd0;
                        state <= S_ROW_INIT;
                    end else begin
                        state <= S_OUTPUT_INIT;
                    end
                end

                S_OUTPUT_INIT: begin
                    hardbit_index <= 16'd0;
                    hardbit_bit <= 3'd0;
                    hardbit_shift <= 8'h00;
                    output_last <= 1'b0;
                    state <= S_OUTPUT_BYTE;
                end

                S_OUTPUT_BYTE: begin
                    if (hardbit_valid && !hardbit_ready) begin
                        hardbit_valid <= 1'b1;
                    end else if (hardbit_valid && hardbit_ready && output_last) begin
                        hardbit_valid <= 1'b0;
                        state <= S_DONE;
                    end else begin
                        logic [7:0] next_shift;
                        next_shift = {hardbit_shift[6:0], lq_mem[hardbit_index][15]};
                        hardbit_shift <= next_shift;
                        hardbit_index <= hardbit_index + 16'd1;

                        if ((hardbit_bit == 3'd7) || (hardbit_index + 16'd1 >= b_bits)) begin
                            hardbit_data <= next_shift << (3'd7 - hardbit_bit);
                            hardbit_valid <= 1'b1;
                            hardbit_shift <= 8'h00;
                            hardbit_bit <= 3'd0;
                            output_last <= (hardbit_index + 16'd1 >= b_bits);
                        end else begin
                            hardbit_bit <= hardbit_bit + 3'd1;
                        end
                    end
                end

                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
