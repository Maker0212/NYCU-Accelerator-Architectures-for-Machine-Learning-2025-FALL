// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* verilator lint_off MULTITOP */
`include "hw/PE.v"
`include "hw/systolic_array.v"
`include "hw/global_buffer_bram.v"
`include "hw/TPU.v"
/* verilator lint_on MULTITOP */

module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output              rsp_valid,
  input               rsp_ready,
  output     [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  // Function decoding helpers
  wire [2:0] funct3 = cmd_payload_function_id[2:0];
  wire [6:0] funct7 = cmd_payload_function_id[9:3];

  // Dot-product and quantization state
  reg  signed [8:0]  input_offset;
  reg         [31:0] quant_multiplier;
  reg  signed [7:0]  quant_shift;
  reg  signed [31:0] accumulator;

  // TPU interface signals
  reg         tpu_in_valid;
  reg  [15:0] tpu_K;
  reg  [7:0]  tpu_M, tpu_N;
  reg         tpu_a_bank;
  reg         tpu_b_bank;
  reg         tpu_c_bank;
  wire        tpu_busy;

  wire        tpu_A_wr_en;
  wire [15:0] tpu_A_index;
  wire [31:0] tpu_A_data_in;
  wire [31:0] tpu_A_data_out;

  wire        tpu_B_wr_en;
  wire [15:0] tpu_B_index0;
  wire [15:0] tpu_B_index1;
  wire [31:0] tpu_B_data_in;
  wire [31:0] tpu_B_data_out0;
  wire [31:0] tpu_B_data_out1;

  wire        tpu_C_wr_en;
  wire [15:0] tpu_C_index;
  wire [127:0] tpu_C_data_in;
  wire [127:0] tpu_C_data_out;

  // Global buffer register interface
  reg         gbuff_A_wr_en;
  reg  [15:0] gbuff_A_index;
  reg  [31:0] gbuff_A_data_in;

  reg         gbuff_B_wr_en;
  reg  [15:0] gbuff_B_index;
  reg  [31:0] gbuff_B_data_in;

  reg  [15:0] gbuff_C_index;

  wire [10:0] c_write_index = tpu_C_index[10:0];
  wire [10:0] c_read_index = gbuff_C_index[10:0];
  wire       c_write_bank = tpu_c_bank;
  wire       c0_wr_en = tpu_C_wr_en && !c_write_bank;
  wire       c1_wr_en = tpu_C_wr_en && c_write_bank;
  wire [10:0] c0_index = c0_wr_en ? c_write_index : c_read_index;
  wire [10:0] c1_index = c1_wr_en ? c_write_index : c_read_index;
  wire       c_read_bank = gbuff_C_index[15];
  wire [127:0] c_data_out0;
  wire [127:0] c_data_out1;
  wire [127:0] c_read_data = c_read_bank ? c_data_out1 : c_data_out0;
  assign tpu_C_data_out = 128'd0;

  // Global buffers feed the TPU
  wire [13:0] a_write_index = gbuff_A_index[13:0];
  wire        a_write_bank = gbuff_A_index[15];
  wire        a0_wr_en = gbuff_A_wr_en && !a_write_bank;
  wire        a1_wr_en = gbuff_A_wr_en && a_write_bank;
  wire [13:0] a0_index = a0_wr_en ? a_write_index : tpu_A_index[13:0];
  wire [13:0] a1_index = a1_wr_en ? a_write_index : tpu_A_index[13:0];
  wire [31:0] tpu_A_data_out0;
  wire [31:0] tpu_A_data_out1;
  assign tpu_A_data_out = tpu_a_bank ? tpu_A_data_out1 : tpu_A_data_out0;

  wire [13:0] b_write_index = gbuff_B_index[13:0];
  wire        b_write_bank = gbuff_B_index[15];
  wire        b0a_wr_en = gbuff_B_wr_en && !b_write_bank;
  wire        b0b_wr_en = gbuff_B_wr_en && !b_write_bank;
  wire        b1a_wr_en = gbuff_B_wr_en && b_write_bank;
  wire        b1b_wr_en = gbuff_B_wr_en && b_write_bank;
  wire [13:0] b0a_index = b0a_wr_en ? b_write_index : tpu_B_index0[13:0];
  wire [13:0] b0b_index = b0b_wr_en ? b_write_index : tpu_B_index1[13:0];
  wire [13:0] b1a_index = b1a_wr_en ? b_write_index : tpu_B_index0[13:0];
  wire [13:0] b1b_index = b1b_wr_en ? b_write_index : tpu_B_index1[13:0];
  wire [31:0] b0a_data_out;
  wire [31:0] b0b_data_out;
  wire [31:0] b1a_data_out;
  wire [31:0] b1b_data_out;
  assign tpu_B_data_out0 = tpu_b_bank ? b1a_data_out : b0a_data_out;
  assign tpu_B_data_out1 = tpu_b_bank ? b1b_data_out : b0b_data_out;

  global_buffer_bram #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
  ) gbuff_A0 (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(a0_wr_en),
    .index(a0_index),
    .data_in(gbuff_A_data_in),
    .data_out(tpu_A_data_out0)
  );

  global_buffer_bram #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
  ) gbuff_A1 (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(a1_wr_en),
    .index(a1_index),
    .data_in(gbuff_A_data_in),
    .data_out(tpu_A_data_out1)
  );

  global_buffer_bram #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
  ) gbuff_B0a (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(b0a_wr_en),
    .index(b0a_index),
    .data_in(gbuff_B_data_in),
    .data_out(b0a_data_out)
  );

  global_buffer_bram #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
  ) gbuff_B0b (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(b0b_wr_en),
    .index(b0b_index),
    .data_in(gbuff_B_data_in),
    .data_out(b0b_data_out)
  );

  global_buffer_bram #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
  ) gbuff_B1a (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(b1a_wr_en),
    .index(b1a_index),
    .data_in(gbuff_B_data_in),
    .data_out(b1a_data_out)
  );

  global_buffer_bram #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
  ) gbuff_B1b (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(b1b_wr_en),
    .index(b1b_index),
    .data_in(gbuff_B_data_in),
    .data_out(b1b_data_out)
  );

  global_buffer_bram #(
    .ADDR_BITS(11),
    .DATA_BITS(128)
  ) gbuff_C0 (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(c0_wr_en),
    .index(c0_index),
    .data_in(tpu_C_data_in),
    .data_out(c_data_out0)
  );

  global_buffer_bram #(
    .ADDR_BITS(11),
    .DATA_BITS(128)
  ) gbuff_C1 (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(c1_wr_en),
    .index(c1_index),
    .data_in(tpu_C_data_in),
    .data_out(c_data_out1)
  );

  TPU tpu (
    .clk(clk),
    .rst_n(~reset),
    .in_valid(tpu_in_valid),
    .K(tpu_K),
    .M(tpu_M),
    .N(tpu_N),
    .busy(tpu_busy),
    .A_wr_en(tpu_A_wr_en),
    .A_index(tpu_A_index),
    .A_data_in(tpu_A_data_in),
    .A_data_out(tpu_A_data_out),
    .B_wr_en(tpu_B_wr_en),
    .B_index0(tpu_B_index0),
    .B_index1(tpu_B_index1),
    .B_data_in(tpu_B_data_in),
    .B_data_out0(tpu_B_data_out0),
    .B_data_out1(tpu_B_data_out1),
    .C_wr_en(tpu_C_wr_en),
    .C_index(tpu_C_index),
    .C_data_in(tpu_C_data_in),
    .C_data_out(tpu_C_data_out)
  );

  // SIMD LUT
  reg [7:0] simd_lut [0:255];
  integer lut_init_idx;

  // Transpose state
  reg [31:0] transpose_reg0;
  reg [31:0] transpose_reg1;
  reg [31:0] transpose_reg2;
  reg [31:0] transpose_val3;
  reg [31:0] transpose_addr;
  reg [2:0]  transpose_write_count;

  // Read pipeline bookkeeping
  reg        read_waiting;
  reg [1:0]  read_word_sel;

  // Handshake tracking
  reg rsp_valid_reg;
  reg [31:0] rsp_payload_reg;

  assign rsp_valid = rsp_valid_reg;
  assign rsp_payload_outputs_0 = rsp_payload_reg;
  assign cmd_ready = (~rsp_valid_reg) && (~read_waiting) && (transpose_write_count == 0);

  // Helpers for byte extraction
  function automatic signed [8:0] get_signed_byte;
    input [31:0] word;
    input integer lane;
    reg [7:0] byte_val;
    begin
      case (lane)
        0: byte_val = word[7:0];
        1: byte_val = word[15:8];
        2: byte_val = word[23:16];
        default: byte_val = word[31:24];
      endcase
      get_signed_byte = {byte_val[7], byte_val};
    end
  endfunction

  function automatic signed [31:0] dot4_with_offset;
    input [31:0] inputs;
    input [31:0] filters;
    input signed [8:0] offset;
    integer lane;
    reg signed [31:0] acc;
    reg signed [8:0] in_val;
    reg signed [8:0] filt_val;
    reg signed [17:0] prod;
    begin
      acc = 0;
      for (lane = 0; lane < 4; lane = lane + 1) begin
        in_val = get_signed_byte(inputs, lane) + offset;
        filt_val = get_signed_byte(filters, lane);
        prod = in_val * filt_val;
        acc = acc + {{14{prod[17]}}, prod};
      end
      dot4_with_offset = acc;
    end
  endfunction

  function automatic signed [31:0] tail_dot_with_offset;
    input [31:0] inputs;
    input [31:0] filters;
    input signed [8:0] offset;
    integer lane;
    integer lane_count;
    reg signed [31:0] acc;
    reg signed [8:0] in_val;
    reg signed [8:0] filt_val;
    reg signed [17:0] prod;
    begin
      acc = 0;
      lane_count = (filters >> 24) & 8'hFF;
      for (lane = 0; lane < 3; lane = lane + 1) begin
        if (lane < lane_count) begin
          in_val = get_signed_byte(inputs, lane) + offset;
          filt_val = get_signed_byte(filters, lane);
          prod = in_val * filt_val;
          acc = acc + {{14{prod[17]}}, prod};
        end
      end
      tail_dot_with_offset = acc;
    end
  endfunction

  function automatic signed [31:0] apply_quantized_multiplier;
    input signed [31:0] value;
    input [31:0] multiplier;
    input signed [7:0] shift;
    integer total_shift;
    integer left_shift;
    reg signed [63:0] product;
    reg signed [63:0] rounded;
    begin
      product = value * $signed(multiplier);
      total_shift = 31 - shift;
      if (total_shift > 0) begin
        rounded = product + (64'sd1 <<< (total_shift - 1));
        apply_quantized_multiplier = rounded >>> total_shift;
      end else begin
        left_shift = -total_shift;
        apply_quantized_multiplier = product <<< left_shift;
      end
    end
  endfunction

  function automatic [31:0] select_c_word;
    input [127:0] data;
    input [1:0] sel;
    begin
      case (sel)
        2'd0: select_c_word = data[31:0];
        2'd1: select_c_word = data[63:32];
        2'd2: select_c_word = data[95:64];
        default: select_c_word = data[127:96];
      endcase
    end
  endfunction

  // Main control
  always @(posedge clk or posedge reset) begin
    if (reset) begin
      input_offset      <= 9'sd0;
      quant_multiplier  <= 32'd0;
      quant_shift       <= 8'sd0;
      accumulator       <= 32'sd0;
      rsp_valid_reg     <= 1'b0;
      rsp_payload_reg   <= 32'd0;
      read_waiting      <= 1'b0;
      read_word_sel     <= 2'd0;
      gbuff_A_wr_en     <= 1'b0;
      gbuff_B_wr_en     <= 1'b0;
      gbuff_A_index     <= 16'd0;
      gbuff_B_index     <= 16'd0;
      gbuff_C_index     <= 16'd0;
      gbuff_A_data_in   <= 32'd0;
      gbuff_B_data_in   <= 32'd0;
      tpu_in_valid      <= 1'b0;
      tpu_K             <= 8'd0;
      tpu_M             <= 8'd0;
      tpu_N             <= 8'd0;
      tpu_a_bank        <= 1'b0;
      tpu_b_bank        <= 1'b0;
      tpu_c_bank        <= 1'b0;
      transpose_write_count <= 3'd0;
    end else if (transpose_write_count > 0) begin
      gbuff_A_wr_en <= 1'b1;
      gbuff_A_index <= transpose_addr[31:16];
      transpose_addr <= transpose_addr + 32'h00010000;
      transpose_write_count <= transpose_write_count - 3'd1;
      
      case (transpose_write_count)
        3'd4: gbuff_A_data_in <= {transpose_reg0[7:0], transpose_reg1[7:0], transpose_reg2[7:0], transpose_val3[7:0]};
        3'd3: gbuff_A_data_in <= {transpose_reg0[15:8], transpose_reg1[15:8], transpose_reg2[15:8], transpose_val3[15:8]};
        3'd2: gbuff_A_data_in <= {transpose_reg0[23:16], transpose_reg1[23:16], transpose_reg2[23:16], transpose_val3[23:16]};
        3'd1: begin
           gbuff_A_data_in <= {transpose_reg0[31:24], transpose_reg1[31:24], transpose_reg2[31:24], transpose_val3[31:24]};
           rsp_valid_reg <= 1'b1;
           rsp_payload_reg <= 32'd0;
        end
      endcase
    end else begin
      // Defaults
      tpu_in_valid  <= 1'b0;
      gbuff_A_wr_en <= 1'b0;
      gbuff_B_wr_en <= 1'b0;

      if (rsp_valid_reg && rsp_ready) begin
        rsp_valid_reg <= 1'b0;
      end

      if (read_waiting && ~rsp_valid_reg) begin
        rsp_payload_reg <= select_c_word(c_read_data, read_word_sel);
        rsp_valid_reg   <= 1'b1;
        read_waiting    <= 1'b0;
      end else if (cmd_valid && cmd_ready) begin
        case (funct3)
          3'd0: begin
            case (funct7)
              7'd0: begin
                input_offset    <= {cmd_payload_inputs_0[7], cmd_payload_inputs_0[7:0]};
                rsp_payload_reg <= cmd_payload_inputs_0;
              end
              7'd1: begin
                rsp_payload_reg <= dot4_with_offset(cmd_payload_inputs_0,
                                                    cmd_payload_inputs_1,
                                                    input_offset);
              end
              7'd2: begin
                rsp_payload_reg <= tail_dot_with_offset(cmd_payload_inputs_0,
                                                        cmd_payload_inputs_1,
                                                        input_offset);
              end
              default: begin
                rsp_payload_reg <= cmd_payload_inputs_0;
              end
            endcase
            rsp_valid_reg <= 1'b1;
          end

          3'd1: begin
            case (funct7)
              7'd0: begin
                quant_multiplier <= cmd_payload_inputs_0;
                rsp_payload_reg  <= cmd_payload_inputs_0;
              end
              7'd1: begin
                quant_shift      <= $signed(cmd_payload_inputs_0[7:0]);
                rsp_payload_reg  <= cmd_payload_inputs_0;
              end
              7'd2: begin
                begin : apply_mul_block
                  reg signed [31:0] quant_val;
                  quant_val = apply_quantized_multiplier(
                      $signed(cmd_payload_inputs_0),
                      quant_multiplier, quant_shift);
                  rsp_payload_reg <= quant_val;
                end
              end
              default: begin
                rsp_payload_reg <= cmd_payload_inputs_0;
              end
            endcase
            rsp_valid_reg <= 1'b1;
          end

          3'd2: begin
            case (funct7)
              7'd0: begin
                accumulator      <= 32'sd0;
                rsp_payload_reg  <= 32'd0;
              end
              7'd1: begin
                begin : acc_dot_block
                  reg signed [31:0] dot_val;
                  dot_val = dot4_with_offset(cmd_payload_inputs_0,
                                             cmd_payload_inputs_1,
                                             input_offset);
                  accumulator     <= accumulator + dot_val;
                  rsp_payload_reg <= accumulator + dot_val;
                end
              end
              7'd2: begin
                rsp_payload_reg  <= accumulator;
              end
              7'd3: begin
                begin : acc_quant_block
                  reg signed [31:0] quant_val;
                  quant_val = apply_quantized_multiplier(accumulator,
                      quant_multiplier, quant_shift);
                  accumulator     <= quant_val;
                  rsp_payload_reg <= quant_val;
                end
              end
              7'd4: begin
                begin : acc_bias_block
                  reg signed [31:0] bias_val;
                  bias_val = accumulator + $signed(cmd_payload_inputs_0);
                  accumulator     <= bias_val;
                  rsp_payload_reg <= bias_val;
                end
              end
              default: begin
                rsp_payload_reg <= cmd_payload_inputs_0;
              end
            endcase
            rsp_valid_reg <= 1'b1;
          end

          3'd3: begin
            case (funct7)
              7'd0: begin
                gbuff_A_wr_en   <= 1'b1;
                gbuff_A_index   <= cmd_payload_inputs_0[31:16];
                gbuff_A_data_in <= cmd_payload_inputs_1;
                rsp_payload_reg <= 32'd0;
                rsp_valid_reg   <= 1'b1;
              end
              7'd1: begin
                gbuff_B_wr_en   <= 1'b1;
                gbuff_B_index   <= cmd_payload_inputs_0[31:16];
                gbuff_B_data_in <= cmd_payload_inputs_1;
                rsp_payload_reg <= 32'd0;
                rsp_valid_reg   <= 1'b1;
              end
              7'd2: begin
                tpu_K           <= cmd_payload_inputs_0[31:16];
                tpu_M           <= cmd_payload_inputs_0[15:8];
                tpu_N           <= cmd_payload_inputs_0[7:0];
                tpu_in_valid    <= 1'b1;
                rsp_payload_reg <= {31'd0, tpu_busy};
                rsp_valid_reg   <= 1'b1;
              end
              7'd3: begin
                gbuff_C_index   <= cmd_payload_inputs_0[31:16];
                read_word_sel   <= cmd_payload_inputs_0[1:0];
                read_waiting    <= 1'b1;
              end
              7'd4: begin
                rsp_payload_reg <= {31'd0, tpu_busy};
                rsp_valid_reg   <= 1'b1;
              end
              7'd5: begin
                tpu_a_bank      <= cmd_payload_inputs_0[0];
                rsp_payload_reg <= {31'd0, tpu_a_bank};
                rsp_valid_reg   <= 1'b1;
              end
              7'd6: begin
                tpu_b_bank      <= cmd_payload_inputs_0[0];
                rsp_payload_reg <= {31'd0, tpu_b_bank};
                rsp_valid_reg   <= 1'b1;
              end
              7'd7: begin
                tpu_c_bank      <= cmd_payload_inputs_0[0];
                rsp_payload_reg <= {31'd0, tpu_c_bank};
                rsp_valid_reg   <= 1'b1;
              end
              default: begin
                rsp_payload_reg <= cmd_payload_inputs_0;
                rsp_valid_reg   <= 1'b1;
              end
            endcase
          end

          3'd5: begin
            case (funct7)
              7'd0: begin // Write LUT
                simd_lut[cmd_payload_inputs_0[7:0]]        <= cmd_payload_inputs_1[7:0];
                simd_lut[cmd_payload_inputs_0[7:0] + 8'd1] <= cmd_payload_inputs_1[15:8];
                simd_lut[cmd_payload_inputs_0[7:0] + 8'd2] <= cmd_payload_inputs_1[23:16];
                simd_lut[cmd_payload_inputs_0[7:0] + 8'd3] <= cmd_payload_inputs_1[31:24];
                rsp_payload_reg <= 32'd0;
                rsp_valid_reg   <= 1'b1;
              end
              7'd1: begin // Read LUT
                rsp_payload_reg <= {
                    simd_lut[cmd_payload_inputs_0[31:24]],
                    simd_lut[cmd_payload_inputs_0[23:16]],
                    simd_lut[cmd_payload_inputs_0[15:8]],
                    simd_lut[cmd_payload_inputs_0[7:0]]
                };
                rsp_valid_reg   <= 1'b1;
              end
              7'd10: begin // Set Transpose Reg 0
                transpose_reg0 <= cmd_payload_inputs_0;
                rsp_valid_reg  <= 1'b1;
              end
              7'd11: begin // Set Transpose Reg 1
                transpose_reg1 <= cmd_payload_inputs_0;
                rsp_valid_reg  <= 1'b1;
              end
              7'd12: begin // Set Transpose Reg 2
                transpose_reg2 <= cmd_payload_inputs_0;
                rsp_valid_reg  <= 1'b1;
              end
              7'd13: begin // Write Transposed
                transpose_addr <= cmd_payload_inputs_0;
                transpose_val3 <= cmd_payload_inputs_1;
                transpose_write_count <= 3'd4;
              end
              default: begin
                rsp_payload_reg <= cmd_payload_inputs_0;
                rsp_valid_reg   <= 1'b1;
              end
            endcase
          end

          default: begin
            rsp_payload_reg <= cmd_payload_inputs_0;
            rsp_valid_reg   <= 1'b1;
          end
        endcase
      end
    end
  end

endmodule
