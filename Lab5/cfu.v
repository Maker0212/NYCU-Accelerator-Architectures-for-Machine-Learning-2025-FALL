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

`include "PE.v"
`include "systolic_array.v"
`include "global_buffer_bram.v"
`include "TPU.v"

/* verilator lint_on MULTITOP */

module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output              rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  // CFU Operations:
  // funct7 = 0: Write to Global Buffer A
  //   - inputs_0[31:16] = address (index)
  //   - inputs_0[15:0]  = not used
  //   - inputs_1[31:0]  = data (32-bit word containing 4 int8 values)
  //
  // funct7 = 1: Write to Global Buffer B
  //   - inputs_0[31:16] = address (index)
  //   - inputs_0[15:0]  = not used
  //   - inputs_1[31:0]  = data (32-bit word containing 4 int8 values)
  //
  // funct7 = 2: Start TPU computation
  //   - inputs_0[31:24] = K
  //   - inputs_0[23:16] = M
  //   - inputs_0[15:8]  = N
  //   - inputs_0[7:0]   = not used
  //   - outputs_0       = busy status (1 if busy, 0 if done)
  //
  // funct7 = 3: Read from Global Buffer C
  //   - inputs_0[31:16] = address (index)
  //   - inputs_0[15:0]  = word select (0-3, selects which 32-bit word from 128-bit data)
  //   - outputs_0       = data (32-bit int32 value)
  //
  // funct7 = 4: Check TPU busy status
  //   - outputs_0       = busy status (1 if busy, 0 if idle)

  wire [6:0] funct7 = cmd_payload_function_id[9:3];

  // TPU signals
  reg         tpu_in_valid;
  reg  [7:0]  tpu_K, tpu_M, tpu_N;
  wire        tpu_busy;

  wire        tpu_A_wr_en;
  wire [15:0] tpu_A_index;
  wire [31:0] tpu_A_data_in;
  wire [31:0] tpu_A_data_out;

  wire        tpu_B_wr_en;
  wire [15:0] tpu_B_index;
  wire [31:0] tpu_B_data_in;
  wire [31:0] tpu_B_data_out;

  wire        tpu_C_wr_en;
  wire [15:0] tpu_C_index;
  wire [127:0] tpu_C_data_in;
  wire [127:0] tpu_C_data_out;

  // Global Buffer control signals
  reg         gbuff_A_wr_en;
  reg  [15:0] gbuff_A_index;
  reg  [31:0] gbuff_A_data_in;
  
  reg         gbuff_B_wr_en;
  reg  [15:0] gbuff_B_index;
  reg  [31:0] gbuff_B_data_in;
  
  reg  [15:0] gbuff_C_index;
  
  // Actual buffer write enable and index (mux between CFU and TPU)
  wire        buf_A_wr_en = gbuff_A_wr_en | tpu_A_wr_en;
  wire [15:0] buf_A_index = gbuff_A_wr_en ? gbuff_A_index : tpu_A_index;
  wire [31:0] buf_A_data_in = gbuff_A_wr_en ? gbuff_A_data_in : tpu_A_data_in;
  
  wire        buf_B_wr_en = gbuff_B_wr_en | tpu_B_wr_en;
  wire [15:0] buf_B_index = gbuff_B_wr_en ? gbuff_B_index : tpu_B_index;
  wire [31:0] buf_B_data_in = gbuff_B_wr_en ? gbuff_B_data_in : tpu_B_data_in;
  
  wire [15:0] buf_C_index = tpu_busy ? tpu_C_index : gbuff_C_index;

  // Instantiate Global Buffers (BRAM)
  // Using 8-bit addresses (256 entries) - sufficient for 16x16 matrices
  // 16x16 int8 matrix needs 64 words (256 bytes / 4 bytes per word)
  global_buffer_bram #(
    .ADDR_BITS(8),
    .DATA_BITS(32)
  ) gbuff_A (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(buf_A_wr_en),
    .index(buf_A_index[7:0]),
    .data_in(buf_A_data_in),
    .data_out(tpu_A_data_out)
  );

  global_buffer_bram #(
    .ADDR_BITS(8),
    .DATA_BITS(32)
  ) gbuff_B (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(buf_B_wr_en),
    .index(buf_B_index[7:0]),
    .data_in(buf_B_data_in),
    .data_out(tpu_B_data_out)
  );

  global_buffer_bram #(
    .ADDR_BITS(8),
    .DATA_BITS(128)
  ) gbuff_C (
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(tpu_C_wr_en),
    .index(buf_C_index[7:0]),
    .data_in(tpu_C_data_in),
    .data_out(tpu_C_data_out)
  );

  // Instantiate TPU
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
    .B_index(tpu_B_index),
    .B_data_in(tpu_B_data_in),
    .B_data_out(tpu_B_data_out),
    .C_wr_en(tpu_C_wr_en),
    .C_index(tpu_C_index),
    .C_data_in(tpu_C_data_in),
    .C_data_out(tpu_C_data_out)
  );

  // Small shadow buffer for C to provide deterministic read timing
  reg [127:0] c_cache [0:255];
  integer c_cache_init_idx;

  // Handshaking with delay for shadow-buffer read operations
  reg        read_waiting;
  reg        read_valid;
  reg [7:0]  read_addr;
  reg [1:0]  word_sel_reg;

  wire       is_read_cmd = (funct7 == 7'd3);
  wire       accept_cmd  = cmd_valid && cmd_ready;
  assign rsp_valid = read_valid ? 1'b1 : (accept_cmd && !is_read_cmd);
  assign cmd_ready = ~(read_waiting || read_valid);

  // Mirror TPU C-writes into shadow cache
  always @(posedge clk or posedge reset) begin
    if (reset) begin
      for (c_cache_init_idx = 0; c_cache_init_idx < 256; c_cache_init_idx = c_cache_init_idx + 1)
        c_cache[c_cache_init_idx] <= 128'd0;
    end else if (tpu_C_wr_en) begin
      c_cache[tpu_C_index[7:0]] <= tpu_C_data_in;
    end
  end

  // CFU Operation Logic
  always @(posedge clk) begin
    if (reset) begin
      gbuff_A_wr_en <= 1'b0;
      gbuff_B_wr_en <= 1'b0;
      tpu_in_valid <= 1'b0;
      gbuff_A_index <= 16'd0;
      gbuff_B_index <= 16'd0;
      gbuff_C_index <= 16'd0;
      gbuff_A_data_in <= 32'd0;
      gbuff_B_data_in <= 32'd0;
      tpu_K <= 8'd0;
      tpu_M <= 8'd0;
      tpu_N <= 8'd0;
      rsp_payload_outputs_0 <= 32'd0;
      read_waiting <= 1'b0;
      read_valid <= 1'b0;
      word_sel_reg <= 2'd0;
      read_addr <= 8'd0;
    end else begin
      // Default: disable write enables
      gbuff_A_wr_en <= 1'b0;
      gbuff_B_wr_en <= 1'b0;
      tpu_in_valid <= 1'b0;
      
      // Handle read pipeline
      if (read_waiting) begin
        read_waiting <= 1'b0;
        read_valid   <= 1'b1;
        case (word_sel_reg)
          2'd0: rsp_payload_outputs_0 <= c_cache[read_addr][31:0];
          2'd1: rsp_payload_outputs_0 <= c_cache[read_addr][63:32];
          2'd2: rsp_payload_outputs_0 <= c_cache[read_addr][95:64];
          2'd3: rsp_payload_outputs_0 <= c_cache[read_addr][127:96];
        endcase
      end else if (read_valid && rsp_ready) begin
        read_valid <= 1'b0;
      end else if (accept_cmd) begin
        case (funct7)
          // Write to Global Buffer A
          7'd0: begin
            gbuff_A_wr_en <= 1'b1;
            gbuff_A_index <= cmd_payload_inputs_0[31:16];
            gbuff_A_data_in <= cmd_payload_inputs_1;
            rsp_payload_outputs_0 <= 32'd0;
          end
          
          // Write to Global Buffer B
          7'd1: begin
            gbuff_B_wr_en <= 1'b1;
            gbuff_B_index <= cmd_payload_inputs_0[31:16];
            gbuff_B_data_in <= cmd_payload_inputs_1;
            rsp_payload_outputs_0 <= 32'd0;
          end
          
          // Start TPU computation
          7'd2: begin
            tpu_K <= cmd_payload_inputs_0[31:24];
            tpu_M <= cmd_payload_inputs_0[23:16];
            tpu_N <= cmd_payload_inputs_0[15:8];
            tpu_in_valid <= 1'b1;
            rsp_payload_outputs_0 <= {31'd0, tpu_busy};
          end
          
          // Read from Global Buffer C
          7'd3: begin
            gbuff_C_index <= cmd_payload_inputs_0[31:16];
            word_sel_reg <= cmd_payload_inputs_0[1:0];  // Save word select
            read_addr    <= cmd_payload_inputs_0[23:16];
            read_waiting <= 1'b1;  // response next cycle from shadow buffer
          end
          
          // Check TPU busy status
          7'd4: begin
            rsp_payload_outputs_0 <= {31'd0, tpu_busy};
          end
          
          default: begin
            rsp_payload_outputs_0 <= 32'd0;
          end
        endcase
      end
    end
  end

endmodule
