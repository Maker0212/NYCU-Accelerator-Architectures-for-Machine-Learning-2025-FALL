// ---------------------------------------------------------------------------
// TPU : 4x4 systolic array with tiling (supports arbitrary M,N,K up to <128)
// A/B SRAM: index @ posedge, data valid after next negedge
// C SRAM : 128b write, wr_en sampled at negedge
// Mapping rules:
//   - A : Type-A (transposed/column-major-in-4pack)   -> stride = ceil(M/4)
//   - B : Type-B (row-major-in-4pack)                 -> stride = ceil(N/4)
//   - C : Type-B (row-major-in-4pack, 128b per 4 cols, MSB = col0)
// ---------------------------------------------------------------------------
// Note: PE.v and systolic_array.v should be included before this file

module TPU(
    input         clk,
    input         rst_n,
    input         in_valid,
    input  [15:0] K,
    input  [7:0]  M,
    input  [7:0]  N,
    output reg    busy,

    output        A_wr_en,
    output [15:0] A_index,
    output [31:0] A_data_in,
    input  [31:0] A_data_out,

    output        B_wr_en,
    output [15:0] B_index0,
    output [15:0] B_index1,
    output [31:0] B_data_in,
    input  [31:0] B_data_out0,
    input  [31:0] B_data_out1,

    output        C_wr_en,
    output [15:0] C_index,
    output [127:0] C_data_in,
    input  [127:0] C_data_out
  );

  // ------------------------------
  // External buffer connections
  // ------------------------------
  reg         c_wr_en_r;
  reg  [15:0] a_idx_r, b_idx_r0, b_idx_r1, c_idx_r;
  reg  [127:0] c_din_r;

  assign A_wr_en   = 1'b0;
  assign B_wr_en   = 1'b0;
  assign A_data_in = 32'd0;
  assign B_data_in = 32'd0;

  assign A_index   = a_idx_r;
  assign B_index0  = b_idx_r0;
  assign B_index1  = b_idx_r1;

  assign C_wr_en   = c_wr_en_r;
  assign C_index   = c_idx_r;
  assign C_data_in = c_din_r;

  // ------------------------------
  // States
  // ------------------------------
  localparam S_IDLE  = 4'd0;
  localparam S_RD0   = 4'd1; // tile entry: issue k=0 index & latch tile params
  localparam S_LD0   = 4'd2; // latch first words
  localparam S_RUN   = 4'd3; // stream k=0..K-1
  localparam S_FLUSH = 4'd4; // push zeros for (m_q-1)+(n_q-1)
  localparam S_WR    = 4'd5; // write m_q rows

  reg [3:0] state, state_nxt;

  // ------------------------------
  // Totals & tiling
  // ------------------------------
  reg [7:0] M_tot, N_tot;
  reg [15:0] K_tot;     // latched totals
  reg [7:0] blkM, blkN;              // ceil(M/4), ceil(N/4)
  reg [7:0] ib, jb;                  // tile indices: 0..blkM-1, 0..blkN-1

  // tile-local sizes (1..4) and K
  reg [7:0] m_q, n_q, n_q2;
  reg [15:0] k_q;
  reg        has_second_block;

  // pending start request (captures 1-cycle pulses until S_IDLE consumes them)
  reg        start_pending;
  reg [7:0]  M_req, N_req;
  reg [15:0] K_req;
  reg        in_valid_d;

  wire       idle        = (state == S_IDLE);
  wire [7:0] M_sane      = (M==0)?8'd1:M;
  wire [7:0] N_sane      = (N==0)?8'd1:N;
  wire [15:0] K_sane     = (K==0)?16'd1:K;
  wire       start_pulse = in_valid && !in_valid_d;
  wire       start_fire_live = idle && in_valid;
  wire       start_fire_pend = idle && !in_valid && start_pending;
  wire       start_fire  = start_fire_live || start_fire_pend;
  wire       use_pending = start_fire_pend;
  wire [7:0] start_M     = use_pending ? M_req : M_sane;
  wire [7:0] start_N     = use_pending ? N_req : N_sane;
  wire [15:0] start_K    = use_pending ? K_req : K_sane;

  // ------------------------------
  // Streaming words for current k
  // ------------------------------
  reg [31:0] a_word_cur, b_word_cur0, b_word_cur1;

  // pick byte helper (MSB-first)
  function [7:0] pick8;
    input [31:0] w;
    input [1:0] idx;
    begin
      case (idx)
        2'd0:
          pick8 = w[31:24];
        2'd1:
          pick8 = w[23:16];
        2'd2:
          pick8 = w[15:8 ];
        default:
          pick8 = w[7:0];
      endcase
    end
  endfunction

  function [8:0] pick8_signed;
    input [31:0] w;
    input [1:0] idx;
    reg [7:0] val;
    begin
      case (idx)
        2'd0:
          val = w[31:24];
        2'd1:
          val = w[23:16];
        2'd2:
          val = w[15:8 ];
        default:
          val = w[7:0];
      endcase
      pick8_signed = {val[7], val};
    end
  endfunction

  // ------------------------------
  // 4x4 SA
  // ------------------------------
  wire       sa_start = (state == S_LD0);
  wire       sa_vld   = (state == S_RUN) || (state == S_FLUSH);

  reg  [7:0] sa_a0, sa_a1, sa_a2, sa_a3;
  reg  [8:0] sa0_b0, sa0_b1, sa0_b2, sa0_b3;
  reg  [8:0] sa1_b0, sa1_b1, sa1_b2, sa1_b3;

  wire [31:0] c00_0, c01_0, c02_0, c03_0;
  wire [31:0] c10_0, c11_0, c12_0, c13_0;
  wire [31:0] c20_0, c21_0, c22_0, c23_0;
  wire [31:0] c30_0, c31_0, c32_0, c33_0;

  wire [31:0] c00_1, c01_1, c02_1, c03_1;
  wire [31:0] c10_1, c11_1, c12_1, c13_1;
  wire [31:0] c20_1, c21_1, c22_1, c23_1;
  wire [31:0] c30_1, c31_1, c32_1, c33_1;

  wire [31:0]  sa_a_bus = { sa_a0, sa_a1, sa_a2, sa_a3 };
  wire [35:0]  sa_b_bus0 = { sa0_b0, sa0_b1, sa0_b2, sa0_b3 };
  wire [35:0]  sa_b_bus1 = { sa1_b0, sa1_b1, sa1_b2, sa1_b3 };
  wire [511:0] sa_c_bus0;
  wire [511:0] sa_c_bus1;

  systolic_array_4x4 SA4_0 (
                         .clk  (clk),
                         .rst_n(rst_n),
                         .start(sa_start),
                         .vld  (sa_vld),
                         .a_bus(sa_a_bus),
                         .b_bus(sa_b_bus0),
                         .c_bus(sa_c_bus0)
                       );

  systolic_array_4x4 SA4_1 (
                         .clk  (clk),
                         .rst_n(rst_n),
                         .start(sa_start),
                         .vld  (sa_vld),
                         .a_bus(sa_a_bus),
                         .b_bus(sa_b_bus1),
                         .c_bus(sa_c_bus1)
                       );

  assign c00_0 = sa_c_bus0[ 31:  0];
  assign c01_0 = sa_c_bus0[ 63: 32];
  assign c02_0 = sa_c_bus0[ 95: 64];
  assign c03_0 = sa_c_bus0[127: 96];
  assign c10_0 = sa_c_bus0[159:128];
  assign c11_0 = sa_c_bus0[191:160];
  assign c12_0 = sa_c_bus0[223:192];
  assign c13_0 = sa_c_bus0[255:224];
  assign c20_0 = sa_c_bus0[287:256];
  assign c21_0 = sa_c_bus0[319:288];
  assign c22_0 = sa_c_bus0[351:320];
  assign c23_0 = sa_c_bus0[383:352];
  assign c30_0 = sa_c_bus0[415:384];
  assign c31_0 = sa_c_bus0[447:416];
  assign c32_0 = sa_c_bus0[479:448];
  assign c33_0 = sa_c_bus0[511:480];

  assign c00_1 = sa_c_bus1[ 31:  0];
  assign c01_1 = sa_c_bus1[ 63: 32];
  assign c02_1 = sa_c_bus1[ 95: 64];
  assign c03_1 = sa_c_bus1[127: 96];
  assign c10_1 = sa_c_bus1[159:128];
  assign c11_1 = sa_c_bus1[191:160];
  assign c12_1 = sa_c_bus1[223:192];
  assign c13_1 = sa_c_bus1[255:224];
  assign c20_1 = sa_c_bus1[287:256];
  assign c21_1 = sa_c_bus1[319:288];
  assign c22_1 = sa_c_bus1[351:320];
  assign c23_1 = sa_c_bus1[383:352];
  assign c30_1 = sa_c_bus1[415:384];
  assign c31_1 = sa_c_bus1[447:416];
  assign c32_1 = sa_c_bus1[479:448];
  assign c33_1 = sa_c_bus1[511:480];

  // ------------------------------
  // Counters
  // ------------------------------
  reg [15:0] k_cnt;                   // 0..K-1
  reg [2:0] flush_cnt, flush_need;   // <=6 when m_q=n_q=4
  reg [1:0] row_cnt;                 // 0..m_q-1
  reg       write_half;
  wire      row_write_done = (!has_second_block) || write_half;

  // ------------------------------
  // SA boundary feed
  // ------------------------------
  always @*
  begin
  // default: zeros (S_RD0 / S_LD0 / S_FLUSH)
    sa_a0 = 8'd0;
    sa_a1 = 8'd0;
    sa_a2 = 8'd0;
    sa_a3 = 8'd0;
    sa0_b0 = 9'd0;
    sa0_b1 = 9'd0;
    sa0_b2 = 9'd0;
    sa0_b3 = 9'd0;
    sa1_b0 = 9'd0;
    sa1_b1 = 9'd0;
    sa1_b2 = 9'd0;
    sa1_b3 = 9'd0;
    if (state == S_RUN)
    begin
      if (0 < m_q)
        sa_a0 = pick8(a_word_cur, 2'd0);
      if (1 < m_q)
        sa_a1 = pick8(a_word_cur, 2'd1);
      if (2 < m_q)
        sa_a2 = pick8(a_word_cur, 2'd2);
      if (3 < m_q)
        sa_a3 = pick8(a_word_cur, 2'd3);
      if (0 < n_q)
        sa0_b0 = pick8_signed(b_word_cur0, 2'd0);
      if (1 < n_q)
        sa0_b1 = pick8_signed(b_word_cur0, 2'd1);
      if (2 < n_q)
        sa0_b2 = pick8_signed(b_word_cur0, 2'd2);
      if (3 < n_q)
        sa0_b3 = pick8_signed(b_word_cur0, 2'd3);
      if (0 < n_q2)
        sa1_b0 = pick8_signed(b_word_cur1, 2'd0);
      if (1 < n_q2)
        sa1_b1 = pick8_signed(b_word_cur1, 2'd1);
      if (2 < n_q2)
        sa1_b2 = pick8_signed(b_word_cur1, 2'd2);
      if (3 < n_q2)
        sa1_b3 = pick8_signed(b_word_cur1, 2'd3);
    end
  end

  // ------------------------------
  // Pack a row for C writeback (Type-B row word, MSB holds col0)
  // ------------------------------
  reg [127:0] c_din_pack0;
  reg [127:0] c_din_pack1;
  always @*
  begin
    case (row_cnt)
      2'd0:
        c_din_pack0 = { (n_q>0)?c00_0:32'd0, (n_q>1)?c01_0:32'd0,
                        (n_q>2)?c02_0:32'd0, (n_q>3)?c03_0:32'd0 };
      2'd1:
        c_din_pack0 = { (n_q>0)?c10_0:32'd0, (n_q>1)?c11_0:32'd0,
                        (n_q>2)?c12_0:32'd0, (n_q>3)?c13_0:32'd0 };
      2'd2:
        c_din_pack0 = { (n_q>0)?c20_0:32'd0, (n_q>1)?c21_0:32'd0,
                        (n_q>2)?c22_0:32'd0, (n_q>3)?c23_0:32'd0 };
      default:
        c_din_pack0 = { (n_q>0)?c30_0:32'd0, (n_q>1)?c31_0:32'd0,
                        (n_q>2)?c32_0:32'd0, (n_q>3)?c33_0:32'd0 };
    endcase
  end

  always @*
  begin
    case (row_cnt)
      2'd0:
        c_din_pack1 = { (n_q2>0)?c00_1:32'd0, (n_q2>1)?c01_1:32'd0,
                        (n_q2>2)?c02_1:32'd0, (n_q2>3)?c03_1:32'd0 };
      2'd1:
        c_din_pack1 = { (n_q2>0)?c10_1:32'd0, (n_q2>1)?c11_1:32'd0,
                        (n_q2>2)?c12_1:32'd0, (n_q2>3)?c13_1:32'd0 };
      2'd2:
        c_din_pack1 = { (n_q2>0)?c20_1:32'd0, (n_q2>1)?c21_1:32'd0,
                        (n_q2>2)?c22_1:32'd0, (n_q2>3)?c23_1:32'd0 };
      default:
        c_din_pack1 = { (n_q2>0)?c30_1:32'd0, (n_q2>1)?c31_1:32'd0,
                        (n_q2>2)?c32_1:32'd0, (n_q2>3)?c33_1:32'd0 };
    endcase
  end

  // ------------------------------
  // Next-state
  // ------------------------------
  always @*
  begin
    state_nxt = state;
    case (state)
      S_IDLE :
        state_nxt = (in_valid || start_pending) ? S_RD0 : S_IDLE;
      S_RD0  :
        state_nxt = S_LD0;
      S_LD0  :
        state_nxt = S_RUN;
      S_RUN  :
        state_nxt = (k_cnt == k_q-1) ? S_FLUSH : S_RUN;
      S_FLUSH:
        state_nxt = (flush_need == 0) ? S_WR
                    : ((flush_cnt + 3'd1) >= flush_need) ? S_WR : S_FLUSH;
      S_WR   : begin
        if (!row_write_done) begin
          state_nxt = S_WR;
        end else if (row_cnt == m_q-1) begin
          if (jb + (has_second_block ? 8'd2 : 8'd1) < blkN)
            state_nxt = S_RD0;  // next col tile
          else if (ib + 8'd1 < blkM)
            state_nxt = S_RD0;  // next row tile
          else
            state_nxt = S_IDLE; // all tiles done
        end else begin
          state_nxt = S_WR;
        end
      end
      default:
        state_nxt = S_IDLE;
    endcase
  end

  // ------------------------------
  // Sequential
  // ------------------------------
  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      state <= S_IDLE;
      busy  <= 1'b0;

      a_idx_r <= 16'd0;
      b_idx_r0 <= 16'd0;
      b_idx_r1 <= 16'd0;
      c_idx_r <= 16'd0;
      c_wr_en_r <= 1'b0;
      c_din_r <= 128'd0;

      M_tot <= 0;
      N_tot <= 0;
      K_tot <= 0;
      blkM  <= 0;
      blkN  <= 0;
      ib <= 0;
      jb <= 0;

      m_q <= 0;
      n_q <= 0;
      n_q2 <= 0;
      has_second_block <= 1'b0;
      k_q <= 0;
      a_word_cur <= 32'd0;
      b_word_cur0 <= 32'd0;
      b_word_cur1 <= 32'd0;

      start_pending <= 1'b0;
      in_valid_d <= 1'b0;
      M_req <= 8'd0;
      N_req <= 8'd0;
      K_req <= 8'd0;

      k_cnt <= 0;
      flush_cnt <= 0;
      flush_need <= 0;
      row_cnt <= 0;
      write_half <= 1'b0;
      c_wr_en_r <= 1'b0;
    end
    else
    begin
      state <= state_nxt;
      c_wr_en_r <= 1'b0;
      in_valid_d <= in_valid;

      // capture in_valid pulses that arrive while we're not idle
      if (start_pulse && !idle)
      begin
        start_pending <= 1'b1;
        M_req <= M_sane;
        N_req <= N_sane;
        K_req <= K_sane;
      end
      else if (start_fire_pend)
      begin
        start_pending <= 1'b0;
      end

      case (state)

        // -------------------------------------------------------------------
        // Accept a new job
        // -------------------------------------------------------------------
        S_IDLE:
        begin
          if (in_valid || start_pending)
            busy <= 1'b1;
          else
            busy <= 1'b0;

          if (start_fire)
          begin
            M_tot <= start_M;
            N_tot <= start_N;
            K_tot <= start_K;
            k_q   <= start_K;

            blkM  <= (start_M + 8'd3) >> 2; // ceil(M/4)
            blkN  <= (start_N + 8'd3) >> 2; // ceil(N/4)

            ib <= 8'd0;
            jb <= 8'd0;
          end
        end

        // -------------------------------------------------------------------
        // Tile entry: compute tile-local params; issue k=0 base addresses
        // -------------------------------------------------------------------
        S_RD0:
        begin
          // tile-local sizes
          begin : _tile_sizes
            reg [8:0] M9, N9, start_row, start_col0, start_col1;
            reg [8:0] remM, remN0, remN1;
            reg [7:0] m_q_next, n_q_next, n_q2_next, max_n_q;
            reg       second_block;
            M9        = {1'b0, M_tot};
            N9        = {1'b0, N_tot};
            start_row = ({1'b0, ib} << 2);   // ib*4
            start_col0 = ({1'b0, jb} << 2);   // jb*4
            start_col1 = ({1'b0, (jb + 8'd1)} << 2);  // (jb+1)*4
            remM      = (M9 > start_row) ? (M9 - start_row) : 9'd0;
            remN0     = (N9 > start_col0) ? (N9 - start_col0) : 9'd0;
            remN1     = (N9 > start_col1) ? (N9 - start_col1) : 9'd0;
            m_q_next  = (remM > 9'd4) ? 8'd4 : remM[7:0];
            n_q_next  = (remN0 > 9'd4) ? 8'd4 : remN0[7:0];
            second_block = (jb + 8'd1 < blkN);
            n_q2_next = second_block ? ((remN1 > 9'd4) ? 8'd4 : remN1[7:0]) : 8'd0;
            max_n_q   = (n_q2_next > n_q_next) ? n_q2_next : n_q_next;
            m_q       <= m_q_next;
            n_q       <= n_q_next;
            n_q2      <= n_q2_next;
            has_second_block <= second_block;
            flush_cnt <= 3'd0;
            flush_need <= (m_q_next - 8'd1) + (max_n_q - 8'd1);
          end

          // reset tile counters
          k_cnt   <= 8'd0;
          row_cnt <= 2'd0;
          write_half <= 1'b0;

          // issue k=0 base addresses (Type-A / Type-B)
          begin : _ab_base
            reg [15:0] a_blk_base;
            reg [15:0] b_blk_base0;
            reg [15:0] b_blk_base1;
            reg        second_block;
            a_blk_base = {8'd0, ib} * {8'd0, K_tot};
            b_blk_base0 = {8'd0, jb} * {8'd0, K_tot};
            second_block = (jb + 8'd1 < blkN);
            if (second_block)
              b_blk_base1 = {8'd0, (jb + 8'd1)} * {8'd0, K_tot};
            else
              b_blk_base1 = b_blk_base0;
            a_idx_r   <= a_blk_base;
            b_idx_r0  <= b_blk_base0;
            b_idx_r1  <= b_blk_base1;
          end
        end

        // -------------------------------------------------------------------
        // Latch first words (k=0), pre-issue k=1 if exists, and pulse sa_start
        // -------------------------------------------------------------------
        S_LD0:
        begin
          a_word_cur <= A_data_out;  // k=0
          b_word_cur0 <= B_data_out0;
          if (has_second_block)
            b_word_cur1 <= B_data_out1;
          else
            b_word_cur1 <= 32'd0;
          if (k_q > 8'd1)
          begin
            a_idx_r <= a_idx_r + 16'd1; // issue k=1
            b_idx_r0 <= b_idx_r0 + 16'd1;
            if (has_second_block)
              b_idx_r1 <= b_idx_r1 + 16'd1;
          end
        end
        // -------------------------------------------------------------------
        // Stream k=0..K-1 : step by stride
        // -------------------------------------------------------------------
        S_RUN:
        begin
          // 1) latch the word requested in the previous cycle
          if (k_cnt < k_q-1)
          begin
            a_word_cur <= A_data_out;  // becomes k_cnt+1
            b_word_cur0 <= B_data_out0;
            if (has_second_block)
              b_word_cur1 <= B_data_out1;
            else
              b_word_cur1 <= 32'd0;
          end

          // 2) issue address for (k_cnt+2), if any
          if (k_cnt + 8'd1 < k_q-1)
          begin
            a_idx_r <= a_idx_r + 16'd1;
            b_idx_r0 <= b_idx_r0 + 16'd1;
            if (has_second_block)
              b_idx_r1 <= b_idx_r1 + 16'd1;
          end

          // 3) advance k
          if (k_cnt != k_q-1)
            k_cnt <= k_cnt + 8'd1;
        end

        // -------------------------------------------------------------------
        // Flush zeros for (m_q-1)+(n_q-1) cycles before writeback.
        // -------------------------------------------------------------------
        S_FLUSH:
        begin
          if (flush_need != 0 && flush_cnt < flush_need)
            flush_cnt <= flush_cnt + 3'd1;
        end

        // -------------------------------------------------------------------
        // Write back m_q rows (128b each)
        // C_index = ((ib*4 + row_cnt) * blkN) + jb
        // -------------------------------------------------------------------
        S_WR:
        begin
          c_wr_en_r <= 1'b1;
          // compute c_idx directly to avoid stale bases
          // C buffer stores rows in Type-B format: 4 int32s per entry
          // For row r, column block cb (0..N_tot/4-1): index = r * (N_tot/4) + cb
          begin : _cidx
            reg [15:0] ib4, row_glb;
            reg [15:0] n_blocks;  // Number of column blocks per row
            reg [1:0]  row_for_idx;
            reg [7:0]  col_block;

            row_for_idx = row_cnt;

            ib4          = ({8'd0, ib} << 2);                  // ib*4
            row_glb      = ib4 + {14'd0, row_for_idx};         // ib*4 + row (global row index)
            n_blocks     = ({8'd0, N_tot} + 8'd3) >> 2;        // ceil(N_tot/4) = number of blocks per row
            col_block    = jb + ((has_second_block && write_half) ? 8'd1 : 8'd0);
            c_idx_r      <= row_glb * n_blocks + {14'd0, 2'd0, col_block};  // row_glb * (N_tot/4) + col_block
          end

          if (has_second_block && write_half)
            c_din_r   <= c_din_pack1;
          else
            c_din_r   <= c_din_pack0;
          c_wr_en_r <= 1'b1;

          if (has_second_block && !write_half) begin
            write_half <= 1'b1;
          end else begin
            write_half <= 1'b0;
            if (row_cnt != (m_q-1))
            begin
              row_cnt <= row_cnt + 2'd1;
            end
            else
            begin
              if (jb + (has_second_block ? 8'd2 : 8'd1) < blkN)
              begin
                jb <= jb + (has_second_block ? 8'd2 : 8'd1);         // next column tile(s)
              end
              else if (ib + 8'd1 < blkM)
              begin
                ib <= ib + 8'd1;         // next row tile
                jb <= 8'd0;
              end
              row_cnt <= 2'd0;
            end
          end
        end

      endcase
    end
  end
endmodule
