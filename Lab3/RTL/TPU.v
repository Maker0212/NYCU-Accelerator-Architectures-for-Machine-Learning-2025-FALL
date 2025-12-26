// ---------------------------------------------------------------------------
// TPU : 4x4 systolic array with tiling (supports arbitrary M,N,K up to <128)
// A/B SRAM: index @ posedge, data valid after next negedge
// C SRAM : 128b write, wr_en sampled at negedge
// Mapping rules:
//   - A : Type-A (transposed/column-major-in-4pack)   -> stride = ceil(M/4)
//   - B : Type-B (row-major-in-4pack)                 -> stride = ceil(N/4)
//   - C : Type-B (row-major-in-4pack, 128b per 4 cols, MSB = col0)
// ---------------------------------------------------------------------------
`include "PE.v"
`include "systolic_array.v"

module TPU(
    input         clk,
    input         rst_n,
    input         in_valid,
    input  [7:0]  K,
    input  [7:0]  M,
    input  [7:0]  N,
    output reg    busy,

    output        A_wr_en,
    output [15:0] A_index,
    output [31:0] A_data_in,
    input  [31:0] A_data_out,

    output        B_wr_en,
    output [15:0] B_index,
    output [31:0] B_data_in,
    input  [31:0] B_data_out,

    output        C_wr_en,
    output [15:0] C_index,
    output [127:0] C_data_in,
    input  [127:0] C_data_out
  );

  // ------------------------------
  // External buffer connections
  // ------------------------------
  reg         c_wr_en_r;
  reg  [15:0] a_idx_r, b_idx_r, c_idx_r;
  reg  [127:0] c_din_r;

  assign A_wr_en   = 1'b0;
  assign B_wr_en   = 1'b0;
  assign A_data_in = 32'd0;
  assign B_data_in = 32'd0;

  assign A_index   = a_idx_r;
  assign B_index   = b_idx_r;

  assign C_wr_en   = c_wr_en_r;
  assign C_index   = c_idx_r;
  assign C_data_in = c_din_r;

  // ------------------------------
  // States
  // ------------------------------
  localparam S_IDLE  = 4'd0;
  localparam S_RD0   = 4'd1; // tile entry: issue k=0 index & latch tile params
  localparam S_LD0   = 4'd2; // latch first words
  localparam S_CLR   = 4'd3; // clear accumulators
  localparam S_RUN   = 4'd4; // stream k=0..K-1
  localparam S_FLUSH = 4'd5; // push zeros for (m_q-1)+(n_q-1)
  localparam S_WR    = 4'd6; // write m_q rows
  localparam S_DONE  = 4'd7;

  reg [3:0] state, state_nxt;

  // ------------------------------
  // Totals & tiling
  // ------------------------------
  reg [7:0] M_tot, N_tot, K_tot;     // latched totals
  reg [7:0] blkM, blkN;              // ceil(M/4), ceil(N/4)
  reg [7:0] ib, jb;                  // tile indices: 0..blkM-1, 0..blkN-1

  // tile-local sizes (1..4) and K
  reg [7:0] m_q, n_q, k_q;

  // ------------------------------
  // Streaming words for current k
  // ------------------------------
  reg [31:0] a_word_cur, b_word_cur;

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

  // ------------------------------
  // 4x4 SA
  // ------------------------------
  wire       sa_start = (state == S_CLR);
  wire       sa_vld   = (state == S_RUN) || (state == S_FLUSH);

  reg  [7:0] sa_a0, sa_a1, sa_a2, sa_a3;
  reg  [7:0] sa_b0, sa_b1, sa_b2, sa_b3;

  wire [31:0] c00, c01, c02, c03;
  wire [31:0] c10, c11, c12, c13;
  wire [31:0] c20, c21, c22, c23;
  wire [31:0] c30, c31, c32, c33;

  wire [31:0]  sa_a_bus = { sa_a0, sa_a1, sa_a2, sa_a3 };
  wire [31:0]  sa_b_bus = { sa_b0, sa_b1, sa_b2, sa_b3 };
  wire [511:0] sa_c_bus;

  systolic_array_4x4 SA4 (
                       .clk  (clk),
                       .rst_n(rst_n),
                       .start(sa_start),
                       .vld  (sa_vld),
                       .a_bus(sa_a_bus),
                       .b_bus(sa_b_bus),
                       .c_bus(sa_c_bus)
                     );

  assign c00 = sa_c_bus[ 31:  0];
  assign c01 = sa_c_bus[ 63: 32];
  assign c02 = sa_c_bus[ 95: 64];
  assign c03 = sa_c_bus[127: 96];
  assign c10 = sa_c_bus[159:128];
  assign c11 = sa_c_bus[191:160];
  assign c12 = sa_c_bus[223:192];
  assign c13 = sa_c_bus[255:224];
  assign c20 = sa_c_bus[287:256];
  assign c21 = sa_c_bus[319:288];
  assign c22 = sa_c_bus[351:320];
  assign c23 = sa_c_bus[383:352];
  assign c30 = sa_c_bus[415:384];
  assign c31 = sa_c_bus[447:416];
  assign c32 = sa_c_bus[479:448];
  assign c33 = sa_c_bus[511:480];

  // ------------------------------
  // Counters
  // ------------------------------
  reg [7:0] k_cnt;                   // 0..K-1
  reg [2:0] flush_cnt, flush_need;   // <=6 when m_q=n_q=4
  reg [1:0] row_cnt;                 // 0..m_q-1

  // ------------------------------
  // SA boundary feed
  // ------------------------------
  always @*
  begin
    // default: zeros (S_RD0 / S_LD0 / S_CLR / S_FLUSH)
    sa_a0 = 8'd0;
    sa_a1 = 8'd0;
    sa_a2 = 8'd0;
    sa_a3 = 8'd0;
    sa_b0 = 8'd0;
    sa_b1 = 8'd0;
    sa_b2 = 8'd0;
    sa_b3 = 8'd0;
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
        sa_b0 = pick8(b_word_cur, 2'd0);
      if (1 < n_q)
        sa_b1 = pick8(b_word_cur, 2'd1);
      if (2 < n_q)
        sa_b2 = pick8(b_word_cur, 2'd2);
      if (3 < n_q)
        sa_b3 = pick8(b_word_cur, 2'd3);
    end
  end

  // ------------------------------
  // Pack a row for C writeback (Type-B row word, MSB holds col0)
  // ------------------------------
  reg [127:0] c_din_pack;
  always @*
  begin
    case (row_cnt)
      2'd0:
        c_din_pack = { (n_q>0)?c00:32'd0, (n_q>1)?c01:32'd0,
                       (n_q>2)?c02:32'd0, (n_q>3)?c03:32'd0 };
      2'd1:
        c_din_pack = { (n_q>0)?c10:32'd0, (n_q>1)?c11:32'd0,
                       (n_q>2)?c12:32'd0, (n_q>3)?c13:32'd0 };
      2'd2:
        c_din_pack = { (n_q>0)?c20:32'd0, (n_q>1)?c21:32'd0,
                       (n_q>2)?c22:32'd0, (n_q>3)?c23:32'd0 };
      default:
        c_din_pack = { (n_q>0)?c30:32'd0, (n_q>1)?c31:32'd0,
                       (n_q>2)?c32:32'd0, (n_q>3)?c33:32'd0 };
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
        state_nxt = in_valid ? S_RD0 : S_IDLE;
      S_RD0  :
        state_nxt = S_LD0;
      S_LD0  :
        state_nxt = S_CLR;
      S_CLR  :
        state_nxt = S_RUN;
      S_RUN  :
        state_nxt = (k_cnt == k_q-1) ? S_FLUSH : S_RUN;
      S_FLUSH:
        state_nxt = (flush_need==0) ? S_WR
        : (flush_cnt == (flush_need-1)) ? S_WR : S_FLUSH;
      S_WR   :
      begin
        if (row_cnt == m_q-1)
        begin
          if (jb + 8'd1 < blkN)
            state_nxt = S_RD0;  // next col tile
          else if (ib + 8'd1 < blkM)
            state_nxt = S_RD0;  // next row tile
          else
            state_nxt = S_DONE; // all tiles done
        end
        else
          state_nxt = S_WR;
      end
      S_DONE :
        state_nxt = S_IDLE;
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
      b_idx_r <= 16'd0;
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
      k_q <= 0;
      a_word_cur <= 32'd0;
      b_word_cur <= 32'd0;

      k_cnt <= 0;
      flush_cnt <= 0;
      flush_need <= 0;
      row_cnt <= 0;
      c_wr_en_r <= 1'b0;
    end
    else
    begin
      state <= state_nxt;
      c_wr_en_r <= 1'b0;

      case (state)

        // -------------------------------------------------------------------
        // Accept a new job
        // -------------------------------------------------------------------
        S_IDLE:
        begin
          busy <= in_valid ? 1'b1 : 1'b0;
          if (in_valid)
          begin
            M_tot <= (M==0)?8'd1:M;
            N_tot <= (N==0)?8'd1:N;
            K_tot <= (K==0)?8'd1:K;
            k_q   <= (K==0)?8'd1:K;

            blkM  <= (((M==0)?8'd1:M) + 8'd3) >> 2; // ceil(M/4)
            blkN  <= (((N==0)?8'd1:N) + 8'd3) >> 2; // ceil(N/4)

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
            reg [8:0] M9, N9, start_row, start_col, remM, remN;
            M9        = {1'b0, M_tot};
            N9        = {1'b0, N_tot};
            start_row = ({1'b0, ib} << 2);   // ib*4
            start_col = ({1'b0, jb} << 2);   // jb*4
            remM      = (M9 > start_row) ? (M9 - start_row) : 9'd0;
            remN      = (N9 > start_col) ? (N9 - start_col) : 9'd0;
            m_q       <= (remM > 9'd4) ? 8'd4 : remM[7:0];
            n_q       <= (remN > 9'd4) ? 8'd4 : remN[7:0];
            flush_cnt <= 3'd0;
            flush_need <= ( ((remM > 9'd4) ? 8'd4 : remM[7:0]) - 8'd1 )
            + ( ((remN > 9'd4) ? 8'd4 : remN[7:0]) - 8'd1 );
          end

          // reset tile counters
          k_cnt   <= 8'd0;
          row_cnt <= 2'd0;

          // issue k=0 base addresses (Type-A / Type-B)
          begin : _ab_base
            reg [15:0] a_blk_base;
            reg [15:0] b_blk_base;
            a_blk_base = {8'd0, ib} * {8'd0, K_tot};
            b_blk_base = {8'd0, jb} * {8'd0, K_tot};
            a_idx_r   <= a_blk_base;
            b_idx_r   <= b_blk_base;
          end
        end

        // -------------------------------------------------------------------
        // Latch first words (k=0), and pre-issue k=1 if exists
        // -------------------------------------------------------------------
        S_LD0:
        begin
          a_word_cur <= A_data_out;  // k=0
          b_word_cur <= B_data_out;
          if (k_q > 8'd1)
          begin
            a_idx_r <= a_idx_r + 16'd1; // issue k=1
            b_idx_r <= b_idx_r + 16'd1;
          end
        end

        // -------------------------------------------------------------------
        // Clear accumulators (sa_start asserted combinationally)
        // -------------------------------------------------------------------
        S_CLR:
        begin
          // nothing sequential
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
            b_word_cur <= B_data_out;
          end

          // 2) issue address for (k_cnt+2), if any
          if (k_cnt + 8'd1 < k_q-1)
          begin
            a_idx_r <= a_idx_r + 16'd1;
            b_idx_r <= b_idx_r + 16'd1;
          end

          // 3) advance k
          if (k_cnt != k_q-1)
            k_cnt <= k_cnt + 8'd1;
        end

        // -------------------------------------------------------------------
        // Flush zeros for (m_q-1)+(n_q-1)
        // -------------------------------------------------------------------
        S_FLUSH:
        begin
          if (flush_need != 0 && flush_cnt != (flush_need-1))
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
          begin : _cidx
            reg [15:0] ib4, row_glb;
            reg [15:0] col_blk_base;
            reg [1:0]  row_for_idx;

            row_for_idx = row_cnt;

            ib4          = ({8'd0, ib} << 2);                  // ib*4
            row_glb      = ib4 + {14'd0, row_for_idx};         // ib*4 + row
            col_blk_base = {8'd0, jb} * {8'd0, M_tot};         // jb*M
            c_idx_r      <= col_blk_base + row_glb;            // jb*M + row
          end


          c_din_r   <= c_din_pack;
          c_wr_en_r <= 1'b1;

          if (row_cnt != (m_q-1))
          begin
            row_cnt <= row_cnt + 2'd1;
          end
          else
          begin
            if (jb + 8'd1 < blkN)
            begin
              jb <= jb + 8'd1;         // next column tile
            end
            else if (ib + 8'd1 < blkM)
            begin
              ib <= ib + 8'd1;         // next row tile
              jb <= 8'd0;
            end
            row_cnt <= 2'd0;
          end
        end

        // -------------------------------------------------------------------
        // All tiles done
        // -------------------------------------------------------------------
        S_DONE:
        begin
          busy <= 1'b0;
        end

      endcase
    end
  end
endmodule
