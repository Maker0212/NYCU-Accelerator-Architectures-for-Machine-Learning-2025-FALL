module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,  // in0
  input      [31:0]   cmd_payload_inputs_1,  // unused
  output              rsp_valid,
  input               rsp_ready,
  output     [31:0]   rsp_payload_outputs_0, // out0
  input               reset,
  input               clk
);

  assign rsp_valid = cmd_valid;
  assign cmd_ready = rsp_ready;
  wire [2:0] funct3 = cmd_payload_function_id[2:0];
  wire [6:0] funct7 = cmd_payload_function_id[9:3];
  wire       is_op0 = (funct3 == 3'd0);

  localparam signed [31:0] MIN_RAW_Q4 = -(8 << 27);
  localparam signed [31:0] MAX_RAW_Q4 =  (8 << 27) - 1;

  // 1. Logistic 用節點：σ(k*0.25), k=0..32，Q0.31, σ(x) = 1 / (1 + e^{-x}), Y_sig[0]  = σ(0*0.25)
  reg [31:0] Y_sig[0:32];
  initial begin
    Y_sig[ 0]=32'h40000000; Y_sig[ 1]=32'h47F5664B; Y_sig[ 2]=32'h4FACBF53; Y_sig[ 3]=32'h56EF53DF;
    Y_sig[ 4]=32'h5D9353D7; Y_sig[ 5]=32'h637E8FD5; Y_sig[ 6]=32'h68A647CB; Y_sig[ 7]=32'h6D0CCA17;
    Y_sig[ 8]=32'h70BDF56A; Y_sig[ 9]=32'h73CB96B7; Y_sig[10]=32'h764A4777; Y_sig[11]=32'h784F14A5;
    Y_sig[12]=32'h79EDF2F1; Y_sig[13]=32'h7B38DF60; Y_sig[14]=32'h7C3F7F30; Y_sig[15]=32'h7D0F13DA;
    Y_sig[16]=32'h7DB2A0BC; Y_sig[17]=32'h7E3329BF; Y_sig[18]=32'h7E97FAD8; Y_sig[19]=32'h7EE6EED6;
    Y_sig[20]=32'h7F24B04C; Y_sig[21]=32'h7F54F269; Y_sig[22]=32'h7F7AA136; Y_sig[23]=32'h7F98099B;
    Y_sig[24]=32'h7FEAFA22; Y_sig[25]=32'h7FC0DD61; Y_sig[26]=32'h7FCECF2C; Y_sig[27]=32'h7FD9AD70;
    Y_sig[28]=32'h7FE2258C; Y_sig[29]=32'h7FE8BEDA; Y_sig[30]=32'h7FEDE2F5; Y_sig[31]=32'h7FF1E43A;
    Y_sig[32]=32'h7FF502E1;
  end

  // 2. e^{-x} 節點：x=k*0.25, k=0..32，Q0.31  （Softmax exp 用）, e^{-x}
  reg [31:0] Y_exp[0:32];
  initial begin
    Y_exp[ 0]=32'h7FFFFFFF; Y_exp[ 1]=32'h63AFBE7B; Y_exp[ 2]=32'h4DA2CBF2; Y_exp[ 3]=32'h3C7681D8;
    Y_exp[ 4]=32'h2F16AC6C; Y_exp[ 5]=32'h24AC306E; Y_exp[ 6]=32'h1C8F8772; Y_exp[ 7]=32'h163E397E;
    Y_exp[ 8]=32'h1152AAA4; Y_exp[ 9]=32'h0D7DB8C7; Y_exp[10]=32'h0A81C2E0; Y_exp[11]=32'h082EC9C5;
    Y_exp[12]=32'h065F6C33; Y_exp[13]=32'h04F68DA1; Y_exp[14]=32'h03DD8203; Y_exp[15]=32'h0302A127;
    Y_exp[16]=32'h02582AB7; Y_exp[17]=32'h01D36911; Y_exp[18]=32'h016C0504; Y_exp[19]=32'h011B7FAE;
    Y_exp[20]=32'h00DCC9FF; Y_exp[21]=32'h00ABF360; Y_exp[22]=32'h0085EA53; Y_exp[23]=32'h00684B1A;
    Y_exp[24]=32'h00513948; Y_exp[25]=32'h003F41D3; Y_exp[26]=32'h003143C3; Y_exp[27]=32'h00265E0D;
    Y_exp[28]=32'h001DE16C; Y_exp[29]=32'h00174560; Y_exp[30]=32'h00121F9C; Y_exp[31]=32'h000E1D55;
    Y_exp[32]=32'h000AFE11;
  end

  // function_id=1 : Logistic σ(x)
  wire is_logi = is_op0 && (funct7 == 7'd1);
  wire signed [31:0] x_q4      = cmd_payload_inputs_0;   // Q4.27
  wire signed [31:0] x_clamped = (x_q4 < MIN_RAW_Q4) ? MIN_RAW_Q4 :
                                 (x_q4 > MAX_RAW_Q4) ? MAX_RAW_Q4 : x_q4;
  wire neg = x_clamped[31];
  wire signed [31:0] x_pos = neg ? (~x_clamped + 32'sd1) : x_clamped; // |x|
  // 0.25 步距：scaled = |x| * 4
  wire [32:0] scaled = {1'b0, x_pos} << 2; // 乘 4（每 0.25 一格 變每 1 一格）
  wire [26:0] frac   = scaled[26:0];       // Q0.27 小數
  wire [4:0]  idx0   = scaled[31:27];      // 0..31 整數
  wire [5:0]  idx1   = {1'b0, idx0} + 6'd1;// 下一點索引（idx0+1）

  //y=y0​+(y1​−y0​)*t
  wire signed [31:0] y0_sig = Y_sig[idx0];
  wire signed [31:0] y1_sig = Y_sig[idx1];
  wire signed [31:0] dy_sig = y1_sig - y0_sig;
  wire signed [63:0] mul_sig = $signed(dy_sig) * $signed({1'b0, frac}); // Q0.31 × Q0.27 = Q0.58
  wire signed [31:0] y_pos_q0 = y0_sig + $signed(mul_sig >>> 27); // 除以2^27
  
  //對稱性：σ(-x) = 1 - σ(x)
  wire [31:0] one_q0 = 32'h7FFF_FFFF;
  wire [31:0] logi_q0 = neg ? (one_q0 - y_pos_q0) : y_pos_q0;



  // function_id=2 : exp_on_negative_values(a)
  wire is_exp  = is_op0 && (funct7 == 7'd2);
  wire signed [31:0] a_q5 = cmd_payload_inputs_0;        // Q5.26
  // b = -a（若 a>=0 則設 0），接著轉到 Q4.27（左移 1）
  wire signed [31:0] b_q5 = a_q5[31] ? (~a_q5 + 32'sd1) : 32'sd0;
  wire signed [31:0] b_q4 = b_q5 << 1;  // 同值，換小數位

  // clamp b 到 [0,8)
  wire signed [31:0] b_clamped_q4 = (b_q4 > MAX_RAW_Q4) ? MAX_RAW_Q4 : b_q4;

  // 0.25 步距插值
  wire [32:0] e_scaled = {1'b0, b_clamped_q4} << 2;  // *4
  wire [26:0] e_frac   = e_scaled[26:0];
  wire [4:0]  e_idx0   = e_scaled[31:27];            // 0..31
  wire [5:0]  e_idx1   = {1'b0, e_idx0} + 6'd1;

  //取 LUT 的兩個端點值
  wire signed [31:0] y0_exp = Y_exp[e_idx0];
  wire signed [31:0] y1_exp = Y_exp[e_idx1];
  
  //內插：y ≈ y0 + (y1 - y0) * t
  wire signed [31:0] dy_exp = y1_exp - y0_exp;
  wire signed [63:0] mul_exp = $signed(dy_exp) * $signed({1'b0, e_frac});
  wire signed [31:0] exp_q0  = y0_exp + $signed(mul_exp >>> 27);  // Q0.31

  assign rsp_payload_outputs_0 =
      is_logi ? logi_q0 :
      is_exp  ? exp_q0  :
                32'h0000_0000;

endmodule
