// Custom Function Unit (CFU) 模組
// 用於加速卷積 / fully connected 的 dot product 與 MAC 計算
module Cfu (
    input               cmd_valid,               // CPU 送來的命令有效
    output              cmd_ready,               // CFU 是否準備好接收新命令
    input      [9:0]    cmd_payload_function_id, // 功能 ID，決定執行哪個運算
    input      [31:0]   cmd_payload_inputs_0,    // 輸入資料 x，打包的 4 個 int8
    input      [31:0]   cmd_payload_inputs_1,    // 權重資料 w，打包的 4 個 int8
    output reg          rsp_valid,               // 回覆有效
    input               rsp_ready,               // CPU 是否準備好接收回覆
    output reg [31:0]   rsp_payload_outputs_0,   // 回覆資料（通常是 MAC 結果）
    input               reset,
    input               clk
);

    // -------- 模型相關參數 --------
    // InputOffset：修正輸入 zero-point（int8 量化偏移），權重是對稱量化所以不用修正
    reg  signed [8:0] InputOffset;  // 範圍 [-256,255]，9bit 以避免溢位
	
    // 內部累加器，存放部分和或 MAC 結果
	reg signed [31:0] acc_reg;

	// 暫存變數：單一元素的輸入 / 權重 / 乘積
	reg signed [8:0] sx, sw;
	reg signed [17:0] prod;
	
    // 工具函式：將 8-bit 符號數 sign-extend 成 9-bit
    function automatic signed [8:0] sext8(input [7:0] x);
        sext8 = {x[7], x};
    endfunction

    // === 將 inputs_0 unpack 成 4 個 int8 並加上 input_offset ===
    wire signed [8:0] x0 = sext8(cmd_payload_inputs_0[ 7: 0]) + InputOffset;
    wire signed [8:0] x1 = sext8(cmd_payload_inputs_0[15: 8]) + InputOffset;
    wire signed [8:0] x2 = sext8(cmd_payload_inputs_0[23:16]) + InputOffset;
    wire signed [8:0] x3 = sext8(cmd_payload_inputs_0[31:24]) + InputOffset;

    // === 將 inputs_1 unpack 成 4 個 int8（權重不用加 offset） ===
    wire signed [8:0] w0 = sext8(cmd_payload_inputs_1[ 7: 0]);
    wire signed [8:0] w1 = sext8(cmd_payload_inputs_1[15: 8]);
    wire signed [8:0] w2 = sext8(cmd_payload_inputs_1[23:16]);
    wire signed [8:0] w3 = sext8(cmd_payload_inputs_1[31:24]);

    // === 進行 4 個乘法：每個 9x9=18 bit ===
    wire signed [17:0] p0 = x0 * w0;
    wire signed [17:0] p1 = x1 * w1;
    wire signed [17:0] p2 = x2 * w2;
    wire signed [17:0] p3 = x3 * w3;

    // === 把結果 sign-extend 到 32 bit 再加總 ===
    wire signed [31:0] sum_prods =
        $signed({{14{p0[17]}}, p0}) +
        $signed({{14{p1[17]}}, p1}) +
        $signed({{14{p2[17]}}, p2}) +
        $signed({{14{p3[17]}}, p3});

    // === 硬體 handshake ===
    // 當前一次 rsp_valid 還沒被取走時，不能再接新命令
    assign cmd_ready = ~rsp_valid;

    always @(posedge clk) begin
        if (reset) begin
            // reset 時清空狀態
            rsp_valid             <= 1'b0;
            rsp_payload_outputs_0 <= 32'd0;
            InputOffset           <= 9'sd0;
			acc_reg               <= 32'd0;
        end else begin
            // 若回覆送出且 CPU 收走，清 rsp_valid
            if (rsp_valid && rsp_ready)
                rsp_valid <= 1'b0;

            // 有新命令時依 function_id 執行
            if (cmd_valid && cmd_ready) begin
                case (cmd_payload_function_id)
                    // -----------------------------
                    // FID=0: 設定 InputOffset
                    // -----------------------------
                    10'd0: begin
                        InputOffset           <= $signed(cmd_payload_inputs_0[8:0]);
                        rsp_payload_outputs_0 <= 32'd0; // 回覆 0 表示完成
                        rsp_valid             <= 1'b1;
                    end

                    // -----------------------------
                    // FID=1: 4-way dot product
                    // 輸出 sum_i (w_i * (x_i + offset))
                    // 同時存入累加器 acc_reg
                    // -----------------------------
                    10'd1: begin
                        rsp_payload_outputs_0 <= sum_prods;
                        rsp_valid             <= 1'b1;
						acc_reg <= acc_reg + sum_prods;
                    end

                    // -----------------------------
                    // FID=2: 回傳累加器的值並清零
                    // -----------------------------
					10'd2: begin
					  rsp_payload_outputs_0 <= acc_reg;
					  rsp_valid             <= 1'b1;
					  acc_reg               <= 32'sd0;
					end

                    // -----------------------------
                    // FID=3: scalar accumulate
                    // acc_reg += ( (int8)x + InputOffset ) * (int8)w
                    // -----------------------------
					10'd3: begin
					  sx = {cmd_payload_inputs_0[7], cmd_payload_inputs_0[7:0]} + InputOffset;
					  sw = {cmd_payload_inputs_1[7], cmd_payload_inputs_1[7:0]};
					  prod = sx * sw;
					  acc_reg <= acc_reg + {{14{prod[17]}}, prod}; // sign-extend to 32
					  rsp_payload_outputs_0 <= 32'sd0; // 回覆值不重要
					  rsp_valid <= 1'b1;
					end

                    // -----------------------------
                    // FID=4: 把 bias 加進累加器
                    // -----------------------------
					10'd4: begin
					  acc_reg <= acc_reg + $signed(cmd_payload_inputs_0);
					  rsp_payload_outputs_0 <= 32'sd0;
					  rsp_valid <= 1'b1;
					end

                    // -----------------------------
                    // Default: 未定義功能 → 回傳錯誤碼
                    // -----------------------------
                    default: begin
                        rsp_payload_outputs_0 <= 32'hDEAD_BEEF;
                        rsp_valid             <= 1'b1;
                    end
                endcase
            end
        end
    end
endmodule
