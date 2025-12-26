module PE (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clr_acc,   // pulse to clear accumulator
    input  wire        vld,       // shift/accumulate this cycle
    input  wire [7:0]  a_in,
    input  wire [7:0]  b_in,
    output reg  [7:0]  a_out,
    output reg  [7:0]  b_out,
    output reg signed [31:0] acc
  );
  wire signed [15:0] prod = $signed(a_in) * $signed(b_in);

  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      a_out <= 0;
      b_out <= 0;
      acc <= 0;
    end
    else
    begin
      a_out <= a_in;
      b_out <= b_in;
      if (clr_acc)
        acc <= 0;
      else if (vld)
        acc <= acc + $signed(prod);
    end
  end
endmodule
