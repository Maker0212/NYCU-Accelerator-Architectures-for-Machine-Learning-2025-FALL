// 2x2 systolic array module removed for CFU integration
// Only 4x4 version is used in TPU

// 4x4 systolic array with flat ports + boundary skew (Verilog-2005)
module systolic_array_4x4 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,       // pulse: clear all accumulators
    input  wire        vld,         // 1 while streaming/flush
    input  wire [31:0] a_bus,       // MSB..LSB = {A[0,k],A[1,k],A[2,k],A[3,k]}
    input  wire [31:0] b_bus,       // MSB..LSB = {B[k,0],B[k,1],B[k,2],B[k,3]}
    output wire [511:0] c_bus
  );

  // ---- left boundary: A moves left->right (each row has 5 taps) ----
  wire [7:0] a_r0_t0, a_r0_t1, a_r0_t2, a_r0_t3, a_r0_t4;
  wire [7:0] a_r1_t0, a_r1_t1, a_r1_t2, a_r1_t3, a_r1_t4;
  wire [7:0] a_r2_t0, a_r2_t1, a_r2_t2, a_r2_t3, a_r2_t4;
  wire [7:0] a_r3_t0, a_r3_t1, a_r3_t2, a_r3_t3, a_r3_t4;

  // add boundary skew for A: row1:+1, row2:+2, row3:+3
  reg [7:0] a1_d0;
  reg [7:0] a2_d1, a2_d2;
  reg [7:0] a3_d1, a3_d2, a3_d3;

  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      a1_d0 <= 8'd0;
      a2_d1 <= 8'd0;
      a2_d2 <= 8'd0;
      a3_d1 <= 8'd0;
      a3_d2 <= 8'd0;
      a3_d3 <= 8'd0;
    end
    else if (start)
    begin
      a1_d0 <= 8'd0;
      a2_d1 <= 8'd0;
      a2_d2 <= 8'd0;
      a3_d1 <= 8'd0;
      a3_d2 <= 8'd0;
      a3_d3 <= 8'd0;
    end
    else if (vld)
    begin
      a1_d0 <= a_bus[23:16];                 // row1 delay 1
      a2_d1 <= a_bus[15:8 ];
      a2_d2 <= a2_d1; // row2 delay 2
      a3_d1 <= a_bus[7 :0 ];
      a3_d2 <= a3_d1;
      a3_d3 <= a3_d2; // row3 delay 3
    end
  end

  assign a_r0_t0 = a_bus[31:24]; // row0 no delay
  assign a_r1_t0 = a1_d0;
  assign a_r2_t0 = a2_d2;
  assign a_r3_t0 = a3_d3;

  // ---- top boundary: B moves top->down (each col has 5 taps) ----
  wire [7:0] b_t0_c0, b_t1_c0, b_t2_c0, b_t3_c0, b_t4_c0;
  wire [7:0] b_t0_c1, b_t1_c1, b_t2_c1, b_t3_c1, b_t4_c1;
  wire [7:0] b_t0_c2, b_t1_c2, b_t2_c2, b_t3_c2, b_t4_c2;
  wire [7:0] b_t0_c3, b_t1_c3, b_t2_c3, b_t3_c3, b_t4_c3;

  // add boundary skew for B: col1:+1, col2:+2, col3:+3
  reg [7:0] b1_d0;
  reg [7:0] b2_d1, b2_d2;
  reg [7:0] b3_d1, b3_d2, b3_d3;

  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      b1_d0 <= 8'd0;
      b2_d1 <= 8'd0;
      b2_d2 <= 8'd0;
      b3_d1 <= 8'd0;
      b3_d2 <= 8'd0;
      b3_d3 <= 8'd0;
    end
    else if (start)
    begin
      b1_d0 <= 8'd0;
      b2_d1 <= 8'd0;
      b2_d2 <= 8'd0;
      b3_d1 <= 8'd0;
      b3_d2 <= 8'd0;
      b3_d3 <= 8'd0;
    end
    else if (vld)
    begin
      b1_d0 <= b_bus[23:16];                 // col1 delay 1
      b2_d1 <= b_bus[15:8 ];
      b2_d2 <= b2_d1; // col2 delay 2
      b3_d1 <= b_bus[7 :0 ];
      b3_d2 <= b3_d1;
      b3_d3 <= b3_d2; // col3 delay 3
    end
  end

  assign b_t0_c0 = b_bus[31:24]; // col0 no delay
  assign b_t0_c1 = b1_d0;
  assign b_t0_c2 = b2_d2;
  assign b_t0_c3 = b3_d3;

  // ---- 16 accumulators (results) ----
  wire [31:0] c00, c01, c02, c03;
  wire [31:0] c10, c11, c12, c13;
  wire [31:0] c20, c21, c22, c23;
  wire [31:0] c30, c31, c32, c33;

  // Row 0
  PE pe00 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r0_t0), .b_in(b_t0_c0), .a_out(a_r0_t1), .b_out(b_t1_c0), .acc(c00));
  PE pe01 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r0_t1), .b_in(b_t0_c1), .a_out(a_r0_t2), .b_out(b_t1_c1), .acc(c01));
  PE pe02 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r0_t2), .b_in(b_t0_c2), .a_out(a_r0_t3), .b_out(b_t1_c2), .acc(c02));
  PE pe03 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r0_t3), .b_in(b_t0_c3), .a_out(a_r0_t4), .b_out(b_t1_c3), .acc(c03));

  // Row 1
  PE pe10 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r1_t0), .b_in(b_t1_c0), .a_out(a_r1_t1), .b_out(b_t2_c0), .acc(c10));
  PE pe11 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r1_t1), .b_in(b_t1_c1), .a_out(a_r1_t2), .b_out(b_t2_c1), .acc(c11));
  PE pe12 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r1_t2), .b_in(b_t1_c2), .a_out(a_r1_t3), .b_out(b_t2_c2), .acc(c12));
  PE pe13 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r1_t3), .b_in(b_t1_c3), .a_out(a_r1_t4), .b_out(b_t2_c3), .acc(c13));

  // Row 2
  PE pe20 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r2_t0), .b_in(b_t2_c0), .a_out(a_r2_t1), .b_out(b_t3_c0), .acc(c20));
  PE pe21 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r2_t1), .b_in(b_t2_c1), .a_out(a_r2_t2), .b_out(b_t3_c1), .acc(c21));
  PE pe22 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r2_t2), .b_in(b_t2_c2), .a_out(a_r2_t3), .b_out(b_t3_c2), .acc(c22));
  PE pe23 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r2_t3), .b_in(b_t2_c3), .a_out(a_r2_t4), .b_out(b_t3_c3), .acc(c23));

  // Row 3
  PE pe30 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r3_t0), .b_in(b_t3_c0), .a_out(a_r3_t1), .b_out(b_t4_c0), .acc(c30));
  PE pe31 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r3_t1), .b_in(b_t3_c1), .a_out(a_r3_t2), .b_out(b_t4_c1), .acc(c31));
  PE pe32 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r3_t2), .b_in(b_t3_c2), .a_out(a_r3_t3), .b_out(b_t4_c2), .acc(c32));
  PE pe33 (.clk(clk), .rst_n(rst_n), .clr_acc(start), .vld(vld),
           .a_in(a_r3_t3), .b_in(b_t3_c3), .a_out(a_r3_t4), .b_out(b_t4_c3), .acc(c33));


  //   // systolic_array_4x4
  //   always @(posedge clk) if (vld)
  //     begin
  //       $display("SKW A_in={r0:%02x r1d1:%02x r2d2:%02x r3d3:%02x}  B_in={c0:%02x c1d1:%02x c2d2:%02x c3d3:%02x}",
  //                a_r0_t0, a_r1_t0, a_r2_t0, a_r3_t0,
  //                b_t0_c0, b_t0_c1, b_t0_c2, b_t0_c3);
  //     end


  // Pack row-major 512b: {row3,row2,row1,row0}, each row {c3,c2,c1,c0}
  assign c_bus = { c33, c32, c31, c30,
                   c23, c22, c21, c20,
                   c13, c12, c11, c10,
                   c03, c02, c01, c00 };
endmodule
