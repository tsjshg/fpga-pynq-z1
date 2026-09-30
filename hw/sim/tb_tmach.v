`timescale 1ns / 1ps
`include "tbh_cfg.vh"
// axis_tmach の機能検証。上流の tvalid を 1割の確率で落として意地悪する。
module tb_tmach;
  localparam NIN = `NIN, NEXP = `NEXP;
  reg clk = 0, rstn = 0;
  always #3.5 clk = ~clk;

  reg [63:0] indat [0:NIN-1];
  reg [31:0] expd  [0:NEXP-1];
  initial begin $readmemh("tbh_in.hex", indat); $readmemh("tbh_exp.hex", expd); end

  reg [63:0] td; reg tv = 0, tl = 0;
  integer ip = 0;
  wire s_ready;
  always @(posedge clk) begin
    if (!rstn) begin tv <= 1'b0; ip <= 0; tl <= 1'b0; end
    else if (!tv || s_ready) begin
      if (ip < NIN && ({$random} % 10) != 0) begin
        td <= indat[ip]; tl <= (ip == NIN-1); tv <= 1'b1; ip <= ip + 1;
      end else begin tv <= 1'b0; tl <= 1'b0; end
    end
  end

  wire [31:0] m_tdata; wire m_tvalid, m_tlast;
  axis_tmach dut (.aclk(clk), .aresetn(rstn),
    .s_axis_tdata(td), .s_axis_tkeep(8'hFF), .s_axis_tlast(tl),
    .s_axis_tvalid(tv), .s_axis_tready(s_ready),
    .m_axis_tdata(m_tdata), .m_axis_tlast(m_tlast),
    .m_axis_tvalid(m_tvalid), .m_axis_tready(1'b1));

  integer op = 0, bad = 0, j;
  always @(posedge clk) if (rstn && m_tvalid) begin
    if (op >= NEXP) begin $display("★ 余分な出力 %0d 語目", op); bad = bad + 1; end
    else if (m_tdata !== expd[op]) begin
      if (bad < 12) $display("★ 不一致 %0d 語目: PL=%0d 期待=%0d", op, $signed(m_tdata), $signed(expd[op]));
      bad = bad + 1;
    end
    if (m_tlast !== ((op == NEXP-1) ? 1'b1 : 1'b0)) begin
      $display("★ TLAST の位置が違う: %0d 語目で tlast=%b", op, m_tlast); bad = bad + 1;
    end
    op = op + 1;
  end

  initial begin
    repeat (20) @(posedge clk); rstn = 1;
    for (j = 0; j < NIN*12 + 4000; j = j + 1) begin
      @(posedge clk);
      if (op >= NEXP) j = NIN*12 + 4000;
    end
    repeat (60) @(posedge clk);
    $display("--------------------------------------------------");
    $display("入力 %0d/%0d ビート  出力 %0d/%0d 語", ip, NIN, op, NEXP);
    if (op != NEXP) begin $display("★ 語数が合わない"); bad = bad + 1; end
    if (bad == 0) $display("=== 一致。%0d 語すべて ===", NEXP);
    else          $display("=== ★不一致 %0d 件 ===", bad);
    $display("--------------------------------------------------");
    $finish;
  end
endmodule
