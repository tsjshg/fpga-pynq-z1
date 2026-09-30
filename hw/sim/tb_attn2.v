`timescale 1ns / 1ps
`include "tb_cfg.vh"
// =====================================================================
//  axis_attn2 の機能検証。rtl/gen_attn2_tb.py が作った刺激と期待値を使う。
//
//  実機での照合と違い、ここでは意地悪をする:
//    ・上流の tvalid を 1割の確率で落とす（DMA のバーストの切れ目を模す）
//    ・下流の tready を 4回に1回落とす（5a で取りこぼしていた経路）
//  どちらも 5a で実際に踏んだ/踏みかけた形。
// =====================================================================
module tb_attn2;
  localparam NIN  = `NIN;
  localparam NEXP = `NEXP;

  reg clk = 0, rstn = 0;
  always #3.5 clk = ~clk;              // 142.86 MHz

  reg  [63:0] indat [0:NIN-1];
  reg  [31:0] expd  [0:NEXP-1];
  initial begin
    $readmemh("tb_in.hex",  indat);
    $readmemh("tb_exp.hex", expd);
  end

  // ---- 上流 ----
  reg  [63:0] td; reg tv = 0, tl = 0;
  integer ip = 0;
  wire s_tready;

  always @(posedge clk) begin
    if (!rstn) begin tv <= 1'b0; ip <= 0; tl <= 1'b0; end
    else if (!tv || s_tready) begin
      if (ip < NIN && ({$random} % 10) != 0) begin
        td <= indat[ip]; tl <= (ip == NIN-1); tv <= 1'b1; ip <= ip + 1;
      end else begin
        tv <= 1'b0; tl <= 1'b0;
      end
    end
  end

  // ---- 下流 ----
  reg m_ready = 0;
  always @(posedge clk) m_ready <= rstn && (({$random} % 4) != 0);

  wire [31:0] m_tdata;
  wire        m_tvalid, m_tlast;

  axis_attn2 dut (
    .aclk(clk), .aresetn(rstn),
    .s_axis_tdata(td), .s_axis_tkeep(8'hFF), .s_axis_tlast(tl),
    .s_axis_tvalid(tv), .s_axis_tready(s_tready),
    .m_axis_tdata(m_tdata), .m_axis_tlast(m_tlast),
    .m_axis_tvalid(m_tvalid), .m_axis_tready(m_ready)
  );

  integer op = 0, bad = 0, j;
  always @(posedge clk) begin
    if (rstn && m_tvalid && m_ready) begin
      if (op >= NEXP) begin
        $display("★ 余分な出力 %0d 語目 = %0d", op, $signed(m_tdata));
        bad = bad + 1;
      end else if (m_tdata !== expd[op]) begin
        if (bad < 20)
          $display("★ 不一致 %0d 語目 (グループ%0d 要素%0d): PL=%0d 期待=%0d",
                   op, op/195, op%195, $signed(m_tdata), $signed(expd[op]));
        bad = bad + 1;
      end
      // TLAST は最後の1語だけに立つはず
      if (m_tlast !== ((op == NEXP-1) ? 1'b1 : 1'b0)) begin
        $display("★ TLAST の位置が違う: %0d 語目で tlast=%b", op, m_tlast);
        bad = bad + 1;
      end
      op = op + 1;
    end
  end

  initial begin
    repeat (20) @(posedge clk);
    rstn = 1;
    // 十分待つ。入力 NIN ビート + 出力と排出のぶん。
    for (j = 0; j < NIN*8 + NEXP*8 + 5000; j = j + 1) begin
      @(posedge clk);
      if (op >= NEXP) j = NIN*8 + NEXP*8 + 5000;
    end
    repeat (50) @(posedge clk);

    $display("--------------------------------------------------");
    $display("入力 %0d / %0d ビート投入", ip, NIN);
    $display("出力 %0d / %0d 語受信", op, NEXP);
    if (op != NEXP) begin
      $display("★ 語数が合わない");
      bad = bad + 1;
    end
    if (bad == 0) $display("=== 一致。%0d 語すべて ===", NEXP);
    else          $display("=== ★不一致 %0d 件 ===", bad);
    $display("--------------------------------------------------");
    $finish;
  end
endmodule
