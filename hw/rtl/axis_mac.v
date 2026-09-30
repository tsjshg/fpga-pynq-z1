`timescale 1ns / 1ps
// =====================================================================
//  AXI4-Stream INT8 MAC アレイ
//
//  ヌルシンク (axis_sink.v) の置き換え。捨てる代わりに内積を計算する。
//  段階2で測った 1.13 GB/s（HP 1本）を、計算させても保てるかを試す。
//
//  ストリームの形式（1回の転送）:
//     [ x: L バイト ][ 重み: L バイトの行 × R 行 ]
//  先頭の L バイトを入力ベクトルとして内部に取り込み、
//  以降は L バイトごとに x との内積を取って 32bit で吐く。
//  x を別の経路で配らなくて済むので、AXI スレーブも BRAM 制御器も要らない。
//  x の分は 1/(R+1) の帯域しか食わない（R=65535 なら 0.002%）。
//
//  tready は常に 1。上流を絶対に止めない。出力は入力の 1/(L/4) の量しか
//  出ないので、下流の FIFO が詰まることはない（詰まれば結果の数が合わず、
//  Python 側の照合で必ず露見する）。
//
//  演算は5段パイプライン:
//     0: 重みと x の取り込み   1: 8個の積   2: 4個の和   3: 2個の和   4: 累算
//
//  【0段目を後から足した理由】
//  最初は4段で作ったが 142.86 MHz でタイミングが 0.5ns 足りなかった。
//  臨界パスは「bc カウンタ → 分散RAM の非同期読み出し → LUT の掛け算 → レジスタ」で、
//  RAM 読みと掛け算を同じサイクルに詰め込んでいたのが原因。
//  読んだ値を一度レジスタで受けて、重みのほうも同じだけ遅らせて揃える。
//  1クロックあたり 8 MAC。142.86 MHz なら 1.14 G MAC/s で、
//  ちょうど HP 1本の道幅 (8 B/クロック) と釣り合う。
// =====================================================================
module axis_mac #(
  parameter integer L = 512               // 1行の長さ（バイト）。8の倍数であること
)(
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET aresetn" *)
  input  wire         aclk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
  (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input  wire         aresetn,

  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA"  *)
  input  wire [63:0]  s_axis_tdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TKEEP"  *)
  input  wire [7:0]   s_axis_tkeep,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TLAST"  *)
  input  wire         s_axis_tlast,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *)
  input  wire         s_axis_tvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *)
  output wire         s_axis_tready,

  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TDATA"  *)
  output wire [31:0]  m_axis_tdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TLAST"  *)
  output wire         m_axis_tlast,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TVALID" *)
  output wire         m_axis_tvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TREADY" *)
  input  wire         m_axis_tready
);

  localparam integer BPR = L/8;             // 1行あたりのビート数
  localparam integer AW  = (BPR <= 1) ? 1 : $clog2(BPR);

  // x は 64bit 幅で持つ。1ビートで 8 個ぶん同時に読む必要があるため、
  // 8個ずつまとめた形で置く。L=512 なら 64 語 = 512 バイト。
  // 分散RAM（LUTRAM）に載る大きさなので BRAM を使わない。
  reg [63:0] xw [0:BPR-1];

  reg [AW-1:0] bc;                          // 行内のビート位置
  reg          in_header;                   // 先頭 L バイト（x）を取り込み中か

  // 上流は絶対に止めない。これが帯域を保つための唯一の条件。
  assign s_axis_tready = 1'b1;
  wire fire = s_axis_tvalid & s_axis_tready;

  always @(posedge aclk) begin
    if (!aresetn) begin
      in_header <= 1'b1;
      bc        <= {AW{1'b0}};
    end else if (fire) begin
      if (s_axis_tlast) begin
        // 転送の終わり。次の転送はまた x の取り込みから始める。
        in_header <= 1'b1;
        bc        <= {AW{1'b0}};
      end else if (bc == BPR-1) begin
        bc        <= {AW{1'b0}};
        in_header <= 1'b0;
      end else begin
        bc <= bc + 1'b1;
      end
    end
  end

  always @(posedge aclk)
    if (fire && in_header) xw[bc] <= s_axis_tdata;

  wire [63:0] xv      = xw[bc];             // 分散RAM の非同期読み出し
  wire        compute = fire & ~in_header;
  wire        row_end = compute & (bc == BPR-1);

  // ---- 0段目: 重みと x を受けるだけ ----
  // 分散RAM の読み出しをここで切る。ここを削ると掛け算と同じサイクルになって
  // タイミングが 0.5ns 足りなくなる（実測済み）。
  reg [63:0] w0, x0;
  reg v0, e0, l0;
  always @(posedge aclk) begin
    if (!aresetn) begin v0<=1'b0; e0<=1'b0; l0<=1'b0; end
    else begin
      v0 <= compute;
      e0 <= row_end;
      l0 <= compute & s_axis_tlast;
    end
    w0 <= s_axis_tdata;
    x0 <= xv;
  end

  // ---- 1段目: 8個の積 ----
  // use_dsp を付けないと 8x8 は小さすぎると判断されて LUT で組まれ、
  // 9段の論理段数になって間に合わない。DSP48E1 は 25x18 なので過剰だが、
  // 8個しか使わないのに 220 個あるので気にしない。
  (* use_dsp = "yes" *) reg signed [15:0] p0,p1,p2,p3,p4,p5,p6,p7;
  reg v1, e1, l1;
  always @(posedge aclk) begin
    if (!aresetn) begin v1<=1'b0; e1<=1'b0; l1<=1'b0; end
    else begin v1 <= v0; e1 <= e0; l1 <= l0; end
    p0 <= $signed(w0[ 7: 0]) * $signed(x0[ 7: 0]);
    p1 <= $signed(w0[15: 8]) * $signed(x0[15: 8]);
    p2 <= $signed(w0[23:16]) * $signed(x0[23:16]);
    p3 <= $signed(w0[31:24]) * $signed(x0[31:24]);
    p4 <= $signed(w0[39:32]) * $signed(x0[39:32]);
    p5 <= $signed(w0[47:40]) * $signed(x0[47:40]);
    p6 <= $signed(w0[55:48]) * $signed(x0[55:48]);
    p7 <= $signed(w0[63:56]) * $signed(x0[63:56]);
  end

  // ---- 2段目: 4個の和 ----
  reg signed [16:0] q0,q1,q2,q3;
  reg v2, e2, l2;
  always @(posedge aclk) begin
    if (!aresetn) begin v2<=1'b0; e2<=1'b0; l2<=1'b0; end
    else begin v2<=v1; e2<=e1; l2<=l1; end
    q0 <= p0+p1; q1 <= p2+p3; q2 <= p4+p5; q3 <= p6+p7;
  end

  // ---- 3段目: 2個の和 ----
  reg signed [17:0] r0,r1;
  reg v3, e3, l3;
  always @(posedge aclk) begin
    if (!aresetn) begin v3<=1'b0; e3<=1'b0; l3<=1'b0; end
    else begin v3<=v2; e3<=e2; l3<=l2; end
    r0 <= q0+q1; r1 <= q2+q3;
  end

  // ---- 4段目: 累算して行の終わりで吐く ----
  // L=512, 重みも x も INT8 なので最大でも 512*127*128 = 8.3M。32bit で足りる。
  wire signed [18:0] tot = r0 + r1;
  reg signed [31:0] acc;
  reg signed [31:0] out_d;
  reg out_v, out_l;

  always @(posedge aclk) begin
    if (!aresetn) begin
      acc <= 32'sd0; out_v <= 1'b0; out_l <= 1'b0; out_d <= 32'sd0;
    end else begin
      out_v <= 1'b0;
      if (v3) begin
        // 最終行が L に満たなくても TLAST で必ず吐く。
        // そうしないと DMA に TLAST が届かず転送が終わらない。
        if (e3 | l3) begin
          out_d <= acc + tot;
          out_v <= 1'b1;
          out_l <= l3;
          acc   <= 32'sd0;
        end else begin
          acc <= acc + tot;
        end
      end
    end
  end

  assign m_axis_tdata  = out_d;
  assign m_axis_tvalid = out_v;
  assign m_axis_tlast  = out_l;

endmodule
