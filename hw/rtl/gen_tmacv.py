#!/usr/bin/env python3
"""axis_tmacv.v（三値 MAC アレイ・見出し付き・行長可変）を生成する。

1ビート64bit = 8バイト = 40個の三値重み。40項の加算木は手書きすると
必ず間違えるので、ここで機械的に書き出す。

符号化: バイト v は 5 個の三値を 3進数で持つ。
        v = Σ (w_d + 1) * 3^d   (d = 0..4, w ∈ {-1,0,+1})
        3^5 = 243 ≤ 256 なので 1 バイトに 5 個ちょうど入る。
        = 1.6 ビット/重み。BitNet b1.58 の log2(3)=1.585 に対応する。

段構成: 0 取り込み / 1 復号 / 2 項の生成 / 3-8 加算木6段 / 9 累算
        1段に1レベルずつしか置かない。段階3で「分散RAM読み＋演算」を
        1サイクルに詰めて 0.5ns 足りなくなった反省。面積は余っている。

【axis_tmach.v からの変更点（段階8）】
見出しに **行長（1行のビート数 BPR）** も入れて、**行の長さを組ごとに変えられる**ようにした。

  段階7: [R][x 640B][行 × R]            … 行長は 640 固定
  段階8: [R,BPR][x BPR*8B][行 × R]      … 組ごとに変わる

L=640 固定だと入力次元 1536 は 1920 に、4096 は 4480 に膨らむ（詰め物 20%）。
40 の倍数なら何でもよくすれば 1560 と 4120 で済む（詰め物 1.3%）。
bitnet_b1_58-large で **169.7 → 147.8 MB**。しかも 1 塊に収まるので
**組が 387 → 97 に減り、CPU 側での部分和の足し合わせも要らなくなる。**

x の置き場は BPRMAX まで持つ。**深くしても資源は増えない**——
`x0 <= xm[bc]` は「登録つき読み出し」なので Vivado は最初から BRAM に載せており
（深さ 16 でも RAMB36 を1個ずつ使っていた）、128 にしても同じタイルを
使い切るだけ。BRAM 27 / LUT +36 / WNS も +0.039 → +0.043 と変わらなかった。
"""
WPB = 40          # 1ビートあたりの重み数
L   = 640         # 1行の重み数。WPB の倍数であること

def rom_entry(v):
    """バイト値から 5 個の三値コード(2bit)へ。00=0, 01=+1, 10=-1"""
    bits = 0
    for d in range(5):
        w = (v // (3**d)) % 3 - 1        # -1, 0, +1
        code = {0: 0b00, 1: 0b01, -1: 0b10}[w]
        bits |= code << (2*d)
    return bits

out = []
A = out.append

A(f"""`timescale 1ns / 1ps
// =====================================================================
//  三値 MAC アレイ・見出し付き・行長可変  ※ rtl/gen_tmacv.py が生成。直接編集しない
//
//  段階3の axis_mac.v（INT8・8 MAC）の置き換え。
//  1バイトに三値を5個詰めるので、同じ 1 バイトから 5 回の演算が出る。
//  掛け算は「足す・引く・何もしない」の3択なので DSP を使わない。
//
//  ストリームの形式（見出し付き・行長可変。TLAST まで繰り返せる）:
//     [ 見出し 8B: [31:0]=R 行数, [47:32]=BPR 1行のビート数 ]
//     [ x: BPR*{WPB} バイト (INT8) ][ 重み: BPR*8 バイト/行 × R 行 ] × 何組でも
//  1ビート = 三値 {WPB} 個。1行 = BPR*{WPB} 重み。BPR は 1..{{BPRMAX}}
//  1行 = 三値 {L} 個 = {L//5} バイト。1ビート8バイトで {WPB} 個ぶん進む。
//
//  x の読み出しは 5 個のメモリに分けてある。1ビートで x を {WPB} 個
//  （= 64bit 語で5語ぶん）同時に読む必要があり、分散RAM は1語ずつしか
//  読めないため。語番号を 5 で割った余りで分ければ、5つとも同じアドレスで
//  引けるので、アドレス計算は1本で済む。
// =====================================================================
module axis_tmacv #(
  parameter integer BPRMAX = 128            // 1行の最大ビート数（= 重み {WPB*128} 個まで）
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

  localparam integer WPB = {WPB};                 // 1ビートあたりの重み数
  localparam integer AW  = $clog2(BPRMAX);        // 行内ビート位置の幅
  localparam integer NS  = 9;                     // 累算段に届くまでの段数

  assign s_axis_tready = 1'b1;                    // 上流は絶対に止めない
  wire fire = s_axis_tvalid & s_axis_tready;
""")

# ---- 復号 ROM ----
A("  // ---- 3進復号表。1バイト → 三値5個（2bitずつ、00=0 / 01=+1 / 10=-1） ----")
A("  function [9:0] dec3 (input [7:0] v);")
A("    case (v)")
for v in range(256):
    A(f"      8'd{v}: dec3 = 10'b{rom_entry(v):010b};")
A("      default: dec3 = 10'b0;")
A("    endcase")
A("  endfunction")
A("")

# ---- x メモリと制御 ----
A(f"""  // ---- x の格納。64bit 語を 5 本のメモリに撒く ----
  reg [63:0] xm0 [0:BPRMAX-1];
  reg [63:0] xm1 [0:BPRMAX-1];
  reg [63:0] xm2 [0:BPRMAX-1];
  reg [63:0] xm3 [0:BPRMAX-1];
  reg [63:0] xm4 [0:BPRMAX-1];

  reg [2:0]     hj;          // 語番号 mod 5
  reg [AW-1:0]  ha;          // 語番号 / 5
  reg [AW-1:0]  bc;          // 行内のビート位置
  reg [31:0]    rleft;       // この組で残っている行数
  reg [AW-1:0]  bpr_m1;      // この組の 1行のビート数 − 1（見出しで受ける）
  reg [1:0]     st;          // 0=見出し 1=x 取り込み 2=計算

  localparam [1:0] ST_R = 2'd0, ST_X = 2'd1, ST_W = 2'd2;

  // 見出し 1 ビートで行数を受け、x を {L//8} ビート取り込み、R 行流したらまた見出しへ。
  // TLAST が来たらどこにいても見出しへ戻る（転送の切れ目で必ず揃う）。
  always @(posedge aclk) begin
    if (!aresetn) begin
      st <= ST_R; hj <= 3'd0; ha <= {{AW{{1'b0}}}}; bc <= {{AW{{1'b0}}}};
      rleft <= 32'd0; bpr_m1 <= {{AW{{1'b0}}}};
    end else if (fire) begin
      if (s_axis_tlast) begin
        st <= ST_R; hj <= 3'd0; ha <= {{AW{{1'b0}}}}; bc <= {{AW{{1'b0}}}};
      end else begin
        case (st)
          ST_R: begin
                  rleft  <= s_axis_tdata[31:0];
                  bpr_m1 <= s_axis_tdata[47:32] - 16'd1;      // 1行のビート数 − 1（下位 AW bit を使う）
                  hj <= 3'd0; ha <= {{AW{{1'b0}}}}; st <= ST_X;
                end
          ST_X: begin
                  if (hj == 3'd4) begin
                    hj <= 3'd0;
                    if (ha == bpr_m1) begin st <= ST_W; ha <= {{AW{{1'b0}}}}; bc <= {{AW{{1'b0}}}}; end
                    else ha <= ha + 1'b1;
                  end else hj <= hj + 1'b1;
                end
          default: begin        // ST_W
                  if (bc == bpr_m1) begin
                    bc <= {{AW{{1'b0}}}};
                    if (rleft == 32'd1) st <= ST_R;
                    else rleft <= rleft - 1'b1;
                  end else bc <= bc + 1'b1;
                end
        endcase
      end
    end
  end

  wire hw = fire & (st == ST_X);
  always @(posedge aclk) begin
    if (hw && hj == 3'd0) xm0[ha] <= s_axis_tdata;
    if (hw && hj == 3'd1) xm1[ha] <= s_axis_tdata;
    if (hw && hj == 3'd2) xm2[ha] <= s_axis_tdata;
    if (hw && hj == 3'd3) xm3[ha] <= s_axis_tdata;
    if (hw && hj == 3'd4) xm4[ha] <= s_axis_tdata;
  end

  wire [319:0] xv = {{xm4[bc], xm3[bc], xm2[bc], xm1[bc], xm0[bc]}};

  wire compute = fire & (st == ST_W);
  wire row_end = compute & (bc == bpr_m1);

  // ---- 制御フラグを段数ぶん遅らせる ----
  reg [NS-1:0] v_sr, e_sr, l_sr;
  always @(posedge aclk) begin
    if (!aresetn) begin v_sr <= 0; e_sr <= 0; l_sr <= 0; end
    else begin
      v_sr <= {{v_sr[NS-2:0], compute}};
      e_sr <= {{e_sr[NS-2:0], row_end}};
      l_sr <= {{l_sr[NS-2:0], compute & s_axis_tlast}};
    end
  end

  // ---- 0段目: 重みと x を受ける ----
  reg  [63:0]  w0;
  reg  [319:0] x0;
  always @(posedge aclk) begin w0 <= s_axis_tdata; x0 <= xv; end

  // ---- 1段目: 復号 ----
  reg  [79:0]  t1;
  reg  [319:0] x1;
  always @(posedge aclk) begin
    x1 <= x0;
""")
for i in range(8):
    A(f"    t1[{10*i+9}:{10*i}] <= dec3(w0[{8*i+7}:{8*i}]);")
A("  end")
A("")

# ---- 項の生成 ----
A("""  // ---- 2段目: 項の生成。掛け算ではなく「そのまま／符号反転／ゼロ」の3択 ----
  // 9bit に広げてから反転する。x=-128 でも桁あふれしない。""")
for m in range(WPB):
    A(f"  reg signed [8:0] u{m};")
A("  always @(posedge aclk) begin")
for m in range(WPB):
    A(f"    case (t1[{2*m+1}:{2*m}])")
    A(f"      2'b01:   u{m} <=  $signed({{x1[{8*m+7}], x1[{8*m+7}:{8*m}]}});")
    A(f"      2'b10:   u{m} <= -$signed({{x1[{8*m+7}], x1[{8*m+7}:{8*m}]}});")
    A(f"      default: u{m} <= 9'sd0;")
    A("    endcase")
A("  end")
A("")

# ---- 加算木 ----
A("  // ---- 3〜8段目: 40項の加算木。1段に1レベルだけ置く ----")
cur = [f"u{m}" for m in range(WPB)]
width = 9
stage = 3
while len(cur) > 1:
    nxt, decls, asgn = [], [], []
    width += 1
    i = 0
    k = 0
    while i + 1 < len(cur):
        nm = f"s{stage}_{k}"
        decls.append(f"  reg signed [{width-1}:0] {nm};")
        asgn.append(f"    {nm} <= {cur[i]} + {cur[i+1]};")
        nxt.append(nm); i += 2; k += 1
    if i < len(cur):                       # 余りは幅を合わせて素通し
        nm = f"s{stage}_{k}"
        decls.append(f"  reg signed [{width-1}:0] {nm};")
        asgn.append(f"    {nm} <= {cur[i]};")
        nxt.append(nm)
    A(f"  // {stage}段目: {len(cur)} → {len(nxt)} 項 ({width}bit)")
    out.extend(decls)
    A("  always @(posedge aclk) begin")
    out.extend(asgn)
    A("  end")
    A("")
    cur = nxt
    stage += 1

TOTAL = cur[0]
# レジスタ段数 = 取り込み・復号・項の生成の3段 + 加算木の段数。
# ループを抜けた時点の stage が 3 + 加算木の段数 になっているのでそれが答え。
# （ここを stage-1 にすると行の区切りが1ビートずれて全行不一致になる）
NS_actual = stage
A(f"""  // ---- 累算。行末（または TLAST）で吐く ----
  // 三値 {L} 項 × |x|≤127 なら最大 81,280。32bit で十分。
  reg signed [31:0] acc, out_d;
  reg out_v, out_l;

  always @(posedge aclk) begin
    if (!aresetn) begin
      acc <= 32'sd0; out_v <= 1'b0; out_l <= 1'b0; out_d <= 32'sd0;
    end else begin
      out_v <= 1'b0;
      if (v_sr[NS-1]) begin
        if (e_sr[NS-1] | l_sr[NS-1]) begin
          out_d <= acc + {TOTAL};
          out_v <= 1'b1;
          out_l <= l_sr[NS-1];
          acc   <= 32'sd0;
        end else begin
          acc <= acc + {TOTAL};
        end
      end
    end
  end

  assign m_axis_tdata  = out_d;
  assign m_axis_tvalid = out_v;
  assign m_axis_tlast  = out_l;

endmodule""")

src = "\n".join(out) + "\n"
src = src.replace("localparam integer NS  = 9;", f"localparam integer NS  = {NS_actual};")
open("rtl/axis_tmacv.v","w").write(src)
print(f"axis_tmacv.v を生成: {len(src.splitlines())} 行 / 段数 NS={NS_actual} / 行長可変（最大128ビート＝{WPB*128}重み）/ 1ビート {WPB} 重み")
