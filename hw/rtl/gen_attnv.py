#!/usr/bin/env python3
"""axis_attnv.v（attention コア・bitnet_b1_58-large 用）を生成する。段階9。

段階5b の axis_attn2（SmolLM2 用・head_dim 64・GQA 3:1）の作り直し。
bitnet_b1_58-large は **head_dim 96 で GQA が無い**（q 16 : kv 16）。

  ストリーム（見出し付き。TLAST まで何組でも詰められる）:
     [ 見出し 8B: [31:0]=T, [49:32]=m ][ q: 96 B ][ K: T×96 B ][ V: T×96 B ] × 組
  出力: out[d] を int32 で 96 個 + 分母 Σe を 1 個 = 97 語

  第1相 (K): s = Σ_d q[d]·K[t][d]                    … 96 項 = 12 ビート
             s8 = clamp((s·m) >>> SHM)                … ★ m は見出しで受ける
             mx = max_t s8                             … 同時に最大値を追う
  第2相 (V): e = EROM[mx - s8]、out[d] += e·V[t][d]、den += e

**m を見出しで受けるのが段階5b からの一番の変更。**
段階5b は s8 = clamp(s >> 8) の固定で、q と K の量子化尺度を無視していた。
そのため指数の温度が層・ヘッドごとにずれ、実モデルで perplexity が
35.56 → 58.25 に悪化する（numpy で実測）。m = TAU·sq·sk/√HD を見出しに
載せると **35.90** まで戻る。掛け算器 1 個で品質がほぼ元通りになる。

m は 18bit 符号なし、固定シフト SHM=24（m_float = m/2^24）。
実測の値域は 2003〜52633 で、18bit（最大 262143）に 5 倍の余裕がある。
s は |s| ≤ 96·127·127 = 1,548,384 < 2^21 なので 22bit 符号付きで足りる。

**GQA が無いので 1 バイトあたり 1 MAC**（段階5b は 2.99）。
つまり完全に帯域律速で、掛け算器は 1 ビート 8 バイトぶんの 8 個で足りる。
"""
import math

HD    = 96           # head_dim
BPB   = 8            # 1ビートのバイト数
BPR   = HD//BPB      # 1つの t あたりのビート数 = 12
TMAX  = 2048
SHM   = 24           # スコア倍率の固定シフト
TAU   = 8.0
NOUT  = HD           # 出力の本体語数 = 96
STRIDE= 16           # 累算器の i ごとの間隔（BPR=12 を 16 に切り上げ）
# 出力アドレス = {ocnt[2:0], ocnt[6:3]}。d = 8b+i なので i*STRIDE + b になる。
DEN   = ((96 & 7) << 4) | (96 >> 3)      # ocnt=96 を同じ式に通した番地 = 12
OMAX  = 128
NOTOT = NOUT + 1

EROM = [max(0, min(255, int(round(255.0*math.exp(-d/TAU))))) for d in range(256)]
NZ   = sum(1 for v in EROM if v > 0)

out=[]; A=out.append

A(f'''`timescale 1ns / 1ps
// =====================================================================
//  attention コア（head_dim {HD} / GQA なし / softmax 入り）
//  ※ rtl/gen_attnv.py が生成。直接編集しない
//
//  ストリーム（見出し付き。TLAST まで何組でも）:
//     [ 見出し 8B: [31:0]=T, [{32+18-1}:32]=m ][ q: {HD} B ][ K: T×{HD} B ][ V: T×{HD} B ]
//  出力: out[d] を int32 で {NOUT} 個 + 分母 den を 1 個 = {NOTOT} 語
//
//  第1相 (K): s = Σ q·K（{HD} 項 = {BPR} ビート）→ s8 = clamp((s·m) >>> {SHM}) を BRAM へ
//             同時に mx = max s8 を追う
//  第2相 (V): e = EROM[mx - s8]、out += e·V、den += e
//
//  ★ m を見出しで受ける。段階5b は clamp(s >> 8) の固定で、量子化尺度を
//    無視していたため実モデルで perplexity 35.56 → 58.25 に悪化した。
//    m = TAU·sq·sk/√HD を渡すと 35.90 に戻る（掛け算器 1 個ぶんの改修）。
//
//  EROM[d] = round(255·exp(-d/{TAU}))、{NZ} 番地までが非零。
//  割り算はしない。分子 {NOUT} 語と分母 1 語を出して下流に任せる。
//
//  GQA が無いので 1 バイトあたり 1 MAC。掛け算器は
//  第1相 {BPB} + スコア倍率 1 + 第2相 {BPB} = {2*BPB+1} 個。
// =====================================================================
module axis_attnv #(
  parameter integer HD   = {HD},
  parameter integer TMAX = {TMAX},
  parameter integer SHM  = {SHM}
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

  localparam integer BPR   = HD/8;           // 1つの t あたりのビート数 = {BPR}
  localparam integer NOUT  = HD;             // {NOUT}
  localparam integer NOTOT = NOUT + 1;       // {NOTOT}
  localparam integer TW    = $clog2(TMAX+1);

  localparam [2:0] ST_HDR=3'd0, ST_Q=3'd1, ST_K=3'd2, ST_KDR=3'd3,
                   ST_V=3'd4, ST_DRN=3'd5, ST_OUT=3'd6;

  reg [2:0]     st;
  reg [TW-1:0]  Tn;                          // 見出しの文脈長
  reg [17:0]    mscale;                      // 見出しのスコア倍率
  reg [TW-1:0]  tcnt;
  reg [3:0]     bcnt;                        // 行内のビート位置 0..{BPR-1}
  reg [3:0]     qcnt;                        // q 取り込みの進捗 0..{BPR-1}
  reg [7:0]     ocnt;                        // 出力の進捗 0..{NOTOT-1}
  reg [3:0]     dcnt, kdcnt;
  reg           last_seen;

  assign s_axis_tready = (st != ST_OUT) && (st != ST_DRN) && (st != ST_KDR);
  wire fire = s_axis_tvalid & s_axis_tready;

  reg  ovld;
  wire oload = (!ovld) | m_axis_tready;

  // ---- q の格納。ビート b は次元 8b..8b+7 ----
  reg [63:0] qm [0:BPR-1];
  always @(posedge aclk) if (fire && st==ST_Q) qm[qcnt] <= s_axis_tdata;
  wire [63:0] qw = qm[bcnt];

  // ---- スコアの置き場 ----
  reg signed [7:0] sm [0:TMAX-1];
  reg signed [7:0] mx;

  always @(posedge aclk) begin
    if (!aresetn) begin
      st<=ST_HDR; Tn<=0; mscale<=0; tcnt<=0; bcnt<=0; qcnt<=0; ocnt<=0;
      dcnt<=0; kdcnt<=0; last_seen<=1'b0;
    end else begin
      case (st)
        ST_HDR: if (fire) begin
                  Tn     <= s_axis_tdata[TW-1:0];
                  mscale <= s_axis_tdata[49:32];
                  qcnt <= 0; st <= ST_Q;
                  last_seen <= s_axis_tlast;
                end
        ST_Q:   if (fire) begin
                  if (qcnt == BPR-1) begin qcnt<=0; tcnt<=0; bcnt<=0; st<=ST_K; end
                  else qcnt <= qcnt + 1'b1;
                  if (s_axis_tlast) last_seen <= 1'b1;
                end
        ST_K:   if (fire) begin
                  if (bcnt == BPR-1) begin
                    bcnt <= 0;
                    if (tcnt == Tn-1) begin tcnt<=0; kdcnt<=0; st<=ST_KDR; end
                    else tcnt <= tcnt + 1'b1;
                  end else bcnt <= bcnt + 1'b1;
                  if (s_axis_tlast) last_seen <= 1'b1;
                end
        // 最後の K 行のスコアが BRAM に入り mx が確定するまで待つ。
        // 第1相は 取込1+積1+加算木3+累算1+倍率1+clamp1 = 8 段なので、余裕を見て 14。
        ST_KDR: if (kdcnt == 4'd13) begin tcnt<=0; bcnt<=0; st<=ST_V; end
                else kdcnt <= kdcnt + 1'b1;
        ST_V:   if (fire) begin
                  if (bcnt == BPR-1) begin
                    bcnt <= 0;
                    if (tcnt == Tn-1) begin tcnt<=0; dcnt<=0; st<=ST_DRN; end
                    else tcnt <= tcnt + 1'b1;
                  end else bcnt <= bcnt + 1'b1;
                  if (s_axis_tlast) last_seen <= 1'b1;
                end
        ST_DRN: if (dcnt == 4'd8) begin ocnt<=0; st<=ST_OUT; end
                else dcnt <= dcnt + 1'b1;
        ST_OUT: if (oload) begin
                  if (ocnt == NOTOT-1) begin ocnt<=0; st<=ST_HDR; last_seen<=1'b0; end
                  else ocnt <= ocnt + 1'b1;
                end
        default: st <= ST_HDR;
      endcase
    end
  end

  wire in_k = fire & (st==ST_K);
  wire in_v = fire & (st==ST_V);
  wire k_rowend = in_k & (bcnt == BPR-1);
  wire v_row0   = in_v & (bcnt == 4'd0);

  // ================= 第1相: QK^T =================
  // 0段: 取り込み
  reg [63:0] kw0, qr0;
  reg kv0, ke0;
  always @(posedge aclk) begin
    if (!aresetn) begin kv0<=1'b0; ke0<=1'b0; end
    else begin kv0 <= in_k; ke0 <= k_rowend; end
    kw0 <= s_axis_tdata; qr0 <= qw;
  end
''')

# 1段: 積
A("  // 1段: 積（8個）。use_dsp を付けないと LUT で組まれて論理段数が足りなくなる")
A("  (* use_dsp = \"yes\" *) reg signed [15:0] " + ",".join(f"kp{i}" for i in range(BPB)) + ";")
A("  reg kv1, ke1;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin kv1<=1'b0; ke1<=1'b0; end")
A("    else begin kv1<=kv0; ke1<=ke0; end")
for i in range(BPB):
    A(f"    kp{i} <= $signed(kw0[{8*i+7}:{8*i}]) * $signed(qr0[{8*i+7}:{8*i}]);")
A("  end")
A("")

# 加算木 8→4→2→1
stage=2; cur=[f"kp{i}" for i in range(BPB)]; width=16
while len(cur) > 1:
    width += 1
    A(f"  // {stage}段: {len(cur)} → {len(cur)//2} 項 ({width}bit)")
    nxt=[]
    for k in range(len(cur)//2):
        nm=f"ks{stage}_{k}"; A(f"  reg signed [{width-1}:0] {nm};"); nxt.append(nm)
    A(f"  reg kv{stage}, ke{stage};")
    A("  always @(posedge aclk) begin")
    A(f"    if (!aresetn) begin kv{stage}<=1'b0; ke{stage}<=1'b0; end")
    A(f"    else begin kv{stage}<=kv{stage-1}; ke{stage}<=ke{stage-1}; end")
    for k,nm in enumerate(nxt):
        A(f"    {nm} <= {cur[2*k]} + {cur[2*k+1]};")
    A("  end")
    A("")
    cur=nxt; stage+=1
KL = stage-1

A(f"  // {stage}段: 累算。行末で 1 行ぶんの s が揃う")
A("  // 【重要】行番号の遅延は演算パイプラインと *同じ有効ビット* で進める")
for i in range(KL+1):
    A(f"  reg [TW-1:0] kt{i};")
A("  always @(posedge aclk) begin")
A("    if (in_k) kt0 <= tcnt;")
for i in range(1, KL+1):
    A(f"    if (kv{i-1}) kt{i} <= kt{i-1};")
A("  end")
A("  reg signed [31:0] kacc;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) kacc <= 32'sd0;")
A(f"    else if (kv{KL}) begin")
A(f"      if (ke{KL}) kacc <= 32'sd0; else kacc <= kacc + {cur[0]};")
A("    end")
A("  end")
A("")
A(f"  // {stage+1}段: 行の合計。|s| ≤ {HD}·127·127 = {HD*127*127} < 2^21 なので 22bit で足りる")
A(f"  wire signed [31:0] ktot = kacc + {cur[0]};   // 式に添字は付けられないので一度 wire に")
A("  reg signed [21:0] ksum;")
A("  reg [TW-1:0] kw_addr;  reg kw_en;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) kw_en <= 1'b0;")
A(f"    else kw_en <= kv{KL} & ke{KL};")
A("    ksum    <= ktot[21:0];")
A(f"    kw_addr <= kt{KL};")
A("  end")
A("")
A(f"  // {stage+2}段: ★ スコア倍率。見出しの m（18bit）を掛けて >>> {SHM}")
A("  // これが段階5b からの本質的な変更。22bit × 18bit なので DSP 1 個に収まる。")
A("  (* use_dsp = \"yes\" *) reg signed [40:0] kmul;")
A("  reg [TW-1:0] km_addr;  reg km_en;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) km_en <= 1'b0; else km_en <= kw_en;")
A("    kmul    <= ksum * $signed({1'b0, mscale});")
A("    km_addr <= kw_addr;")
A("  end")
A("")
A(f"  // {stage+3}段: clamp して 8bit に落とす（掛け算と同じサイクルには置かない）")
A("  function signed [7:0] clamp8(input signed [40:0] v);")
A("    reg signed [40:0] t;")
A("    begin")
A(f"      t = v >>> {SHM};")
A("      if (t >  127) clamp8 =  8'sd127;")
A("      else if (t < -128) clamp8 = 8'sh80;")
A("      else clamp8 = t[7:0];")
A("    end")
A("  endfunction")
A("  reg signed [7:0] kc;")
A("  reg [TW-1:0] kc_addr;  reg kc_en;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) kc_en <= 1'b0; else kc_en <= km_en;")
A("    kc      <= clamp8(kmul);")
A("    kc_addr <= km_addr;")
A("  end")
A("")
A("  // 置き場への書き込みと最大値の更新。どちらも浅い。")
A("  always @(posedge aclk) if (kc_en) sm[kc_addr] <= kc;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn)                mx <= 8'sh80;")
A("    else if (st==ST_HDR)         mx <= 8'sh80;")
A("    else if (kc_en && kc > mx)   mx <= kc;")
A("  end")
A("")

# exp の表
A("  // ================= exp の表（256語） =================")
A(f"  // EROM[d] = round(255·exp(-d/{TAU}))。d = mx - s8 なので必ず 0..255。")
A(f"  // {NZ} 番地から先は 0 なので、合成すると LUT の論理に畳まれる。")
A("  reg [7:0] erom [0:255];")
A("  initial begin")
for d in range(0, 256, 4):
    A("    " + " ".join(f"erom[{d+k:3d}]=8'd{EROM[d+k]};" for k in range(4)))
A("  end")
A("")

# 第2相
A("  // ================= 第2相: softmax + AV =================")
A("  // 【重要】データと有効ビットの段数を数え合わせる（段階5a で1つずれて全滅した）")
A("  //   データ: s_axis_tdata → vw0(1) → vw1(2) → vw2(3) → vp(4)")
A("  //   指数  : sm[tcnt] → sr(1) → vd(2) → ve(3)")
A("  //   有効  : in_v → vv0(1) → vv1(2) → vv2(3) → vv3(4)")
A("  //   行の頭: v_row0 → vr0(1) → vr1(2) → vr2(3)   ← ve と同じ3段")
A("  reg [63:0] vw0, vw1, vw2;")
A("  reg vv0, vv1, vv2;")
A("  // vv3 は 96×32 個の累算器の CE を駆動する。1本だと配線が遅延の 93% を占め")
A("  // （実測 6.3ns）、混雑で AXI DMA 側まで巻き添えにした。複製させる。")
A("  (* max_fanout = 64 *) reg vv3;")
A("  reg vr0, vr1, vr2;")
A("  reg signed [7:0] sr;")
A("  reg [7:0] vd, ve;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin vv0<=1'b0; vv1<=1'b0; vv2<=1'b0; vv3<=1'b0;")
A("                        vr0<=1'b0; vr1<=1'b0; vr2<=1'b0; end")
A("    else begin vv0<=in_v; vv1<=vv0; vv2<=vv1; vv3<=vv2;")
A("               vr0<=v_row0; vr1<=vr0; vr2<=vr1; end")
A("    vw0 <= s_axis_tdata; vw1 <= vw0; vw2 <= vw1;")
A("    sr  <= sm[tcnt];")
A("    vd  <= mx - sr;            // mx は必ず s8 以上なので 9bit 引き算の下位 8bit でよい")
A("    ve  <= erom[vd];           // ★ ここが softmax。DDR には触らない")
A("  end")
A("")
A("  // 4段: 積（8個）。V は符号付き8bit、指数は符号なし8bit")
A("  (* use_dsp = \"yes\" *) reg signed [17:0] " + ",".join(f"vp{i}" for i in range(BPB)) + ";")
A("  always @(posedge aclk) begin")
for i in range(BPB):
    A(f"    vp{i} <= $signed(vw2[{8*i+7}:{8*i}]) * $signed({{1'b0, ve}});")
A("  end")
A("")
A("  // 5段: 累算器へ。")
A("  // 【重要】ビート位置で添字を引くとアドレス解読に扇形に広がって落ちる")
A("  // （段階5a の実測 WNS -0.480ns）。b は 0→11 を順に回るだけなので累算器を回す。")
A(f"  // 配置は oacc[i*{STRIDE} + b]。b が回転位置。回転長 = 1行のビート数 {BPR}。")
A(f"  // 分母は {DEN} 番地。出力アドレスの入れ替え式に ocnt={NOUT} を通すとそこに当たる。")
A(f"  reg signed [31:0] oacc [0:{OMAX-1}];")
A("  integer oi;")
A("  // クリアも 128×32 個を駆動するので、1段遅らせたうえで複製させる。")
A("  // ST_HDR の次の組の最初の V ビートまでには 1+12+12T サイクルあるので遅らせて安全。")
A("  (* max_fanout = 64 *) reg oclr;")
A("  always @(posedge aclk) oclr <= (st==ST_HDR);")
A("  always @(posedge aclk) begin")
A("    if (oclr) begin")
A(f"      for (oi=0; oi<{OMAX}; oi=oi+1) oacc[oi] <= 32'sd0;")
A("    end else begin")
A("      if (vv3) begin")
for i in range(BPB):
    base = i*STRIDE
    for j in range(BPR-1):
        A(f"        oacc[{base+j}] <= oacc[{base+j+1}];")
    A(f"        oacc[{base+BPR-1}] <= oacc[{base}] + $signed(vp{i});")
A("      end")
A("      if (vr2) oacc[%d] <= oacc[%d] + $signed({24'd0, ve});" % (DEN, DEN))
A("    end")
A("  end")
A("")
A(f'''  // ---- 出力 ----
  // 出す順は out[d]（d = 8b+i）、置き場は i*{STRIDE} + b。ビットを入れ替えるだけ。
  // ocnt={NOUT}（分母）もこの式で {DEN} 番地に当たる。
  wire [6:0] oaddr = {{ocnt[2:0], ocnt[6:3]}};

  reg signed [31:0] odat;
  reg olast;
  always @(posedge aclk) begin
    if (!aresetn) begin ovld<=1'b0; olast<=1'b0; end
    else if (oload) begin
      ovld  <= (st==ST_OUT);
      olast <= (st==ST_OUT) && (ocnt==NOTOT-1) && last_seen;
      odat  <= oacc[oaddr];
    end
  end

  assign m_axis_tdata  = odat;
  assign m_axis_tvalid = ovld;
  assign m_axis_tlast  = olast;

endmodule''')

src="\n".join(out)+"\n"
open("rtl/axis_attnv.v","w").write(src)
print(f"axis_attnv.v を生成: {len(src.splitlines())} 行")
print(f"  HD={HD} BPR={BPR} / 掛け算器 {2*BPB+1} 個 / 第1相 {KL+1}段 + 倍率1 + clamp1")
print(f"  出力 {NOTOT} 語（本体 {NOUT} + 分母 1）/ 分母の番地 {DEN} / 配列 {OMAX}")
print(f"  m は 18bit・固定シフト {SHM} / EROM: TAU={TAU} 非零 {NZ} 語")
