#!/usr/bin/env python3
"""axis_attn2.v（softmax 入り attention コア）を生成する。段階5b。

段階5a（rtl/gen_attn.py / axis_attn.v）との違いは softmax だけ。
帯域・MAC 数は変わらない。**softmax は DDR を一切余分に読まない**、
つまり「ただで付く」ことを実機で示すのがこの段階の目的。

  ストリーム（1グループ = 1層の1 KV ヘッドぶん）:
     [ヘッダ 8B: T][q: NQ*HD B][K: T*HD B][V: T*HD B]
  出力: out[h][d] を int32 で NQ*HD 個 + 分母 Σe を NQ 個 = 195 語

  第1相 (K): s[h][t]  = Σ_d q[h][d]*K[t][d]
             s8[h][t] = clamp(s >> SHIFT)         … BRAM へ
             mx[h]    = max_t s8[h][t]            … 同時に最大値を追う
  第2相 (V): e[h][t]  = EROM[mx[h] - s8[h][t]]    … ★ここが exp の表引き
             out[h][d] += e[h][t] * V[t][d]
             den[h]   += e[h][t]

割り算は回路でやらない。1グループにつき NQ 回しかないので、
下流（出力射影の行列積）のスケールに畳むか CPU で割ればよい。
分子と分母を両方出すのはそのため。

EROM[d] = round(255 * exp(-d/TAU))、d = 0..255 の 256 語。
TAU は「s8 の1目盛りが softmax の指数で何に当たるか」で、実重みの
量子化スケールが決まるまでは暫定値。**表を焼き直すだけで変えられる**
のがこの作りの肝で、回路の形は TAU に依存しない。
"""
import math

HD    = 64      # head_dim
NQ    = 3       # 1つの KV ヘッドを共有する q ヘッド数（SmolLM2-135M は 9/3）
BPB   = 8       # 1ビートのバイト数
BPR   = HD//BPB # 1行（1つの t）あたりのビート数 = 8
TMAX  = 2048
SHIFT = 8
TAU   = 8.0
NOUT  = NQ*HD   # 192

# 分母の置き場。出力アドレスは oaddr = {ocnt[7:6], ocnt[2:0], ocnt[5:3]} で引く。
# ocnt = 192,193,194 をこの式に通すと 192,200,208 になるので、そこに置けば
# **出力マルチプレクサに 2:1 を足さずに済む**（5a の出力経路は余裕 0.436ns しかない）。
ODEN  = [((192+k) >> 6 << 6) | (((192+k) & 7) << 3) | (((192+k) >> 3) & 7) for k in range(NQ)]
OMAX  = max(ODEN) + 1          # 209
NOTOT = NOUT + NQ              # 195

EROM = [max(0, min(255, int(round(255.0 * math.exp(-d/TAU))))) for d in range(256)]
NZ   = sum(1 for v in EROM if v > 0)

out=[]; A=out.append

A(f'''`timescale 1ns / 1ps
// =====================================================================
//  attention コア（softmax 入り）  ※ rtl/gen_attn2.py が生成。直接編集しない
//
//  ストリーム（1グループ = 1層の1 KV ヘッドぶん）:
//     [ヘッダ 8B: T][q: {NQ}×{HD} B][K: T×{HD} B][V: T×{HD} B]
//  出力: out[h][d] を int32 で {NOUT} 個 + 分母 den[h] を {NQ} 個 = {NOTOT} 語
//
//  第1相 (K): s = Σ q·K、s8 = clamp(s >> {SHIFT}) を BRAM へ。同時に mx[h] を追う
//  第2相 (V): e = EROM[mx[h] - s8]、out += e·V、den += e
//
//  ★ softmax は DDR を1バイトも余分に読まない。表引きと引き算だけ。
//  ★ 割り算はしない。分子と分母を出して下流に任せる（1グループに {NQ} 回だけ）。
//
//  EROM[d] = round(255·exp(-d/{TAU}))、{NZ} 番地までが非零。
//  TAU は実重みの量子化スケールが決まるまでの暫定値。表を焼き直せば変わる。
//
//  掛け算器は第1相 {NQ*BPB} 個 + 第2相 {NQ*BPB} 個 = {2*NQ*BPB} 個。
// =====================================================================
module axis_attn2 #(
  parameter integer HD    = {HD},
  parameter integer NQ    = {NQ},
  parameter integer TMAX  = {TMAX},
  parameter integer SHIFT = {SHIFT}
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
  localparam integer QB    = NQ*HD/8;        // q の取り込みビート数 = {NQ*HD//8}
  localparam integer NOUT  = NQ*HD;          // 出力の本体語数 = {NOUT}
  localparam integer NOTOT = NOUT + NQ;      // 分母を足した総語数 = {NOTOT}
  localparam integer TW    = $clog2(TMAX+1);

  localparam [2:0] ST_HDR=3'd0, ST_Q=3'd1, ST_K=3'd2, ST_KDR=3'd3,
                   ST_V=3'd4, ST_DRN=3'd5, ST_OUT=3'd6;

  reg [2:0]     st;
  reg [TW-1:0]  Tn;                          // ヘッダで受け取る文脈長
  reg [TW-1:0]  tcnt;                        // 現在の t
  reg [2:0]     bcnt;                        // 行内のビート位置 0..BPR-1
  reg [7:0]     qcnt;                        // q 取り込みの進捗
  reg [8:0]     ocnt;                        // 出力の進捗 0..{NOTOT-1}
  reg [3:0]     dcnt;                        // 第2相の排出
  reg [3:0]     kdcnt;                       // 第1相の排出（mx が確定するまで待つ）
  reg           last_seen;                   // 入力の TLAST を見た

  // 出力中と排出中は上流を止める。それ以外は常に受ける。
  assign s_axis_tready = (st != ST_OUT) && (st != ST_DRN) && (st != ST_KDR);
  wire fire = s_axis_tvalid & s_axis_tready;

  // 出力レジスタの読み込み許可。空いているか、いま吐き出されているとき。
  // 【重要】5a はここが無条件で、上流が詰まるとビートを取りこぼす作りだった。
  // FIFO が 1024 段あって実際には詰まらないので表に出なかったが、直しておく。
  reg           ovld;
  wire          oload = (!ovld) | m_axis_tready;

  // ---- q の格納。ビート b は q[h] の次元 8b..8b+7 に対応する ----
  reg [63:0] qm [0:QB-1];
  always @(posedge aclk) if (fire && st==ST_Q) qm[qcnt] <= s_axis_tdata;

  // NQ ヘッドぶんを同時に引くので、ヘッドごとに別々に読み出す''')

for h in range(NQ):
    A(f"  wire [63:0] qw{h} = qm[{h*BPR} + bcnt];")
A("")

A(f'''  // ---- スコアの置き場。第1相で書き、第2相で読む ----''')
for h in range(NQ):
    A(f"  reg signed [7:0] sm{h} [0:TMAX-1];")
A("")
A("  // ---- 各ヘッドのスコア最大値。softmax の指数を負に寄せるために使う ----")
for h in range(NQ):
    A(f"  reg signed [7:0] mx{h};")
A("")

A(f'''  // ---- 状態遷移 ----
  always @(posedge aclk) begin
    if (!aresetn) begin
      st<=ST_HDR; Tn<=0; tcnt<=0; bcnt<=0; qcnt<=0; ocnt<=0;
      dcnt<=0; kdcnt<=0; last_seen<=1'b0;
    end else begin
      case (st)
        ST_HDR: if (fire) begin
                  Tn <= s_axis_tdata[TW-1:0];
                  qcnt <= 0; st <= ST_Q;
                  last_seen <= s_axis_tlast;
                end
        ST_Q:   if (fire) begin
                  if (qcnt == QB-1) begin qcnt<=0; tcnt<=0; bcnt<=0; st<=ST_K; end
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
        // 【段階5b で足した】最後の K 行のスコアが BRAM に書かれ、mx が確定する
        // までの待ち。第1相は取り込み1+積1+加算木3+加算1+clamp1 = 7段あるので、
        // ここを省くと第2相の先頭が「まだ最大でない mx」で指数を引いてしまう。
        // 12 サイクル待つ（必要なのは 7）。1グループ 16T ビートに対して誤差。
        ST_KDR: if (kdcnt == 4'd11) begin tcnt<=0; bcnt<=0; st<=ST_V; end
                else kdcnt <= kdcnt + 1'b1;
        ST_V:   if (fire) begin
                  if (bcnt == BPR-1) begin
                    bcnt <= 0;
                    if (tcnt == Tn-1) begin tcnt<=0; dcnt<=0; st<=ST_DRN; end
                    else tcnt <= tcnt + 1'b1;
                  end else bcnt <= bcnt + 1'b1;
                  if (s_axis_tlast) last_seen <= 1'b1;
                end
        // 最後の V ビートの計算がパイプラインを抜けるまで出力を待つ。
        // 第2相は取り込み1+引き算1+表引き1+掛け算1 = 4段、累算がその次なので 5。
        // 余裕を見て 9 サイクル。
        ST_DRN: if (dcnt == 4'd8) begin ocnt<=0; st<=ST_OUT; end
                else dcnt <= dcnt + 1'b1;
        ST_OUT: if (oload) begin
                  if (ocnt == NOTOT-1) begin
                    ocnt <= 0;
                    st <= ST_HDR;
                    last_seen <= 1'b0;
                  end else ocnt <= ocnt + 1'b1;
                end
        default: st <= ST_HDR;
      endcase
    end
  end

  wire in_k = fire & (st==ST_K);
  wire in_v = fire & (st==ST_V);
  wire k_rowend = in_k & (bcnt == BPR-1);
  // 分母は t ごとに1回だけ足す。行の先頭ビートでだけ立てる。
  wire v_row0 = in_v & (bcnt == 3'd0);

  // ================= 第1相: QK^T =================
  // 段階3の axis_mac と同じ形。ただし NQ 本を並列に回す。
''')

# ---- 第1相のパイプライン ----
A("  // 0段: 取り込み")
A("  reg [63:0] kw0;")
for h in range(NQ):
    A(f"  reg [63:0] qr{h};")
A("  reg kv0, ke0;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin kv0<=1'b0; ke0<=1'b0; end")
A("    else begin kv0 <= in_k; ke0 <= k_rowend; end")
A("    kw0 <= s_axis_tdata;")
for h in range(NQ):
    A(f"    qr{h} <= qw{h};")
A("  end")
A("")
A("  // 1段: 積（8個 × NQ）")
for h in range(NQ):
    A("  (* use_dsp = \"yes\" *) reg signed [15:0] " + ",".join(f"kp{h}_{i}" for i in range(BPB)) + ";")
A("  reg kv1, ke1;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin kv1<=1'b0; ke1<=1'b0; end")
A("    else begin kv1<=kv0; ke1<=ke0; end")
for h in range(NQ):
    for i in range(BPB):
        A(f"    kp{h}_{i} <= $signed(kw0[{8*i+7}:{8*i}]) * $signed(qr{h}[{8*i+7}:{8*i}]);")
A("  end")
A("")
stage=2
cur = {h: [f"kp{h}_{i}" for i in range(BPB)] for h in range(NQ)}
width = 16
while len(cur[0]) > 1:
    width += 1
    A(f"  // {stage}段: {len(cur[0])} → {len(cur[0])//2} 項 ({width}bit) × NQ")
    nxt={}
    for h in range(NQ):
        names=[]
        for k in range(len(cur[h])//2):
            nm=f"ks{stage}_{h}_{k}"
            A(f"  reg signed [{width-1}:0] {nm};")
            names.append(nm)
        nxt[h]=names
    A(f"  reg kv{stage}, ke{stage};")
    A("  always @(posedge aclk) begin")
    A(f"    if (!aresetn) begin kv{stage}<=1'b0; ke{stage}<=1'b0; end")
    A(f"    else begin kv{stage}<=kv{stage-1}; ke{stage}<=ke{stage-1}; end")
    for h in range(NQ):
        for k,nm in enumerate(nxt[h]):
            A(f"    {nm} <= {cur[h][2*k]} + {cur[h][2*k+1]};")
    A("  end")
    A("")
    cur=nxt; stage+=1

KL = stage-1
A(f"  // {stage}段: 累算。行末で s8 を作って置き場へ書く")
A("  // 【重要】行番号の遅延は、演算パイプラインと *同じ有効ビット* で進めること。")
A("  // 無条件に毎クロック進めると、DMA のバーストの切れ目で tvalid が落ちたときに")
A("  // 演算側は止まるのに行番号だけ先へ進み、スコアが別の行の番地に書かれる。")
for i in range(KL+1):
    A(f"  reg [TW-1:0] kt{i};")
A("  always @(posedge aclk) begin")
A("    if (in_k) kt0 <= tcnt;")
for i in range(1, KL+1):
    A(f"    if (kv{i-1}) kt{i} <= kt{i-1};")
A("  end")
for h in range(NQ):
    A(f"  reg signed [31:0] kacc{h};")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin " + " ".join(f"kacc{h}<=0;" for h in range(NQ)) + " end")
A(f"    else if (kv{KL}) begin")
for h in range(NQ):
    A(f"      if (ke{KL}) kacc{h} <= 0; else kacc{h} <= kacc{h} + {cur[h][0]};")
A("    end")
A("  end")
A("")
A("  // clamp して 8bit に落とす。第2相ではこれで指数の表を引く")
A("  function signed [7:0] clamp8(input signed [31:0] v);")
A("    reg signed [31:0] s;")
A("    begin")
A(f"      s = v >>> {SHIFT};")
A("      if (s >  127) clamp8 =  8'sd127;")
A("      else if (s < -128) clamp8 = 8'sh80;")
A("      else clamp8 = s[7:0];")
A("    end")
A("  endfunction")
A("")
A("  // 【重要】32bit の加算と clamp を同じサイクルに置くと論理12段になって間に合わない")
A("  // （5a の実測 WNS -0.134ns）。加算 → clamp → 書き込み+最大値 の3段に割る。")
A("  // 段階5b では最大値の比較が増えたので、clamp と書き込みもさらに割った。")
for h in range(NQ):
    A(f"  reg signed [31:0] ksum{h};")
A("  reg [TW-1:0] kw_addr;  reg kw_en;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) kw_en <= 1'b0;")
A(f"    else kw_en <= kv{KL} & ke{KL};")
for h in range(NQ):
    A(f"    ksum{h} <= kacc{h} + {cur[h][0]};")
A(f"    kw_addr <= kt{KL};")
A("  end")
A("")
for h in range(NQ):
    A(f"  reg signed [7:0] kc{h};")
A("  reg [TW-1:0] kc_addr;  reg kc_en;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) kc_en <= 1'b0;")
A("    else kc_en <= kw_en;")
for h in range(NQ):
    A(f"    kc{h} <= clamp8(ksum{h});")
A("    kc_addr <= kw_addr;")
A("  end")
A("")
A("  // 置き場への書き込みと、最大値の更新。どちらも浅い。")
A("  always @(posedge aclk) begin")
A("    if (kc_en) begin")
for h in range(NQ):
    A(f"      sm{h}[kc_addr] <= kc{h};")
A("    end")
A("  end")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin " + " ".join(f"mx{h} <= 8'sh80;" for h in range(NQ)) + " end")
A("    else if (st==ST_HDR) begin " + " ".join(f"mx{h} <= 8'sh80;" for h in range(NQ)) + " end")
A("    else if (kc_en) begin")
for h in range(NQ):
    A(f"      if (kc{h} > mx{h}) mx{h} <= kc{h};")
A("    end")
A("  end")
A("")

# ---- exp の表 ----
A("  // ================= exp の表（256語 × NQ 本） =================")
A(f"  // EROM[d] = round(255·exp(-d/{TAU}))。d = mx - s8 なので必ず 0..255 に収まる。")
A(f"  // {NZ} 番地から先は 0。3ヘッドが別々の番地を同時に引くので3本に複製する。")
for h in range(NQ):
    A(f"  reg [7:0] erom{h} [0:255];")
A("  initial begin")
for d in range(256):
    A("    " + " ".join(f"erom{h}[{d:3d}]=8'd{EROM[d]};" for h in range(NQ)))
A("  end")
A("")

# ---- 第2相 ----
A("  // ================= 第2相: softmax + AV =================")
A("  // 内積ではなく外積の累算。1ビートの V 8バイトが 8 個の別々の累算器に入る。")
A("  // 【重要】データと有効ビットの段数を数え合わせること（5a で1つずれて全滅した）。")
A("  //   データ  : s_axis_tdata → vw0(1) → vw1(2) → vw2(3) → vp(4)")
A("  //   指数    : sm[tcnt] → sr(1) → vd(2) → ve(3)")
A("  //   有効    : in_v → vv0(1) → vv1(2) → vv2(3) → vv3(4)")
A("  //   行の頭  : v_row0 → vr0(1) → vr1(2) → vr2(3)  ← ve と同じ3段")
A("  reg [63:0] vw0, vw1, vw2;")
A("  reg vv0, vv1, vv2, vv3;")
A("  reg vr0, vr1, vr2;")
for h in range(NQ):
    A(f"  reg signed [7:0] sr{h};")
for h in range(NQ):
    A(f"  reg [7:0] vd{h};")
for h in range(NQ):
    A(f"  reg [7:0] ve{h};")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin vv0<=1'b0; vv1<=1'b0; vv2<=1'b0; vv3<=1'b0;")
A("                        vr0<=1'b0; vr1<=1'b0; vr2<=1'b0; end")
A("    else begin vv0<=in_v; vv1<=vv0; vv2<=vv1; vv3<=vv2;")
A("               vr0<=v_row0; vr1<=vr0; vr2<=vr1; end")
A("    vw0 <= s_axis_tdata;  vw1 <= vw0;  vw2 <= vw1;")
for h in range(NQ):
    A(f"    sr{h} <= sm{h}[tcnt];")
A("    // mx は必ずその行の s8 以上なので、差は 0..255。9bit 引き算の下位8bit でよい。")
for h in range(NQ):
    A(f"    vd{h} <= mx{h} - sr{h};")
A("    // ★ ここが softmax。BRAM を1回引くだけ。DDR には触らない。")
for h in range(NQ):
    A(f"    ve{h} <= erom{h}[vd{h}];")
A("  end")
A("")
A("  // 4段目: 積（8個 × NQ）。V は符号付き8bit、指数は符号なし8bit。")
for h in range(NQ):
    A("  (* use_dsp = \"yes\" *) reg signed [17:0] " + ",".join(f"vp{h}_{i}" for i in range(BPB)) + ";")
A("  always @(posedge aclk) begin")
for h in range(NQ):
    for i in range(BPB):
        A(f"    vp{h}_{i} <= $signed(vw2[{8*i+7}:{8*i}]) * $signed({{1'b0, ve{h}}});")
A("  end")
A("")
A("  // 5段目: 累算器へ。")
A("  // 【重要】ビート位置 b で添字を引く書き方にすると、b が 192 個の累算器の")
A("  // アドレス解読に扇形に広がって配線が伸びる（5a の実測 WNS -0.480ns で落ちた）。")
A("  // b は 0→7 を順に回るだけなので、アドレスで選ばず累算器のほうを回す。")
A(f"  // 配置は oacc[h*{HD} + i*{BPB} + b]。b が回転位置。")
A("  //")
A(f"  // 分母は {ODEN} 番地に置く。出力アドレスの入れ替え式にこの番地が")
A("  // そのまま出てくるので、出力マルチプレクサに 2:1 を足さなくて済む。")
A(f"  reg signed [31:0] oacc [0:{OMAX-1}];")
A("  integer oi;")
A("  always @(posedge aclk) begin")
A("    if (st==ST_HDR) begin")
A(f"      for (oi=0; oi<{OMAX}; oi=oi+1) oacc[oi] <= 32'sd0;")
A("    end else begin")
A("      if (vv3) begin")
for h in range(NQ):
    for i in range(BPB):
        base = h*HD + i*BPB
        for j in range(BPB-1):
            A(f"        oacc[{base+j}] <= oacc[{base+j+1}];")
        A(f"        oacc[{base+BPB-1}] <= oacc[{base}] + $signed(vp{h}_{i});")
A("      end")
A("      if (vr2) begin")
for h in range(NQ):
    A(f"        oacc[{ODEN[h]}] <= oacc[{ODEN[h]}] + $signed({{24'd0, ve{h}}});")
A("      end")
A("    end")
A("  end")
A("")
A(f'''  // ---- 出力 ----
  // 出す順は out[h][d]（d = 8b+i）だが、置き場は h*{HD} + i*{BPB} + b。
  // ocnt の中の b と i のビット位置を入れ替えるだけで引ける。
  // ocnt = {NOUT}..{NOTOT-1}（分母）もこの式を通すと {ODEN} になる。
  wire [7:0] oaddr = {{ocnt[7:6], ocnt[2:0], ocnt[5:3]}};

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
open("rtl/axis_attn2.v","w").write(src)

# 照合スクリプトが同じ表を作れるように、表そのものも書き出しておく
with open("rtl/erom.txt","w") as f:
    f.write(f"# EROM[d] = round(255*exp(-d/{TAU}))  TAU={TAU} SHIFT={SHIFT}\n")
    f.write(" ".join(str(v) for v in EROM) + "\n")

print(f"axis_attn2.v を生成: {len(src.splitlines())} 行")
print(f"  HD={HD} NQ={NQ} 掛け算器 {2*NQ*BPB} 個 / 第1相 {KL+1}段")
print(f"  出力 {NOTOT} 語（本体 {NOUT} + 分母 {NQ}）/ 分母の番地 {ODEN} / 配列 {OMAX}")
print(f"  EROM: TAU={TAU} 非零 {NZ} 語 先頭 {EROM[:8]} ... [{NZ-1}]={EROM[NZ-1]} [{NZ}]={EROM[NZ]}")
