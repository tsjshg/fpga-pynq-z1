#!/usr/bin/env python3
"""axis_attn.v（attention コア）を生成する。

段階5。段階3の axis_mac（内積のみ）に対して、こちらは2相になる。

  ストリーム（1グループ = 1層の1 KV ヘッドぶん）:
     [ヘッダ 8B: T][q: NQ*HD B][K: T*HD B][V: T*HD B]
  出力:
     out[h][d] を int32 で NQ*HD 個

  第1相 (K): s[h][t] = Σ_d q[h][d]*K[t][d]     … 内積。段階3と同じ形
             s8[h][t] = clamp(s >> SHIFT)       … オンチップに貯める
  第2相 (V): out[h][d] += s8[h][t] * V[t][d]    … 外積の累算。ここが新しい

  GQA の再利用: K/V 1ヘッドを q NQ ヘッドが共有するので、K を1回流す間に
  NQ 本のスコアを同時に計算する。1バイトあたり NQ MAC。

  softmax はまだ入れない。CPU 実測で 83ms 中 6.94ms しかなく、本丸は QK^T と AV。
  exp の表引きを足すと検証が「回路」と「表」の切り分けになるので、先に経路を固める。
"""
HD    = 64      # head_dim
NQ    = 3       # 1つの KV ヘッドを共有する q ヘッド数（SmolLM2-135M は 9/3）
BPB   = 8       # 1ビートのバイト数
BPR   = HD//BPB # 1行（1つの t）あたりのビート数 = 8
TMAX  = 2048
SHIFT = 8
NOUT  = NQ*HD   # 192

out=[]; A=out.append

A(f'''`timescale 1ns / 1ps
// =====================================================================
//  attention コア  ※ rtl/gen_attn.py が生成。直接編集しない
//
//  ストリーム（1グループ = 1層の1 KV ヘッドぶん）:
//     [ヘッダ 8B: T][q: {NQ}×{HD} B][K: T×{HD} B][V: T×{HD} B]
//  出力: out[h][d] を int32 で {NOUT} 個（グループの終わりにまとめて吐く）
//
//  第1相 (K): s[h][t] = Σ q[h][d]·K[t][d]、s8 = clamp(s >> {SHIFT}) を BRAM へ
//  第2相 (V): out[h][d] += s8[h][t] · V[t][d]
//
//  GQA の再利用により 1 バイトあたり {NQ} MAC。段階3の行列積（1 MAC/B）より濃い。
//  掛け算器は第1相 {NQ*BPB} 個 + 第2相 {NQ*BPB} 個 = {2*NQ*BPB} 個。
//
//  【既知の割り切り】出力の {NOUT} ビートの間は tready を下げて上流を止める。
//  1グループ {25}+16T ビートに対して {NOUT} なので T=512 なら 2.3% の損。
//  出力を次グループと重ねれば消せるが、影の累算器がもう1組要るので今はやらない。
// =====================================================================
module axis_attn #(
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

  localparam integer BPR  = HD/8;            // 1つの t あたりのビート数 = {BPR}
  localparam integer QB   = NQ*HD/8;         // q の取り込みビート数 = {NQ*HD//8}
  localparam integer NOUT = NQ*HD;           // 出力語数 = {NOUT}
  localparam integer TW   = $clog2(TMAX+1);

  localparam [2:0] ST_HDR=3'd0, ST_Q=3'd1, ST_K=3'd2, ST_V=3'd3, ST_DRN=3'd4, ST_OUT=3'd5;

  reg [2:0]     st;
  reg [TW-1:0]  Tn;                          // ヘッダで受け取る文脈長
  reg [TW-1:0]  tcnt;                        // 現在の t
  reg [2:0]     bcnt;                        // 行内のビート位置 0..BPR-1
  reg [7:0]     qcnt;                        // q 取り込みの進捗
  reg [8:0]     ocnt;                        // 出力の進捗
  reg [2:0]     dcnt;                        // 排出の進捗
  reg           last_seen;                   // 入力の TLAST を見た

  // 出力中と排出中は上流を止める。それ以外は常に受ける。
  assign s_axis_tready = (st != ST_OUT) && (st != ST_DRN);
  wire fire = s_axis_tvalid & s_axis_tready;

  // ---- q の格納。ビート b は q[h] の次元 8b..8b+7 に対応する ----
  reg [63:0] qm [0:QB-1];
  always @(posedge aclk) if (fire && st==ST_Q) qm[qcnt] <= s_axis_tdata;
  wire [63:0] qw = qm[{{{{5{{1'b0}}}},bcnt}} + (hsel*BPR)];   // 使わない（下で展開）
''')

# q は NQ ヘッドぶんを同時に読む必要があるので、ヘッド別に読み出しを展開する
A("  // NQ ヘッドぶんを同時に引くので、ヘッドごとに別々に読み出す")
for h in range(NQ):
    A(f"  wire [63:0] qw{h} = qm[{h*BPR} + bcnt];")
A("")

A(f'''  // ---- スコアの置き場。第1相で書き、第2相で読む ----
  reg signed [7:0] sm0 [0:TMAX-1];
  reg signed [7:0] sm1 [0:TMAX-1];
  reg signed [7:0] sm2 [0:TMAX-1];

  // ---- 状態遷移 ----
  always @(posedge aclk) begin
    if (!aresetn) begin
      st<=ST_HDR; Tn<=0; tcnt<=0; bcnt<=0; qcnt<=0; ocnt<=0; dcnt<=0; last_seen<=1'b0;
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
                    if (tcnt == Tn-1) begin tcnt<=0; st<=ST_V; end
                    else tcnt <= tcnt + 1'b1;
                  end else bcnt <= bcnt + 1'b1;
                  if (s_axis_tlast) last_seen <= 1'b1;
                end
        ST_V:   if (fire) begin
                  if (bcnt == BPR-1) begin
                    bcnt <= 0;
                    if (tcnt == Tn-1) begin tcnt<=0; dcnt<=0; st<=ST_DRN; end
                    else tcnt <= tcnt + 1'b1;
                  end else bcnt <= bcnt + 1'b1;
                  if (s_axis_tlast) last_seen <= 1'b1;
                end
        // 【重要】最後の V ビートの計算がパイプラインを抜けるまで出力を待つ。
        // これを入れずに ST_V から直接 ST_OUT へ行くと、先頭3個が
        // 完成前の累算器を読んでちょうど1項ぶん足りなくなる（実機で確認済み）。
        ST_DRN: if (dcnt == 3'd4) begin ocnt<=0; st<=ST_OUT; end
                else dcnt <= dcnt + 1'b1;
        ST_OUT: if (m_axis_tready) begin
                  if (ocnt == NOUT-1) begin
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
    A(f"  (* use_dsp = \"yes\" *) reg signed [15:0] " + ",".join(f"kp{h}_{i}" for i in range(BPB)) + ";")
A("  reg kv1, ke1;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin kv1<=1'b0; ke1<=1'b0; end")
A("    else begin kv1<=kv0; ke1<=ke0; end")
for h in range(NQ):
    for i in range(BPB):
        A(f"    kp{h}_{i} <= $signed(kw0[{8*i+7}:{8*i}]) * $signed(qr{h}[{8*i+7}:{8*i}]);")
A("  end")
A("")
# 加算木 8 -> 4 -> 2 -> 1
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

KLAST = stage-1
KL = KLAST
KL = stage-1
A(f"  // {stage}段: 累算。行末で s8 を作って置き場へ書く")
A("  // 【重要】行番号の遅延は、演算パイプラインと *同じ有効ビット* で進めること。")
A("  // 無条件に毎クロック進めると、DMA のバーストの切れ目で tvalid が落ちたときに")
A("  // 演算側は止まるのに行番号だけ先へ進み、スコアが別の行の番地に書かれる。")
A("  // 合計は保存されるので「V=全1」の検査は通り、個々の値だけ入れ替わる（実機で踏んだ）。")
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
A(f"    else if (kv{KLAST}) begin")
for h in range(NQ):
    A(f"      if (ke{KLAST}) kacc{h} <= 0; else kacc{h} <= kacc{h} + {cur[h][0]};")
A("    end")
A("  end")
A("")
A("  // clamp して 8bit に落とす。第2相ではこれを重みとして使う")
A("  function signed [7:0] clamp8(input signed [31:0] v);")
A("    reg signed [31:0] s;")
A("    begin")
A(f"      s = v >>> {SHIFT};")
A("      if (s >  127) clamp8 =  8'sd127;")
A("      else if (s < -128) clamp8 = -8'sd128;")
A("      else clamp8 = s[7:0];")
A("    end")
A("  endfunction")
A("")
A("  // 【重要】32bit の加算と clamp を同じサイクルに置くと")
A("  // 「kacc → 加算 → シフト → 比較 → BRAM の D 入力」で論理12段になって間に合わない")
A("  // （実測 WNS -0.134ns）。加算を1段、clamp と書き込みを次段に分ける。")
for h in range(NQ):
    A(f"  reg signed [31:0] ksum{h};")
A("  reg [TW-1:0] kw_addr;")
A("  reg kw_en;")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) kw_en <= 1'b0;")
A(f"    else kw_en <= kv{KLAST} & ke{KLAST};")
for h in range(NQ):
    A(f"    ksum{h} <= kacc{h} + {cur[h][0]};")
A(f"    kw_addr <= kt{KLAST};")
A("  end")
A("  always @(posedge aclk) begin")
A("    if (kw_en) begin")
for h in range(NQ):
    A(f"      sm{h}[kw_addr] <= clamp8(ksum{h});")
A("    end")
A("  end")
A("")

# ---- 第2相 ----
A("  // ================= 第2相: AV =================")
A("  // 内積ではなく外積の累算。1ビートの V 8バイトが 8 個の別々の累算器に入る。")
A("  reg [63:0] vw0;")
A("  reg vv0, vv1;")
for h in range(NQ):
    A(f"  reg signed [7:0] sr{h};")
A("  always @(posedge aclk) begin")
A("    if (!aresetn) begin vv0<=1'b0; vv1<=1'b0; end")
A("    else begin vv0<=in_v; vv1<=vv0; end")
A("    vw0 <= s_axis_tdata;")
for h in range(NQ):
    A(f"    sr{h} <= sm{h}[tcnt];")
A("  end")
A("")
A("  // 1段: 積（8個 × NQ）")
for h in range(NQ):
    A(f"  (* use_dsp = \"yes\" *) reg signed [15:0] " + ",".join(f"vp{h}_{i}" for i in range(BPB)) + ";")
A("  always @(posedge aclk) begin")
for h in range(NQ):
    for i in range(BPB):
        A(f"    vp{h}_{i} <= $signed(vw0[{8*i+7}:{8*i}]) * sr{h};")
A("  end")
A("")
A("  // 2段: 累算器へ。")
A("  // 【重要】ビート位置 b で添字を引く書き方にすると、b が 192 個の累算器の")
A("  // アドレス解読に扇形に広がって配線が伸びる（実測 WNS -0.480ns で落ちた）。")
A("  // b は 0→7 を順に回るだけなので、アドレスで選ばず累算器のほうを回す。")
A("  // こうすると添字が全部定数になり、b はデータ経路から消える。")
A(f"  // 配置は oacc[h*{HD} + i*{BPB} + b]。b が回転位置。")
A(f"  reg signed [31:0] oacc [0:{NOUT-1}];")
A("  integer oi;")
A("  always @(posedge aclk) begin")
A("    if (st==ST_HDR) begin")
A(f"      for (oi=0; oi<{NOUT}; oi=oi+1) oacc[oi] <= 32'sd0;")
A("    // 【重要】受けるのは vv2 ではなく vv1。")
A("    //   データ  : s_axis_tdata → vw0(1段) → vp(2段)")
A("    //   有効ビット: in_v → vv0(1段) → vv1(2段)")
A("    // ここを vv2(3段) にすると積を1サイクル遅れて取り込み、")
A("    // ビート b の積がビート b-1 の累算器に入る（実機で踏んだ）。")
A("    end else if (vv1) begin")
for h in range(NQ):
    for i in range(BPB):
        base = h*HD + i*BPB
        for j in range(BPB-1):
            A(f"      oacc[{base+j}] <= oacc[{base+j+1}];")
        A(f"      oacc[{base+BPB-1}] <= oacc[{base}] + $signed(vp{h}_{i});")
A("    end")
A("  end")
A("")
A(f'''  // ---- 出力 ----
  // 出す順は out[h][d]（d = 8b+i）だが、置き場は h*{HD} + i*{BPB} + b。
  // ocnt の中の b と i のビット位置を入れ替えるだけで引ける。
  wire [7:0] oaddr = {{ocnt[7:6], ocnt[2:0], ocnt[5:3]}};
  reg signed [31:0] odat;
  always @(posedge aclk) odat <= oacc[oaddr];

  reg ovld, olast;
  always @(posedge aclk) begin
    if (!aresetn) begin ovld<=1'b0; olast<=1'b0; end
    else begin
      ovld  <= (st==ST_OUT);
      olast <= (st==ST_OUT) && (ocnt==NOUT-1) && last_seen;
    end
  end

  assign m_axis_tdata  = odat;
  assign m_axis_tvalid = ovld;
  assign m_axis_tlast  = olast;

endmodule''')

src="\n".join(out)+"\n"
# 使わなかった qw の行を落とす
src = "\n".join(l for l in src.split("\n") if "使わない（下で展開）" not in l)
open("rtl/axis_attn.v","w").write(src)
print(f"axis_attn.v を生成: {len(src.splitlines())} 行 / HD={HD} NQ={NQ} 掛け算器 {2*NQ*BPB} 個 / 第1相 {KLAST+1}段")
