`timescale 1ns / 1ps
// =====================================================================
//  attention コア  ※ rtl/gen_attn.py が生成。直接編集しない
//
//  ストリーム（1グループ = 1層の1 KV ヘッドぶん）:
//     [ヘッダ 8B: T][q: 3×64 B][K: T×64 B][V: T×64 B]
//  出力: out[h][d] を int32 で 192 個（グループの終わりにまとめて吐く）
//
//  第1相 (K): s[h][t] = Σ q[h][d]·K[t][d]、s8 = clamp(s >> 8) を BRAM へ
//  第2相 (V): out[h][d] += s8[h][t] · V[t][d]
//
//  GQA の再利用により 1 バイトあたり 3 MAC。段階3の行列積（1 MAC/B）より濃い。
//  掛け算器は第1相 24 個 + 第2相 24 個 = 48 個。
//
//  【既知の割り切り】出力の 192 ビートの間は tready を下げて上流を止める。
//  1グループ 25+16T ビートに対して 192 なので T=512 なら 2.3% の損。
//  出力を次グループと重ねれば消せるが、影の累算器がもう1組要るので今はやらない。
// =====================================================================
module axis_attn #(
  parameter integer HD    = 64,
  parameter integer NQ    = 3,
  parameter integer TMAX  = 2048,
  parameter integer SHIFT = 8
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

  localparam integer BPR  = HD/8;            // 1つの t あたりのビート数 = 8
  localparam integer QB   = NQ*HD/8;         // q の取り込みビート数 = 24
  localparam integer NOUT = NQ*HD;           // 出力語数 = 192
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

  // NQ ヘッドぶんを同時に引くので、ヘッドごとに別々に読み出す
  wire [63:0] qw0 = qm[0 + bcnt];
  wire [63:0] qw1 = qm[8 + bcnt];
  wire [63:0] qw2 = qm[16 + bcnt];

  // ---- スコアの置き場。第1相で書き、第2相で読む ----
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

  // 0段: 取り込み
  reg [63:0] kw0;
  reg [63:0] qr0;
  reg [63:0] qr1;
  reg [63:0] qr2;
  reg kv0, ke0;
  always @(posedge aclk) begin
    if (!aresetn) begin kv0<=1'b0; ke0<=1'b0; end
    else begin kv0 <= in_k; ke0 <= k_rowend; end
    kw0 <= s_axis_tdata;
    qr0 <= qw0;
    qr1 <= qw1;
    qr2 <= qw2;
  end

  // 1段: 積（8個 × NQ）
  (* use_dsp = "yes" *) reg signed [15:0] kp0_0,kp0_1,kp0_2,kp0_3,kp0_4,kp0_5,kp0_6,kp0_7;
  (* use_dsp = "yes" *) reg signed [15:0] kp1_0,kp1_1,kp1_2,kp1_3,kp1_4,kp1_5,kp1_6,kp1_7;
  (* use_dsp = "yes" *) reg signed [15:0] kp2_0,kp2_1,kp2_2,kp2_3,kp2_4,kp2_5,kp2_6,kp2_7;
  reg kv1, ke1;
  always @(posedge aclk) begin
    if (!aresetn) begin kv1<=1'b0; ke1<=1'b0; end
    else begin kv1<=kv0; ke1<=ke0; end
    kp0_0 <= $signed(kw0[7:0]) * $signed(qr0[7:0]);
    kp0_1 <= $signed(kw0[15:8]) * $signed(qr0[15:8]);
    kp0_2 <= $signed(kw0[23:16]) * $signed(qr0[23:16]);
    kp0_3 <= $signed(kw0[31:24]) * $signed(qr0[31:24]);
    kp0_4 <= $signed(kw0[39:32]) * $signed(qr0[39:32]);
    kp0_5 <= $signed(kw0[47:40]) * $signed(qr0[47:40]);
    kp0_6 <= $signed(kw0[55:48]) * $signed(qr0[55:48]);
    kp0_7 <= $signed(kw0[63:56]) * $signed(qr0[63:56]);
    kp1_0 <= $signed(kw0[7:0]) * $signed(qr1[7:0]);
    kp1_1 <= $signed(kw0[15:8]) * $signed(qr1[15:8]);
    kp1_2 <= $signed(kw0[23:16]) * $signed(qr1[23:16]);
    kp1_3 <= $signed(kw0[31:24]) * $signed(qr1[31:24]);
    kp1_4 <= $signed(kw0[39:32]) * $signed(qr1[39:32]);
    kp1_5 <= $signed(kw0[47:40]) * $signed(qr1[47:40]);
    kp1_6 <= $signed(kw0[55:48]) * $signed(qr1[55:48]);
    kp1_7 <= $signed(kw0[63:56]) * $signed(qr1[63:56]);
    kp2_0 <= $signed(kw0[7:0]) * $signed(qr2[7:0]);
    kp2_1 <= $signed(kw0[15:8]) * $signed(qr2[15:8]);
    kp2_2 <= $signed(kw0[23:16]) * $signed(qr2[23:16]);
    kp2_3 <= $signed(kw0[31:24]) * $signed(qr2[31:24]);
    kp2_4 <= $signed(kw0[39:32]) * $signed(qr2[39:32]);
    kp2_5 <= $signed(kw0[47:40]) * $signed(qr2[47:40]);
    kp2_6 <= $signed(kw0[55:48]) * $signed(qr2[55:48]);
    kp2_7 <= $signed(kw0[63:56]) * $signed(qr2[63:56]);
  end

  // 2段: 8 → 4 項 (17bit) × NQ
  reg signed [16:0] ks2_0_0;
  reg signed [16:0] ks2_0_1;
  reg signed [16:0] ks2_0_2;
  reg signed [16:0] ks2_0_3;
  reg signed [16:0] ks2_1_0;
  reg signed [16:0] ks2_1_1;
  reg signed [16:0] ks2_1_2;
  reg signed [16:0] ks2_1_3;
  reg signed [16:0] ks2_2_0;
  reg signed [16:0] ks2_2_1;
  reg signed [16:0] ks2_2_2;
  reg signed [16:0] ks2_2_3;
  reg kv2, ke2;
  always @(posedge aclk) begin
    if (!aresetn) begin kv2<=1'b0; ke2<=1'b0; end
    else begin kv2<=kv1; ke2<=ke1; end
    ks2_0_0 <= kp0_0 + kp0_1;
    ks2_0_1 <= kp0_2 + kp0_3;
    ks2_0_2 <= kp0_4 + kp0_5;
    ks2_0_3 <= kp0_6 + kp0_7;
    ks2_1_0 <= kp1_0 + kp1_1;
    ks2_1_1 <= kp1_2 + kp1_3;
    ks2_1_2 <= kp1_4 + kp1_5;
    ks2_1_3 <= kp1_6 + kp1_7;
    ks2_2_0 <= kp2_0 + kp2_1;
    ks2_2_1 <= kp2_2 + kp2_3;
    ks2_2_2 <= kp2_4 + kp2_5;
    ks2_2_3 <= kp2_6 + kp2_7;
  end

  // 3段: 4 → 2 項 (18bit) × NQ
  reg signed [17:0] ks3_0_0;
  reg signed [17:0] ks3_0_1;
  reg signed [17:0] ks3_1_0;
  reg signed [17:0] ks3_1_1;
  reg signed [17:0] ks3_2_0;
  reg signed [17:0] ks3_2_1;
  reg kv3, ke3;
  always @(posedge aclk) begin
    if (!aresetn) begin kv3<=1'b0; ke3<=1'b0; end
    else begin kv3<=kv2; ke3<=ke2; end
    ks3_0_0 <= ks2_0_0 + ks2_0_1;
    ks3_0_1 <= ks2_0_2 + ks2_0_3;
    ks3_1_0 <= ks2_1_0 + ks2_1_1;
    ks3_1_1 <= ks2_1_2 + ks2_1_3;
    ks3_2_0 <= ks2_2_0 + ks2_2_1;
    ks3_2_1 <= ks2_2_2 + ks2_2_3;
  end

  // 4段: 2 → 1 項 (19bit) × NQ
  reg signed [18:0] ks4_0_0;
  reg signed [18:0] ks4_1_0;
  reg signed [18:0] ks4_2_0;
  reg kv4, ke4;
  always @(posedge aclk) begin
    if (!aresetn) begin kv4<=1'b0; ke4<=1'b0; end
    else begin kv4<=kv3; ke4<=ke3; end
    ks4_0_0 <= ks3_0_0 + ks3_0_1;
    ks4_1_0 <= ks3_1_0 + ks3_1_1;
    ks4_2_0 <= ks3_2_0 + ks3_2_1;
  end

  // 5段: 累算。行末で s8 を作って置き場へ書く
  // 【重要】行番号の遅延は、演算パイプラインと *同じ有効ビット* で進めること。
  // 無条件に毎クロック進めると、DMA のバーストの切れ目で tvalid が落ちたときに
  // 演算側は止まるのに行番号だけ先へ進み、スコアが別の行の番地に書かれる。
  // 合計は保存されるので「V=全1」の検査は通り、個々の値だけ入れ替わる（実機で踏んだ）。
  reg [TW-1:0] kt0;
  reg [TW-1:0] kt1;
  reg [TW-1:0] kt2;
  reg [TW-1:0] kt3;
  reg [TW-1:0] kt4;
  always @(posedge aclk) begin
    if (in_k) kt0 <= tcnt;
    if (kv0) kt1 <= kt0;
    if (kv1) kt2 <= kt1;
    if (kv2) kt3 <= kt2;
    if (kv3) kt4 <= kt3;
  end
  reg signed [31:0] kacc0;
  reg signed [31:0] kacc1;
  reg signed [31:0] kacc2;
  always @(posedge aclk) begin
    if (!aresetn) begin kacc0<=0; kacc1<=0; kacc2<=0; end
    else if (kv4) begin
      if (ke4) kacc0 <= 0; else kacc0 <= kacc0 + ks4_0_0;
      if (ke4) kacc1 <= 0; else kacc1 <= kacc1 + ks4_1_0;
      if (ke4) kacc2 <= 0; else kacc2 <= kacc2 + ks4_2_0;
    end
  end

  // clamp して 8bit に落とす。第2相ではこれを重みとして使う
  function signed [7:0] clamp8(input signed [31:0] v);
    reg signed [31:0] s;
    begin
      s = v >>> 8;
      if (s >  127) clamp8 =  8'sd127;
      else if (s < -128) clamp8 = -8'sd128;
      else clamp8 = s[7:0];
    end
  endfunction

  // 【重要】32bit の加算と clamp を同じサイクルに置くと
  // 「kacc → 加算 → シフト → 比較 → BRAM の D 入力」で論理12段になって間に合わない
  // （実測 WNS -0.134ns）。加算を1段、clamp と書き込みを次段に分ける。
  reg signed [31:0] ksum0;
  reg signed [31:0] ksum1;
  reg signed [31:0] ksum2;
  reg [TW-1:0] kw_addr;
  reg kw_en;
  always @(posedge aclk) begin
    if (!aresetn) kw_en <= 1'b0;
    else kw_en <= kv4 & ke4;
    ksum0 <= kacc0 + ks4_0_0;
    ksum1 <= kacc1 + ks4_1_0;
    ksum2 <= kacc2 + ks4_2_0;
    kw_addr <= kt4;
  end
  always @(posedge aclk) begin
    if (kw_en) begin
      sm0[kw_addr] <= clamp8(ksum0);
      sm1[kw_addr] <= clamp8(ksum1);
      sm2[kw_addr] <= clamp8(ksum2);
    end
  end

  // ================= 第2相: AV =================
  // 内積ではなく外積の累算。1ビートの V 8バイトが 8 個の別々の累算器に入る。
  reg [63:0] vw0;
  reg vv0, vv1;
  reg signed [7:0] sr0;
  reg signed [7:0] sr1;
  reg signed [7:0] sr2;
  always @(posedge aclk) begin
    if (!aresetn) begin vv0<=1'b0; vv1<=1'b0; end
    else begin vv0<=in_v; vv1<=vv0; end
    vw0 <= s_axis_tdata;
    sr0 <= sm0[tcnt];
    sr1 <= sm1[tcnt];
    sr2 <= sm2[tcnt];
  end

  // 1段: 積（8個 × NQ）
  (* use_dsp = "yes" *) reg signed [15:0] vp0_0,vp0_1,vp0_2,vp0_3,vp0_4,vp0_5,vp0_6,vp0_7;
  (* use_dsp = "yes" *) reg signed [15:0] vp1_0,vp1_1,vp1_2,vp1_3,vp1_4,vp1_5,vp1_6,vp1_7;
  (* use_dsp = "yes" *) reg signed [15:0] vp2_0,vp2_1,vp2_2,vp2_3,vp2_4,vp2_5,vp2_6,vp2_7;
  always @(posedge aclk) begin
    vp0_0 <= $signed(vw0[7:0]) * sr0;
    vp0_1 <= $signed(vw0[15:8]) * sr0;
    vp0_2 <= $signed(vw0[23:16]) * sr0;
    vp0_3 <= $signed(vw0[31:24]) * sr0;
    vp0_4 <= $signed(vw0[39:32]) * sr0;
    vp0_5 <= $signed(vw0[47:40]) * sr0;
    vp0_6 <= $signed(vw0[55:48]) * sr0;
    vp0_7 <= $signed(vw0[63:56]) * sr0;
    vp1_0 <= $signed(vw0[7:0]) * sr1;
    vp1_1 <= $signed(vw0[15:8]) * sr1;
    vp1_2 <= $signed(vw0[23:16]) * sr1;
    vp1_3 <= $signed(vw0[31:24]) * sr1;
    vp1_4 <= $signed(vw0[39:32]) * sr1;
    vp1_5 <= $signed(vw0[47:40]) * sr1;
    vp1_6 <= $signed(vw0[55:48]) * sr1;
    vp1_7 <= $signed(vw0[63:56]) * sr1;
    vp2_0 <= $signed(vw0[7:0]) * sr2;
    vp2_1 <= $signed(vw0[15:8]) * sr2;
    vp2_2 <= $signed(vw0[23:16]) * sr2;
    vp2_3 <= $signed(vw0[31:24]) * sr2;
    vp2_4 <= $signed(vw0[39:32]) * sr2;
    vp2_5 <= $signed(vw0[47:40]) * sr2;
    vp2_6 <= $signed(vw0[55:48]) * sr2;
    vp2_7 <= $signed(vw0[63:56]) * sr2;
  end

  // 2段: 累算器へ。
  // 【重要】ビート位置 b で添字を引く書き方にすると、b が 192 個の累算器の
  // アドレス解読に扇形に広がって配線が伸びる（実測 WNS -0.480ns で落ちた）。
  // b は 0→7 を順に回るだけなので、アドレスで選ばず累算器のほうを回す。
  // こうすると添字が全部定数になり、b はデータ経路から消える。
  // 配置は oacc[h*64 + i*8 + b]。b が回転位置。
  reg signed [31:0] oacc [0:191];
  integer oi;
  always @(posedge aclk) begin
    if (st==ST_HDR) begin
      for (oi=0; oi<192; oi=oi+1) oacc[oi] <= 32'sd0;
    // 【重要】受けるのは vv2 ではなく vv1。
    //   データ  : s_axis_tdata → vw0(1段) → vp(2段)
    //   有効ビット: in_v → vv0(1段) → vv1(2段)
    // ここを vv2(3段) にすると積を1サイクル遅れて取り込み、
    // ビート b の積がビート b-1 の累算器に入る（実機で踏んだ）。
    end else if (vv1) begin
      oacc[0] <= oacc[1];
      oacc[1] <= oacc[2];
      oacc[2] <= oacc[3];
      oacc[3] <= oacc[4];
      oacc[4] <= oacc[5];
      oacc[5] <= oacc[6];
      oacc[6] <= oacc[7];
      oacc[7] <= oacc[0] + $signed(vp0_0);
      oacc[8] <= oacc[9];
      oacc[9] <= oacc[10];
      oacc[10] <= oacc[11];
      oacc[11] <= oacc[12];
      oacc[12] <= oacc[13];
      oacc[13] <= oacc[14];
      oacc[14] <= oacc[15];
      oacc[15] <= oacc[8] + $signed(vp0_1);
      oacc[16] <= oacc[17];
      oacc[17] <= oacc[18];
      oacc[18] <= oacc[19];
      oacc[19] <= oacc[20];
      oacc[20] <= oacc[21];
      oacc[21] <= oacc[22];
      oacc[22] <= oacc[23];
      oacc[23] <= oacc[16] + $signed(vp0_2);
      oacc[24] <= oacc[25];
      oacc[25] <= oacc[26];
      oacc[26] <= oacc[27];
      oacc[27] <= oacc[28];
      oacc[28] <= oacc[29];
      oacc[29] <= oacc[30];
      oacc[30] <= oacc[31];
      oacc[31] <= oacc[24] + $signed(vp0_3);
      oacc[32] <= oacc[33];
      oacc[33] <= oacc[34];
      oacc[34] <= oacc[35];
      oacc[35] <= oacc[36];
      oacc[36] <= oacc[37];
      oacc[37] <= oacc[38];
      oacc[38] <= oacc[39];
      oacc[39] <= oacc[32] + $signed(vp0_4);
      oacc[40] <= oacc[41];
      oacc[41] <= oacc[42];
      oacc[42] <= oacc[43];
      oacc[43] <= oacc[44];
      oacc[44] <= oacc[45];
      oacc[45] <= oacc[46];
      oacc[46] <= oacc[47];
      oacc[47] <= oacc[40] + $signed(vp0_5);
      oacc[48] <= oacc[49];
      oacc[49] <= oacc[50];
      oacc[50] <= oacc[51];
      oacc[51] <= oacc[52];
      oacc[52] <= oacc[53];
      oacc[53] <= oacc[54];
      oacc[54] <= oacc[55];
      oacc[55] <= oacc[48] + $signed(vp0_6);
      oacc[56] <= oacc[57];
      oacc[57] <= oacc[58];
      oacc[58] <= oacc[59];
      oacc[59] <= oacc[60];
      oacc[60] <= oacc[61];
      oacc[61] <= oacc[62];
      oacc[62] <= oacc[63];
      oacc[63] <= oacc[56] + $signed(vp0_7);
      oacc[64] <= oacc[65];
      oacc[65] <= oacc[66];
      oacc[66] <= oacc[67];
      oacc[67] <= oacc[68];
      oacc[68] <= oacc[69];
      oacc[69] <= oacc[70];
      oacc[70] <= oacc[71];
      oacc[71] <= oacc[64] + $signed(vp1_0);
      oacc[72] <= oacc[73];
      oacc[73] <= oacc[74];
      oacc[74] <= oacc[75];
      oacc[75] <= oacc[76];
      oacc[76] <= oacc[77];
      oacc[77] <= oacc[78];
      oacc[78] <= oacc[79];
      oacc[79] <= oacc[72] + $signed(vp1_1);
      oacc[80] <= oacc[81];
      oacc[81] <= oacc[82];
      oacc[82] <= oacc[83];
      oacc[83] <= oacc[84];
      oacc[84] <= oacc[85];
      oacc[85] <= oacc[86];
      oacc[86] <= oacc[87];
      oacc[87] <= oacc[80] + $signed(vp1_2);
      oacc[88] <= oacc[89];
      oacc[89] <= oacc[90];
      oacc[90] <= oacc[91];
      oacc[91] <= oacc[92];
      oacc[92] <= oacc[93];
      oacc[93] <= oacc[94];
      oacc[94] <= oacc[95];
      oacc[95] <= oacc[88] + $signed(vp1_3);
      oacc[96] <= oacc[97];
      oacc[97] <= oacc[98];
      oacc[98] <= oacc[99];
      oacc[99] <= oacc[100];
      oacc[100] <= oacc[101];
      oacc[101] <= oacc[102];
      oacc[102] <= oacc[103];
      oacc[103] <= oacc[96] + $signed(vp1_4);
      oacc[104] <= oacc[105];
      oacc[105] <= oacc[106];
      oacc[106] <= oacc[107];
      oacc[107] <= oacc[108];
      oacc[108] <= oacc[109];
      oacc[109] <= oacc[110];
      oacc[110] <= oacc[111];
      oacc[111] <= oacc[104] + $signed(vp1_5);
      oacc[112] <= oacc[113];
      oacc[113] <= oacc[114];
      oacc[114] <= oacc[115];
      oacc[115] <= oacc[116];
      oacc[116] <= oacc[117];
      oacc[117] <= oacc[118];
      oacc[118] <= oacc[119];
      oacc[119] <= oacc[112] + $signed(vp1_6);
      oacc[120] <= oacc[121];
      oacc[121] <= oacc[122];
      oacc[122] <= oacc[123];
      oacc[123] <= oacc[124];
      oacc[124] <= oacc[125];
      oacc[125] <= oacc[126];
      oacc[126] <= oacc[127];
      oacc[127] <= oacc[120] + $signed(vp1_7);
      oacc[128] <= oacc[129];
      oacc[129] <= oacc[130];
      oacc[130] <= oacc[131];
      oacc[131] <= oacc[132];
      oacc[132] <= oacc[133];
      oacc[133] <= oacc[134];
      oacc[134] <= oacc[135];
      oacc[135] <= oacc[128] + $signed(vp2_0);
      oacc[136] <= oacc[137];
      oacc[137] <= oacc[138];
      oacc[138] <= oacc[139];
      oacc[139] <= oacc[140];
      oacc[140] <= oacc[141];
      oacc[141] <= oacc[142];
      oacc[142] <= oacc[143];
      oacc[143] <= oacc[136] + $signed(vp2_1);
      oacc[144] <= oacc[145];
      oacc[145] <= oacc[146];
      oacc[146] <= oacc[147];
      oacc[147] <= oacc[148];
      oacc[148] <= oacc[149];
      oacc[149] <= oacc[150];
      oacc[150] <= oacc[151];
      oacc[151] <= oacc[144] + $signed(vp2_2);
      oacc[152] <= oacc[153];
      oacc[153] <= oacc[154];
      oacc[154] <= oacc[155];
      oacc[155] <= oacc[156];
      oacc[156] <= oacc[157];
      oacc[157] <= oacc[158];
      oacc[158] <= oacc[159];
      oacc[159] <= oacc[152] + $signed(vp2_3);
      oacc[160] <= oacc[161];
      oacc[161] <= oacc[162];
      oacc[162] <= oacc[163];
      oacc[163] <= oacc[164];
      oacc[164] <= oacc[165];
      oacc[165] <= oacc[166];
      oacc[166] <= oacc[167];
      oacc[167] <= oacc[160] + $signed(vp2_4);
      oacc[168] <= oacc[169];
      oacc[169] <= oacc[170];
      oacc[170] <= oacc[171];
      oacc[171] <= oacc[172];
      oacc[172] <= oacc[173];
      oacc[173] <= oacc[174];
      oacc[174] <= oacc[175];
      oacc[175] <= oacc[168] + $signed(vp2_5);
      oacc[176] <= oacc[177];
      oacc[177] <= oacc[178];
      oacc[178] <= oacc[179];
      oacc[179] <= oacc[180];
      oacc[180] <= oacc[181];
      oacc[181] <= oacc[182];
      oacc[182] <= oacc[183];
      oacc[183] <= oacc[176] + $signed(vp2_6);
      oacc[184] <= oacc[185];
      oacc[185] <= oacc[186];
      oacc[186] <= oacc[187];
      oacc[187] <= oacc[188];
      oacc[188] <= oacc[189];
      oacc[189] <= oacc[190];
      oacc[190] <= oacc[191];
      oacc[191] <= oacc[184] + $signed(vp2_7);
    end
  end

  // ---- 出力 ----
  // 出す順は out[h][d]（d = 8b+i）だが、置き場は h*64 + i*8 + b。
  // ocnt の中の b と i のビット位置を入れ替えるだけで引ける。
  wire [7:0] oaddr = {ocnt[7:6], ocnt[2:0], ocnt[5:3]};
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

endmodule
