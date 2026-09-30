`timescale 1ns / 1ps
// =====================================================================
//  attention コア（head_dim 96 / GQA なし / softmax 入り）
//  ※ rtl/gen_attnv.py が生成。直接編集しない
//
//  ストリーム（見出し付き。TLAST まで何組でも）:
//     [ 見出し 8B: [31:0]=T, [49:32]=m ][ q: 96 B ][ K: T×96 B ][ V: T×96 B ]
//  出力: out[d] を int32 で 96 個 + 分母 den を 1 個 = 97 語
//
//  第1相 (K): s = Σ q·K（96 項 = 12 ビート）→ s8 = clamp((s·m) >>> 24) を BRAM へ
//             同時に mx = max s8 を追う
//  第2相 (V): e = EROM[mx - s8]、out += e·V、den += e
//
//  ★ m を見出しで受ける。段階5b は clamp(s >> 8) の固定で、量子化尺度を
//    無視していたため実モデルで perplexity 35.56 → 58.25 に悪化した。
//    m = TAU·sq·sk/√HD を渡すと 35.90 に戻る（掛け算器 1 個ぶんの改修）。
//
//  EROM[d] = round(255·exp(-d/8.0))、50 番地までが非零。
//  割り算はしない。分子 96 語と分母 1 語を出して下流に任せる。
//
//  GQA が無いので 1 バイトあたり 1 MAC。掛け算器は
//  第1相 8 + スコア倍率 1 + 第2相 8 = 17 個。
// =====================================================================
module axis_attnv #(
  parameter integer HD   = 96,
  parameter integer TMAX = 2048,
  parameter integer SHM  = 24
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

  localparam integer BPR   = HD/8;           // 1つの t あたりのビート数 = 12
  localparam integer NOUT  = HD;             // 96
  localparam integer NOTOT = NOUT + 1;       // 97
  localparam integer TW    = $clog2(TMAX+1);

  localparam [2:0] ST_HDR=3'd0, ST_Q=3'd1, ST_K=3'd2, ST_KDR=3'd3,
                   ST_V=3'd4, ST_DRN=3'd5, ST_OUT=3'd6;

  reg [2:0]     st;
  reg [TW-1:0]  Tn;                          // 見出しの文脈長
  reg [17:0]    mscale;                      // 見出しのスコア倍率
  reg [TW-1:0]  tcnt;
  reg [3:0]     bcnt;                        // 行内のビート位置 0..11
  reg [3:0]     qcnt;                        // q 取り込みの進捗 0..11
  reg [7:0]     ocnt;                        // 出力の進捗 0..96
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

  // 1段: 積（8個）。use_dsp を付けないと LUT で組まれて論理段数が足りなくなる
  (* use_dsp = "yes" *) reg signed [15:0] kp0,kp1,kp2,kp3,kp4,kp5,kp6,kp7;
  reg kv1, ke1;
  always @(posedge aclk) begin
    if (!aresetn) begin kv1<=1'b0; ke1<=1'b0; end
    else begin kv1<=kv0; ke1<=ke0; end
    kp0 <= $signed(kw0[7:0]) * $signed(qr0[7:0]);
    kp1 <= $signed(kw0[15:8]) * $signed(qr0[15:8]);
    kp2 <= $signed(kw0[23:16]) * $signed(qr0[23:16]);
    kp3 <= $signed(kw0[31:24]) * $signed(qr0[31:24]);
    kp4 <= $signed(kw0[39:32]) * $signed(qr0[39:32]);
    kp5 <= $signed(kw0[47:40]) * $signed(qr0[47:40]);
    kp6 <= $signed(kw0[55:48]) * $signed(qr0[55:48]);
    kp7 <= $signed(kw0[63:56]) * $signed(qr0[63:56]);
  end

  // 2段: 8 → 4 項 (17bit)
  reg signed [16:0] ks2_0;
  reg signed [16:0] ks2_1;
  reg signed [16:0] ks2_2;
  reg signed [16:0] ks2_3;
  reg kv2, ke2;
  always @(posedge aclk) begin
    if (!aresetn) begin kv2<=1'b0; ke2<=1'b0; end
    else begin kv2<=kv1; ke2<=ke1; end
    ks2_0 <= kp0 + kp1;
    ks2_1 <= kp2 + kp3;
    ks2_2 <= kp4 + kp5;
    ks2_3 <= kp6 + kp7;
  end

  // 3段: 4 → 2 項 (18bit)
  reg signed [17:0] ks3_0;
  reg signed [17:0] ks3_1;
  reg kv3, ke3;
  always @(posedge aclk) begin
    if (!aresetn) begin kv3<=1'b0; ke3<=1'b0; end
    else begin kv3<=kv2; ke3<=ke2; end
    ks3_0 <= ks2_0 + ks2_1;
    ks3_1 <= ks2_2 + ks2_3;
  end

  // 4段: 2 → 1 項 (19bit)
  reg signed [18:0] ks4_0;
  reg kv4, ke4;
  always @(posedge aclk) begin
    if (!aresetn) begin kv4<=1'b0; ke4<=1'b0; end
    else begin kv4<=kv3; ke4<=ke3; end
    ks4_0 <= ks3_0 + ks3_1;
  end

  // 5段: 累算。行末で 1 行ぶんの s が揃う
  // 【重要】行番号の遅延は演算パイプラインと *同じ有効ビット* で進める
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
  reg signed [31:0] kacc;
  always @(posedge aclk) begin
    if (!aresetn) kacc <= 32'sd0;
    else if (kv4) begin
      if (ke4) kacc <= 32'sd0; else kacc <= kacc + ks4_0;
    end
  end

  // 6段: 行の合計。|s| ≤ 96·127·127 = 1548384 < 2^21 なので 22bit で足りる
  wire signed [31:0] ktot = kacc + ks4_0;   // 式に添字は付けられないので一度 wire に
  reg signed [21:0] ksum;
  reg [TW-1:0] kw_addr;  reg kw_en;
  always @(posedge aclk) begin
    if (!aresetn) kw_en <= 1'b0;
    else kw_en <= kv4 & ke4;
    ksum    <= ktot[21:0];
    kw_addr <= kt4;
  end

  // 7段: ★ スコア倍率。見出しの m（18bit）を掛けて >>> 24
  // これが段階5b からの本質的な変更。22bit × 18bit なので DSP 1 個に収まる。
  (* use_dsp = "yes" *) reg signed [40:0] kmul;
  reg [TW-1:0] km_addr;  reg km_en;
  always @(posedge aclk) begin
    if (!aresetn) km_en <= 1'b0; else km_en <= kw_en;
    kmul    <= ksum * $signed({1'b0, mscale});
    km_addr <= kw_addr;
  end

  // 8段: clamp して 8bit に落とす（掛け算と同じサイクルには置かない）
  function signed [7:0] clamp8(input signed [40:0] v);
    reg signed [40:0] t;
    begin
      t = v >>> 24;
      if (t >  127) clamp8 =  8'sd127;
      else if (t < -128) clamp8 = 8'sh80;
      else clamp8 = t[7:0];
    end
  endfunction
  reg signed [7:0] kc;
  reg [TW-1:0] kc_addr;  reg kc_en;
  always @(posedge aclk) begin
    if (!aresetn) kc_en <= 1'b0; else kc_en <= km_en;
    kc      <= clamp8(kmul);
    kc_addr <= km_addr;
  end

  // 置き場への書き込みと最大値の更新。どちらも浅い。
  always @(posedge aclk) if (kc_en) sm[kc_addr] <= kc;
  always @(posedge aclk) begin
    if (!aresetn)                mx <= 8'sh80;
    else if (st==ST_HDR)         mx <= 8'sh80;
    else if (kc_en && kc > mx)   mx <= kc;
  end

  // ================= exp の表（256語） =================
  // EROM[d] = round(255·exp(-d/8.0))。d = mx - s8 なので必ず 0..255。
  // 50 番地から先は 0 なので、合成すると LUT の論理に畳まれる。
  reg [7:0] erom [0:255];
  initial begin
    erom[  0]=8'd255; erom[  1]=8'd225; erom[  2]=8'd199; erom[  3]=8'd175;
    erom[  4]=8'd155; erom[  5]=8'd136; erom[  6]=8'd120; erom[  7]=8'd106;
    erom[  8]=8'd94; erom[  9]=8'd83; erom[ 10]=8'd73; erom[ 11]=8'd64;
    erom[ 12]=8'd57; erom[ 13]=8'd50; erom[ 14]=8'd44; erom[ 15]=8'd39;
    erom[ 16]=8'd35; erom[ 17]=8'd30; erom[ 18]=8'd27; erom[ 19]=8'd24;
    erom[ 20]=8'd21; erom[ 21]=8'd18; erom[ 22]=8'd16; erom[ 23]=8'd14;
    erom[ 24]=8'd13; erom[ 25]=8'd11; erom[ 26]=8'd10; erom[ 27]=8'd9;
    erom[ 28]=8'd8; erom[ 29]=8'd7; erom[ 30]=8'd6; erom[ 31]=8'd5;
    erom[ 32]=8'd5; erom[ 33]=8'd4; erom[ 34]=8'd4; erom[ 35]=8'd3;
    erom[ 36]=8'd3; erom[ 37]=8'd2; erom[ 38]=8'd2; erom[ 39]=8'd2;
    erom[ 40]=8'd2; erom[ 41]=8'd2; erom[ 42]=8'd1; erom[ 43]=8'd1;
    erom[ 44]=8'd1; erom[ 45]=8'd1; erom[ 46]=8'd1; erom[ 47]=8'd1;
    erom[ 48]=8'd1; erom[ 49]=8'd1; erom[ 50]=8'd0; erom[ 51]=8'd0;
    erom[ 52]=8'd0; erom[ 53]=8'd0; erom[ 54]=8'd0; erom[ 55]=8'd0;
    erom[ 56]=8'd0; erom[ 57]=8'd0; erom[ 58]=8'd0; erom[ 59]=8'd0;
    erom[ 60]=8'd0; erom[ 61]=8'd0; erom[ 62]=8'd0; erom[ 63]=8'd0;
    erom[ 64]=8'd0; erom[ 65]=8'd0; erom[ 66]=8'd0; erom[ 67]=8'd0;
    erom[ 68]=8'd0; erom[ 69]=8'd0; erom[ 70]=8'd0; erom[ 71]=8'd0;
    erom[ 72]=8'd0; erom[ 73]=8'd0; erom[ 74]=8'd0; erom[ 75]=8'd0;
    erom[ 76]=8'd0; erom[ 77]=8'd0; erom[ 78]=8'd0; erom[ 79]=8'd0;
    erom[ 80]=8'd0; erom[ 81]=8'd0; erom[ 82]=8'd0; erom[ 83]=8'd0;
    erom[ 84]=8'd0; erom[ 85]=8'd0; erom[ 86]=8'd0; erom[ 87]=8'd0;
    erom[ 88]=8'd0; erom[ 89]=8'd0; erom[ 90]=8'd0; erom[ 91]=8'd0;
    erom[ 92]=8'd0; erom[ 93]=8'd0; erom[ 94]=8'd0; erom[ 95]=8'd0;
    erom[ 96]=8'd0; erom[ 97]=8'd0; erom[ 98]=8'd0; erom[ 99]=8'd0;
    erom[100]=8'd0; erom[101]=8'd0; erom[102]=8'd0; erom[103]=8'd0;
    erom[104]=8'd0; erom[105]=8'd0; erom[106]=8'd0; erom[107]=8'd0;
    erom[108]=8'd0; erom[109]=8'd0; erom[110]=8'd0; erom[111]=8'd0;
    erom[112]=8'd0; erom[113]=8'd0; erom[114]=8'd0; erom[115]=8'd0;
    erom[116]=8'd0; erom[117]=8'd0; erom[118]=8'd0; erom[119]=8'd0;
    erom[120]=8'd0; erom[121]=8'd0; erom[122]=8'd0; erom[123]=8'd0;
    erom[124]=8'd0; erom[125]=8'd0; erom[126]=8'd0; erom[127]=8'd0;
    erom[128]=8'd0; erom[129]=8'd0; erom[130]=8'd0; erom[131]=8'd0;
    erom[132]=8'd0; erom[133]=8'd0; erom[134]=8'd0; erom[135]=8'd0;
    erom[136]=8'd0; erom[137]=8'd0; erom[138]=8'd0; erom[139]=8'd0;
    erom[140]=8'd0; erom[141]=8'd0; erom[142]=8'd0; erom[143]=8'd0;
    erom[144]=8'd0; erom[145]=8'd0; erom[146]=8'd0; erom[147]=8'd0;
    erom[148]=8'd0; erom[149]=8'd0; erom[150]=8'd0; erom[151]=8'd0;
    erom[152]=8'd0; erom[153]=8'd0; erom[154]=8'd0; erom[155]=8'd0;
    erom[156]=8'd0; erom[157]=8'd0; erom[158]=8'd0; erom[159]=8'd0;
    erom[160]=8'd0; erom[161]=8'd0; erom[162]=8'd0; erom[163]=8'd0;
    erom[164]=8'd0; erom[165]=8'd0; erom[166]=8'd0; erom[167]=8'd0;
    erom[168]=8'd0; erom[169]=8'd0; erom[170]=8'd0; erom[171]=8'd0;
    erom[172]=8'd0; erom[173]=8'd0; erom[174]=8'd0; erom[175]=8'd0;
    erom[176]=8'd0; erom[177]=8'd0; erom[178]=8'd0; erom[179]=8'd0;
    erom[180]=8'd0; erom[181]=8'd0; erom[182]=8'd0; erom[183]=8'd0;
    erom[184]=8'd0; erom[185]=8'd0; erom[186]=8'd0; erom[187]=8'd0;
    erom[188]=8'd0; erom[189]=8'd0; erom[190]=8'd0; erom[191]=8'd0;
    erom[192]=8'd0; erom[193]=8'd0; erom[194]=8'd0; erom[195]=8'd0;
    erom[196]=8'd0; erom[197]=8'd0; erom[198]=8'd0; erom[199]=8'd0;
    erom[200]=8'd0; erom[201]=8'd0; erom[202]=8'd0; erom[203]=8'd0;
    erom[204]=8'd0; erom[205]=8'd0; erom[206]=8'd0; erom[207]=8'd0;
    erom[208]=8'd0; erom[209]=8'd0; erom[210]=8'd0; erom[211]=8'd0;
    erom[212]=8'd0; erom[213]=8'd0; erom[214]=8'd0; erom[215]=8'd0;
    erom[216]=8'd0; erom[217]=8'd0; erom[218]=8'd0; erom[219]=8'd0;
    erom[220]=8'd0; erom[221]=8'd0; erom[222]=8'd0; erom[223]=8'd0;
    erom[224]=8'd0; erom[225]=8'd0; erom[226]=8'd0; erom[227]=8'd0;
    erom[228]=8'd0; erom[229]=8'd0; erom[230]=8'd0; erom[231]=8'd0;
    erom[232]=8'd0; erom[233]=8'd0; erom[234]=8'd0; erom[235]=8'd0;
    erom[236]=8'd0; erom[237]=8'd0; erom[238]=8'd0; erom[239]=8'd0;
    erom[240]=8'd0; erom[241]=8'd0; erom[242]=8'd0; erom[243]=8'd0;
    erom[244]=8'd0; erom[245]=8'd0; erom[246]=8'd0; erom[247]=8'd0;
    erom[248]=8'd0; erom[249]=8'd0; erom[250]=8'd0; erom[251]=8'd0;
    erom[252]=8'd0; erom[253]=8'd0; erom[254]=8'd0; erom[255]=8'd0;
  end

  // ================= 第2相: softmax + AV =================
  // 【重要】データと有効ビットの段数を数え合わせる（段階5a で1つずれて全滅した）
  //   データ: s_axis_tdata → vw0(1) → vw1(2) → vw2(3) → vp(4)
  //   指数  : sm[tcnt] → sr(1) → vd(2) → ve(3)
  //   有効  : in_v → vv0(1) → vv1(2) → vv2(3) → vv3(4)
  //   行の頭: v_row0 → vr0(1) → vr1(2) → vr2(3)   ← ve と同じ3段
  reg [63:0] vw0, vw1, vw2;
  reg vv0, vv1, vv2;
  // vv3 は 96×32 個の累算器の CE を駆動する。1本だと配線が遅延の 93% を占め
  // （実測 6.3ns）、混雑で AXI DMA 側まで巻き添えにした。複製させる。
  (* max_fanout = 64 *) reg vv3;
  reg vr0, vr1, vr2;
  reg signed [7:0] sr;
  reg [7:0] vd, ve;
  always @(posedge aclk) begin
    if (!aresetn) begin vv0<=1'b0; vv1<=1'b0; vv2<=1'b0; vv3<=1'b0;
                        vr0<=1'b0; vr1<=1'b0; vr2<=1'b0; end
    else begin vv0<=in_v; vv1<=vv0; vv2<=vv1; vv3<=vv2;
               vr0<=v_row0; vr1<=vr0; vr2<=vr1; end
    vw0 <= s_axis_tdata; vw1 <= vw0; vw2 <= vw1;
    sr  <= sm[tcnt];
    vd  <= mx - sr;            // mx は必ず s8 以上なので 9bit 引き算の下位 8bit でよい
    ve  <= erom[vd];           // ★ ここが softmax。DDR には触らない
  end

  // 4段: 積（8個）。V は符号付き8bit、指数は符号なし8bit
  (* use_dsp = "yes" *) reg signed [17:0] vp0,vp1,vp2,vp3,vp4,vp5,vp6,vp7;
  always @(posedge aclk) begin
    vp0 <= $signed(vw2[7:0]) * $signed({1'b0, ve});
    vp1 <= $signed(vw2[15:8]) * $signed({1'b0, ve});
    vp2 <= $signed(vw2[23:16]) * $signed({1'b0, ve});
    vp3 <= $signed(vw2[31:24]) * $signed({1'b0, ve});
    vp4 <= $signed(vw2[39:32]) * $signed({1'b0, ve});
    vp5 <= $signed(vw2[47:40]) * $signed({1'b0, ve});
    vp6 <= $signed(vw2[55:48]) * $signed({1'b0, ve});
    vp7 <= $signed(vw2[63:56]) * $signed({1'b0, ve});
  end

  // 5段: 累算器へ。
  // 【重要】ビート位置で添字を引くとアドレス解読に扇形に広がって落ちる
  // （段階5a の実測 WNS -0.480ns）。b は 0→11 を順に回るだけなので累算器を回す。
  // 配置は oacc[i*16 + b]。b が回転位置。回転長 = 1行のビート数 12。
  // 分母は 12 番地。出力アドレスの入れ替え式に ocnt=96 を通すとそこに当たる。
  reg signed [31:0] oacc [0:127];
  integer oi;
  // クリアも 128×32 個を駆動するので、1段遅らせたうえで複製させる。
  // ST_HDR の次の組の最初の V ビートまでには 1+12+12T サイクルあるので遅らせて安全。
  (* max_fanout = 64 *) reg oclr;
  always @(posedge aclk) oclr <= (st==ST_HDR);
  always @(posedge aclk) begin
    if (oclr) begin
      for (oi=0; oi<128; oi=oi+1) oacc[oi] <= 32'sd0;
    end else begin
      if (vv3) begin
        oacc[0] <= oacc[1];
        oacc[1] <= oacc[2];
        oacc[2] <= oacc[3];
        oacc[3] <= oacc[4];
        oacc[4] <= oacc[5];
        oacc[5] <= oacc[6];
        oacc[6] <= oacc[7];
        oacc[7] <= oacc[8];
        oacc[8] <= oacc[9];
        oacc[9] <= oacc[10];
        oacc[10] <= oacc[11];
        oacc[11] <= oacc[0] + $signed(vp0);
        oacc[16] <= oacc[17];
        oacc[17] <= oacc[18];
        oacc[18] <= oacc[19];
        oacc[19] <= oacc[20];
        oacc[20] <= oacc[21];
        oacc[21] <= oacc[22];
        oacc[22] <= oacc[23];
        oacc[23] <= oacc[24];
        oacc[24] <= oacc[25];
        oacc[25] <= oacc[26];
        oacc[26] <= oacc[27];
        oacc[27] <= oacc[16] + $signed(vp1);
        oacc[32] <= oacc[33];
        oacc[33] <= oacc[34];
        oacc[34] <= oacc[35];
        oacc[35] <= oacc[36];
        oacc[36] <= oacc[37];
        oacc[37] <= oacc[38];
        oacc[38] <= oacc[39];
        oacc[39] <= oacc[40];
        oacc[40] <= oacc[41];
        oacc[41] <= oacc[42];
        oacc[42] <= oacc[43];
        oacc[43] <= oacc[32] + $signed(vp2);
        oacc[48] <= oacc[49];
        oacc[49] <= oacc[50];
        oacc[50] <= oacc[51];
        oacc[51] <= oacc[52];
        oacc[52] <= oacc[53];
        oacc[53] <= oacc[54];
        oacc[54] <= oacc[55];
        oacc[55] <= oacc[56];
        oacc[56] <= oacc[57];
        oacc[57] <= oacc[58];
        oacc[58] <= oacc[59];
        oacc[59] <= oacc[48] + $signed(vp3);
        oacc[64] <= oacc[65];
        oacc[65] <= oacc[66];
        oacc[66] <= oacc[67];
        oacc[67] <= oacc[68];
        oacc[68] <= oacc[69];
        oacc[69] <= oacc[70];
        oacc[70] <= oacc[71];
        oacc[71] <= oacc[72];
        oacc[72] <= oacc[73];
        oacc[73] <= oacc[74];
        oacc[74] <= oacc[75];
        oacc[75] <= oacc[64] + $signed(vp4);
        oacc[80] <= oacc[81];
        oacc[81] <= oacc[82];
        oacc[82] <= oacc[83];
        oacc[83] <= oacc[84];
        oacc[84] <= oacc[85];
        oacc[85] <= oacc[86];
        oacc[86] <= oacc[87];
        oacc[87] <= oacc[88];
        oacc[88] <= oacc[89];
        oacc[89] <= oacc[90];
        oacc[90] <= oacc[91];
        oacc[91] <= oacc[80] + $signed(vp5);
        oacc[96] <= oacc[97];
        oacc[97] <= oacc[98];
        oacc[98] <= oacc[99];
        oacc[99] <= oacc[100];
        oacc[100] <= oacc[101];
        oacc[101] <= oacc[102];
        oacc[102] <= oacc[103];
        oacc[103] <= oacc[104];
        oacc[104] <= oacc[105];
        oacc[105] <= oacc[106];
        oacc[106] <= oacc[107];
        oacc[107] <= oacc[96] + $signed(vp6);
        oacc[112] <= oacc[113];
        oacc[113] <= oacc[114];
        oacc[114] <= oacc[115];
        oacc[115] <= oacc[116];
        oacc[116] <= oacc[117];
        oacc[117] <= oacc[118];
        oacc[118] <= oacc[119];
        oacc[119] <= oacc[120];
        oacc[120] <= oacc[121];
        oacc[121] <= oacc[122];
        oacc[122] <= oacc[123];
        oacc[123] <= oacc[112] + $signed(vp7);
      end
      if (vr2) oacc[12] <= oacc[12] + $signed({24'd0, ve});
    end
  end

  // ---- 出力 ----
  // 出す順は out[d]（d = 8b+i）、置き場は i*16 + b。ビットを入れ替えるだけ。
  // ocnt=96（分母）もこの式で 12 番地に当たる。
  wire [6:0] oaddr = {ocnt[2:0], ocnt[6:3]};

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

endmodule
