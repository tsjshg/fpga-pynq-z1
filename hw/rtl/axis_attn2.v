`timescale 1ns / 1ps
// =====================================================================
//  attention コア（softmax 入り）  ※ rtl/gen_attn2.py が生成。直接編集しない
//
//  ストリーム（1グループ = 1層の1 KV ヘッドぶん）:
//     [ヘッダ 8B: T][q: 3×64 B][K: T×64 B][V: T×64 B]
//  出力: out[h][d] を int32 で 192 個 + 分母 den[h] を 3 個 = 195 語
//
//  第1相 (K): s = Σ q·K、s8 = clamp(s >> 8) を BRAM へ。同時に mx[h] を追う
//  第2相 (V): e = EROM[mx[h] - s8]、out += e·V、den += e
//
//  ★ softmax は DDR を1バイトも余分に読まない。表引きと引き算だけ。
//  ★ 割り算はしない。分子と分母を出して下流に任せる（1グループに 3 回だけ）。
//
//  EROM[d] = round(255·exp(-d/8.0))、50 番地までが非零。
//  TAU は実重みの量子化スケールが決まるまでの暫定値。表を焼き直せば変わる。
//
//  掛け算器は第1相 24 個 + 第2相 24 個 = 48 個。
// =====================================================================
module axis_attn2 #(
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

  localparam integer BPR   = HD/8;           // 1つの t あたりのビート数 = 8
  localparam integer QB    = NQ*HD/8;        // q の取り込みビート数 = 24
  localparam integer NOUT  = NQ*HD;          // 出力の本体語数 = 192
  localparam integer NOTOT = NOUT + NQ;      // 分母を足した総語数 = 195
  localparam integer TW    = $clog2(TMAX+1);

  localparam [2:0] ST_HDR=3'd0, ST_Q=3'd1, ST_K=3'd2, ST_KDR=3'd3,
                   ST_V=3'd4, ST_DRN=3'd5, ST_OUT=3'd6;

  reg [2:0]     st;
  reg [TW-1:0]  Tn;                          // ヘッダで受け取る文脈長
  reg [TW-1:0]  tcnt;                        // 現在の t
  reg [2:0]     bcnt;                        // 行内のビート位置 0..BPR-1
  reg [7:0]     qcnt;                        // q 取り込みの進捗
  reg [8:0]     ocnt;                        // 出力の進捗 0..194
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

  // NQ ヘッドぶんを同時に引くので、ヘッドごとに別々に読み出す
  wire [63:0] qw0 = qm[0 + bcnt];
  wire [63:0] qw1 = qm[8 + bcnt];
  wire [63:0] qw2 = qm[16 + bcnt];

  // ---- スコアの置き場。第1相で書き、第2相で読む ----
  reg signed [7:0] sm0 [0:TMAX-1];
  reg signed [7:0] sm1 [0:TMAX-1];
  reg signed [7:0] sm2 [0:TMAX-1];

  // ---- 各ヘッドのスコア最大値。softmax の指数を負に寄せるために使う ----
  reg signed [7:0] mx0;
  reg signed [7:0] mx1;
  reg signed [7:0] mx2;

  // ---- 状態遷移 ----
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

  // clamp して 8bit に落とす。第2相ではこれで指数の表を引く
  function signed [7:0] clamp8(input signed [31:0] v);
    reg signed [31:0] s;
    begin
      s = v >>> 8;
      if (s >  127) clamp8 =  8'sd127;
      else if (s < -128) clamp8 = 8'sh80;
      else clamp8 = s[7:0];
    end
  endfunction

  // 【重要】32bit の加算と clamp を同じサイクルに置くと論理12段になって間に合わない
  // （5a の実測 WNS -0.134ns）。加算 → clamp → 書き込み+最大値 の3段に割る。
  // 段階5b では最大値の比較が増えたので、clamp と書き込みもさらに割った。
  reg signed [31:0] ksum0;
  reg signed [31:0] ksum1;
  reg signed [31:0] ksum2;
  reg [TW-1:0] kw_addr;  reg kw_en;
  always @(posedge aclk) begin
    if (!aresetn) kw_en <= 1'b0;
    else kw_en <= kv4 & ke4;
    ksum0 <= kacc0 + ks4_0_0;
    ksum1 <= kacc1 + ks4_1_0;
    ksum2 <= kacc2 + ks4_2_0;
    kw_addr <= kt4;
  end

  reg signed [7:0] kc0;
  reg signed [7:0] kc1;
  reg signed [7:0] kc2;
  reg [TW-1:0] kc_addr;  reg kc_en;
  always @(posedge aclk) begin
    if (!aresetn) kc_en <= 1'b0;
    else kc_en <= kw_en;
    kc0 <= clamp8(ksum0);
    kc1 <= clamp8(ksum1);
    kc2 <= clamp8(ksum2);
    kc_addr <= kw_addr;
  end

  // 置き場への書き込みと、最大値の更新。どちらも浅い。
  always @(posedge aclk) begin
    if (kc_en) begin
      sm0[kc_addr] <= kc0;
      sm1[kc_addr] <= kc1;
      sm2[kc_addr] <= kc2;
    end
  end
  always @(posedge aclk) begin
    if (!aresetn) begin mx0 <= 8'sh80; mx1 <= 8'sh80; mx2 <= 8'sh80; end
    else if (st==ST_HDR) begin mx0 <= 8'sh80; mx1 <= 8'sh80; mx2 <= 8'sh80; end
    else if (kc_en) begin
      if (kc0 > mx0) mx0 <= kc0;
      if (kc1 > mx1) mx1 <= kc1;
      if (kc2 > mx2) mx2 <= kc2;
    end
  end

  // ================= exp の表（256語 × NQ 本） =================
  // EROM[d] = round(255·exp(-d/8.0))。d = mx - s8 なので必ず 0..255 に収まる。
  // 50 番地から先は 0。3ヘッドが別々の番地を同時に引くので3本に複製する。
  reg [7:0] erom0 [0:255];
  reg [7:0] erom1 [0:255];
  reg [7:0] erom2 [0:255];
  initial begin
    erom0[  0]=8'd255; erom1[  0]=8'd255; erom2[  0]=8'd255;
    erom0[  1]=8'd225; erom1[  1]=8'd225; erom2[  1]=8'd225;
    erom0[  2]=8'd199; erom1[  2]=8'd199; erom2[  2]=8'd199;
    erom0[  3]=8'd175; erom1[  3]=8'd175; erom2[  3]=8'd175;
    erom0[  4]=8'd155; erom1[  4]=8'd155; erom2[  4]=8'd155;
    erom0[  5]=8'd136; erom1[  5]=8'd136; erom2[  5]=8'd136;
    erom0[  6]=8'd120; erom1[  6]=8'd120; erom2[  6]=8'd120;
    erom0[  7]=8'd106; erom1[  7]=8'd106; erom2[  7]=8'd106;
    erom0[  8]=8'd94; erom1[  8]=8'd94; erom2[  8]=8'd94;
    erom0[  9]=8'd83; erom1[  9]=8'd83; erom2[  9]=8'd83;
    erom0[ 10]=8'd73; erom1[ 10]=8'd73; erom2[ 10]=8'd73;
    erom0[ 11]=8'd64; erom1[ 11]=8'd64; erom2[ 11]=8'd64;
    erom0[ 12]=8'd57; erom1[ 12]=8'd57; erom2[ 12]=8'd57;
    erom0[ 13]=8'd50; erom1[ 13]=8'd50; erom2[ 13]=8'd50;
    erom0[ 14]=8'd44; erom1[ 14]=8'd44; erom2[ 14]=8'd44;
    erom0[ 15]=8'd39; erom1[ 15]=8'd39; erom2[ 15]=8'd39;
    erom0[ 16]=8'd35; erom1[ 16]=8'd35; erom2[ 16]=8'd35;
    erom0[ 17]=8'd30; erom1[ 17]=8'd30; erom2[ 17]=8'd30;
    erom0[ 18]=8'd27; erom1[ 18]=8'd27; erom2[ 18]=8'd27;
    erom0[ 19]=8'd24; erom1[ 19]=8'd24; erom2[ 19]=8'd24;
    erom0[ 20]=8'd21; erom1[ 20]=8'd21; erom2[ 20]=8'd21;
    erom0[ 21]=8'd18; erom1[ 21]=8'd18; erom2[ 21]=8'd18;
    erom0[ 22]=8'd16; erom1[ 22]=8'd16; erom2[ 22]=8'd16;
    erom0[ 23]=8'd14; erom1[ 23]=8'd14; erom2[ 23]=8'd14;
    erom0[ 24]=8'd13; erom1[ 24]=8'd13; erom2[ 24]=8'd13;
    erom0[ 25]=8'd11; erom1[ 25]=8'd11; erom2[ 25]=8'd11;
    erom0[ 26]=8'd10; erom1[ 26]=8'd10; erom2[ 26]=8'd10;
    erom0[ 27]=8'd9; erom1[ 27]=8'd9; erom2[ 27]=8'd9;
    erom0[ 28]=8'd8; erom1[ 28]=8'd8; erom2[ 28]=8'd8;
    erom0[ 29]=8'd7; erom1[ 29]=8'd7; erom2[ 29]=8'd7;
    erom0[ 30]=8'd6; erom1[ 30]=8'd6; erom2[ 30]=8'd6;
    erom0[ 31]=8'd5; erom1[ 31]=8'd5; erom2[ 31]=8'd5;
    erom0[ 32]=8'd5; erom1[ 32]=8'd5; erom2[ 32]=8'd5;
    erom0[ 33]=8'd4; erom1[ 33]=8'd4; erom2[ 33]=8'd4;
    erom0[ 34]=8'd4; erom1[ 34]=8'd4; erom2[ 34]=8'd4;
    erom0[ 35]=8'd3; erom1[ 35]=8'd3; erom2[ 35]=8'd3;
    erom0[ 36]=8'd3; erom1[ 36]=8'd3; erom2[ 36]=8'd3;
    erom0[ 37]=8'd2; erom1[ 37]=8'd2; erom2[ 37]=8'd2;
    erom0[ 38]=8'd2; erom1[ 38]=8'd2; erom2[ 38]=8'd2;
    erom0[ 39]=8'd2; erom1[ 39]=8'd2; erom2[ 39]=8'd2;
    erom0[ 40]=8'd2; erom1[ 40]=8'd2; erom2[ 40]=8'd2;
    erom0[ 41]=8'd2; erom1[ 41]=8'd2; erom2[ 41]=8'd2;
    erom0[ 42]=8'd1; erom1[ 42]=8'd1; erom2[ 42]=8'd1;
    erom0[ 43]=8'd1; erom1[ 43]=8'd1; erom2[ 43]=8'd1;
    erom0[ 44]=8'd1; erom1[ 44]=8'd1; erom2[ 44]=8'd1;
    erom0[ 45]=8'd1; erom1[ 45]=8'd1; erom2[ 45]=8'd1;
    erom0[ 46]=8'd1; erom1[ 46]=8'd1; erom2[ 46]=8'd1;
    erom0[ 47]=8'd1; erom1[ 47]=8'd1; erom2[ 47]=8'd1;
    erom0[ 48]=8'd1; erom1[ 48]=8'd1; erom2[ 48]=8'd1;
    erom0[ 49]=8'd1; erom1[ 49]=8'd1; erom2[ 49]=8'd1;
    erom0[ 50]=8'd0; erom1[ 50]=8'd0; erom2[ 50]=8'd0;
    erom0[ 51]=8'd0; erom1[ 51]=8'd0; erom2[ 51]=8'd0;
    erom0[ 52]=8'd0; erom1[ 52]=8'd0; erom2[ 52]=8'd0;
    erom0[ 53]=8'd0; erom1[ 53]=8'd0; erom2[ 53]=8'd0;
    erom0[ 54]=8'd0; erom1[ 54]=8'd0; erom2[ 54]=8'd0;
    erom0[ 55]=8'd0; erom1[ 55]=8'd0; erom2[ 55]=8'd0;
    erom0[ 56]=8'd0; erom1[ 56]=8'd0; erom2[ 56]=8'd0;
    erom0[ 57]=8'd0; erom1[ 57]=8'd0; erom2[ 57]=8'd0;
    erom0[ 58]=8'd0; erom1[ 58]=8'd0; erom2[ 58]=8'd0;
    erom0[ 59]=8'd0; erom1[ 59]=8'd0; erom2[ 59]=8'd0;
    erom0[ 60]=8'd0; erom1[ 60]=8'd0; erom2[ 60]=8'd0;
    erom0[ 61]=8'd0; erom1[ 61]=8'd0; erom2[ 61]=8'd0;
    erom0[ 62]=8'd0; erom1[ 62]=8'd0; erom2[ 62]=8'd0;
    erom0[ 63]=8'd0; erom1[ 63]=8'd0; erom2[ 63]=8'd0;
    erom0[ 64]=8'd0; erom1[ 64]=8'd0; erom2[ 64]=8'd0;
    erom0[ 65]=8'd0; erom1[ 65]=8'd0; erom2[ 65]=8'd0;
    erom0[ 66]=8'd0; erom1[ 66]=8'd0; erom2[ 66]=8'd0;
    erom0[ 67]=8'd0; erom1[ 67]=8'd0; erom2[ 67]=8'd0;
    erom0[ 68]=8'd0; erom1[ 68]=8'd0; erom2[ 68]=8'd0;
    erom0[ 69]=8'd0; erom1[ 69]=8'd0; erom2[ 69]=8'd0;
    erom0[ 70]=8'd0; erom1[ 70]=8'd0; erom2[ 70]=8'd0;
    erom0[ 71]=8'd0; erom1[ 71]=8'd0; erom2[ 71]=8'd0;
    erom0[ 72]=8'd0; erom1[ 72]=8'd0; erom2[ 72]=8'd0;
    erom0[ 73]=8'd0; erom1[ 73]=8'd0; erom2[ 73]=8'd0;
    erom0[ 74]=8'd0; erom1[ 74]=8'd0; erom2[ 74]=8'd0;
    erom0[ 75]=8'd0; erom1[ 75]=8'd0; erom2[ 75]=8'd0;
    erom0[ 76]=8'd0; erom1[ 76]=8'd0; erom2[ 76]=8'd0;
    erom0[ 77]=8'd0; erom1[ 77]=8'd0; erom2[ 77]=8'd0;
    erom0[ 78]=8'd0; erom1[ 78]=8'd0; erom2[ 78]=8'd0;
    erom0[ 79]=8'd0; erom1[ 79]=8'd0; erom2[ 79]=8'd0;
    erom0[ 80]=8'd0; erom1[ 80]=8'd0; erom2[ 80]=8'd0;
    erom0[ 81]=8'd0; erom1[ 81]=8'd0; erom2[ 81]=8'd0;
    erom0[ 82]=8'd0; erom1[ 82]=8'd0; erom2[ 82]=8'd0;
    erom0[ 83]=8'd0; erom1[ 83]=8'd0; erom2[ 83]=8'd0;
    erom0[ 84]=8'd0; erom1[ 84]=8'd0; erom2[ 84]=8'd0;
    erom0[ 85]=8'd0; erom1[ 85]=8'd0; erom2[ 85]=8'd0;
    erom0[ 86]=8'd0; erom1[ 86]=8'd0; erom2[ 86]=8'd0;
    erom0[ 87]=8'd0; erom1[ 87]=8'd0; erom2[ 87]=8'd0;
    erom0[ 88]=8'd0; erom1[ 88]=8'd0; erom2[ 88]=8'd0;
    erom0[ 89]=8'd0; erom1[ 89]=8'd0; erom2[ 89]=8'd0;
    erom0[ 90]=8'd0; erom1[ 90]=8'd0; erom2[ 90]=8'd0;
    erom0[ 91]=8'd0; erom1[ 91]=8'd0; erom2[ 91]=8'd0;
    erom0[ 92]=8'd0; erom1[ 92]=8'd0; erom2[ 92]=8'd0;
    erom0[ 93]=8'd0; erom1[ 93]=8'd0; erom2[ 93]=8'd0;
    erom0[ 94]=8'd0; erom1[ 94]=8'd0; erom2[ 94]=8'd0;
    erom0[ 95]=8'd0; erom1[ 95]=8'd0; erom2[ 95]=8'd0;
    erom0[ 96]=8'd0; erom1[ 96]=8'd0; erom2[ 96]=8'd0;
    erom0[ 97]=8'd0; erom1[ 97]=8'd0; erom2[ 97]=8'd0;
    erom0[ 98]=8'd0; erom1[ 98]=8'd0; erom2[ 98]=8'd0;
    erom0[ 99]=8'd0; erom1[ 99]=8'd0; erom2[ 99]=8'd0;
    erom0[100]=8'd0; erom1[100]=8'd0; erom2[100]=8'd0;
    erom0[101]=8'd0; erom1[101]=8'd0; erom2[101]=8'd0;
    erom0[102]=8'd0; erom1[102]=8'd0; erom2[102]=8'd0;
    erom0[103]=8'd0; erom1[103]=8'd0; erom2[103]=8'd0;
    erom0[104]=8'd0; erom1[104]=8'd0; erom2[104]=8'd0;
    erom0[105]=8'd0; erom1[105]=8'd0; erom2[105]=8'd0;
    erom0[106]=8'd0; erom1[106]=8'd0; erom2[106]=8'd0;
    erom0[107]=8'd0; erom1[107]=8'd0; erom2[107]=8'd0;
    erom0[108]=8'd0; erom1[108]=8'd0; erom2[108]=8'd0;
    erom0[109]=8'd0; erom1[109]=8'd0; erom2[109]=8'd0;
    erom0[110]=8'd0; erom1[110]=8'd0; erom2[110]=8'd0;
    erom0[111]=8'd0; erom1[111]=8'd0; erom2[111]=8'd0;
    erom0[112]=8'd0; erom1[112]=8'd0; erom2[112]=8'd0;
    erom0[113]=8'd0; erom1[113]=8'd0; erom2[113]=8'd0;
    erom0[114]=8'd0; erom1[114]=8'd0; erom2[114]=8'd0;
    erom0[115]=8'd0; erom1[115]=8'd0; erom2[115]=8'd0;
    erom0[116]=8'd0; erom1[116]=8'd0; erom2[116]=8'd0;
    erom0[117]=8'd0; erom1[117]=8'd0; erom2[117]=8'd0;
    erom0[118]=8'd0; erom1[118]=8'd0; erom2[118]=8'd0;
    erom0[119]=8'd0; erom1[119]=8'd0; erom2[119]=8'd0;
    erom0[120]=8'd0; erom1[120]=8'd0; erom2[120]=8'd0;
    erom0[121]=8'd0; erom1[121]=8'd0; erom2[121]=8'd0;
    erom0[122]=8'd0; erom1[122]=8'd0; erom2[122]=8'd0;
    erom0[123]=8'd0; erom1[123]=8'd0; erom2[123]=8'd0;
    erom0[124]=8'd0; erom1[124]=8'd0; erom2[124]=8'd0;
    erom0[125]=8'd0; erom1[125]=8'd0; erom2[125]=8'd0;
    erom0[126]=8'd0; erom1[126]=8'd0; erom2[126]=8'd0;
    erom0[127]=8'd0; erom1[127]=8'd0; erom2[127]=8'd0;
    erom0[128]=8'd0; erom1[128]=8'd0; erom2[128]=8'd0;
    erom0[129]=8'd0; erom1[129]=8'd0; erom2[129]=8'd0;
    erom0[130]=8'd0; erom1[130]=8'd0; erom2[130]=8'd0;
    erom0[131]=8'd0; erom1[131]=8'd0; erom2[131]=8'd0;
    erom0[132]=8'd0; erom1[132]=8'd0; erom2[132]=8'd0;
    erom0[133]=8'd0; erom1[133]=8'd0; erom2[133]=8'd0;
    erom0[134]=8'd0; erom1[134]=8'd0; erom2[134]=8'd0;
    erom0[135]=8'd0; erom1[135]=8'd0; erom2[135]=8'd0;
    erom0[136]=8'd0; erom1[136]=8'd0; erom2[136]=8'd0;
    erom0[137]=8'd0; erom1[137]=8'd0; erom2[137]=8'd0;
    erom0[138]=8'd0; erom1[138]=8'd0; erom2[138]=8'd0;
    erom0[139]=8'd0; erom1[139]=8'd0; erom2[139]=8'd0;
    erom0[140]=8'd0; erom1[140]=8'd0; erom2[140]=8'd0;
    erom0[141]=8'd0; erom1[141]=8'd0; erom2[141]=8'd0;
    erom0[142]=8'd0; erom1[142]=8'd0; erom2[142]=8'd0;
    erom0[143]=8'd0; erom1[143]=8'd0; erom2[143]=8'd0;
    erom0[144]=8'd0; erom1[144]=8'd0; erom2[144]=8'd0;
    erom0[145]=8'd0; erom1[145]=8'd0; erom2[145]=8'd0;
    erom0[146]=8'd0; erom1[146]=8'd0; erom2[146]=8'd0;
    erom0[147]=8'd0; erom1[147]=8'd0; erom2[147]=8'd0;
    erom0[148]=8'd0; erom1[148]=8'd0; erom2[148]=8'd0;
    erom0[149]=8'd0; erom1[149]=8'd0; erom2[149]=8'd0;
    erom0[150]=8'd0; erom1[150]=8'd0; erom2[150]=8'd0;
    erom0[151]=8'd0; erom1[151]=8'd0; erom2[151]=8'd0;
    erom0[152]=8'd0; erom1[152]=8'd0; erom2[152]=8'd0;
    erom0[153]=8'd0; erom1[153]=8'd0; erom2[153]=8'd0;
    erom0[154]=8'd0; erom1[154]=8'd0; erom2[154]=8'd0;
    erom0[155]=8'd0; erom1[155]=8'd0; erom2[155]=8'd0;
    erom0[156]=8'd0; erom1[156]=8'd0; erom2[156]=8'd0;
    erom0[157]=8'd0; erom1[157]=8'd0; erom2[157]=8'd0;
    erom0[158]=8'd0; erom1[158]=8'd0; erom2[158]=8'd0;
    erom0[159]=8'd0; erom1[159]=8'd0; erom2[159]=8'd0;
    erom0[160]=8'd0; erom1[160]=8'd0; erom2[160]=8'd0;
    erom0[161]=8'd0; erom1[161]=8'd0; erom2[161]=8'd0;
    erom0[162]=8'd0; erom1[162]=8'd0; erom2[162]=8'd0;
    erom0[163]=8'd0; erom1[163]=8'd0; erom2[163]=8'd0;
    erom0[164]=8'd0; erom1[164]=8'd0; erom2[164]=8'd0;
    erom0[165]=8'd0; erom1[165]=8'd0; erom2[165]=8'd0;
    erom0[166]=8'd0; erom1[166]=8'd0; erom2[166]=8'd0;
    erom0[167]=8'd0; erom1[167]=8'd0; erom2[167]=8'd0;
    erom0[168]=8'd0; erom1[168]=8'd0; erom2[168]=8'd0;
    erom0[169]=8'd0; erom1[169]=8'd0; erom2[169]=8'd0;
    erom0[170]=8'd0; erom1[170]=8'd0; erom2[170]=8'd0;
    erom0[171]=8'd0; erom1[171]=8'd0; erom2[171]=8'd0;
    erom0[172]=8'd0; erom1[172]=8'd0; erom2[172]=8'd0;
    erom0[173]=8'd0; erom1[173]=8'd0; erom2[173]=8'd0;
    erom0[174]=8'd0; erom1[174]=8'd0; erom2[174]=8'd0;
    erom0[175]=8'd0; erom1[175]=8'd0; erom2[175]=8'd0;
    erom0[176]=8'd0; erom1[176]=8'd0; erom2[176]=8'd0;
    erom0[177]=8'd0; erom1[177]=8'd0; erom2[177]=8'd0;
    erom0[178]=8'd0; erom1[178]=8'd0; erom2[178]=8'd0;
    erom0[179]=8'd0; erom1[179]=8'd0; erom2[179]=8'd0;
    erom0[180]=8'd0; erom1[180]=8'd0; erom2[180]=8'd0;
    erom0[181]=8'd0; erom1[181]=8'd0; erom2[181]=8'd0;
    erom0[182]=8'd0; erom1[182]=8'd0; erom2[182]=8'd0;
    erom0[183]=8'd0; erom1[183]=8'd0; erom2[183]=8'd0;
    erom0[184]=8'd0; erom1[184]=8'd0; erom2[184]=8'd0;
    erom0[185]=8'd0; erom1[185]=8'd0; erom2[185]=8'd0;
    erom0[186]=8'd0; erom1[186]=8'd0; erom2[186]=8'd0;
    erom0[187]=8'd0; erom1[187]=8'd0; erom2[187]=8'd0;
    erom0[188]=8'd0; erom1[188]=8'd0; erom2[188]=8'd0;
    erom0[189]=8'd0; erom1[189]=8'd0; erom2[189]=8'd0;
    erom0[190]=8'd0; erom1[190]=8'd0; erom2[190]=8'd0;
    erom0[191]=8'd0; erom1[191]=8'd0; erom2[191]=8'd0;
    erom0[192]=8'd0; erom1[192]=8'd0; erom2[192]=8'd0;
    erom0[193]=8'd0; erom1[193]=8'd0; erom2[193]=8'd0;
    erom0[194]=8'd0; erom1[194]=8'd0; erom2[194]=8'd0;
    erom0[195]=8'd0; erom1[195]=8'd0; erom2[195]=8'd0;
    erom0[196]=8'd0; erom1[196]=8'd0; erom2[196]=8'd0;
    erom0[197]=8'd0; erom1[197]=8'd0; erom2[197]=8'd0;
    erom0[198]=8'd0; erom1[198]=8'd0; erom2[198]=8'd0;
    erom0[199]=8'd0; erom1[199]=8'd0; erom2[199]=8'd0;
    erom0[200]=8'd0; erom1[200]=8'd0; erom2[200]=8'd0;
    erom0[201]=8'd0; erom1[201]=8'd0; erom2[201]=8'd0;
    erom0[202]=8'd0; erom1[202]=8'd0; erom2[202]=8'd0;
    erom0[203]=8'd0; erom1[203]=8'd0; erom2[203]=8'd0;
    erom0[204]=8'd0; erom1[204]=8'd0; erom2[204]=8'd0;
    erom0[205]=8'd0; erom1[205]=8'd0; erom2[205]=8'd0;
    erom0[206]=8'd0; erom1[206]=8'd0; erom2[206]=8'd0;
    erom0[207]=8'd0; erom1[207]=8'd0; erom2[207]=8'd0;
    erom0[208]=8'd0; erom1[208]=8'd0; erom2[208]=8'd0;
    erom0[209]=8'd0; erom1[209]=8'd0; erom2[209]=8'd0;
    erom0[210]=8'd0; erom1[210]=8'd0; erom2[210]=8'd0;
    erom0[211]=8'd0; erom1[211]=8'd0; erom2[211]=8'd0;
    erom0[212]=8'd0; erom1[212]=8'd0; erom2[212]=8'd0;
    erom0[213]=8'd0; erom1[213]=8'd0; erom2[213]=8'd0;
    erom0[214]=8'd0; erom1[214]=8'd0; erom2[214]=8'd0;
    erom0[215]=8'd0; erom1[215]=8'd0; erom2[215]=8'd0;
    erom0[216]=8'd0; erom1[216]=8'd0; erom2[216]=8'd0;
    erom0[217]=8'd0; erom1[217]=8'd0; erom2[217]=8'd0;
    erom0[218]=8'd0; erom1[218]=8'd0; erom2[218]=8'd0;
    erom0[219]=8'd0; erom1[219]=8'd0; erom2[219]=8'd0;
    erom0[220]=8'd0; erom1[220]=8'd0; erom2[220]=8'd0;
    erom0[221]=8'd0; erom1[221]=8'd0; erom2[221]=8'd0;
    erom0[222]=8'd0; erom1[222]=8'd0; erom2[222]=8'd0;
    erom0[223]=8'd0; erom1[223]=8'd0; erom2[223]=8'd0;
    erom0[224]=8'd0; erom1[224]=8'd0; erom2[224]=8'd0;
    erom0[225]=8'd0; erom1[225]=8'd0; erom2[225]=8'd0;
    erom0[226]=8'd0; erom1[226]=8'd0; erom2[226]=8'd0;
    erom0[227]=8'd0; erom1[227]=8'd0; erom2[227]=8'd0;
    erom0[228]=8'd0; erom1[228]=8'd0; erom2[228]=8'd0;
    erom0[229]=8'd0; erom1[229]=8'd0; erom2[229]=8'd0;
    erom0[230]=8'd0; erom1[230]=8'd0; erom2[230]=8'd0;
    erom0[231]=8'd0; erom1[231]=8'd0; erom2[231]=8'd0;
    erom0[232]=8'd0; erom1[232]=8'd0; erom2[232]=8'd0;
    erom0[233]=8'd0; erom1[233]=8'd0; erom2[233]=8'd0;
    erom0[234]=8'd0; erom1[234]=8'd0; erom2[234]=8'd0;
    erom0[235]=8'd0; erom1[235]=8'd0; erom2[235]=8'd0;
    erom0[236]=8'd0; erom1[236]=8'd0; erom2[236]=8'd0;
    erom0[237]=8'd0; erom1[237]=8'd0; erom2[237]=8'd0;
    erom0[238]=8'd0; erom1[238]=8'd0; erom2[238]=8'd0;
    erom0[239]=8'd0; erom1[239]=8'd0; erom2[239]=8'd0;
    erom0[240]=8'd0; erom1[240]=8'd0; erom2[240]=8'd0;
    erom0[241]=8'd0; erom1[241]=8'd0; erom2[241]=8'd0;
    erom0[242]=8'd0; erom1[242]=8'd0; erom2[242]=8'd0;
    erom0[243]=8'd0; erom1[243]=8'd0; erom2[243]=8'd0;
    erom0[244]=8'd0; erom1[244]=8'd0; erom2[244]=8'd0;
    erom0[245]=8'd0; erom1[245]=8'd0; erom2[245]=8'd0;
    erom0[246]=8'd0; erom1[246]=8'd0; erom2[246]=8'd0;
    erom0[247]=8'd0; erom1[247]=8'd0; erom2[247]=8'd0;
    erom0[248]=8'd0; erom1[248]=8'd0; erom2[248]=8'd0;
    erom0[249]=8'd0; erom1[249]=8'd0; erom2[249]=8'd0;
    erom0[250]=8'd0; erom1[250]=8'd0; erom2[250]=8'd0;
    erom0[251]=8'd0; erom1[251]=8'd0; erom2[251]=8'd0;
    erom0[252]=8'd0; erom1[252]=8'd0; erom2[252]=8'd0;
    erom0[253]=8'd0; erom1[253]=8'd0; erom2[253]=8'd0;
    erom0[254]=8'd0; erom1[254]=8'd0; erom2[254]=8'd0;
    erom0[255]=8'd0; erom1[255]=8'd0; erom2[255]=8'd0;
  end

  // ================= 第2相: softmax + AV =================
  // 内積ではなく外積の累算。1ビートの V 8バイトが 8 個の別々の累算器に入る。
  // 【重要】データと有効ビットの段数を数え合わせること（5a で1つずれて全滅した）。
  //   データ  : s_axis_tdata → vw0(1) → vw1(2) → vw2(3) → vp(4)
  //   指数    : sm[tcnt] → sr(1) → vd(2) → ve(3)
  //   有効    : in_v → vv0(1) → vv1(2) → vv2(3) → vv3(4)
  //   行の頭  : v_row0 → vr0(1) → vr1(2) → vr2(3)  ← ve と同じ3段
  reg [63:0] vw0, vw1, vw2;
  reg vv0, vv1, vv2, vv3;
  reg vr0, vr1, vr2;
  reg signed [7:0] sr0;
  reg signed [7:0] sr1;
  reg signed [7:0] sr2;
  reg [7:0] vd0;
  reg [7:0] vd1;
  reg [7:0] vd2;
  reg [7:0] ve0;
  reg [7:0] ve1;
  reg [7:0] ve2;
  always @(posedge aclk) begin
    if (!aresetn) begin vv0<=1'b0; vv1<=1'b0; vv2<=1'b0; vv3<=1'b0;
                        vr0<=1'b0; vr1<=1'b0; vr2<=1'b0; end
    else begin vv0<=in_v; vv1<=vv0; vv2<=vv1; vv3<=vv2;
               vr0<=v_row0; vr1<=vr0; vr2<=vr1; end
    vw0 <= s_axis_tdata;  vw1 <= vw0;  vw2 <= vw1;
    sr0 <= sm0[tcnt];
    sr1 <= sm1[tcnt];
    sr2 <= sm2[tcnt];
    // mx は必ずその行の s8 以上なので、差は 0..255。9bit 引き算の下位8bit でよい。
    vd0 <= mx0 - sr0;
    vd1 <= mx1 - sr1;
    vd2 <= mx2 - sr2;
    // ★ ここが softmax。BRAM を1回引くだけ。DDR には触らない。
    ve0 <= erom0[vd0];
    ve1 <= erom1[vd1];
    ve2 <= erom2[vd2];
  end

  // 4段目: 積（8個 × NQ）。V は符号付き8bit、指数は符号なし8bit。
  (* use_dsp = "yes" *) reg signed [17:0] vp0_0,vp0_1,vp0_2,vp0_3,vp0_4,vp0_5,vp0_6,vp0_7;
  (* use_dsp = "yes" *) reg signed [17:0] vp1_0,vp1_1,vp1_2,vp1_3,vp1_4,vp1_5,vp1_6,vp1_7;
  (* use_dsp = "yes" *) reg signed [17:0] vp2_0,vp2_1,vp2_2,vp2_3,vp2_4,vp2_5,vp2_6,vp2_7;
  always @(posedge aclk) begin
    vp0_0 <= $signed(vw2[7:0]) * $signed({1'b0, ve0});
    vp0_1 <= $signed(vw2[15:8]) * $signed({1'b0, ve0});
    vp0_2 <= $signed(vw2[23:16]) * $signed({1'b0, ve0});
    vp0_3 <= $signed(vw2[31:24]) * $signed({1'b0, ve0});
    vp0_4 <= $signed(vw2[39:32]) * $signed({1'b0, ve0});
    vp0_5 <= $signed(vw2[47:40]) * $signed({1'b0, ve0});
    vp0_6 <= $signed(vw2[55:48]) * $signed({1'b0, ve0});
    vp0_7 <= $signed(vw2[63:56]) * $signed({1'b0, ve0});
    vp1_0 <= $signed(vw2[7:0]) * $signed({1'b0, ve1});
    vp1_1 <= $signed(vw2[15:8]) * $signed({1'b0, ve1});
    vp1_2 <= $signed(vw2[23:16]) * $signed({1'b0, ve1});
    vp1_3 <= $signed(vw2[31:24]) * $signed({1'b0, ve1});
    vp1_4 <= $signed(vw2[39:32]) * $signed({1'b0, ve1});
    vp1_5 <= $signed(vw2[47:40]) * $signed({1'b0, ve1});
    vp1_6 <= $signed(vw2[55:48]) * $signed({1'b0, ve1});
    vp1_7 <= $signed(vw2[63:56]) * $signed({1'b0, ve1});
    vp2_0 <= $signed(vw2[7:0]) * $signed({1'b0, ve2});
    vp2_1 <= $signed(vw2[15:8]) * $signed({1'b0, ve2});
    vp2_2 <= $signed(vw2[23:16]) * $signed({1'b0, ve2});
    vp2_3 <= $signed(vw2[31:24]) * $signed({1'b0, ve2});
    vp2_4 <= $signed(vw2[39:32]) * $signed({1'b0, ve2});
    vp2_5 <= $signed(vw2[47:40]) * $signed({1'b0, ve2});
    vp2_6 <= $signed(vw2[55:48]) * $signed({1'b0, ve2});
    vp2_7 <= $signed(vw2[63:56]) * $signed({1'b0, ve2});
  end

  // 5段目: 累算器へ。
  // 【重要】ビート位置 b で添字を引く書き方にすると、b が 192 個の累算器の
  // アドレス解読に扇形に広がって配線が伸びる（5a の実測 WNS -0.480ns で落ちた）。
  // b は 0→7 を順に回るだけなので、アドレスで選ばず累算器のほうを回す。
  // 配置は oacc[h*64 + i*8 + b]。b が回転位置。
  //
  // 分母は [192, 200, 208] 番地に置く。出力アドレスの入れ替え式にこの番地が
  // そのまま出てくるので、出力マルチプレクサに 2:1 を足さなくて済む。
  reg signed [31:0] oacc [0:208];
  integer oi;
  always @(posedge aclk) begin
    if (st==ST_HDR) begin
      for (oi=0; oi<209; oi=oi+1) oacc[oi] <= 32'sd0;
    end else begin
      if (vv3) begin
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
      if (vr2) begin
        oacc[192] <= oacc[192] + $signed({24'd0, ve0});
        oacc[200] <= oacc[200] + $signed({24'd0, ve1});
        oacc[208] <= oacc[208] + $signed({24'd0, ve2});
      end
    end
  end

  // ---- 出力 ----
  // 出す順は out[h][d]（d = 8b+i）だが、置き場は h*64 + i*8 + b。
  // ocnt の中の b と i のビット位置を入れ替えるだけで引ける。
  // ocnt = 192..194（分母）もこの式を通すと [192, 200, 208] になる。
  wire [7:0] oaddr = {ocnt[7:6], ocnt[2:0], ocnt[5:3]};

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
