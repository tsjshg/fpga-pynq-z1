`timescale 1ns / 1ps
// =====================================================================
//  三値 (BitNet b1.58) MAC アレイ  ※ rtl/gen_tmac.py が生成。直接編集しない
//
//  段階3の axis_mac.v（INT8・8 MAC）の置き換え。
//  1バイトに三値を5個詰めるので、同じ 1 バイトから 5 回の演算が出る。
//  掛け算は「足す・引く・何もしない」の3択なので DSP を使わない。
//
//  ストリームの形式（段階3と同じ考え方）:
//     [ x: 640 バイト (INT8) ][ 重み: 128 バイト/行 × R 行 ]
//  1行 = 三値 640 個 = 128 バイト。1ビート8バイトで 40 個ぶん進む。
//
//  x の読み出しは 5 個のメモリに分けてある。1ビートで x を 40 個
//  （= 64bit 語で5語ぶん）同時に読む必要があり、分散RAM は1語ずつしか
//  読めないため。語番号を 5 で割った余りで分ければ、5つとも同じアドレスで
//  引けるので、アドレス計算は1本で済む。
// =====================================================================
module axis_tmac #(
  parameter integer L = 640                 // 1行の重み数。40 の倍数
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

  localparam integer WPB = 40;                 // 1ビートあたりの重み数
  localparam integer BPR = L/WPB;                 // 1行あたりのビート数
  localparam integer HB  = L/8;                   // x の取り込みに要するビート数
  localparam integer AW  = (BPR <= 1) ? 1 : $clog2(BPR);
  localparam integer NS  = 9;                     // 累算段に届くまでの段数

  assign s_axis_tready = 1'b1;                    // 上流は絶対に止めない
  wire fire = s_axis_tvalid & s_axis_tready;

  // ---- 3進復号表。1バイト → 三値5個（2bitずつ、00=0 / 01=+1 / 10=-1） ----
  function [9:0] dec3 (input [7:0] v);
    case (v)
      8'd0: dec3 = 10'b1010101010;
      8'd1: dec3 = 10'b1010101000;
      8'd2: dec3 = 10'b1010101001;
      8'd3: dec3 = 10'b1010100010;
      8'd4: dec3 = 10'b1010100000;
      8'd5: dec3 = 10'b1010100001;
      8'd6: dec3 = 10'b1010100110;
      8'd7: dec3 = 10'b1010100100;
      8'd8: dec3 = 10'b1010100101;
      8'd9: dec3 = 10'b1010001010;
      8'd10: dec3 = 10'b1010001000;
      8'd11: dec3 = 10'b1010001001;
      8'd12: dec3 = 10'b1010000010;
      8'd13: dec3 = 10'b1010000000;
      8'd14: dec3 = 10'b1010000001;
      8'd15: dec3 = 10'b1010000110;
      8'd16: dec3 = 10'b1010000100;
      8'd17: dec3 = 10'b1010000101;
      8'd18: dec3 = 10'b1010011010;
      8'd19: dec3 = 10'b1010011000;
      8'd20: dec3 = 10'b1010011001;
      8'd21: dec3 = 10'b1010010010;
      8'd22: dec3 = 10'b1010010000;
      8'd23: dec3 = 10'b1010010001;
      8'd24: dec3 = 10'b1010010110;
      8'd25: dec3 = 10'b1010010100;
      8'd26: dec3 = 10'b1010010101;
      8'd27: dec3 = 10'b1000101010;
      8'd28: dec3 = 10'b1000101000;
      8'd29: dec3 = 10'b1000101001;
      8'd30: dec3 = 10'b1000100010;
      8'd31: dec3 = 10'b1000100000;
      8'd32: dec3 = 10'b1000100001;
      8'd33: dec3 = 10'b1000100110;
      8'd34: dec3 = 10'b1000100100;
      8'd35: dec3 = 10'b1000100101;
      8'd36: dec3 = 10'b1000001010;
      8'd37: dec3 = 10'b1000001000;
      8'd38: dec3 = 10'b1000001001;
      8'd39: dec3 = 10'b1000000010;
      8'd40: dec3 = 10'b1000000000;
      8'd41: dec3 = 10'b1000000001;
      8'd42: dec3 = 10'b1000000110;
      8'd43: dec3 = 10'b1000000100;
      8'd44: dec3 = 10'b1000000101;
      8'd45: dec3 = 10'b1000011010;
      8'd46: dec3 = 10'b1000011000;
      8'd47: dec3 = 10'b1000011001;
      8'd48: dec3 = 10'b1000010010;
      8'd49: dec3 = 10'b1000010000;
      8'd50: dec3 = 10'b1000010001;
      8'd51: dec3 = 10'b1000010110;
      8'd52: dec3 = 10'b1000010100;
      8'd53: dec3 = 10'b1000010101;
      8'd54: dec3 = 10'b1001101010;
      8'd55: dec3 = 10'b1001101000;
      8'd56: dec3 = 10'b1001101001;
      8'd57: dec3 = 10'b1001100010;
      8'd58: dec3 = 10'b1001100000;
      8'd59: dec3 = 10'b1001100001;
      8'd60: dec3 = 10'b1001100110;
      8'd61: dec3 = 10'b1001100100;
      8'd62: dec3 = 10'b1001100101;
      8'd63: dec3 = 10'b1001001010;
      8'd64: dec3 = 10'b1001001000;
      8'd65: dec3 = 10'b1001001001;
      8'd66: dec3 = 10'b1001000010;
      8'd67: dec3 = 10'b1001000000;
      8'd68: dec3 = 10'b1001000001;
      8'd69: dec3 = 10'b1001000110;
      8'd70: dec3 = 10'b1001000100;
      8'd71: dec3 = 10'b1001000101;
      8'd72: dec3 = 10'b1001011010;
      8'd73: dec3 = 10'b1001011000;
      8'd74: dec3 = 10'b1001011001;
      8'd75: dec3 = 10'b1001010010;
      8'd76: dec3 = 10'b1001010000;
      8'd77: dec3 = 10'b1001010001;
      8'd78: dec3 = 10'b1001010110;
      8'd79: dec3 = 10'b1001010100;
      8'd80: dec3 = 10'b1001010101;
      8'd81: dec3 = 10'b0010101010;
      8'd82: dec3 = 10'b0010101000;
      8'd83: dec3 = 10'b0010101001;
      8'd84: dec3 = 10'b0010100010;
      8'd85: dec3 = 10'b0010100000;
      8'd86: dec3 = 10'b0010100001;
      8'd87: dec3 = 10'b0010100110;
      8'd88: dec3 = 10'b0010100100;
      8'd89: dec3 = 10'b0010100101;
      8'd90: dec3 = 10'b0010001010;
      8'd91: dec3 = 10'b0010001000;
      8'd92: dec3 = 10'b0010001001;
      8'd93: dec3 = 10'b0010000010;
      8'd94: dec3 = 10'b0010000000;
      8'd95: dec3 = 10'b0010000001;
      8'd96: dec3 = 10'b0010000110;
      8'd97: dec3 = 10'b0010000100;
      8'd98: dec3 = 10'b0010000101;
      8'd99: dec3 = 10'b0010011010;
      8'd100: dec3 = 10'b0010011000;
      8'd101: dec3 = 10'b0010011001;
      8'd102: dec3 = 10'b0010010010;
      8'd103: dec3 = 10'b0010010000;
      8'd104: dec3 = 10'b0010010001;
      8'd105: dec3 = 10'b0010010110;
      8'd106: dec3 = 10'b0010010100;
      8'd107: dec3 = 10'b0010010101;
      8'd108: dec3 = 10'b0000101010;
      8'd109: dec3 = 10'b0000101000;
      8'd110: dec3 = 10'b0000101001;
      8'd111: dec3 = 10'b0000100010;
      8'd112: dec3 = 10'b0000100000;
      8'd113: dec3 = 10'b0000100001;
      8'd114: dec3 = 10'b0000100110;
      8'd115: dec3 = 10'b0000100100;
      8'd116: dec3 = 10'b0000100101;
      8'd117: dec3 = 10'b0000001010;
      8'd118: dec3 = 10'b0000001000;
      8'd119: dec3 = 10'b0000001001;
      8'd120: dec3 = 10'b0000000010;
      8'd121: dec3 = 10'b0000000000;
      8'd122: dec3 = 10'b0000000001;
      8'd123: dec3 = 10'b0000000110;
      8'd124: dec3 = 10'b0000000100;
      8'd125: dec3 = 10'b0000000101;
      8'd126: dec3 = 10'b0000011010;
      8'd127: dec3 = 10'b0000011000;
      8'd128: dec3 = 10'b0000011001;
      8'd129: dec3 = 10'b0000010010;
      8'd130: dec3 = 10'b0000010000;
      8'd131: dec3 = 10'b0000010001;
      8'd132: dec3 = 10'b0000010110;
      8'd133: dec3 = 10'b0000010100;
      8'd134: dec3 = 10'b0000010101;
      8'd135: dec3 = 10'b0001101010;
      8'd136: dec3 = 10'b0001101000;
      8'd137: dec3 = 10'b0001101001;
      8'd138: dec3 = 10'b0001100010;
      8'd139: dec3 = 10'b0001100000;
      8'd140: dec3 = 10'b0001100001;
      8'd141: dec3 = 10'b0001100110;
      8'd142: dec3 = 10'b0001100100;
      8'd143: dec3 = 10'b0001100101;
      8'd144: dec3 = 10'b0001001010;
      8'd145: dec3 = 10'b0001001000;
      8'd146: dec3 = 10'b0001001001;
      8'd147: dec3 = 10'b0001000010;
      8'd148: dec3 = 10'b0001000000;
      8'd149: dec3 = 10'b0001000001;
      8'd150: dec3 = 10'b0001000110;
      8'd151: dec3 = 10'b0001000100;
      8'd152: dec3 = 10'b0001000101;
      8'd153: dec3 = 10'b0001011010;
      8'd154: dec3 = 10'b0001011000;
      8'd155: dec3 = 10'b0001011001;
      8'd156: dec3 = 10'b0001010010;
      8'd157: dec3 = 10'b0001010000;
      8'd158: dec3 = 10'b0001010001;
      8'd159: dec3 = 10'b0001010110;
      8'd160: dec3 = 10'b0001010100;
      8'd161: dec3 = 10'b0001010101;
      8'd162: dec3 = 10'b0110101010;
      8'd163: dec3 = 10'b0110101000;
      8'd164: dec3 = 10'b0110101001;
      8'd165: dec3 = 10'b0110100010;
      8'd166: dec3 = 10'b0110100000;
      8'd167: dec3 = 10'b0110100001;
      8'd168: dec3 = 10'b0110100110;
      8'd169: dec3 = 10'b0110100100;
      8'd170: dec3 = 10'b0110100101;
      8'd171: dec3 = 10'b0110001010;
      8'd172: dec3 = 10'b0110001000;
      8'd173: dec3 = 10'b0110001001;
      8'd174: dec3 = 10'b0110000010;
      8'd175: dec3 = 10'b0110000000;
      8'd176: dec3 = 10'b0110000001;
      8'd177: dec3 = 10'b0110000110;
      8'd178: dec3 = 10'b0110000100;
      8'd179: dec3 = 10'b0110000101;
      8'd180: dec3 = 10'b0110011010;
      8'd181: dec3 = 10'b0110011000;
      8'd182: dec3 = 10'b0110011001;
      8'd183: dec3 = 10'b0110010010;
      8'd184: dec3 = 10'b0110010000;
      8'd185: dec3 = 10'b0110010001;
      8'd186: dec3 = 10'b0110010110;
      8'd187: dec3 = 10'b0110010100;
      8'd188: dec3 = 10'b0110010101;
      8'd189: dec3 = 10'b0100101010;
      8'd190: dec3 = 10'b0100101000;
      8'd191: dec3 = 10'b0100101001;
      8'd192: dec3 = 10'b0100100010;
      8'd193: dec3 = 10'b0100100000;
      8'd194: dec3 = 10'b0100100001;
      8'd195: dec3 = 10'b0100100110;
      8'd196: dec3 = 10'b0100100100;
      8'd197: dec3 = 10'b0100100101;
      8'd198: dec3 = 10'b0100001010;
      8'd199: dec3 = 10'b0100001000;
      8'd200: dec3 = 10'b0100001001;
      8'd201: dec3 = 10'b0100000010;
      8'd202: dec3 = 10'b0100000000;
      8'd203: dec3 = 10'b0100000001;
      8'd204: dec3 = 10'b0100000110;
      8'd205: dec3 = 10'b0100000100;
      8'd206: dec3 = 10'b0100000101;
      8'd207: dec3 = 10'b0100011010;
      8'd208: dec3 = 10'b0100011000;
      8'd209: dec3 = 10'b0100011001;
      8'd210: dec3 = 10'b0100010010;
      8'd211: dec3 = 10'b0100010000;
      8'd212: dec3 = 10'b0100010001;
      8'd213: dec3 = 10'b0100010110;
      8'd214: dec3 = 10'b0100010100;
      8'd215: dec3 = 10'b0100010101;
      8'd216: dec3 = 10'b0101101010;
      8'd217: dec3 = 10'b0101101000;
      8'd218: dec3 = 10'b0101101001;
      8'd219: dec3 = 10'b0101100010;
      8'd220: dec3 = 10'b0101100000;
      8'd221: dec3 = 10'b0101100001;
      8'd222: dec3 = 10'b0101100110;
      8'd223: dec3 = 10'b0101100100;
      8'd224: dec3 = 10'b0101100101;
      8'd225: dec3 = 10'b0101001010;
      8'd226: dec3 = 10'b0101001000;
      8'd227: dec3 = 10'b0101001001;
      8'd228: dec3 = 10'b0101000010;
      8'd229: dec3 = 10'b0101000000;
      8'd230: dec3 = 10'b0101000001;
      8'd231: dec3 = 10'b0101000110;
      8'd232: dec3 = 10'b0101000100;
      8'd233: dec3 = 10'b0101000101;
      8'd234: dec3 = 10'b0101011010;
      8'd235: dec3 = 10'b0101011000;
      8'd236: dec3 = 10'b0101011001;
      8'd237: dec3 = 10'b0101010010;
      8'd238: dec3 = 10'b0101010000;
      8'd239: dec3 = 10'b0101010001;
      8'd240: dec3 = 10'b0101010110;
      8'd241: dec3 = 10'b0101010100;
      8'd242: dec3 = 10'b0101010101;
      8'd243: dec3 = 10'b1010101010;
      8'd244: dec3 = 10'b1010101000;
      8'd245: dec3 = 10'b1010101001;
      8'd246: dec3 = 10'b1010100010;
      8'd247: dec3 = 10'b1010100000;
      8'd248: dec3 = 10'b1010100001;
      8'd249: dec3 = 10'b1010100110;
      8'd250: dec3 = 10'b1010100100;
      8'd251: dec3 = 10'b1010100101;
      8'd252: dec3 = 10'b1010001010;
      8'd253: dec3 = 10'b1010001000;
      8'd254: dec3 = 10'b1010001001;
      8'd255: dec3 = 10'b1010000010;
      default: dec3 = 10'b0;
    endcase
  endfunction

  // ---- x の格納。64bit 語を 5 本のメモリに撒く ----
  reg [63:0] xm0 [0:BPR-1];
  reg [63:0] xm1 [0:BPR-1];
  reg [63:0] xm2 [0:BPR-1];
  reg [63:0] xm3 [0:BPR-1];
  reg [63:0] xm4 [0:BPR-1];

  reg [2:0]     hj;          // 語番号 mod 5
  reg [AW-1:0]  ha;          // 語番号 / 5
  reg [AW-1:0]  bc;          // 行内のビート位置
  reg           in_header;

  always @(posedge aclk) begin
    if (!aresetn) begin
      in_header <= 1'b1; hj <= 3'd0; ha <= {AW{1'b0}}; bc <= {AW{1'b0}};
    end else if (fire) begin
      if (s_axis_tlast) begin
        in_header <= 1'b1; hj <= 3'd0; ha <= {AW{1'b0}}; bc <= {AW{1'b0}};
      end else if (in_header) begin
        if (hj == 3'd4) begin
          hj <= 3'd0;
          if (ha == BPR-1) begin in_header <= 1'b0; ha <= {AW{1'b0}}; end
          else ha <= ha + 1'b1;
        end else hj <= hj + 1'b1;
      end else begin
        bc <= (bc == BPR-1) ? {AW{1'b0}} : bc + 1'b1;
      end
    end
  end

  wire hw = fire & in_header;
  always @(posedge aclk) begin
    if (hw && hj == 3'd0) xm0[ha] <= s_axis_tdata;
    if (hw && hj == 3'd1) xm1[ha] <= s_axis_tdata;
    if (hw && hj == 3'd2) xm2[ha] <= s_axis_tdata;
    if (hw && hj == 3'd3) xm3[ha] <= s_axis_tdata;
    if (hw && hj == 3'd4) xm4[ha] <= s_axis_tdata;
  end

  wire [319:0] xv = {xm4[bc], xm3[bc], xm2[bc], xm1[bc], xm0[bc]};

  wire compute = fire & ~in_header;
  wire row_end = compute & (bc == BPR-1);

  // ---- 制御フラグを段数ぶん遅らせる ----
  reg [NS-1:0] v_sr, e_sr, l_sr;
  always @(posedge aclk) begin
    if (!aresetn) begin v_sr <= 0; e_sr <= 0; l_sr <= 0; end
    else begin
      v_sr <= {v_sr[NS-2:0], compute};
      e_sr <= {e_sr[NS-2:0], row_end};
      l_sr <= {l_sr[NS-2:0], compute & s_axis_tlast};
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

    t1[9:0] <= dec3(w0[7:0]);
    t1[19:10] <= dec3(w0[15:8]);
    t1[29:20] <= dec3(w0[23:16]);
    t1[39:30] <= dec3(w0[31:24]);
    t1[49:40] <= dec3(w0[39:32]);
    t1[59:50] <= dec3(w0[47:40]);
    t1[69:60] <= dec3(w0[55:48]);
    t1[79:70] <= dec3(w0[63:56]);
  end

  // ---- 2段目: 項の生成。掛け算ではなく「そのまま／符号反転／ゼロ」の3択 ----
  // 9bit に広げてから反転する。x=-128 でも桁あふれしない。
  reg signed [8:0] u0;
  reg signed [8:0] u1;
  reg signed [8:0] u2;
  reg signed [8:0] u3;
  reg signed [8:0] u4;
  reg signed [8:0] u5;
  reg signed [8:0] u6;
  reg signed [8:0] u7;
  reg signed [8:0] u8;
  reg signed [8:0] u9;
  reg signed [8:0] u10;
  reg signed [8:0] u11;
  reg signed [8:0] u12;
  reg signed [8:0] u13;
  reg signed [8:0] u14;
  reg signed [8:0] u15;
  reg signed [8:0] u16;
  reg signed [8:0] u17;
  reg signed [8:0] u18;
  reg signed [8:0] u19;
  reg signed [8:0] u20;
  reg signed [8:0] u21;
  reg signed [8:0] u22;
  reg signed [8:0] u23;
  reg signed [8:0] u24;
  reg signed [8:0] u25;
  reg signed [8:0] u26;
  reg signed [8:0] u27;
  reg signed [8:0] u28;
  reg signed [8:0] u29;
  reg signed [8:0] u30;
  reg signed [8:0] u31;
  reg signed [8:0] u32;
  reg signed [8:0] u33;
  reg signed [8:0] u34;
  reg signed [8:0] u35;
  reg signed [8:0] u36;
  reg signed [8:0] u37;
  reg signed [8:0] u38;
  reg signed [8:0] u39;
  always @(posedge aclk) begin
    case (t1[1:0])
      2'b01:   u0 <=  $signed({x1[7], x1[7:0]});
      2'b10:   u0 <= -$signed({x1[7], x1[7:0]});
      default: u0 <= 9'sd0;
    endcase
    case (t1[3:2])
      2'b01:   u1 <=  $signed({x1[15], x1[15:8]});
      2'b10:   u1 <= -$signed({x1[15], x1[15:8]});
      default: u1 <= 9'sd0;
    endcase
    case (t1[5:4])
      2'b01:   u2 <=  $signed({x1[23], x1[23:16]});
      2'b10:   u2 <= -$signed({x1[23], x1[23:16]});
      default: u2 <= 9'sd0;
    endcase
    case (t1[7:6])
      2'b01:   u3 <=  $signed({x1[31], x1[31:24]});
      2'b10:   u3 <= -$signed({x1[31], x1[31:24]});
      default: u3 <= 9'sd0;
    endcase
    case (t1[9:8])
      2'b01:   u4 <=  $signed({x1[39], x1[39:32]});
      2'b10:   u4 <= -$signed({x1[39], x1[39:32]});
      default: u4 <= 9'sd0;
    endcase
    case (t1[11:10])
      2'b01:   u5 <=  $signed({x1[47], x1[47:40]});
      2'b10:   u5 <= -$signed({x1[47], x1[47:40]});
      default: u5 <= 9'sd0;
    endcase
    case (t1[13:12])
      2'b01:   u6 <=  $signed({x1[55], x1[55:48]});
      2'b10:   u6 <= -$signed({x1[55], x1[55:48]});
      default: u6 <= 9'sd0;
    endcase
    case (t1[15:14])
      2'b01:   u7 <=  $signed({x1[63], x1[63:56]});
      2'b10:   u7 <= -$signed({x1[63], x1[63:56]});
      default: u7 <= 9'sd0;
    endcase
    case (t1[17:16])
      2'b01:   u8 <=  $signed({x1[71], x1[71:64]});
      2'b10:   u8 <= -$signed({x1[71], x1[71:64]});
      default: u8 <= 9'sd0;
    endcase
    case (t1[19:18])
      2'b01:   u9 <=  $signed({x1[79], x1[79:72]});
      2'b10:   u9 <= -$signed({x1[79], x1[79:72]});
      default: u9 <= 9'sd0;
    endcase
    case (t1[21:20])
      2'b01:   u10 <=  $signed({x1[87], x1[87:80]});
      2'b10:   u10 <= -$signed({x1[87], x1[87:80]});
      default: u10 <= 9'sd0;
    endcase
    case (t1[23:22])
      2'b01:   u11 <=  $signed({x1[95], x1[95:88]});
      2'b10:   u11 <= -$signed({x1[95], x1[95:88]});
      default: u11 <= 9'sd0;
    endcase
    case (t1[25:24])
      2'b01:   u12 <=  $signed({x1[103], x1[103:96]});
      2'b10:   u12 <= -$signed({x1[103], x1[103:96]});
      default: u12 <= 9'sd0;
    endcase
    case (t1[27:26])
      2'b01:   u13 <=  $signed({x1[111], x1[111:104]});
      2'b10:   u13 <= -$signed({x1[111], x1[111:104]});
      default: u13 <= 9'sd0;
    endcase
    case (t1[29:28])
      2'b01:   u14 <=  $signed({x1[119], x1[119:112]});
      2'b10:   u14 <= -$signed({x1[119], x1[119:112]});
      default: u14 <= 9'sd0;
    endcase
    case (t1[31:30])
      2'b01:   u15 <=  $signed({x1[127], x1[127:120]});
      2'b10:   u15 <= -$signed({x1[127], x1[127:120]});
      default: u15 <= 9'sd0;
    endcase
    case (t1[33:32])
      2'b01:   u16 <=  $signed({x1[135], x1[135:128]});
      2'b10:   u16 <= -$signed({x1[135], x1[135:128]});
      default: u16 <= 9'sd0;
    endcase
    case (t1[35:34])
      2'b01:   u17 <=  $signed({x1[143], x1[143:136]});
      2'b10:   u17 <= -$signed({x1[143], x1[143:136]});
      default: u17 <= 9'sd0;
    endcase
    case (t1[37:36])
      2'b01:   u18 <=  $signed({x1[151], x1[151:144]});
      2'b10:   u18 <= -$signed({x1[151], x1[151:144]});
      default: u18 <= 9'sd0;
    endcase
    case (t1[39:38])
      2'b01:   u19 <=  $signed({x1[159], x1[159:152]});
      2'b10:   u19 <= -$signed({x1[159], x1[159:152]});
      default: u19 <= 9'sd0;
    endcase
    case (t1[41:40])
      2'b01:   u20 <=  $signed({x1[167], x1[167:160]});
      2'b10:   u20 <= -$signed({x1[167], x1[167:160]});
      default: u20 <= 9'sd0;
    endcase
    case (t1[43:42])
      2'b01:   u21 <=  $signed({x1[175], x1[175:168]});
      2'b10:   u21 <= -$signed({x1[175], x1[175:168]});
      default: u21 <= 9'sd0;
    endcase
    case (t1[45:44])
      2'b01:   u22 <=  $signed({x1[183], x1[183:176]});
      2'b10:   u22 <= -$signed({x1[183], x1[183:176]});
      default: u22 <= 9'sd0;
    endcase
    case (t1[47:46])
      2'b01:   u23 <=  $signed({x1[191], x1[191:184]});
      2'b10:   u23 <= -$signed({x1[191], x1[191:184]});
      default: u23 <= 9'sd0;
    endcase
    case (t1[49:48])
      2'b01:   u24 <=  $signed({x1[199], x1[199:192]});
      2'b10:   u24 <= -$signed({x1[199], x1[199:192]});
      default: u24 <= 9'sd0;
    endcase
    case (t1[51:50])
      2'b01:   u25 <=  $signed({x1[207], x1[207:200]});
      2'b10:   u25 <= -$signed({x1[207], x1[207:200]});
      default: u25 <= 9'sd0;
    endcase
    case (t1[53:52])
      2'b01:   u26 <=  $signed({x1[215], x1[215:208]});
      2'b10:   u26 <= -$signed({x1[215], x1[215:208]});
      default: u26 <= 9'sd0;
    endcase
    case (t1[55:54])
      2'b01:   u27 <=  $signed({x1[223], x1[223:216]});
      2'b10:   u27 <= -$signed({x1[223], x1[223:216]});
      default: u27 <= 9'sd0;
    endcase
    case (t1[57:56])
      2'b01:   u28 <=  $signed({x1[231], x1[231:224]});
      2'b10:   u28 <= -$signed({x1[231], x1[231:224]});
      default: u28 <= 9'sd0;
    endcase
    case (t1[59:58])
      2'b01:   u29 <=  $signed({x1[239], x1[239:232]});
      2'b10:   u29 <= -$signed({x1[239], x1[239:232]});
      default: u29 <= 9'sd0;
    endcase
    case (t1[61:60])
      2'b01:   u30 <=  $signed({x1[247], x1[247:240]});
      2'b10:   u30 <= -$signed({x1[247], x1[247:240]});
      default: u30 <= 9'sd0;
    endcase
    case (t1[63:62])
      2'b01:   u31 <=  $signed({x1[255], x1[255:248]});
      2'b10:   u31 <= -$signed({x1[255], x1[255:248]});
      default: u31 <= 9'sd0;
    endcase
    case (t1[65:64])
      2'b01:   u32 <=  $signed({x1[263], x1[263:256]});
      2'b10:   u32 <= -$signed({x1[263], x1[263:256]});
      default: u32 <= 9'sd0;
    endcase
    case (t1[67:66])
      2'b01:   u33 <=  $signed({x1[271], x1[271:264]});
      2'b10:   u33 <= -$signed({x1[271], x1[271:264]});
      default: u33 <= 9'sd0;
    endcase
    case (t1[69:68])
      2'b01:   u34 <=  $signed({x1[279], x1[279:272]});
      2'b10:   u34 <= -$signed({x1[279], x1[279:272]});
      default: u34 <= 9'sd0;
    endcase
    case (t1[71:70])
      2'b01:   u35 <=  $signed({x1[287], x1[287:280]});
      2'b10:   u35 <= -$signed({x1[287], x1[287:280]});
      default: u35 <= 9'sd0;
    endcase
    case (t1[73:72])
      2'b01:   u36 <=  $signed({x1[295], x1[295:288]});
      2'b10:   u36 <= -$signed({x1[295], x1[295:288]});
      default: u36 <= 9'sd0;
    endcase
    case (t1[75:74])
      2'b01:   u37 <=  $signed({x1[303], x1[303:296]});
      2'b10:   u37 <= -$signed({x1[303], x1[303:296]});
      default: u37 <= 9'sd0;
    endcase
    case (t1[77:76])
      2'b01:   u38 <=  $signed({x1[311], x1[311:304]});
      2'b10:   u38 <= -$signed({x1[311], x1[311:304]});
      default: u38 <= 9'sd0;
    endcase
    case (t1[79:78])
      2'b01:   u39 <=  $signed({x1[319], x1[319:312]});
      2'b10:   u39 <= -$signed({x1[319], x1[319:312]});
      default: u39 <= 9'sd0;
    endcase
  end

  // ---- 3〜8段目: 40項の加算木。1段に1レベルだけ置く ----
  // 3段目: 40 → 20 項 (10bit)
  reg signed [9:0] s3_0;
  reg signed [9:0] s3_1;
  reg signed [9:0] s3_2;
  reg signed [9:0] s3_3;
  reg signed [9:0] s3_4;
  reg signed [9:0] s3_5;
  reg signed [9:0] s3_6;
  reg signed [9:0] s3_7;
  reg signed [9:0] s3_8;
  reg signed [9:0] s3_9;
  reg signed [9:0] s3_10;
  reg signed [9:0] s3_11;
  reg signed [9:0] s3_12;
  reg signed [9:0] s3_13;
  reg signed [9:0] s3_14;
  reg signed [9:0] s3_15;
  reg signed [9:0] s3_16;
  reg signed [9:0] s3_17;
  reg signed [9:0] s3_18;
  reg signed [9:0] s3_19;
  always @(posedge aclk) begin
    s3_0 <= u0 + u1;
    s3_1 <= u2 + u3;
    s3_2 <= u4 + u5;
    s3_3 <= u6 + u7;
    s3_4 <= u8 + u9;
    s3_5 <= u10 + u11;
    s3_6 <= u12 + u13;
    s3_7 <= u14 + u15;
    s3_8 <= u16 + u17;
    s3_9 <= u18 + u19;
    s3_10 <= u20 + u21;
    s3_11 <= u22 + u23;
    s3_12 <= u24 + u25;
    s3_13 <= u26 + u27;
    s3_14 <= u28 + u29;
    s3_15 <= u30 + u31;
    s3_16 <= u32 + u33;
    s3_17 <= u34 + u35;
    s3_18 <= u36 + u37;
    s3_19 <= u38 + u39;
  end

  // 4段目: 20 → 10 項 (11bit)
  reg signed [10:0] s4_0;
  reg signed [10:0] s4_1;
  reg signed [10:0] s4_2;
  reg signed [10:0] s4_3;
  reg signed [10:0] s4_4;
  reg signed [10:0] s4_5;
  reg signed [10:0] s4_6;
  reg signed [10:0] s4_7;
  reg signed [10:0] s4_8;
  reg signed [10:0] s4_9;
  always @(posedge aclk) begin
    s4_0 <= s3_0 + s3_1;
    s4_1 <= s3_2 + s3_3;
    s4_2 <= s3_4 + s3_5;
    s4_3 <= s3_6 + s3_7;
    s4_4 <= s3_8 + s3_9;
    s4_5 <= s3_10 + s3_11;
    s4_6 <= s3_12 + s3_13;
    s4_7 <= s3_14 + s3_15;
    s4_8 <= s3_16 + s3_17;
    s4_9 <= s3_18 + s3_19;
  end

  // 5段目: 10 → 5 項 (12bit)
  reg signed [11:0] s5_0;
  reg signed [11:0] s5_1;
  reg signed [11:0] s5_2;
  reg signed [11:0] s5_3;
  reg signed [11:0] s5_4;
  always @(posedge aclk) begin
    s5_0 <= s4_0 + s4_1;
    s5_1 <= s4_2 + s4_3;
    s5_2 <= s4_4 + s4_5;
    s5_3 <= s4_6 + s4_7;
    s5_4 <= s4_8 + s4_9;
  end

  // 6段目: 5 → 3 項 (13bit)
  reg signed [12:0] s6_0;
  reg signed [12:0] s6_1;
  reg signed [12:0] s6_2;
  always @(posedge aclk) begin
    s6_0 <= s5_0 + s5_1;
    s6_1 <= s5_2 + s5_3;
    s6_2 <= s5_4;
  end

  // 7段目: 3 → 2 項 (14bit)
  reg signed [13:0] s7_0;
  reg signed [13:0] s7_1;
  always @(posedge aclk) begin
    s7_0 <= s6_0 + s6_1;
    s7_1 <= s6_2;
  end

  // 8段目: 2 → 1 項 (15bit)
  reg signed [14:0] s8_0;
  always @(posedge aclk) begin
    s8_0 <= s7_0 + s7_1;
  end

  // ---- 累算。行末（または TLAST）で吐く ----
  // 三値 640 項 × |x|≤127 なら最大 81,280。32bit で十分。
  reg signed [31:0] acc, out_d;
  reg out_v, out_l;

  always @(posedge aclk) begin
    if (!aresetn) begin
      acc <= 32'sd0; out_v <= 1'b0; out_l <= 1'b0; out_d <= 32'sd0;
    end else begin
      out_v <= 1'b0;
      if (v_sr[NS-1]) begin
        if (e_sr[NS-1] | l_sr[NS-1]) begin
          out_d <= acc + s8_0;
          out_v <= 1'b1;
          out_l <= l_sr[NS-1];
          acc   <= 32'sd0;
        end else begin
          acc <= acc + s8_0;
        end
      end
    end
  end

  assign m_axis_tdata  = out_d;
  assign m_axis_tvalid = out_v;
  assign m_axis_tlast  = out_l;

endmodule
