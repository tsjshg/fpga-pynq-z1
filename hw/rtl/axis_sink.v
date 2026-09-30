`timescale 1ns / 1ps
// =====================================================================
//  AXI4-Stream ヌルシンク
//
//  受け取ったビートを捨てるだけの回路。tready を常に 1 にするので、
//  上流の AXI DMA (MM2S) はバックプレッシャを一切受けず、
//  DDR から読める限りの速度で読み続ける。
//  これで「書き戻し」の分の DDR トラフィックが消え、
//  読み専用の帯域が測れる。
//
//  X_INTERFACE_INFO を明示しているのは、命名規則まかせの推論に頼ると
//  IPI がストリームインタフェースとして束ねてくれないことがあるため。
// =====================================================================
module axis_sink #(
  parameter integer DW = 64
)(
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis, ASSOCIATED_RESET aresetn" *)
  input  wire              aclk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
  (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input  wire              aresetn,

  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA"  *)
  input  wire [DW-1:0]     s_axis_tdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TKEEP"  *)
  input  wire [DW/8-1:0]   s_axis_tkeep,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TLAST"  *)
  input  wire              s_axis_tlast,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *)
  input  wire              s_axis_tvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *)
  output wire              s_axis_tready
);

  // 常に受け取れる。これがこの回路の全て。
  assign s_axis_tready = 1'b1;

  // 受け取ったデータを 1 ビットに畳んで残しておく。値そのものに意味は無いが、
  // これが無いと合成時に上流の TDATA 配線ごと消えてしまう。
  // DONT_TOUCH でこのレジスタを固定し、その入力側（XOR 木と TDATA）を守る。
  (* DONT_TOUCH = "true" *) reg accum;
  always @(posedge aclk) begin
    if (!aresetn)                           accum <= 1'b0;
    else if (s_axis_tvalid & s_axis_tready) accum <= accum ^ (^s_axis_tdata);
  end

endmodule
