# =====================================================================
#  PYNQ-Z1 / XC7Z020  段階3: INT8 MAC アレイ（HP 1本）
#
#    PS7 -M_AXI_GP0-> AXI DMA (制御)
#    AXI DMA -MM2S-> axis_mac -> AXIS FIFO -> AXI DMA S2MM
#    AXI DMA -M_AXI_MM2S/S2MM-> SmartConnect -> PS7 S_AXI_HP0 -> DDR
#
#  ヌルシンク版 (build_sink.tcl) の axis_sink を axis_mac に差し替えたもの。
#  問いは一つだけ: 計算させても段階2の 1.13 GB/s（HP 1本）を保てるか。
#
#  **2本ではなく1本で試すのは、そのほうが厳しい試験になるから。**
#  1本のときは道幅の 98.7% に張り付いているので、演算パイプラインが
#  少しでも詰まれば即座に数字に出る。2本だと道幅に 14% の余裕があり、
#  小さな失速を隠してしまう。
#
#  結果の書き戻しは入力の 1/128（L=512、4バイト/行）しかないので、
#  S2MM を復活させても帯域はほぼ食われない。
#
#  使い方: vivado -mode batch -source build_mac.tcl -tclargs <bd|all> <FCLK MHz>
# =====================================================================
set stage "bd"
set fclk  100
if {$argc > 0} { set stage [lindex $argv 0] }
if {$argc > 1} { set fclk  [lindex $argv 1] }
puts "### stage = $stage / FCLK 要求 = $fclk MHz"

set part      xc7z020clg400-1
set proj      dmamac${fclk}
set bd        design_1
set outdir    [file normalize "./out"]
set tag       mac${fclk}
file mkdir $outdir

create_project $proj ./$proj -part $part -force
add_files -norecurse ./rtl/axis_mac.v
update_compile_order -fileset sources_1
create_bd_design $bd

# ---------- PS7（533項目は build.tcl と共通） ----------
source ./ps7_common.tcl

# 実際に得られる FCLK は PLL の分周で決まるので、要求どおりとは限らない。
# 帯域の計算に使う値なので、Vivado が確定させた実周波数を控えておく。
set fclk_act [get_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $ps7_0]
set fclk_rep [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps7_0]
puts "### FCLK_CLK0 実周波数 = $fclk_rep MHz (要求 $fclk)"

# ---------- AXI DMA（読み 64bit / 書き 32bit / Scatter-Gather 無し） ----------
# 読みと書きは別のマスタなので、幅は独立に決められる。
# 結果は入力の 1/128 しか出ないので、書き側は 32bit で足りる。
set dma [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0 ]
set_property -dict [list \
  CONFIG.c_include_sg {0} \
  CONFIG.c_sg_include_stscntrl_strm {0} \
  CONFIG.c_include_mm2s {1} \
  CONFIG.c_include_s2mm {1} \
  CONFIG.c_include_mm2s_dre {0} \
  CONFIG.c_include_s2mm_dre {0} \
  CONFIG.c_m_axi_mm2s_data_width {64} \
  CONFIG.c_m_axis_mm2s_tdata_width {64} \
  CONFIG.c_mm2s_burst_size {256} \
  CONFIG.c_m_axi_s2mm_data_width {32} \
  CONFIG.c_s_axis_s2mm_tdata_width {32} \
  CONFIG.c_s2mm_burst_size {16} \
  CONFIG.c_sg_length_width {26} \
] $dma

# ---------- MAC アレイ ----------
set mac [ create_bd_cell -type module -reference axis_mac axis_mac_0 ]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins axis_mac_0/s_axis]

# 結果を受ける FIFO。MAC は tready を常に 1 にして上流を止めないので、
# 下流が一瞬でも詰まると結果を落とす。出る量は入力の 1/128 なので
# 1024段あれば絶対に埋まらないが、保険として置く。
set ofifo [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo axis_data_fifo_0 ]
set_property -dict [list CONFIG.FIFO_DEPTH {1024} CONFIG.HAS_TLAST {1} \
                         CONFIG.TDATA_NUM_BYTES {4} CONFIG.HAS_TKEEP {0}] $ofifo
connect_bd_intf_net [get_bd_intf_pins axis_mac_0/m_axis]      [get_bd_intf_pins axis_data_fifo_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_data_fifo_0/M_AXIS] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# ---------- 接続の自動配線 ----------
# 制御: PS M_AXI_GP0 -> DMA S_AXI_LITE
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/ps7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} \
            ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}} \
  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# データ: DMA MM2S -> PS S_AXI_HP0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/axi_dma_0/M_AXI_MM2S} Slave {/ps7_0/S_AXI_HP0} \
            ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}} \
  [get_bd_intf_pins ps7_0/S_AXI_HP0]

# 書き側 S2MM も同じ HP0 へ。上で立てた SmartConnect にぶら下げる。
set smc [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:smartconnect:*"}] 0]
if {$smc eq ""} { error "SmartConnect が見つかりません" }
puts "### データバス: $smc"
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
           Master {/axi_dma_0/M_AXI_S2MM} Slave {/ps7_0/S_AXI_HP0} \
           ddr_seg {Auto} intc_ip {$smc} master_apm {0}" \
  [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]

# MAC と FIFO のクロックとリセットは手で結ぶ。
# apply_bd_automation は AXI インタフェースしか面倒を見ないので、
# ストリームだけで繋いだセルのクロック系は自動配線されない。
# proc_sys_reset はセル名が自動生成なので VLNV で検索する。
set rstcell [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:proc_sys_reset:*"}] 0]
if {$rstcell eq ""} { error "proc_sys_reset セルが見つかりません" }
puts "### リセット源: $rstcell"
connect_bd_net [get_bd_pins axis_mac_0/aclk]    [get_bd_pins ps7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axis_mac_0/aresetn] [get_bd_pins $rstcell/peripheral_aresetn]
connect_bd_net [get_bd_pins axis_data_fifo_0/s_axis_aclk]    [get_bd_pins ps7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axis_data_fifo_0/s_axis_aresetn] [get_bd_pins $rstcell/peripheral_aresetn]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

puts "### ブロックデザインの検証を通過"
puts "### アドレスマップ:"
foreach s [get_bd_addr_segs -quiet] { puts "###   $s" }

if {$stage ne "all"} { puts "### stage=bd のためここで終了"; exit 0 }

# ---------- 合成〜ビットストリーム ----------
make_wrapper -files [get_files ./$proj/$proj.srcs/sources_1/bd/$bd/$bd.bd] -top
add_files -norecurse ./$proj/$proj.gen/sources_1/bd/$bd/hdl/${bd}_wrapper.v
set_property top ${bd}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# -jobs は控えめに。Vivado は IP ごとに別プロセスを立て、1本あたり約2.8GB使う。
# 8並列だとコンテナの 14.6GB を超えて OOM killer に殺される（実際に殺された）。
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
  puts "### 実装に失敗しました"
  exit 1
}

file copy -force ./$proj/$proj.runs/impl_1/${bd}_wrapper.bit $outdir/${tag}.bit
set hwh [glob -nocomplain ./$proj/$proj.gen/sources_1/bd/$bd/hw_handoff/${bd}.hwh]
if {$hwh eq ""} { set hwh [glob -nocomplain ./$proj/$proj.srcs/sources_1/bd/$bd/hw_handoff/${bd}.hwh] }
file copy -force [lindex $hwh 0] $outdir/${tag}.hwh

set wns [get_property STATS.WNS [get_runs impl_1]]
puts "### 完了: $outdir/${tag}.bit と ${tag}.hwh"
puts "### FCLK 実周波数 = $fclk_rep MHz / タイミング WNS = $wns ns"

# 何個の DSP を実際に使ったかを見る。8 MAC のはずだが、
# 8x8 の掛け算は LUT で組まれることもあるので実物を確認する。
open_run impl_1
report_utilization -file $outdir/${tag}_util.txt
# STATS.* は open_run 後だと取れないので、レポートを読んで要点だけ出す。
set fh [open $outdir/${tag}_util.txt r]
set rpt [read $fh]; close $fh
puts "### 資源使用量:"
foreach {label pat} {
  LUT      {\|\s+Slice LUTs\s+\|\s+(\d+)\s+\|}
  FF       {\|\s+Slice Registers\s+\|\s+(\d+)\s+\|}
  LUTRAM   {\|\s+LUT as Distributed RAM\s+\|\s+(\d+)\s+\|}
  BRAM     {\|\s+Block RAM Tile\s+\|\s+([\d.]+)\s+\|}
  DSP      {\|\s+DSPs\s+\|\s+(\d+)\s+\|}
} {
  if {[regexp $pat $rpt -> v]} { puts "###   $label = $v" }
}
puts "### 詳細は $outdir/${tag}_util.txt"
if {$wns < 0} { puts "### 警告: タイミング未達。この周波数の測定値は信用できない" }
exit 0
