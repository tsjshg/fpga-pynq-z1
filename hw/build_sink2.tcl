# =====================================================================
#  PYNQ-Z1 / XC7Z020  読み専用 PL→DDR 帯域測定（HP ポート2本版）
#
#    PS7 -M_AXI_GP0-> AXI DMA ×2 (制御)
#    AXI DMA n -MM2S-> axis_sink n      (受け取って捨てるだけ)
#    AXI DMA 0 -M_AXI_MM2S-> SmartConnect -> PS7 S_AXI_HP0 -> DDR
#    AXI DMA 1 -M_AXI_MM2S-> SmartConnect -> PS7 S_AXI_HP2 -> DDR
#
#  1本版 (build_sink.tcl) では 100/142.86 MHz とも道幅の 99% に張り付き、
#  DDR は半分しか使えていなかった。200 MHz はタイミング未達。
#  そこで幅のほうを倍にする。142.86 MHz × 8B × 2本 = 2.28 GB/s ぶんの道幅で、
#  DDR の理論値 2.1 GB/s を初めて上回る。ここで頭打ちになれば、それが DDR の壁。
#
#  HP0 と HP2 を選ぶのは、Zynq の PS 側スイッチで別ポートに繋がるため
#  （HP0/HP1 が対、HP2/HP3 が対）。同じ対を使うと入口で先に詰まる。
#
#  使い方: vivado -mode batch -source build_sink2.tcl -tclargs <bd|all> <FCLK MHz>
# =====================================================================
set stage "bd"
set fclk  100
if {$argc > 0} { set stage [lindex $argv 0] }
if {$argc > 1} { set fclk  [lindex $argv 1] }
puts "### stage = $stage / FCLK 要求 = $fclk MHz"

set part      xc7z020clg400-1
set proj      dmasink2_${fclk}
set bd        design_1
set outdir    [file normalize "./out"]
set tag       sink2_${fclk}
file mkdir $outdir

create_project $proj ./$proj -part $part -force
add_files -norecurse ./rtl/axis_sink.v
update_compile_order -fileset sources_1
create_bd_design $bd

# ---------- PS7（533項目は build.tcl と共通） ----------
source ./ps7_common.tcl

# HP2 を追加で開ける。ps7_common.tcl は HP0 しか有効にしていない。
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP2 {1}] $ps7_0

# 実際に得られる FCLK は PLL の分周で決まるので、要求どおりとは限らない。
# 帯域の計算に使う値なので、Vivado が確定させた実周波数を控えておく。
set fclk_act [get_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $ps7_0]
set fclk_rep [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps7_0]
puts "### FCLK_CLK0 実周波数 = $fclk_rep MHz (要求 $fclk)"

# ---------- AXI DMA ×2（MM2S のみ / Scatter-Gather 無し） ----------
foreach n {0 1} {
  set dma [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_$n ]
  set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_include_mm2s_dre {0} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_mm2s_burst_size {256} \
    CONFIG.c_sg_length_width {26} \
  ] $dma

  create_bd_cell -type module -reference axis_sink axis_sink_$n
  connect_bd_intf_net [get_bd_intf_pins axi_dma_$n/M_AXIS_MM2S] \
                      [get_bd_intf_pins axis_sink_$n/s_axis]
}

# ---------- 接続の自動配線 ----------
# 制御: PS M_AXI_GP0 -> DMA ×2 の S_AXI_LITE
# 1本目で AXI Interconnect を新設し、2本目はそこにぶら下げる。
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/ps7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} \
            ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}} \
  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

set gpintc [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:axi_interconnect:*"}] 0]
if {$gpintc eq ""} { error "GP0 側の AXI Interconnect が見つかりません" }
puts "### 制御バス: $gpintc"
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
           Master {/ps7_0/M_AXI_GP0} Slave {/axi_dma_1/S_AXI_LITE} \
           ddr_seg {Auto} intc_ip {$gpintc} master_apm {0}" \
  [get_bd_intf_pins axi_dma_1/S_AXI_LITE]

# データ: DMA0 -> HP0 / DMA1 -> HP2。別々の SmartConnect を立てる。
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/axi_dma_0/M_AXI_MM2S} Slave {/ps7_0/S_AXI_HP0} \
            ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}} \
  [get_bd_intf_pins ps7_0/S_AXI_HP0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/axi_dma_1/M_AXI_MM2S} Slave {/ps7_0/S_AXI_HP2} \
            ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}} \
  [get_bd_intf_pins ps7_0/S_AXI_HP2]

# シンクのクロックとリセットは手で結ぶ。
# apply_bd_automation は AXI インタフェースしか面倒を見ないので、
# ストリームだけで繋いだセルのクロック系は自動配線されない。
# proc_sys_reset はセル名が自動生成なので VLNV で検索する。
set rstcell [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:proc_sys_reset:*"}] 0]
if {$rstcell eq ""} { error "proc_sys_reset セルが見つかりません" }
puts "### リセット源: $rstcell"
foreach n {0 1} {
  connect_bd_net [get_bd_pins axis_sink_$n/aclk]    [get_bd_pins ps7_0/FCLK_CLK0]
  connect_bd_net [get_bd_pins axis_sink_$n/aresetn] [get_bd_pins $rstcell/peripheral_aresetn]
}

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
if {$wns < 0} { puts "### 警告: タイミング未達。この周波数の測定値は信用できない" }
exit 0
