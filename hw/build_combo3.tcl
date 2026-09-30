# =====================================================================
#  PYNQ-Z1 / XC7Z020  段階11: 通しで動かすための同居版（HP 2本）
#
#    axi_dma_0 -> axis_tmacv_0 (三値・行長可変) -> FIFO -> dma_0 S2MM ┐
#    axi_dma_2 -> axis_attnv_0 (attention)      -> FIFO -> dma_2 S2MM ┴→ SmartConnect → HP0
#    axi_dma_1 -> axis_tmacv_1                                        ┐
#    axi_dma_3 -> axis_attnv_1                                        ┴→ SmartConnect → HP2
#
#  段階10（build_combo2.tcl）との違いは **attention 側の DMA 2台を SG（散布収集）にした**ことだけ。
#  コアは1ビットも変えていない。
#
#  【なぜ SG か】KV キャッシュは1トークンごとに伸びる。attention のストリームは
#    [見出し][q][K: T 行][V: T 行] × ヘッド
#  なので、T が1増えると **V の位置と、後ろのヘッド全部の位置がずれる**。
#  単純転送（1転送 = 連続した1領域）のままだと、毎トークン KV を丸ごと詰め直すか、
#  ヘッドごとに転送を切る（1トークン 384 回）しかない。
#  SG なら K と V を別々の領域に置いたまま、記述子の鎖で1本のストリームに綴じられる。
#  毎トークン書き換えるのは記述子の長さ欄だけ。
#
#  使い方: vivado -mode batch -source build_combo3.tcl -tclargs <bd|all> <FCLK MHz>
# =====================================================================
set stage "bd"
set fclk  125
if {$argc > 0} { set stage [lindex $argv 0] }
if {$argc > 1} { set fclk  [lindex $argv 1] }
puts "### stage = $stage / FCLK 要求 = $fclk MHz"

set part      xc7z020clg400-1
set proj      dmacombo3_${fclk}
set bd        design_1
set outdir    [file normalize "./out"]
set tag       combo3_${fclk}
file mkdir $outdir

create_project $proj ./$proj -part $part -force
add_files -norecurse ./rtl/axis_tmacv.v
add_files -norecurse ./rtl/axis_attnv.v
update_compile_order -fileset sources_1
create_bd_design $bd

source ./ps7_common.tcl
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP2 {1}] $ps7_0
set fclk_rep [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps7_0]
puts "### FCLK_CLK0 実周波数 = $fclk_rep MHz (要求 $fclk)"

# ---------- DMA 4台 + コア 4個 + FIFO 4個 ----------
# 0,1 = 行列積（三値）/ 2,3 = attention
set kinds {axis_tmacv axis_tmacv axis_attnv axis_attnv}
foreach n {0 1 2 3} {
  set dma [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_$n ]
  set_property -dict [list \
    CONFIG.c_include_sg [expr {$n >= 2 ? 1 : 0}] \
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
    CONFIG.c_sg_length_width {23} \
  ] $dma

  set kind [lindex $kinds $n]
  create_bd_cell -type module -reference $kind ${kind}_$n
  set f [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo axis_data_fifo_$n ]
  set_property -dict [list CONFIG.FIFO_DEPTH {1024} CONFIG.HAS_TLAST {1} \
                           CONFIG.TDATA_NUM_BYTES {4} CONFIG.HAS_TKEEP {0}] $f

  connect_bd_intf_net [get_bd_intf_pins axi_dma_$n/M_AXIS_MM2S]   [get_bd_intf_pins ${kind}_$n/s_axis]
  connect_bd_intf_net [get_bd_intf_pins ${kind}_$n/m_axis]        [get_bd_intf_pins axis_data_fifo_$n/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins axis_data_fifo_$n/M_AXIS] [get_bd_intf_pins axi_dma_$n/S_AXIS_S2MM]
}

# ---------- 制御: GP0 -> 4台の S_AXI_LITE ----------
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/ps7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} \
            ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}} \
  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
set gpintc [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:axi_interconnect:*"}] 0]
if {$gpintc eq ""} { error "GP0 側の AXI Interconnect が見つかりません" }
puts "### 制御バス: $gpintc"
foreach n {1 2 3} {
  # 【罠】ブレースの中では $ が展開されない。変数を埋める行だけ二重引用符で組む
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
             Master {/ps7_0/M_AXI_GP0} Slave {/axi_dma_$n/S_AXI_LITE} \
             ddr_seg {Auto} intc_ip {$gpintc} master_apm {0}" \
    [get_bd_intf_pins axi_dma_$n/S_AXI_LITE]
}

# ---------- データ: HP0 と HP2 に SmartConnect を1つずつ ----------
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/axi_dma_0/M_AXI_MM2S} Slave {/ps7_0/S_AXI_HP0} \
            ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}} \
  [get_bd_intf_pins ps7_0/S_AXI_HP0]
set smc0 [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:smartconnect:*"}] 0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/axi_dma_1/M_AXI_MM2S} Slave {/ps7_0/S_AXI_HP2} \
            ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}} \
  [get_bd_intf_pins ps7_0/S_AXI_HP2]
# 名前の順ではなく「増えたほう」で取る（自動命名に頼らない）
set smc1 [lindex [lsearch -all -inline -not -exact \
            [get_bd_cells -quiet -filter {VLNV =~ "*:smartconnect:*"}] $smc0] 0]
if {$smc1 eq "" || $smc0 eq ""} { error "SmartConnect が2つ揃いません: $smc0 / $smc1" }
puts "### データバス: HP0=$smc0  HP2=$smc1"

# 残りのマスタをぶら下げる。0,2 は HP0 / 1,3 は HP2。
foreach {n hp smc} [list 0 HP0 $smc0  1 HP2 $smc1  2 HP0 $smc0  3 HP2 $smc1] {
  set chs {M_AXI_S2MM M_AXI_MM2S}
  if {$n >= 2} { lappend chs M_AXI_SG }               ;# SG は記述子を読み書きするマスタが1本増える
  foreach ch $chs {
    if {$n <= 1 && $ch eq "M_AXI_MM2S"} { continue }   ;# 0,1 の読みは上で済み
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
      -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
               Master {/axi_dma_$n/$ch} Slave {/ps7_0/S_AXI_$hp} \
               ddr_seg {Auto} intc_ip {$smc} master_apm {0}" \
      [get_bd_intf_pins axi_dma_$n/$ch]
  }
}

# ---------- コアと FIFO のクロック・リセットは手で結ぶ ----------
set rstcell [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:proc_sys_reset:*"}] 0]
if {$rstcell eq ""} { error "proc_sys_reset セルが見つかりません" }
puts "### リセット源: $rstcell"
foreach n {0 1 2 3} {
  set kind [lindex $kinds $n]
  connect_bd_net [get_bd_pins ${kind}_$n/aclk]    [get_bd_pins ps7_0/FCLK_CLK0]
  connect_bd_net [get_bd_pins ${kind}_$n/aresetn] [get_bd_pins $rstcell/peripheral_aresetn]
  connect_bd_net [get_bd_pins axis_data_fifo_$n/s_axis_aclk]    [get_bd_pins ps7_0/FCLK_CLK0]
  connect_bd_net [get_bd_pins axis_data_fifo_$n/s_axis_aresetn] [get_bd_pins $rstcell/peripheral_aresetn]
}

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design
puts "### ブロックデザインの検証を通過"
foreach s [get_bd_addr_segs -quiet] { puts "###   $s" }

if {$stage ne "all"} { puts "### stage=bd のためここで終了"; exit 0 }

make_wrapper -files [get_files ./$proj/$proj.srcs/sources_1/bd/$bd/$bd.bd] -top
add_files -norecurse ./$proj/$proj.gen/sources_1/bd/$bd/hdl/${bd}_wrapper.v
set_property top ${bd}_wrapper [current_fileset]
update_compile_order -fileset sources_1
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { puts "### 実装に失敗しました"; exit 1 }

file copy -force ./$proj/$proj.runs/impl_1/${bd}_wrapper.bit $outdir/${tag}.bit
set hwh [glob -nocomplain ./$proj/$proj.gen/sources_1/bd/$bd/hw_handoff/${bd}.hwh]
if {$hwh eq ""} { set hwh [glob -nocomplain ./$proj/$proj.srcs/sources_1/bd/$bd/hw_handoff/${bd}.hwh] }
file copy -force [lindex $hwh 0] $outdir/${tag}.hwh
set wns [get_property STATS.WNS [get_runs impl_1]]
puts "### 完了: $outdir/${tag}.bit と ${tag}.hwh"
puts "### FCLK 実周波数 = $fclk_rep MHz / タイミング WNS = $wns ns"

open_run impl_1
report_utilization -file $outdir/${tag}_util.txt
set fh [open $outdir/${tag}_util.txt r]; set rpt [read $fh]; close $fh
puts "### 資源使用量:"
foreach {label pat} {
  LUT      {\|\s+Slice LUTs\s+\|\s+(\d+)\s+\|}
  FF       {\|\s+Slice Registers\s+\|\s+(\d+)\s+\|}
  BRAM     {\|\s+Block RAM Tile\s+\|\s+([\d.]+)\s+\|}
  DSP      {\|\s+DSPs\s+\|\s+(\d+)\s+\|}
} { if {[regexp $pat $rpt -> v]} { puts "###   $label = $v" } }
if {$wns < 0} { puts "### 警告: タイミング未達。この周波数の測定値は信用できない" }
exit 0
