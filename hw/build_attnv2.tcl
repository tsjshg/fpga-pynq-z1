# =====================================================================
#  PYNQ-Z1 / XC7Z020  段階9: attention コア（bitnet_b1_58-large 用・head_dim 96・GQA なし）
#
#    PS7 -M_AXI_GP0-> AXI DMA ×N (制御)
#    AXI DMA n -MM2S-> axis_attnv n -> FIFO n -> AXI DMA n S2MM
#    AXI DMA 0 -> SmartConnect -> PS7 S_AXI_HP0 -> DDR
#    AXI DMA 1 -> SmartConnect -> PS7 S_AXI_HP2 -> DDR
#
#  段階5b（SmolLM2 用・head_dim 64・GQA 3:1）の作り直し。
#  bitnet_b1_58-large は head_dim 96 で **GQA が無い**（q 16 : kv 16）。
#
#  変更点:
#    ① head_dim 64 → 96（1行 12 ビート）。GQA なしなので 1バイト 1 MAC
#    ② 見出しに **スコア倍率 m（18bit）** を追加。段階5b は clamp(s>>8) 固定で
#       量子化尺度を無視していたため、実モデルで perplexity 35.56 → 58.25 に
#       悪化した。m = TAU·sq·sk/√HD を渡すと 35.90 に戻る（numpy で実測）。
#       回路側の追加は掛け算器 1 個だけ。
#    ③ 見出し付きなので 1 転送に何組でも詰められる（1トークン 384 組）
#
#  掛け算器は 第1相 8 + 倍率 1 + 第2相 8 = 17 個（段階5b は 48 個）。
#
#  【タイミング】142.86 MHz では通らなかった。
#    1回目 WNS -0.231ns / 2回目（扇形の複製＋転送長カウンタ 26→23bit）-0.066ns。
#  どちらも違反したのは自作コアではなく **AXI DMA の S2MM 転送長カウンタ**。
#  同じ経路が dma_0 は +0.301、dma_1 は -0.066 と置き場所の運で 0.37ns 開いていた。
#
#  **125 MHz で焼くことにした。** 2本ぶんの道幅が 2.000 GB/s になり、DDR の壁
#  2.009 GB/s とほぼ同じ。**帯域律速のこのコアでは損がほとんど出ない**
#  （142.86 MHz の道幅 2.286 は壁より広く、どのみち使い切れていなかった）。
#  **バッファは 8 MiB 以下に切ること**（転送長カウンタ 23bit）。
#
#  使い方: vivado -mode batch -source build_attnv2.tcl -tclargs <bd|all> <FCLK MHz> <本数>
# =====================================================================
set stage  "bd"
set fclk   150
set nport  2
if {$argc > 0} { set stage [lindex $argv 0] }
if {$argc > 1} { set fclk  [lindex $argv 1] }
if {$argc > 2} { set nport [lindex $argv 2] }
if {$nport != 1 && $nport != 2} { error "本数は 1 か 2" }
puts "### stage = $stage / FCLK 要求 = $fclk MHz / HP $nport 本"

set part      xc7z020clg400-1
set proj      dmaattnv_${nport}_${fclk}
set bd        design_1
set outdir    [file normalize "./out"]
set tag       attnv_${nport}_${fclk}
file mkdir $outdir

create_project $proj ./$proj -part $part -force
add_files -norecurse ./rtl/axis_attnv.v
update_compile_order -fileset sources_1
create_bd_design $bd

# ---------- PS7（533項目は build.tcl と共通） ----------
source ./ps7_common.tcl

# HP0 と HP2 を選ぶのは、PS 側スイッチで HP0/HP1、HP2/HP3 が対になっているため。
# 同じ対を使うと DDR に届く前の入口で詰まる（段階2で実測済み）。
if {$nport == 2} { set_property -dict [list CONFIG.PCW_USE_S_AXI_HP2 {1}] $ps7_0 }

set fclk_act [get_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $ps7_0]
set fclk_rep [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps7_0]
puts "### FCLK_CLK0 実周波数 = $fclk_rep MHz (要求 $fclk)"

set ns [expr {$nport - 1}]
set hps {HP0 HP2}

# ---------- AXI DMA ×N（読み 64bit / 書き 32bit / SG 無し） ----------
for {set n 0} {$n <= $ns} {incr n} {
  set dma [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_$n ]
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
    CONFIG.c_sg_length_width {23} \
  ] $dma

  create_bd_cell -type module -reference axis_attnv axis_attnv_$n
  # 1グループ 195 語しか出ないので 1024 段で十分。
  set f [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo axis_data_fifo_$n ]
  set_property -dict [list CONFIG.FIFO_DEPTH {1024} CONFIG.HAS_TLAST {1} \
                           CONFIG.TDATA_NUM_BYTES {4} CONFIG.HAS_TKEEP {0}] $f

  connect_bd_intf_net [get_bd_intf_pins axi_dma_$n/M_AXIS_MM2S]    [get_bd_intf_pins axis_attnv_$n/s_axis]
  connect_bd_intf_net [get_bd_intf_pins axis_attnv_$n/m_axis]      [get_bd_intf_pins axis_data_fifo_$n/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins axis_data_fifo_$n/M_AXIS]  [get_bd_intf_pins axi_dma_$n/S_AXIS_S2MM]
}

# ---------- 制御: PS M_AXI_GP0 -> 各 DMA の S_AXI_LITE ----------
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
            Master {/ps7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} \
            ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}} \
  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

if {$nport == 2} {
  set gpintc [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:axi_interconnect:*"}] 0]
  if {$gpintc eq ""} { error "GP0 側の AXI Interconnect が見つかりません" }
  puts "### 制御バス: $gpintc"
  # 【罠】ブレースの中では $ が展開されない。変数を埋める行だけ二重引用符で組む。
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
             Master {/ps7_0/M_AXI_GP0} Slave {/axi_dma_1/S_AXI_LITE} \
             ddr_seg {Auto} intc_ip {$gpintc} master_apm {0}" \
    [get_bd_intf_pins axi_dma_1/S_AXI_LITE]
}

# ---------- データ: DMA n の読み -> HP ----------
for {set n 0} {$n <= $ns} {incr n} {
  set hp [lindex $hps $n]
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
             Master {/axi_dma_$n/M_AXI_MM2S} Slave {/ps7_0/S_AXI_$hp} \
             ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}" \
    [get_bd_intf_pins ps7_0/S_AXI_$hp]
}

# 書き側 S2MM も、読みと同じ HP ポートへ。上で立った SmartConnect にぶら下げる。
set smcs [get_bd_cells -quiet -filter {VLNV =~ "*:smartconnect:*"}]
if {[llength $smcs] != $nport} { error "SmartConnect が $nport 個見つかりません: $smcs" }
puts "### データバス: $smcs"
for {set n 0} {$n <= $ns} {incr n} {
  set hp  [lindex $hps $n]
  set smc [lindex $smcs $n]
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
             Master {/axi_dma_$n/M_AXI_S2MM} Slave {/ps7_0/S_AXI_$hp} \
             ddr_seg {Auto} intc_ip {$smc} master_apm {0}" \
    [get_bd_intf_pins axi_dma_$n/M_AXI_S2MM]
}

# ---------- コアと FIFO のクロック・リセットは手で結ぶ ----------
# apply_bd_automation は AXI インタフェースしか面倒を見ない。
# proc_sys_reset はセル名が自動生成なので VLNV で検索する。
set rstcell [lindex [get_bd_cells -quiet -filter {VLNV =~ "*:proc_sys_reset:*"}] 0]
if {$rstcell eq ""} { error "proc_sys_reset セルが見つかりません" }
puts "### リセット源: $rstcell"
for {set n 0} {$n <= $ns} {incr n} {
  connect_bd_net [get_bd_pins axis_attnv_$n/aclk]    [get_bd_pins ps7_0/FCLK_CLK0]
  connect_bd_net [get_bd_pins axis_attnv_$n/aresetn] [get_bd_pins $rstcell/peripheral_aresetn]
  connect_bd_net [get_bd_pins axis_data_fifo_$n/s_axis_aclk]    [get_bd_pins ps7_0/FCLK_CLK0]
  connect_bd_net [get_bd_pins axis_data_fifo_$n/s_axis_aresetn] [get_bd_pins $rstcell/peripheral_aresetn]
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

# -jobs は控えめに。Vivado は IP ごとに約2.8GB のプロセスを立てる。
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

open_run impl_1
report_utilization -file $outdir/${tag}_util.txt
# 【罠】STATS.* は open_run 後だと取れない（警告だけ出て空になる）。
# report_utilization の出力を読んで拾う。
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
