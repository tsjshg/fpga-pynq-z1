#!/usr/bin/env python3
"""PYNQ-Z1: HP ポート2本で DDR から「読むだけ」の帯域を測る。

デザイン: AXI DMA 0 -> HP0 / AXI DMA 1 -> HP2、それぞれヌルシンクへ。
1本版 (dma_read_bw.py) では 100/142.86 MHz とも AXI ストリームの道幅
(8 B × FCLK) の 99% に張り付き、DDR には半分しか届かなかった。
道幅を倍にして、初めて DDR 側の壁を叩きに行く。

2台を「同時に」走らせるのが肝。transfer() は非同期に始まるので、
両方 start してから両方 wait する。片方ずつ待つと直列になって意味が無い。

使い方: python3 dma_read_bw2.py <bitstream> [MB/台] [繰り返し回数]
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/sink2_150.bit"
MB  = int(sys.argv[2]) if len(sys.argv) > 2 else 32
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
N   = MB * 1024 * 1024

DDR_PEAK = 2.1   # GB/s  DDR3 16bit @525MHz(=1050MT/s)

print(f"ビットストリーム: {BIT}")
ol = Overlay(BIT)
print("  ロード成功  IP:", list(ol.ip_dict.keys()))

try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
except Exception as e:
    fclk = None
    print(f"  （FCLK の読み出しに失敗: {e}）")

dmas = [ol.axi_dma_0, ol.axi_dma_1]
for i in range(2):
    par = ol.ip_dict[f"axi_dma_{i}"]["parameters"]
    if par.get("C_INCLUDE_S2MM") == "1":
        sys.exit(f"★ axi_dma_{i} に S2MM が残っています。読み専用になりません")

if fclk:
    path_peak = 8 * fclk * 1e6 / 1e9 * 2      # 64bit × 2本
    print(f"  FCLK_CLK0 = {fclk:.2f} MHz → 経路の上限 {path_peak:.2f} GB/s "
          f"(8B × {fclk:.2f}MHz × 2本)")
else:
    path_peak = None

print(f"\nバッファ {MB} MB × 2 本を確保します（連続物理メモリ）")
bufs = []
rng = np.random.default_rng(12345)
for i in range(2):
    b = allocate(shape=(N,), dtype=np.uint8)
    b[:] = rng.integers(0, 256, size=N, dtype=np.uint8)
    b.flush()
    print(f"  DMA{i} 物理アドレス = 0x{b.physical_address:08x}")
    bufs.append(b)

print(f"\n{REP} 回計測します（1回あたり {MB} MB × 2 = {2*MB} MB を読み出し）")
times, solo = [], []
for i in range(REP):
    t0 = time.perf_counter()
    dmas[0].sendchannel.transfer(bufs[0])
    dmas[1].sendchannel.transfer(bufs[1])
    dmas[0].sendchannel.wait()
    dmas[1].sendchannel.wait()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  {i+1}回目  {dt*1000:7.2f} ms   合計 {2*N/dt/1e9:5.2f} GB/s")

# 比較用に1台だけでも回す。2台の値がこれの2倍に届かなければ、
# 詰まっているのは PL の道幅ではなく DDR 側だと言える。
print(f"\n参考: DMA0 だけを単独で回した場合")
for i in range(3):
    t0 = time.perf_counter()
    dmas[0].sendchannel.transfer(bufs[0])
    dmas[0].sendchannel.wait()
    dt = time.perf_counter() - t0
    solo.append(dt)
    print(f"  {i+1}回目  {dt*1000:7.2f} ms   {N/dt/1e9:5.2f} GB/s")

t   = min(times)
bw  = 2 * N / t / 1e9
bw1 = N / min(solo) / 1e9

print("\n=== 結果 ===")
print(f"  1台単独        : {bw1:.2f} GB/s")
print(f"  2台同時（合計）: {bw:.2f} GB/s  ← DDR トラフィック")
print(f"  スケール率     : {bw/bw1:.2f} 倍  （2.00 なら DDR に余裕あり）")
if path_peak:
    print(f"  経路の上限比   : {100*bw/path_peak:5.1f}%  ({path_peak:.2f} GB/s)")
print(f"  DDR の上限比   : {100*bw/DDR_PEAK:5.1f}%  ({DDR_PEAK} GB/s)")

if path_peak and bw / path_peak > 0.90:
    print("\n  → まだ道幅で詰まっている。DDR にはさらに余裕がある")
else:
    print(f"\n  → 道幅を残して頭打ち。ここが DDR の実力 = {bw:.2f} GB/s")

print(f"\n  重みの流量に換算すると:")
print(f"    INT8 (1.0 B/重み)      {bw:6.2f} G重み/s")
print(f"    三値 (約0.2 B/重み)    {bw/0.2:6.2f} G重み/s")

for b in bufs:
    b.freebuffer()
