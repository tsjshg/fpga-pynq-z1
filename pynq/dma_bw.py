#!/usr/bin/env python3
"""PYNQ-Z1: PL→DDR の実効帯域を AXI DMA のループバックで測る。

デザイン: PS7 -HP0- SmartConnect - AXI DMA -MM2S-> AXIS FIFO -> S2MM-
つまり DDR から読んで、PL を一周させて、DDR に書き戻す。

読み出しと書き込みが HP0 を共有するので、DDR から見たトラフィックは
転送量の 2 倍になる。理論ピーク 2.1 GB/s と比べるときはそちらを見ること。
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/home/xilinx/dmabw.bit"
MB  = int(sys.argv[2]) if len(sys.argv) > 2 else 16
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
N   = MB * 1024 * 1024

print(f"ビットストリーム: {BIT}")
ol = Overlay(BIT)
print("  ロード成功")
print("  IP:", list(ol.ip_dict.keys()))

dma = ol.axi_dma_0
print(f"\nバッファ {MB} MB を 2 本確保します（連続物理メモリ）")
src = allocate(shape=(N,), dtype=np.uint8)
dst = allocate(shape=(N,), dtype=np.uint8)
print(f"  src 物理アドレス = 0x{src.physical_address:08x}")
print(f"  dst 物理アドレス = 0x{dst.physical_address:08x}")

rng = np.random.default_rng(12345)
src[:] = rng.integers(0, 256, size=N, dtype=np.uint8)
dst[:] = 0
src.flush()
dst.flush()

print(f"\n{REP} 回計測します（1回あたり {MB} MB を往復）")
best = 0.0
times = []
for i in range(REP):
    dst[:] = 0
    dst.flush()
    t0 = time.perf_counter()
    dma.recvchannel.transfer(dst)
    dma.sendchannel.transfer(src)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    dt = time.perf_counter() - t0
    times.append(dt)
    one_way = N / dt / 1e9
    print(f"  {i+1}回目  {dt*1000:7.2f} ms   片道 {one_way:5.2f} GB/s   "
          f"DDR実トラフィック {2*one_way:5.2f} GB/s")
    best = max(best, one_way)

dst.invalidate()
ok = np.array_equal(np.asarray(src), np.asarray(dst))
print(f"\nデータ照合: {'一致' if ok else '★不一致（転送が壊れている）'}")

t = min(times)
print(f"\n=== 結果 ===")
print(f"  最速             : {t*1000:.2f} ms")
print(f"  PL→DDR 片道      : {N/t/1e9:.2f} GB/s   （重みを読む速度に相当）")
print(f"  DDR 実トラフィック: {2*N/t/1e9:.2f} GB/s  （理論ピーク 2.1 GB/s に対して "
      f"{200*N/t/1e9/2.1:.0f}%）")
# CPU の 0.70 は「読みのみ」の値。片道と比べるのは不公平なので、
# DDR から見た仕事量（＝実トラフィック）で比較する。
print(f"  参考: CPU 2コアの逐次リード実測 = 0.70 GB/s（読みのみ）")
print(f"        DDR の仕事量で比べると PL は CPU の {2*N/t/1e9/0.70:.1f} 倍")

src.freebuffer()
dst.freebuffer()
