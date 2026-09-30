#!/usr/bin/env python3
"""PYNQ-Z1 段階3b: MAC アレイ2組で、DDR の壁 1.97 GB/s を計算しながら保てるか。

デザイン: AXI DMA n -MM2S-> axis_mac n (8 MAC) -> FIFO -> S2MM  (n=0:HP0, 1:HP2)

段階2（ヌルシンク・2本）= 1.97 GB/s、段階3a（MAC・1本）= 1.11 GB/s。
ここが両方を満たせば、「帯域いっぱいまで計算できる回路」が完成したことになる。

使い方: python3 dma_mac_bw2.py <bitstream> [MB/組] [繰り返し回数]
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/mac2_150.bit"
MB  = int(sys.argv[2]) if len(sys.argv) > 2 else 32
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
L   = 512

DDR_PEAK  = 2.1
SINK_2PORT = 1.97       # 段階2の実測（ヌルシンク・HP 2本）
MAC_1PORT  = 1.11       # 段階3a の実測（MAC・HP 1本）

print(f"ビットストリーム: {BIT}")
ol = Overlay(BIT)
print("  ロード成功  IP:", list(ol.ip_dict.keys()))

try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
except Exception as e:
    fclk = None
    print(f"  （FCLK の読み出しに失敗: {e}）")

if fclk:
    path_peak = 8 * fclk * 1e6 / 1e9 * 2
    print(f"  FCLK_CLK0 = {fclk:.2f} MHz → 経路の上限 {path_peak:.2f} GB/s（2本合計）")
    print(f"  MAC 16個ぶんの計算力 = {16*fclk/1000:.2f} G MAC/s")
else:
    path_peak = None

dmas = [ol.axi_dma_0, ol.axi_dma_1]

N    = MB * 1024 * 1024
rows = (N - L) // L
N    = L + rows * L
print(f"\n各組の入力 {N/1024/1024:.2f} MB = x {L} B + {rows} 行 × {L} B（合計 {2*N/1024/1024:.2f} MB）")

srcs, dsts, wants = [], [], []
rng = np.random.default_rng(12345)
for i in range(2):
    s = allocate(shape=(N,), dtype=np.int8)
    d = allocate(shape=(rows,), dtype=np.int32)
    s[:] = rng.integers(-127, 128, size=N, dtype=np.int8)
    s.flush()
    a = np.asarray(s)
    wants.append(a[L:].astype(np.int32).reshape(rows, L) @ a[:L].astype(np.int32))
    print(f"  組{i}: 入力 0x{s.physical_address:08x} / 結果 0x{d.physical_address:08x}")
    srcs.append(s); dsts.append(d)

print(f"\n{REP} 回計測します（2組同時）")
times = []
for i in range(REP):
    for d in dsts:
        d[:] = 0; d.flush()
    t0 = time.perf_counter()
    for n in range(2):
        dmas[n].recvchannel.transfer(dsts[n])
    for n in range(2):
        dmas[n].sendchannel.transfer(srcs[n])
    for n in range(2):
        dmas[n].sendchannel.wait()
        dmas[n].recvchannel.wait()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  {i+1}回目  {dt*1000:7.2f} ms   合計 {2*N/dt/1e9:5.2f} GB/s")

ok = True
for n in range(2):
    dsts[n].invalidate()
    g = np.asarray(dsts[n])
    if np.array_equal(g, wants[n]):
        print(f"\n組{n} の照合: 一致（{rows} 行）")
    else:
        ok = False
        bad = np.flatnonzero(g != wants[n])
        print(f"\n組{n} の照合: ★不一致 {len(bad)}/{rows} 行")
        for j in bad[:3]:
            print(f"    行 {j}: PL={g[j]}  numpy={wants[n][j]}")

t  = min(times)
bw = 2 * N / t / 1e9
wr = 2 * rows * 4 / t / 1e9

print("\n=== 結果 ===")
print(f"  最速            : {t*1000:.2f} ms")
print(f"  重みの読み出し  : {bw:.2f} GB/s")
print(f"  結果の書き戻し  : {wr:.3f} GB/s")
print(f"  DDR トラフィック: {bw+wr:.2f} GB/s  （理論 {DDR_PEAK} の {100*(bw+wr)/DDR_PEAK:.1f}%）")
if path_peak:
    print(f"  経路の上限比    : {100*bw/path_peak:5.1f}%  ({path_peak:.2f} GB/s)")
print(f"\n  段階2 ヌルシンク・2本（計算なし）= {SINK_2PORT:.2f} GB/s")
print(f"  段階3a MAC・1本                  = {MAC_1PORT:.2f} GB/s")
print(f"  今回   MAC・2本                  = {bw:.2f} GB/s  → ヌルシンク比 {100*bw/SINK_2PORT:.1f}% / 1本比 {bw/MAC_1PORT:.2f}倍")
print(f"\n  実効演算性能: {2*rows*L/t/1e9:.2f} G MAC/s")
if ok and bw >= SINK_2PORT * 0.95:
    print("\n  → 帯域いっぱいまで計算できている。段階3 完了")

for b in srcs + dsts:
    b.freebuffer()
