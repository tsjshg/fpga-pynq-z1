#!/usr/bin/env python3
"""PYNQ-Z1 段階3: 計算させても帯域を保てるかを測る。

デザイン: AXI DMA -MM2S-> axis_mac (8 MAC) -> FIFO -> S2MM -> DDR

問いは一つ: 段階2でヌルシンクが出した 1.13 GB/s（HP 1本）を、
実際に内積を計算しながら保てるか。落ちるなら演算パイプラインが
上流を止めているということで、そこが次の設計課題になる。

ストリームの形式:
    [ x: L バイト ][ 重み: L バイト × R 行 ]  →  結果 R 個（int32）

正しさは numpy と突き合わせる。速いが壊れている、では意味がないので。

使い方: python3 dma_mac_bw.py <bitstream> [MB] [繰り返し回数]
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/mac150.bit"
MB  = int(sys.argv[2]) if len(sys.argv) > 2 else 32
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
L   = 512                       # axis_mac.v の parameter L と一致させること

DDR_PEAK  = 2.1                 # GB/s
SINK_1PORT = 1.13               # 段階2の実測（ヌルシンク・HP 1本）

print(f"ビットストリーム: {BIT}")
ol = Overlay(BIT)
print("  ロード成功  IP:", list(ol.ip_dict.keys()))

par = ol.ip_dict["axi_dma_0"]["parameters"]
if par.get("C_INCLUDE_S2MM") != "1":
    sys.exit("★ S2MM がありません。結果を受け取れません")

try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
except Exception as e:
    fclk = None
    print(f"  （FCLK の読み出しに失敗: {e}）")

if fclk:
    path_peak = 8 * fclk * 1e6 / 1e9
    print(f"  FCLK_CLK0 = {fclk:.2f} MHz → 経路の上限 {path_peak:.2f} GB/s")
    print(f"  MAC 8個ぶんの計算力 = {8*fclk/1000:.2f} G MAC/s")
else:
    path_peak = None

# ---- バッファの用意 ----
N    = MB * 1024 * 1024
rows = (N - L) // L                 # 先頭 L バイトは x なので引く
N    = L + rows * L                 # 端数を出さない
print(f"\n入力 {N/1024/1024:.2f} MB = x {L} B + {rows} 行 × {L} B")

src = allocate(shape=(N,), dtype=np.int8)
dst = allocate(shape=(rows,), dtype=np.int32)
print(f"  入力 物理アドレス = 0x{src.physical_address:08x}")
print(f"  結果 物理アドレス = 0x{dst.physical_address:08x}  ({rows*4/1024:.0f} KB)")

rng = np.random.default_rng(12345)
# -128 を避ける。積が 128*128 になっても 16bit に収まるが、
# 端の値でのオーバーフローを気にせず照合したいので範囲を狭めておく。
src[:] = rng.integers(-127, 128, size=N, dtype=np.int8)
src.flush()

# ---- 期待値を numpy で先に計算 ----
a   = np.asarray(src)
x   = a[:L].astype(np.int32)
W   = a[L:].astype(np.int32).reshape(rows, L)
want = W @ x
print(f"  期待値の範囲: {want.min()} 〜 {want.max()}")

# ---- 計測 ----
print(f"\n{REP} 回計測します")
times = []
for i in range(REP):
    dst[:] = 0
    dst.flush()
    t0 = time.perf_counter()
    ol.axi_dma_0.recvchannel.transfer(dst)
    ol.axi_dma_0.sendchannel.transfer(src)
    ol.axi_dma_0.sendchannel.wait()
    ol.axi_dma_0.recvchannel.wait()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  {i+1}回目  {dt*1000:7.2f} ms   読み出し {N/dt/1e9:5.2f} GB/s")

dst.invalidate()
got = np.asarray(dst)
ok  = np.array_equal(got, want)
print(f"\n計算結果の照合: {'一致' if ok else '★不一致'}")
if not ok:
    bad = np.flatnonzero(got != want)
    print(f"  {len(bad)} / {rows} 行が不一致。最初の3件:")
    for j in bad[:3]:
        print(f"    行 {j}: PL={got[j]}  numpy={want[j]}")

t  = min(times)
bw = N / t / 1e9
wr = rows * 4 / t / 1e9

print("\n=== 結果 ===")
print(f"  最速            : {t*1000:.2f} ms")
print(f"  重みの読み出し  : {bw:.2f} GB/s")
print(f"  結果の書き戻し  : {wr:.3f} GB/s  （読みの {100*wr/bw:.1f}%）")
if path_peak:
    print(f"  経路の上限比    : {100*bw/path_peak:5.1f}%  ({path_peak:.2f} GB/s)")
print(f"  DDR の上限比    : {100*(bw+wr)/DDR_PEAK:5.1f}%  （読み書き合計）")
print(f"\n  段階2のヌルシンク（計算なし・HP 1本）= {SINK_1PORT:.2f} GB/s")
print(f"  今回（計算あり）                    = {bw:.2f} GB/s  → {100*bw/SINK_1PORT:.1f}%")
if bw >= SINK_1PORT * 0.97:
    print("\n  → 計算は帯域を食っていない。MAC アレイは上流を止めていない")
else:
    print(f"\n  → {100-100*bw/SINK_1PORT:.1f}% 落ちた。演算パイプラインが上流を止めている")

print(f"\n  実効演算性能: {rows*L/t/1e9:.2f} G MAC/s")

src.freebuffer()
dst.freebuffer()
