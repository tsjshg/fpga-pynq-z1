#!/usr/bin/env python3
"""PYNQ-Z1 段階4: 三値 (BitNet b1.58) MAC アレイ ×2 の実測。

デザイン: AXI DMA n -MM2S-> axis_tmac n (三値40個/クロック) -> FIFO -> S2MM

段階3（INT8）は 1.91 GB/s で 1.90 G MAC/s だった。
三値は 1 バイトに重み 5 個なので、同じ帯域から 5 倍の重みが出るはず。

符号化: バイト v = Σ (w_d + 1) * 3^d   (d=0..4, w ∈ {-1,0,+1})
        3^5 = 243 ≤ 256。1.6 ビット/重み。

使い方: python3 dma_tmac_bw2.py <bitstream> [MB/組] [繰り返し回数]
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/tmac2_150.bit"
MB  = int(sys.argv[2]) if len(sys.argv) > 2 else 32
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
L   = 640                       # axis_tmac.v の L と一致させること
BPR = L // 5                    # 1行のバイト数 = 128

DDR_PEAK   = 2.1
INT8_2PORT = 1.91               # 段階3 の実測（INT8・MAC 16個・HP 2本）

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
    print(f"  三値 80個/クロックぶんの演算力 = {80*fclk/1000:.2f} G重み/s")
else:
    path_peak = None

dmas = [ol.axi_dma_0, ol.axi_dma_1]

# ---- バッファ ----
N    = MB * 1024 * 1024
rows = (N - L) // BPR
N    = L + rows * BPR
print(f"\n各組の入力 {N/1024/1024:.2f} MB = x {L} B + {rows} 行 × {BPR} B")
print(f"  三値の重み {rows*L/1e6:.2f} M 個 / 組（{2*rows*L/1e6:.2f} M 個 合計）")

POW = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
CHUNK = 8192                    # 一度に作る行数

# ボードの RAM は 491 MB しかない。重み行列を丸ごと numpy に持つと
# 262,139 行 × 640 で int8 でも 168 MB、int64 なら 1.25 GB になって落ちる。
# （実際に落とした。rng.integers は dtype を指定しないと int64 を作る）
# 行を small chunk に切って、詰め込みと期待値計算をその場で済ませる。
srcs, dsts, wants = [], [], []
rng = np.random.default_rng(12345)
for i in range(2):
    s = allocate(shape=(N,), dtype=np.uint8)
    d = allocate(shape=(rows,), dtype=np.int32)

    x  = rng.integers(-127, 128, size=L, dtype=np.int8)
    xi = x.astype(np.int32)
    s[:L] = x.view(np.uint8)

    want = np.empty(rows, dtype=np.int32)
    for r0 in range(0, rows, CHUNK):
        r1 = min(r0 + CHUNK, rows)
        Wc = rng.integers(-1, 2, size=(r1-r0, L), dtype=np.int8)     # -1, 0, +1
        packed = ((Wc + 1).astype(np.uint16).reshape(r1-r0, BPR, 5) * POW).sum(
                    axis=2, dtype=np.uint16).astype(np.uint8)
        s[L + r0*BPR : L + r1*BPR] = packed.reshape(-1)
        want[r0:r1] = Wc.astype(np.int32) @ xi
        del Wc, packed
    s.flush()

    wants.append(want)
    print(f"  組{i}: 入力 0x{s.physical_address:08x} / 結果 0x{d.physical_address:08x}")
    srcs.append(s); dsts.append(d)

print(f"  期待値の範囲: {min(w.min() for w in wants)} 〜 {max(w.max() for w in wants)}")

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
    print(f"  {i+1}回目  {dt*1000:7.2f} ms   {2*N/dt/1e9:5.2f} GB/s   {2*rows*L/dt/1e9:5.2f} G重み/s")

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
gw = 2 * rows * L / t / 1e9

print("\n=== 結果 ===")
print(f"  最速            : {t*1000:.2f} ms")
print(f"  重みの読み出し  : {bw:.2f} GB/s")
print(f"  結果の書き戻し  : {wr:.3f} GB/s")
print(f"  DDR トラフィック: {bw+wr:.2f} GB/s  （理論 {DDR_PEAK} の {100*(bw+wr)/DDR_PEAK:.1f}%）")
if path_peak:
    print(f"  経路の上限比    : {100*bw/path_peak:5.1f}%")
print(f"\n  ★ 三値の処理速度: {gw:.2f} G重み/s")
print(f"     段階3 INT8    : {INT8_2PORT:.2f} G重み/s（1バイト=1重み）")
print(f"     倍率          : {gw/INT8_2PORT:.2f} 倍")
print(f"\n  1バイトあたりの重み数（実効）: {gw/bw:.2f} 個")

# 135M を三値化したモデルでの見込み（試算）
model_w = 135e6
print(f"\n  【試算】135M を三値化した場合: 重み {model_w*0.2/1e6:.0f} MB / "
      f"{model_w/gw/1e9*1000:.2f} ms/トークン → {gw*1e9/model_w:.1f} tok/s")

for b in srcs + dsts:
    b.freebuffer()
