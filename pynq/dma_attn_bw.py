#!/usr/bin/env python3
"""PYNQ-Z1 段階5: attention コアの実測（HP 1本）。

デザイン: AXI DMA -MM2S-> axis_attn -> FIFO -> S2MM

ストリーム（1グループ = 1層の1 KV ヘッドぶん）:
    [ヘッダ 8B: T][q: 3×64 B][K: T×64 B][V: T×64 B]
出力: out[h][d] を int32 で 192 個

第1相 (K): s[h][t] = q_h・K[t]、s8 = clamp(s >> 8)
第2相 (V): out[h][d] += s8[h][t]·V[t][d]

softmax はまだ入れていない。スコアをそのまま重みとして使うので、
データ経路・MAC 数・メモリトラフィックは本番と同じまま numpy と厳密照合できる。

使い方: python3 dma_attn_bw.py <bitstream> [T] [グループ数] [繰り返し]
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

BIT    = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/attn150.bit"
T      = int(sys.argv[2]) if len(sys.argv) > 2 else 512
NGRP   = int(sys.argv[3]) if len(sys.argv) > 3 else 32
REP    = int(sys.argv[4]) if len(sys.argv) > 4 else 5

HD, NQ, SHIFT = 64, 3, 8
GRP_IN  = 8 + NQ*HD + 2*T*HD          # 1グループの入力バイト数
GRP_OUT = NQ*HD                       # 1グループの出力語数（int32）

# SmolLM2-135M の実機での換算に使う定数
GROUPS_PER_TOKEN = 30*3               # 30層 × KV 3ヘッド
CPU_ATTN = {128: 19.21, 512: 83.26, 2048: 338.88}   # attnbench.c の実測 (ms)

print(f"ビットストリーム: {BIT}")
ol = Overlay(BIT)
print("  ロード成功  IP:", list(ol.ip_dict.keys()))
try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
except Exception:
    fclk = None
if fclk:
    print(f"  FCLK_CLK0 = {fclk:.2f} MHz → 経路の上限 {8*fclk/1000:.2f} GB/s")

dma = ol.axi_dma_0
N   = GRP_IN * NGRP
print(f"\n文脈長 T = {T} / グループ {NGRP} 個")
print(f"  1グループ {GRP_IN/1024:.1f} KB → 合計 {N/1024/1024:.2f} MB")

src = allocate(shape=(N,), dtype=np.uint8)
dst = allocate(shape=(GRP_OUT*NGRP,), dtype=np.int32)
print(f"  入力 0x{src.physical_address:08x} / 結果 0x{dst.physical_address:08x}")

rng  = np.random.default_rng(7)
want = np.empty(GRP_OUT*NGRP, dtype=np.int32)

off = 0
for g in range(NGRP):
    q = rng.integers(-127, 128, size=(NQ, HD), dtype=np.int8)
    K = rng.integers(-127, 128, size=(T, HD),  dtype=np.int8)
    V = rng.integers(-127, 128, size=(T, HD),  dtype=np.int8)

    hdr = np.zeros(8, dtype=np.uint8)
    hdr[0:4] = np.frombuffer(np.uint32(T).tobytes(), dtype=np.uint8)
    src[off:off+8] = hdr;                                   off += 8
    src[off:off+NQ*HD] = q.reshape(-1).view(np.uint8);      off += NQ*HD
    src[off:off+T*HD]  = K.reshape(-1).view(np.uint8);      off += T*HD
    src[off:off+T*HD]  = V.reshape(-1).view(np.uint8);      off += T*HD

    # 回路と同じ手順で期待値を作る
    s   = q.astype(np.int32) @ K.astype(np.int32).T          # (NQ, T)
    s8  = np.clip(s >> SHIFT, -128, 127).astype(np.int32)    # 算術シフト
    o   = s8 @ V.astype(np.int32)                            # (NQ, HD)
    want[g*GRP_OUT:(g+1)*GRP_OUT] = o.reshape(-1)

assert off == N, f"詰め方が合っていません {off} != {N}"
src.flush()
print(f"  期待値の範囲: {want.min()} 〜 {want.max()}")

print(f"\n{REP} 回計測します")
times = []
for i in range(REP):
    dst[:] = 0; dst.flush()
    t0 = time.perf_counter()
    dma.recvchannel.transfer(dst)
    dma.sendchannel.transfer(src)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  {i+1}回目  {dt*1000:7.2f} ms   {N/dt/1e9:5.2f} GB/s")

dst.invalidate()
got = np.asarray(dst)
ok  = np.array_equal(got, want)
print(f"\n照合: {'一致' if ok else '★不一致'}（{NGRP} グループ × {GRP_OUT} 個）")
if not ok:
    bad = np.flatnonzero(got != want)
    print(f"  {len(bad)}/{len(want)} 個が不一致。最初の5件:")
    for j in bad[:5]:
        print(f"    グループ{j//GRP_OUT} 要素{j%GRP_OUT}: PL={got[j]}  numpy={want[j]}")

t    = min(times)
bw   = N / t / 1e9
macs = NGRP * 2 * NQ * T * HD            # 第1相 + 第2相
print("\n=== 結果 ===")
print(f"  最速          : {t*1000:.2f} ms")
print(f"  読み出し      : {bw:.2f} GB/s")
print(f"  演算          : {macs/t/1e9:.2f} G MAC/s   （1バイトあたり {macs/N:.2f} MAC）")
if fclk:
    print(f"  経路の上限比  : {100*bw/(8*fclk*1e6/1e9):5.1f}%")

# 実機換算（1トークンぶん = 30層 × KV3ヘッド）
per_tok = GRP_IN * GROUPS_PER_TOKEN / (bw*1e9) * 1000
print(f"\n=== SmolLM2-135M への換算（HP 1本のこの値のまま）===")
print(f"  1トークンの attention : {per_tok:.2f} ms   （{GROUPS_PER_TOKEN} グループ）")
if T in CPU_ATTN:
    print(f"  CPU 実測              : {CPU_ATTN[T]:.2f} ms  → **{CPU_ATTN[T]/per_tok:.1f} 倍**")
    gemv = 14.68
    misc = 4.2
    print(f"\n  予算表: 行列積 {gemv:.2f} + attention {per_tok:.2f} + 小物 {misc:.1f}"
          f" = {gemv+per_tok+misc:.2f} ms → {1000/(gemv+per_tok+misc):.1f} tok/s")
    print(f"  （CPU に残した場合は {gemv+CPU_ATTN[T]+misc:.1f} ms → "
          f"{1000/(gemv+CPU_ATTN[T]+misc):.1f} tok/s）")

src.freebuffer(); dst.freebuffer()
