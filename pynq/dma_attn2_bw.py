#!/usr/bin/env python3
"""PYNQ-Z1 段階5b: softmax 入り attention コアの実測（HP 1本 or 2本）。

デザイン: AXI DMA n -MM2S-> axis_attn2 n -> FIFO -> S2MM   （n = 0..本数-1）

ストリーム（1グループ = 1層の1 KV ヘッドぶん）:
    [ヘッダ 8B: T][q: 3×64 B][K: T×64 B][V: T×64 B]
出力: out[h][d] を int32 で 192 個 + 分母 den[h] を 3 個 = 195 語

第1相 (K): s[h][t] = q_h・K[t]、s8 = clamp(s >> 8)、mx[h] = max_t s8
第2相 (V): e = EROM[mx[h] - s8]、out[h][d] += e·V[t][d]、den[h] += e

★ 段階5a との違いは softmax だけ。DDR の読み量は 1 バイトも変わらない。
   帯域が 5a と同じなら「softmax はただで付く」ことの実証になる。
★ 割り算は回路でやらない。分子と分母を出して下流に任せる。

使い方: python3 dma_attn2_bw.py <bitstream> [T] [1本あたりのグループ数] [繰り返し] [本数]
"""
import sys, time, math
import numpy as np
from pynq import Overlay, allocate

BIT   = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/attn2_2_150.bit"
T     = int(sys.argv[2]) if len(sys.argv) > 2 else 512
NGRP  = int(sys.argv[3]) if len(sys.argv) > 3 else 64      # 1本あたり
REP   = int(sys.argv[4]) if len(sys.argv) > 4 else 5
NPORT = int(sys.argv[5]) if len(sys.argv) > 5 else 2

HD, NQ, SHIFT, TAU = 64, 3, 8, 8.0
GRP_IN  = 8 + NQ*HD + 2*T*HD          # 1グループの入力バイト数
GRP_OUT = NQ*HD + NQ                  # 1グループの出力語数（int32）= 195

# 回路の中の表と同じもの。rtl/gen_attn2.py と同じ式。
EROM = np.array([max(0, min(255, int(round(255.0*math.exp(-d/TAU))))) for d in range(256)],
                dtype=np.int32)

GROUPS_PER_TOKEN = 30*3               # 30層 × KV 3ヘッド
# attnbench2.c を -O3 -mfpu=neon で焼いた実測 (ms)。旧 attnbench.c は -O2 で 1.8 倍遅かった
CPU_ATTN = {128: 10.92, 512: 46.19, 2048: 193.25}

print(f"ビットストリーム: {BIT}")
ol = Overlay(BIT)
print("  ロード成功  IP:", list(ol.ip_dict.keys()))
try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
except Exception:
    fclk = None
if fclk:
    print(f"  FCLK_CLK0 = {fclk:.2f} MHz → 経路の上限 {NPORT*8*fclk/1000:.2f} GB/s（{NPORT}本合計）")

dmas = [getattr(ol, f"axi_dma_{n}") for n in range(NPORT)]
N    = GRP_IN * NGRP                  # 1本あたりのバイト数
NTOT = N * NPORT
print(f"\n文脈長 T = {T} / 1本あたり {NGRP} グループ × {NPORT} 本")
print(f"  1グループ {GRP_IN/1024:.1f} KB → 合計 {NTOT/1024/1024:.2f} MB")
print(f"  指数表: TAU={TAU} 非零 {int((EROM>0).sum())} 語")

srcs = [allocate(shape=(N,), dtype=np.uint8) for _ in range(NPORT)]
dsts = [allocate(shape=(GRP_OUT*NGRP,), dtype=np.int32) for _ in range(NPORT)]
for n in range(NPORT):
    print(f"  [{n}] 入力 0x{srcs[n].physical_address:08x} / 結果 0x{dsts[n].physical_address:08x}")

rng   = np.random.default_rng(7)
wants = [np.empty(GRP_OUT*NGRP, dtype=np.int32) for _ in range(NPORT)]
dmax  = 0

for n in range(NPORT):
    off = 0
    for g in range(NGRP):
        q = rng.integers(-127, 128, size=(NQ, HD), dtype=np.int8)
        # K の行ごとに大きさを振る。固定だと score が固まって指数表の端しか引かない。
        sc = rng.uniform(0.05, 1.0, size=(T, 1))
        K  = np.clip(np.round(rng.integers(-127, 128, size=(T, HD)) * sc), -127, 127).astype(np.int8)
        V  = rng.integers(-127, 128, size=(T, HD), dtype=np.int8)

        hdr = np.zeros(8, dtype=np.uint8)
        hdr[0:4] = np.frombuffer(np.uint32(T).tobytes(), dtype=np.uint8)
        srcs[n][off:off+8] = hdr;                                  off += 8
        srcs[n][off:off+NQ*HD] = q.reshape(-1).view(np.uint8);     off += NQ*HD
        srcs[n][off:off+T*HD]  = K.reshape(-1).view(np.uint8);     off += T*HD
        srcs[n][off:off+T*HD]  = V.reshape(-1).view(np.uint8);     off += T*HD

        # 回路と同じ手順で期待値を作る
        s   = q.astype(np.int32) @ K.astype(np.int32).T          # (NQ,T)
        s8  = np.clip(s >> SHIFT, -128, 127).astype(np.int32)    # 算術シフト
        mx  = s8.max(axis=1)
        d   = (mx[:, None] - s8).astype(np.int32)                # 0..255
        dmax = max(dmax, int(d.max()))
        e   = EROM[d]                                            # (NQ,T)
        o   = e @ V.astype(np.int32)                             # (NQ,HD)
        den = e.sum(axis=1)                                      # (NQ,)
        wants[n][g*GRP_OUT:(g+1)*GRP_OUT] = np.concatenate([o.reshape(-1), den])
    assert off == N, f"詰め方が合っていません {off} != {N}"
    srcs[n].flush()

print(f"  指数表を引いた最大の距離 d = {dmax}")
print(f"  期待値の範囲: {min(w.min() for w in wants)} 〜 {max(w.max() for w in wants)}")

# --------------------------------------------------------------------
# 【測り方】1回の計測には PYNQ 側の固定費（transfer/wait の Python 処理）が
# 乗る。実測すると **約 0.8 ms/チャネル対** あり、8ms の測定では 10% になる。
# そこで「全量」と「半量」を測って *差分* から帯域を出す（傾きで測る）。
# 固定費は両方に等しく乗るので差を取ると消える。
#   帯域 = (全量 - 半量) / (全量の時間 - 半量の時間)
# 固定費そのものも切片として出しておく。試算ではなく実測の内訳になる。
# --------------------------------------------------------------------
def run(ngrp):
    nb = GRP_IN * ngrp * NPORT
    ts = []
    for i in range(REP):
        t0 = time.perf_counter()
        for n in range(NPORT):
            dmas[n].recvchannel.transfer(dsts[n], nbytes=GRP_OUT*ngrp*4)
        for n in range(NPORT):
            dmas[n].sendchannel.transfer(srcs[n], nbytes=GRP_IN*ngrp)
        for n in range(NPORT): dmas[n].sendchannel.wait()
        for n in range(NPORT): dmas[n].recvchannel.wait()
        ts.append(time.perf_counter() - t0)
    return nb, min(ts), ts

HALF = NGRP // 2
print(f"\n{REP} 回ずつ、全量({NGRP}グループ/本)と半量({HALF}グループ/本)を測ります")
for n in range(NPORT):
    dsts[n][:] = 0; dsts[n].flush()
nb2, t2, all2 = run(HALF)
print(f"  半量 {nb2/1024/1024:5.2f} MB : " + " ".join(f"{x*1000:6.2f}" for x in all2) + " ms")
for n in range(NPORT):
    dsts[n][:] = 0; dsts[n].flush()
nb1, t1, all1 = run(NGRP)
print(f"  全量 {nb1/1024/1024:5.2f} MB : " + " ".join(f"{x*1000:6.2f}" for x in all1) + " ms")

NTOT = nb1
bw    = (nb1 - nb2) / (t1 - t2) / 1e9      # 傾き＝正味の帯域
fixed = t1 - nb1 / (bw*1e9)                # 切片＝1回あたりの固定費
t     = t1
ok = True
for n in range(NPORT):
    dsts[n].invalidate()
    got = np.asarray(dsts[n])
    if np.array_equal(got, wants[n]):
        print(f"照合[{n}]: 一致（{NGRP} グループ × {GRP_OUT} 語）")
    else:
        ok = False
        bad = np.flatnonzero(got != wants[n])
        print(f"照合[{n}]: ★不一致 {len(bad)}/{len(wants[n])} 語。最初の5件:")
        for j in bad[:5]:
            kind = "分母" if j % GRP_OUT >= NQ*HD else "本体"
            print(f"    グループ{j//GRP_OUT} {kind} 要素{j%GRP_OUT}: PL={got[j]}  numpy={wants[n][j]}")

macs = NPORT * NGRP * 2 * NQ * T * HD            # 第1相 + 第2相
beats   = GRP_IN // 8
stall   = 216          # ST_KDR 12 + ST_DRN 9 + ST_OUT 195
ideal   = beats / (beats + stall)
print("\n=== 結果 ===")
print(f"  全量の最速    : {t*1000:.2f} ms  （うち固定費 {fixed*1000:.2f} ms）")
print(f"  読み出し      : {bw:.2f} GB/s   （固定費を除いた傾き）")
print(f"  演算          : {macs/NTOT*bw:.2f} G MAC/s   （1バイトあたり {macs/NTOT:.2f} MAC）")
if fclk:
    print(f"  経路の上限比  : {100*bw/(NPORT*8*fclk*1e6/1e9):5.1f}%   （{NPORT}本合計 {NPORT*8*fclk/1000:.2f} GB/s）")
print(f"  DDR 実力比    : {100*bw/2.007:5.1f}%   （段階2 を測り直した 2.007 GB/s）")
print(f"  停止の理論値  : {100*ideal:5.1f}%   （1グループ {beats} ビート中 {stall} サイクル止まる）")

per_tok = GRP_IN * GROUPS_PER_TOKEN / (bw*1e9) * 1000
print(f"\n=== SmolLM2-135M への換算 ===")
print(f"  1トークンの attention : {per_tok:.2f} ms   （{GROUPS_PER_TOKEN} グループ）")
if T in CPU_ATTN:
    print(f"  CPU 実測              : {CPU_ATTN[T]:.2f} ms  → **{CPU_ATTN[T]/per_tok:.1f} 倍**")
    gemv, misc = 14.20, 3.47
    print(f"\n  予算表(試算): 行列積 {gemv:.2f} + attention {per_tok:.2f} + 小物 {misc:.1f}"
          f" = {gemv+per_tok+misc:.2f} ms → {1000/(gemv+per_tok+misc):.1f} tok/s")

for b in srcs + dsts: b.freebuffer()
