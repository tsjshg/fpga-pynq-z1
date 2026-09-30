#!/usr/bin/env python3
"""SmolLM2-135M の1トークンぶんの行列積を、全部 PL で回す。

model/make_fpga_job.py が作った仕事を読む:
    job_weights.bin  [x 640B][重み 128B×R] を 181 ブロック連結したもの
    job_expect.bin   期待値 int32（numpy で作った）
    job_index.json   各ブロックの位置と行数

重みは CMA に常駐させる（本物のアクセラレータはそうする）。
転送のたびに start/nbytes をずらして、その場所から流す。

測るもの:
  ① 全ブロックを numpy と突き合わせて、1語でも違わないか
  ② 実際の壁時計時間
  ③ 大きな1転送で測った正味の帯域 → 「帯域だけなら何 ms か」
  ②−③ が転送の呼び出し固定費。ここが本題。

使い方: python3 dma_model_token.py <bitstream> [仕事のディレクトリ] [繰り返し]
"""
import sys, os, json, time
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/tmac2_150.bit"
JOB = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/job"
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 3

meta = json.load(open(os.path.join(JOB, "job_index.json")))
blocks = meta["blocks"]
TOTW, TOTR = meta["total_w"], meta["total_rows"]
print(f"仕事: {meta['n_blocks']} ブロック / 重み {TOTW/1024/1024:.2f} MB / 出力 {TOTR:,} 語")
print(f"  入力文: {meta['prompt']!r}")

ol = Overlay(BIT)
dma = ol.axi_dma_0
try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
    print(f"  FCLK = {fclk:.2f} MHz → 1本の道幅 {8*fclk/1000:.2f} GB/s")
except Exception:
    fclk = None

rows_big = (TOTW - 640) // 128 + 2
src = allocate(shape=(TOTW,), dtype=np.uint8)
dst = allocate(shape=(max(TOTR, rows_big),), dtype=np.int32)
print(f"  CMA: 入力 {src.nbytes/1024/1024:.2f} MB @0x{src.physical_address:08x}"
      f" / 出力 {dst.nbytes/1024:.0f} KB @0x{dst.physical_address:08x}")

# 重みを CMA に流し込む（常駐。ここは測定に含めない）
t0 = time.perf_counter()
with open(os.path.join(JOB, "job_weights.bin"), "rb") as f:
    off = 0
    while True:
        b = f.read(4 << 20)
        if not b: break
        src[off:off+len(b)] = np.frombuffer(b, np.uint8); off += len(b)
assert off == TOTW, f"重みの大きさが合わない {off} != {TOTW}"
src.flush()
print(f"  重みの常駐化: {time.perf_counter()-t0:.2f} 秒（1回だけ。測定には含めない）")

want = np.fromfile(os.path.join(JOB, "job_expect.bin"), dtype=np.int32)
assert want.size == TOTR

# ---- ③ 正味の帯域。中身は無視して大きな1転送の時間だけ測る ----
print("\n正味の帯域を測る（全量と半量の差＝傾き）")
def one(nb):
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        dma.recvchannel.transfer(dst, 0, ((nb - 640)//128 + 1)*4)
        dma.sendchannel.transfer(src, 0, nb)
        dma.sendchannel.wait(); dma.recvchannel.wait()
        ts.append(time.perf_counter() - t)
    return min(ts)
half = ((TOTW//2 - 640)//128)*128 + 640
t_h, t_f = one(half), one(TOTW)
bw = (TOTW - half) / (t_f - t_h) / 1e9
print(f"  半量 {half/1024/1024:.2f} MB {t_h*1000:.2f} ms / 全量 {TOTW/1024/1024:.2f} MB {t_f*1000:.2f} ms")
print(f"  → 正味 {bw:.3f} GB/s（HP 1本）。この 29.31 MB は帯域だけなら {TOTW/bw/1e9*1000:.2f} ms")

# ---- ①② 本番。181 ブロックを順に流す ----
print(f"\n1トークンぶんを {REP} 回流します（{len(blocks)} 転送）")
times = []
for r in range(REP):
    dst[:TOTR] = 0
    t0 = time.perf_counter()
    for b in blocks:
        dma.recvchannel.transfer(dst, b["eoff"], b["rows"]*4)
        dma.sendchannel.transfer(src, b["woff"], b["wlen"])
        dma.sendchannel.wait()
        dma.recvchannel.wait()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  {r+1}回目  {dt*1000:8.2f} ms")

dst.invalidate()
got = np.asarray(dst[:TOTR])
ok = np.array_equal(got, want)
print(f"\n照合: {'一致（全 %s 語）' % f'{TOTR:,}' if ok else '★不一致'}")
if not ok:
    bad = np.flatnonzero(got != want)
    print(f"  {len(bad):,}/{TOTR:,} 語が不一致。最初の5件:")
    for j in bad[:5]:
        blk = max(b for b in blocks if b["eoff"]//4 <= j)
        print(f"    語{j} (ブロック {blk['tag']} 行{j - blk['eoff']//4}): PL={got[j]} numpy={want[j]}")

t = min(times)
t_stream = TOTW / bw / 1e9
print("\n=== 1トークンの行列積 ===")
print(f"  実際にかかった時間 : {t*1000:8.2f} ms")
print(f"  うち帯域ぶん       : {t_stream*1000:8.2f} ms  ({100*t_stream/t:.1f}%)")
print(f"  うち呼び出しの固定費: {(t-t_stream)*1000:8.2f} ms  ({100*(1-t_stream/t):.1f}%)"
      f"  = {len(blocks)} 転送 × {(t-t_stream)/len(blocks)*1e6:.0f} µs")
print(f"  重み {meta.get('real_w', 134.48e6)/1e6:.1f} M（流した {TOTW*5/1e6:.1f} M）")
print(f"\n  この時間だけで {1000/(t*1000):.2f} tok/s。帯域ぶんだけなら {1000/(t_stream*1000):.1f} tok/s")

src.freebuffer(); dst.freebuffer()
