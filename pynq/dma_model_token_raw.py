#!/usr/bin/env python3
"""1転送 724 µs の内訳を切り分ける。

PYNQ の transfer()/wait() を通さず、AXI DMA のレジスタを直に叩く版。
  0x18 MM2S_SA / 0x28 MM2S_LENGTH（長さを書くと転送が始まる）
  0x48 S2MM_DA / 0x58 S2MM_LENGTH
  0x04 MM2S_DMASR / 0x34 S2MM_DMASR の bit1 = Idle

PYNQ の MMIO は .array で uint32 の numpy ビューが取れる。
メソッド呼び出しより添字アクセスのほうがずっと速い。

これで縮むなら「ドライバが遅い」、縮まないなら「インタフェースの設計が悪い」。
"""
import sys, os, json, time
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/tmac2_150.bit"
JOB = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/job"
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 3

meta = json.load(open(os.path.join(JOB, "job_index.json")))
blocks = meta["blocks"]; TOTW, TOTR = meta["total_w"], meta["total_rows"]

ol = Overlay(BIT); dma = ol.axi_dma_0
src = allocate(shape=(TOTW,), dtype=np.uint8)
dst = allocate(shape=(TOTR,), dtype=np.int32)
with open(os.path.join(JOB, "job_weights.bin"), "rb") as f:
    off = 0
    while True:
        b = f.read(4 << 20)
        if not b: break
        src[off:off+len(b)] = np.frombuffer(b, np.uint8); off += len(b)
src.flush()
want = np.fromfile(os.path.join(JOB, "job_expect.bin"), dtype=np.int32)

# チャネルを起動させるために一度だけ PYNQ 経由で流す
dma.recvchannel.transfer(dst, 0, blocks[0]["rows"]*4)
dma.sendchannel.transfer(src, 0, blocks[0]["wlen"])
dma.sendchannel.wait(); dma.recvchannel.wait()

mm  = dma.mmio.array                 # uint32 の numpy ビュー
SA, SLEN = 0x18 >> 2, 0x28 >> 2
DA, DLEN = 0x48 >> 2, 0x58 >> 2
SSR, DSR = 0x04 >> 2, 0x34 >> 2
sp, dp = src.physical_address, dst.physical_address

# 事前に数値を用意しておく（ループ内で辞書を引かない）
jobs = [(dp + b["eoff"], b["rows"]*4, sp + b["woff"], b["wlen"]) for b in blocks]

print(f"{meta['n_blocks']} ブロック / {TOTW/1024/1024:.2f} MB を {REP} 回")
times = []
for r in range(REP):
    dst[:] = 0
    t0 = time.perf_counter()
    for da, dl, sa, sl in jobs:
        mm[DA] = da; mm[DLEN] = dl          # 受け側を先に仕掛ける
        mm[SA] = sa; mm[SLEN] = sl          # 長さを書いた瞬間に走り出す
        while not (mm[SSR] & 2): pass       # MM2S Idle
        while not (mm[DSR] & 2): pass       # S2MM Idle
    dt = time.perf_counter() - t0
    times.append(dt); print(f"  {r+1}回目  {dt*1000:8.2f} ms")

dst.invalidate()
ok = np.array_equal(np.asarray(dst), want)
print(f"\n照合: {'一致（全 %s 語）' % f'{TOTR:,}' if ok else '★不一致'}")
t = min(times)
print(f"\n  実時間 {t*1000:.2f} ms / 1転送あたり {t/len(blocks)*1e6:.0f} µs")
src.freebuffer(); dst.freebuffer()
