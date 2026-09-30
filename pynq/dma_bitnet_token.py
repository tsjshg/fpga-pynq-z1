#!/usr/bin/env python3
"""bitnet_b1_58-large（三値で学習済み・0.7B）の1トークンぶんの行列積を PL で回す。

model/make_bitnet_job.py が作った仕事を読む。見出し付き形式なので
**387 組が数回の転送**にまとまる（段階6 の SmolLM2 は 181 転送だった）。

    [ 見出し 8B: R ][ x: 640 B ][ 重み: 128 B × R 行 ] × 何組でも → TLAST

HP 2本のときは **各組の行が半分ずつ**両ポートに入っている。
同じ添字のバッファを両方同時に走らせる（層の順序は壊れない）。

PYNQ の transfer/wait は1回 724 µs 食うので、レジスタを直に叩く。
  0x18 MM2S_SA / 0x28 MM2S_LENGTH（長さを書くと走り出す）
  0x48 S2MM_DA / 0x58 S2MM_LENGTH / 0x04・0x34 の bit1 = Idle

使い方: python3 dma_bitnet_token.py <bitstream> <仕事のディレクトリ> [繰り返し]
"""
import sys, os, json, time, subprocess
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/tmach2_150.bit"
JOB = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/bnjob2"
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 3

meta = json.load(open(os.path.join(JOB, "index.json")))
NP, NB = meta["n_ports"], meta["n_bufs"]
TOTW = meta["total_w"]
print(f"{meta['model']}")
print(f"  組 {meta['n_blocks']} / HP {NP} 本 × バッファ {NB} 本 = 往復 {NB} 回")
print(f"  重み {TOTW/1024/1024:.1f} MB / 入力文 {meta['prompt']!r}")

subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True)
time.sleep(3)
ol = Overlay(BIT); time.sleep(1)
dmas = [getattr(ol, f"axi_dma_{p}") for p in range(NP)]
try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
    print(f"  FCLK = {fclk:.2f} MHz → 道幅 {NP*8*fclk/1000:.2f} GB/s（{NP}本合計）")
except Exception:
    fclk = None

# ---- 重みを CMA に常駐させる ----
t0 = time.perf_counter()
srcs, dsts, wants = [], [], []
for p in range(NP):
    pm = meta["ports"][p]
    row = []
    for b in pm["bufs"]:
        a = allocate(shape=(b["nbytes"],), dtype=np.uint8)
        with open(os.path.join(JOB, b["file"]), "rb") as f:
            off = 0
            while True:
                c = f.read(4 << 20)
                if not c: break
                a[off:off+len(c)] = np.frombuffer(c, np.uint8); off += len(c)
        assert off == b["nbytes"]
        a.flush(); row.append(a)
    srcs.append(row)
    dsts.append(allocate(shape=(pm["total_rows"],), dtype=np.int32))
    w = np.fromfile(os.path.join(JOB, pm["expect"]), dtype=np.int32)
    assert w.size == pm["total_rows"], f"{w.size} != {pm['total_rows']}"
    wants.append(w)
used = sum(a.nbytes for r in srcs for a in r) + sum(d.nbytes for d in dsts)
print(f"  CMA に常駐: {used/1024/1024:.1f} MB  （{time.perf_counter()-t0:.1f} 秒・測定には含めない）")

# ---- チャネルを起こしてからレジスタ直叩きへ ----
for p in range(NP):
    dmas[p].recvchannel.transfer(dsts[p], 0, meta["ports"][p]["bufs"][0]["rows"]*4)
    dmas[p].sendchannel.transfer(srcs[p][0], 0, meta["ports"][p]["bufs"][0]["nbytes"])
    dmas[p].sendchannel.wait(); dmas[p].recvchannel.wait()
mms = [d.mmio.array for d in dmas]
SA, SLEN, DA, DLEN, SSR, DSR = 0x18>>2, 0x28>>2, 0x48>>2, 0x58>>2, 0x04>>2, 0x34>>2

# 添字を先に数値化しておく（ループ内で辞書を引かない）
plan = []
for k in range(NB):
    step = []
    for p in range(NP):
        b = meta["ports"][p]["bufs"][k]
        eo = sum(meta["ports"][p]["bufs"][j]["rows"] for j in range(k)) * 4
        step.append((mms[p], dsts[p].physical_address + eo, b["rows"]*4,
                     srcs[p][k].physical_address, b["nbytes"]))
    plan.append(step)

print(f"\n{REP} 回流します")
times = []
for r in range(REP):
    for d in dsts: d[:] = 0
    t0 = time.perf_counter()
    for step in plan:
        for mm, da, dl, sa, sl in step:          # 先に両方走らせる
            mm[DA] = da; mm[DLEN] = dl
            mm[SA] = sa; mm[SLEN] = sl
        for mm, _, _, _, _ in step:              # そのあと両方待つ
            while not (mm[SSR] & 2): pass
            while not (mm[DSR] & 2): pass
    dt = time.perf_counter() - t0
    times.append(dt); print(f"  {r+1}回目  {dt*1000:8.2f} ms")

ok = True; tot_words = 0
for p in range(NP):
    dsts[p].invalidate()
    got = np.asarray(dsts[p]); tot_words += got.size
    if np.array_equal(got, wants[p]):
        print(f"照合[ポート{p}]: 一致（{got.size:,} 語）")
    else:
        ok = False
        bad = np.flatnonzero(got != wants[p])
        print(f"照合[ポート{p}]: ★不一致 {len(bad):,}/{got.size:,} 語。最初の5件:")
        for j in bad[:5]:
            blk = [b for b in meta["blocks"] if b["per"][p]["eoff"]//4 <= j][-1]
            print(f"    語{j} ({blk['tag']} 行{j-blk['per'][p]['eoff']//4}): "
                  f"PL={got[j]} numpy={wants[p][j]}")

t = min(times); bw = TOTW/t/1e9
print(f"\n=== 1トークンの行列積（三値・lm_head も三値・HP {NP}本）===")
print(f"  実際にかかった時間 : {t*1000:8.2f} ms")
print(f"  読み出し           : {bw:.3f} GB/s", end="")
if fclk: print(f"   道幅の {100*bw/(NP*8*fclk*1e6/1e9):.1f}% / DDR の壁(2.009)の {100*bw/2.009:.1f}%")
else: print()
print(f"  演算               : {TOTW*5/t/1e9:.2f} G重み/s（三値 {TOTW*5/1e9:.2f} G個）")
print(f"  照合               : {'全 %s 語一致' % f'{tot_words:,}' if ok else '★不一致あり'}")
print(f"\n  行列積だけで {1000/(t*1000):.2f} tok/s")
# attnbench2.c を -O3 -mfpu=neon で焼いた実測（T=512）
CPU_ATTN, CPU_MISC, KV_PER_POS = 96.22, 10.68, 73728
print(f"  予算: 行列積 {t*1000:.1f}（実測） + attention {CPU_ATTN}（CPU 実測）"
      f" + 小物 {CPU_MISC}（実測） = {t*1000+CPU_ATTN+CPU_MISC:.1f} ms"
      f" → {1000/(t*1000+CPU_ATTN+CPU_MISC):.2f} tok/s")
pl_at = KV_PER_POS*512/bw/1e9*1000
print(f"  attention も PL に載せたら（試算）: {t*1000:.1f} + {pl_at:.1f} + {CPU_MISC}"
      f" = {t*1000+pl_at+CPU_MISC:.1f} ms → {1000/(t*1000+pl_at+CPU_MISC):.2f} tok/s")

for r in srcs:
    for a in r: a.freebuffer()
for d in dsts: d.freebuffer()
