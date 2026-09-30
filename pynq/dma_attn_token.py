#!/usr/bin/env python3
"""bitnet_b1_58-large の1トークンぶんの attention を PL で回す（段階9）。

model/make_attn_job.py が作った仕事を読む。

    [ 見出し 8B: [31:0]=T, [49:32]=m ][ q: 96 B ][ K: T×96 B ][ V: T×96 B ] × 組
    → 出力 96 語 + 分母 1 語 = 97 語

1トークンで 24層 × 16ヘッド = 384 組。**1層の 16 ヘッドは互いに独立**なので
ヘッドを半分ずつ両ポートに振り分けてある（層の順序依存は壊れない）。

head_dim 96 / GQA なしなので **1バイトあたり 1 MAC**。完全に帯域律速。

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
print(f"  KV など {TOTW/1024/1024:.1f} MiB / 文脈長 T = {meta['T']}")

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
print(f"\n=== 1トークンの attention（softmax 込み・HP {NP}本）===")
print(f"  実際にかかった時間 : {t*1000:8.2f} ms")
print(f"  読み出し           : {bw:.3f} GB/s", end="")
if fclk: print(f"   道幅の {100*bw/(NP*8*fclk*1e6/1e9):.1f}% / DDR の壁(2.009)の {100*bw/2.009:.1f}%")
else: print()
macs = meta["n_blocks"]*2*meta["T"]*meta["hd"]
print(f"  演算               : {macs/t/1e9:.2f} G MAC/s（1バイトあたり {macs/TOTW:.2f} MAC）")
print(f"  照合               : {'全 %s 語一致' % f'{tot_words:,}' if ok else '★不一致あり'}")
CPU_ATTN = 96.22       # bench/attnbench2.c を -O3 -mfpu=neon で焼いた実測（T=512）
GEMV, MISC = 76.87, 10.68   # 段階8 の実測 / 小物の実測
print(f"\n  CPU 実測 {CPU_ATTN} ms → **{CPU_ATTN/(t*1000):.1f} 倍**")
print(f"  予算(全部実測): 行列積 {GEMV} + attention {t*1000:.2f} + 小物 {MISC}"
      f" = {GEMV+t*1000+MISC:.1f} ms → {1000/(GEMV+t*1000+MISC):.2f} tok/s")
print(f"  （attention を CPU に残すと {GEMV+CPU_ATTN+MISC:.1f} ms → "
      f"{1000/(GEMV+CPU_ATTN+MISC):.2f} tok/s）")

for r in srcs:
    for a in r: a.freebuffer()
for d in dsts: d.freebuffer()
