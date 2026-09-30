#!/usr/bin/env python3
"""行列積コアと attention コアを同じビットストリームで動かす（段階10）。

    axi_dma_0/1 -> axis_tmacv 0/1（三値・行長可変）
    axi_dma_2/3 -> axis_attnv 0/1（attention・head_dim 96）
どちらも HP0 と HP2 を共有する。1トークンの中で交互に動くので、
同時に使うことはない。切り替え回路は無く、**使う DMA を選ぶだけ**。

重みも KV も CMA に常駐させる（合計 約 179 MiB）。段階8 で行長を可変にして
重みを 169.7 → 141.3 MiB に減らしたので、両方を同時に載せられるようになった。

使い方: python3 dma_token_combo.py <bitstream> <行列積の仕事> <attention の仕事> [繰り返し]
"""
import sys, os, json, time, subprocess
import numpy as np
from pynq import Overlay, allocate

BIT  = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/combo2_125.bit"
JOBG = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/bnjobv"
JOBA = sys.argv[3] if len(sys.argv) > 3 else "/home/claude/atjob"
REP  = int(sys.argv[4]) if len(sys.argv) > 4 else 3

SA, SLEN, DA, DLEN, SSR, DSR = 0x18>>2, 0x28>>2, 0x48>>2, 0x58>>2, 0x04>>2, 0x34>>2

mg = json.load(open(os.path.join(JOBG, "index.json")))
ma = json.load(open(os.path.join(JOBA, "index.json")))
print(f"{mg['model']}")
print(f"  行列積  : {mg['n_blocks']:4d} 組 / 往復 {mg['n_bufs']} 回 / {mg['total_w']/1024/1024:6.1f} MiB")
print(f"  attention: {ma['n_blocks']:4d} 組 / 往復 {ma['n_bufs']} 回 / {ma['total_w']/1024/1024:6.1f} MiB"
      f" / T = {ma['T']}")

subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True)
time.sleep(3)
ol = Overlay(BIT); time.sleep(1)
try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
    print(f"  FCLK = {fclk:.2f} MHz → 道幅 {2*8*fclk/1000:.3f} GB/s（2本合計）")
except Exception:
    fclk = None

def load(job, meta, dmas):
    """仕事を CMA に常駐させて、レジスタ直叩き用の数値を作る。"""
    srcs, dsts, wants = [], [], []
    for p in range(meta["n_ports"]):
        pm = meta["ports"][p]; row = []
        for b in pm["bufs"]:
            a = allocate(shape=(b["nbytes"],), dtype=np.uint8)
            with open(os.path.join(job, b["file"]), "rb") as f:
                off = 0
                while True:
                    c = f.read(4 << 20)
                    if not c: break
                    a[off:off+len(c)] = np.frombuffer(c, np.uint8); off += len(c)
            assert off == b["nbytes"]; a.flush(); row.append(a)
        srcs.append(row)
        dsts.append(allocate(shape=(pm["total_rows"],), dtype=np.int32))
        w = np.fromfile(os.path.join(job, pm["expect"]), dtype=np.int32)
        assert w.size == pm["total_rows"]; wants.append(w)
    # チャネルを起こす
    for p, d in enumerate(dmas):
        d.recvchannel.transfer(dsts[p], 0, meta["ports"][p]["bufs"][0]["rows"]*4)
        d.sendchannel.transfer(srcs[p][0], 0, meta["ports"][p]["bufs"][0]["nbytes"])
        d.sendchannel.wait(); d.recvchannel.wait()
    mms = [d.mmio.array for d in dmas]
    plan = []
    for k in range(meta["n_bufs"]):
        step = []
        for p in range(meta["n_ports"]):
            b = meta["ports"][p]["bufs"][k]
            eo = sum(meta["ports"][p]["bufs"][j]["rows"] for j in range(k))*4
            step.append((mms[p], dsts[p].physical_address+eo, b["rows"]*4,
                         srcs[p][k].physical_address, b["nbytes"]))
        plan.append(step)
    return srcs, dsts, wants, plan

t0 = time.perf_counter()
dg = [getattr(ol, f"axi_dma_{n}") for n in (0, 1)]
da = [getattr(ol, f"axi_dma_{n}") for n in (2, 3)]
sg, tg, wg, pg = load(JOBG, mg, dg)
sa, ta, wa, pa = load(JOBA, ma, da)
used = sum(a.nbytes for r in sg+sa for a in r) + sum(d.nbytes for d in tg+ta)
print(f"  CMA に常駐: {used/1024/1024:.1f} MiB  （{time.perf_counter()-t0:.1f} 秒・測定には含めない）")

def run(plan):
    for step in plan:
        for mm, dd, dl, ss, sl in step:
            mm[DA] = dd; mm[DLEN] = dl
            mm[SA] = ss; mm[SLEN] = sl
        for mm, _, _, _, _ in step:
            while not (mm[SSR] & 2): pass
            while not (mm[DSR] & 2): pass

print(f"\n{REP} 回流します（行列積 → attention の順）")
tot, tgem, tatt = [], [], []
for r in range(REP):
    for d in tg+ta: d[:] = 0
    t0 = time.perf_counter(); run(pg); t1 = time.perf_counter(); run(pa); t2 = time.perf_counter()
    tot.append(t2-t0); tgem.append(t1-t0); tatt.append(t2-t1)
    print(f"  {r+1}回目  行列積 {(t1-t0)*1000:7.2f} + attention {(t2-t1)*1000:6.2f}"
          f" = {(t2-t0)*1000:7.2f} ms")

ok = True
for nm, dsts, wants in (("行列積", tg, wg), ("attention", ta, wa)):
    for p in range(len(dsts)):
        dsts[p].invalidate()
        g = np.asarray(dsts[p])
        if np.array_equal(g, wants[p]):
            print(f"照合[{nm} ポート{p}]: 一致（{g.size:,} 語）")
        else:
            ok = False
            bad = np.flatnonzero(g != wants[p])
            print(f"照合[{nm} ポート{p}]: ★不一致 {len(bad):,}/{g.size:,} 語")

g, a, t = min(tgem), min(tatt), min(tot)
MISC = 10.68
print(f"\n=== 1トークン（行列積 + attention を同じビットストリームで）===")
print(f"  行列積    : {g*1000:7.2f} ms   {mg['total_w']/g/1e9:.3f} GB/s")
print(f"  attention : {a*1000:7.2f} ms   {ma['total_w']/a/1e9:.3f} GB/s")
print(f"  合計(PL)  : {t*1000:7.2f} ms")
print(f"  照合      : {'全語一致' if ok else '★不一致あり'}")
print(f"\n  ＋ 小物 {MISC}（CPU 実測） = {t*1000+MISC:.1f} ms → **{1000/(t*1000+MISC):.2f} tok/s**")
print(f"  （段階9 の別々のビットストリーム: 76.87 + 20.22 + {MISC} = 107.8 ms → 9.28 tok/s）")

for r in sg+sa:
    for x in r: x.freebuffer()
for d in tg+ta: d.freebuffer()
