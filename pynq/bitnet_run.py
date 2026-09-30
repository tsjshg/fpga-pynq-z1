#!/usr/bin/env python3
"""bitnet_b1_58-large を PYNQ-Z1 の上で通しで動かす（段階11）。

    文章 → トークン化 → [1トークンずつ: 埋め込み → 24層 → lm_head] → 次のトークン → …

行列積と attention は全部 PL、正規化・RoPE・silu・残差・量子化は CPU（e2e_core.py）。
KV キャッシュは CMA に INT8 で常駐し、毎トークン1行ずつ積み増す。

【DMA】
  axi_dma_0/1（単純転送）… 行列積。組ごとに1転送。x は重みバッファの空きに直接書く
  axi_dma_2/3（SG）      … attention。記述子の鎖で [見出し q K] と [V] を綴じる
    1ヘッドの置き場: [見出し 8][q 96][K: Tmax×96][V: Tmax×96]
    記述子 A = 見出し〜K の T 行（104+96T バイト）、記述子 B = V の T 行（96T バイト）
    1層 8ヘッド × 2 = 16 記述子を1本のストリームにする（先頭に SOF、末尾に EOF）。
    毎トークン書き換えるのは長さ欄だけ。層ごとに TAILDESC を1回書けば走る。

使い方（root で、しかもログインシェルで動かす。PYNQ の venv と XILINX_XRT は
/etc/profile.d でしか入らないので、sudo python3 … では pynq が見つからない）。
同じフォルダの run.sh がそのように起動し直す:
  ./run.sh gen "文章" [生成数]   生成
  ./run.sh eval                  Mac と比べる一式（ref_in.npz が要る）

必要なもの（make_bundle.sh が1つのフォルダにまとめる）:
  このファイル・e2e_core.py・e2e_cpu.c・dma_glue.c・bpe.py・tokenizer.json・
  pack_e2e.py の出力（g*.bin・index.json・params.npz・emb.f32）・combo3_125.bit/.hwh
  ビットストリームは既定でこのファイルと同じフォルダから読む（環境変数 BIT で変えられる）。
画面の表示は英語（公開用）。コメントは日本語のまま。
"""
import sys, os, json, time, math, subprocess
import numpy as np
from pynq import Overlay, allocate, MMIO

D   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import e2e_core as C
from bpe import SPBPE

BIT = os.environ.get("BIT", os.path.join(D, "combo3_125.bit"))

# 単純転送のレジスタ（32bit 語の添字）
SA, SLEN, DA, DLEN, SSR, DSR = 0x18>>2, 0x28>>2, 0x48>>2, 0x58>>2, 0x04>>2, 0x34>>2
# SG のレジスタ
MCR, MSR, MCUR, MTAIL = 0x00>>2, 0x04>>2, 0x08>>2, 0x10>>2
SCR, SSR2, SCUR, STAIL = 0x30>>2, 0x34>>2, 0x38>>2, 0x40>>2
SOF, EOF, CMPLT = 1 << 27, 1 << 26, 1 << 31
DW = 16                     # 記述子 1個 = 16 語（64 バイト境界）


def glue():
    """dma_glue.c を焼いて読み込む（ボード専用）。"""
    import ctypes
    so, src = os.path.join(D, "libdmaglue.so"), os.path.join(D, "dma_glue.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O2", "-mfpu=neon", "-shared", "-fPIC", "-o", so, src])
    L = ctypes.CDLL(so)
    P, I = ctypes.c_void_p, ctypes.c_int
    L.dma_gemv.argtypes = [P, P, I, P]; L.dma_gemv.restype = I
    L.dma_attn.argtypes = [P, I, I, I, I, P, P, P, P, P]; L.dma_attn.restype = I
    return L


def fill(buf, path):
    off = 0
    with open(path, "rb") as f:
        while True:
            c = f.read(4 << 20)
            if not c: break
            buf[off:off+len(c)] = np.frombuffer(c, np.uint8); off += len(c)
    return off


class CmaShort(RuntimeError):
    pass


class FpgaBackend:
    def _alloc(self, n, dt):
        """CMA から取る。落ちたらそこまでの合計を添えて止まる。"""
        try:
            a = allocate(shape=(n,), dtype=dt)
        except RuntimeError as e:
            got = self.cma
            for x in self.keep: x.freebuffer()          # 取った分は返してから止まる（やり直せるように）
            self.keep = []
            raise CmaShort(f"CMA allocation failed: {np.dtype(dt).itemsize*n/1024/1024:.1f} MiB "
                           f"after {got/1024/1024:.1f} MiB ({e})")
        self.cma += a.nbytes; self.keep.append(a)
        return a

    def __init__(self, ol, meta, tmax):
        self.cma = 0
        self.tmax = tmax
        self.t_g = self.t_a = 0.0
        self.n_g = self.n_a = 0
        self.keep = []
        NP = meta["n_ports"]; assert NP == 2

        # ---------- 行列積（DMA 0/1・単純転送） ----------
        gm = [getattr(ol, f"axi_dma_{n}").mmio.array for n in (0, 1)]
        for mm in gm:                       # 走らせておく（RS=1）
            mm[0x00>>2] = 1; mm[0x30>>2] = 1
        wb = []
        for p in range(NP):
            row = []
            for b in meta["bufs"][p]:
                # 中身はあとで読む。先に全部取っておく（ファイルを読むとページキャッシュが
                # CMA の空きに入り込み、後の確保が失敗しやすくなる）
                row.append(self._alloc(b["nbytes"], np.uint8))
            wb.append(row)
        gd = [self._alloc(meta["rows_per_port"][p], np.int32) for p in range(NP)]
        self.gplan, self.gx, self.gout, self.gtab = [], [], [], []
        self.L = glue()
        for b in meta["blocks"]:
            L = b["bpr"]*C.WPB
            plan, xs, outs = [], [], []
            for p, q in enumerate(b["per"]):
                src = wb[p][q["buf"]]
                plan.append((gm[p], gd[p].physical_address + q["eoff"]*4, q["rows"]*4,
                             src.physical_address + q["off"], q["nbytes"]))
                # PynqBuffer のまま触ると1回 100 µs かかる（切り出すたびに Python の
                # __array_finalize__ が走る）。同じメモリの素の ndarray にしておく
                xs.append(src[q["off"]+8 : q["off"]+8+L].view(np.ndarray))
                outs.append(gd[p][q["eoff"] : q["eoff"]+q["rows"]].view(np.ndarray))
            # 状態レジスタを転送中に叩き続けると、転送そのものが 1回 150 µs ほど遅れる
            # （実測。叩かずに待てば理論どおりに終わる）。バイト数から終わる時刻を見積もり、
            # その 9 割までは時計だけ見て待つ。1ポートの道幅は 8 B × 125 MHz = 1.0 GB/s
            self.gplan.append(plan); self.gx.append(xs); self.gout.append(outs)
            (m0, d0, dl0, s0, sl0), (m1, d1, dl1, s1, sl1) = plan
            self.gtab.append(np.array([m0.ctypes.data, m1.ctypes.data, d0, dl0, s0, sl0,
                                       d1, dl1, s1, sl1, xs[0].ctypes.data, xs[1].ctypes.data,
                                       outs[0].ctypes.data, outs[0].size,
                                       outs[1].ctypes.data, outs[1].size], np.uint32))
        self.pg = [t.ctypes.data for t in self.gtab]

        # ---------- attention（DMA 2/3・SG） ----------
        self.slot = 8 + C.HD + 2*tmax*C.HD
        self.voff = 8 + C.HD + tmax*C.HD
        per_buf = max(1, (8 << 20) // (8*self.slot))           # 1バッファ 8 MiB 以下（段階10 で確実だった大きさ）
        self.am = [MMIO(ol.ip_dict[f"axi_dma_{n}"]["phys_addr"], 0x10000).array for n in (2, 3)]
        self.kv, self.desc, self.sdesc, self.aout = [], [], [], []
        self._maddr, self._saddr = [], []
        for p in range(NP):
            kvb = []
            for l0 in range(0, C.NL, per_buf):
                n = min(per_buf, C.NL - l0)
                a = self._alloc(n*8*self.slot, np.uint8); a[:] = 0
                kvb.append(a)
            views = []
            for l in range(C.NL):
                a = kvb[l // per_buf]
                views.append((a, (l % per_buf)*8*self.slot))
            ds = self._alloc(C.NL*16*DW, np.uint32); ds[:] = 0
            sd = self._alloc(C.NL*DW, np.uint32); sd[:] = 0
            ao = self._alloc(C.NL*8*97, np.int32); ao[:] = 0
            dv = ds.reshape(C.NL*16, DW)
            for l in range(C.NL):
                a, o = views[l]
                for hl in range(8):
                    base = a.physical_address + o + hl*self.slot
                    for j, addr in enumerate((base, base + self.voff)):
                        i = l*16 + hl*2 + j
                        dv[i, 0] = ds.physical_address + ((i+1) % (C.NL*16))*DW*4
                        dv[i, 2] = addr
            sv = sd.reshape(C.NL, DW)
            for l in range(C.NL):
                sv[l, 0] = sd.physical_address + ((l+1) % C.NL)*DW*4
                sv[l, 2] = ao.physical_address + l*8*97*4
                sv[l, 6] = 8*97*4
            self.kv.append([a[o:o+8*self.slot].view(np.ndarray).reshape(8, self.slot)
                            for a, o in views])
            self._maddr.append(ds.physical_address); self._saddr.append(sd.physical_address)
            self.desc.append(dv.view(np.ndarray)); self.sdesc.append(sv.view(np.ndarray))
            self.aout.append(ao.view(np.ndarray).reshape(C.NL, 8, 97))
            # 起こす: リセット → 先頭記述子 → RS=1
            mm = self.am[p]
            mm[MCR] = 4
            t0 = time.time()
            while mm[MCR] & 4:
                if time.time() - t0 > 1: raise RuntimeError("DMA reset did not complete")
            mm[MCUR] = ds.physical_address; mm[MCR] = 1
            mm[SCUR] = sd.physical_address; mm[SCR] = 1
            if (mm[MSR] & 1) or (mm[SSR2] & 1):
                raise RuntimeError(f"SG DMA {p} is still halted: MM2S {mm[MSR]:#x} S2MM {mm[SSR2]:#x}")
        # 長さ欄の雛形
        hl = np.tile(np.repeat(np.arange(8), 2), C.NL)
        j  = np.tile(np.array([0, 1]), C.NL*8)
        self.flag = np.where((j == 0) & (hl == 0), SOF, 0) | np.where((j == 1) & (hl == 7), EOF, 0)
        self.isA = (j == 0)
        self.curT = -1
        # C から叩くための層ごとの数表
        self.atab = []
        for l in range(C.NL):
            self.atab.append(np.array(
                [self.am[0].ctypes.data, self.am[1].ctypes.data,
                 self.kv[0][l].ctypes.data, self.kv[1][l].ctypes.data,
                 self._maddr[0] + (l*16 + 15)*DW*4, self._maddr[1] + (l*16 + 15)*DW*4,
                 self._saddr[0] + l*DW*4, self._saddr[1] + l*DW*4,
                 self.sdesc[0][l, 7:].ctypes.data, self.sdesc[1][l, 7:].ctypes.data,
                 self.aout[0][l].ctypes.data, self.aout[1][l].ctypes.data], np.uint32))
        self.pa = [t.ctypes.data for t in self.atab]
        # ---------- 最後に重みを読み込む ----------
        for p in range(NP):
            for a, b in zip(wb[p], meta["bufs"][p]):
                assert fill(a, os.path.join(D, b["file"])) == b["nbytes"]

    def bind(self, r):
        """Runner の作業領域の番地を覚える（ndarray.ctypes.data は1回 37 µs かかる）。"""
        self.r = r
        self.pxq, self.pacc = r.xq.ctypes.data, r.acc.ctypes.data
        self.pqi, self.pkq, self.pvq = r.qi.ctypes.data, r.kq.ctypes.data, r.vq.ctypes.data
        self.pmm, self.pao = r.mm.ctypes.data, r.ao.ctypes.data

    def gemv(self, bi, xq, out):
        t0 = time.perf_counter()
        n = self.L.dma_gemv(self.pg[bi], self.pxq, xq.size, self.pacc)
        self.t_g += time.perf_counter() - t0; self.n_g += 1
        return n

    def _set_T(self, T):
        """このトークンの長さを全記述子に書き、完了印を消す。"""
        ctrl = (self.flag | np.where(self.isA, 8 + C.HD + C.HD*T, C.HD*T)).astype(np.uint32)
        for p in range(2):
            self.desc[p][:, 6] = ctrl
            self.desc[p][:, 7] = 0
            self.sdesc[p][:, 7] = 0
        self.curT = T

    def attn(self, l, pos, kq, vq, qi, mm, out):
        t0 = time.perf_counter()
        T = pos + 1
        assert T <= self.tmax
        if T != self.curT:
            self._set_T(T)
        rc = self.L.dma_attn(self.pa[l], self.slot, 8 + C.HD + pos*C.HD, self.voff + pos*C.HD,
                             T, self.pmm, self.pqi, self.pkq, self.pvq, self.pao)
        if rc:
            m = self.am[-1 - rc]
            raise RuntimeError(f"attention layer {l} port {-1-rc} did not finish: "
                               f"MM2S {m[MSR]:#x} S2MM {m[SSR2]:#x}")
        self.t_a += time.perf_counter() - t0; self.n_a += 1

    def check(self):
        for p in range(2):
            m = self.am[p]
            if (m[MSR] | m[SSR2]) & 0x770:
                raise RuntimeError(f"SG DMA {p} error: MM2S {m[MSR]:#x} S2MM {m[SSR2]:#x}")


def main():
    if os.geteuid() != 0:
        # ビットストリームの書き込み・DMA のレジスタ・drop_caches はどれも root が要る。
        # root でないと pyxrt が「Could not open device」で落ちて、原因が分かりにくい
        args = " ".join(f'"{a}"' if " " in a else a for a in sys.argv[1:])   # 外側が ' なので " で包む
        sys.exit("This must run as root (loading the bitstream and driving the DMAs need it),\n"
                 "inside a login shell (PYNQ's venv is only set up by /etc/profile.d). Use:\n\n"
                 f"  {D}/run.sh {args}\n\n"
                 "or equivalently:\n\n"
                 f"  sudo bash -lc 'cd {D} && python3 bitnet_run.py {args}'\n")
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    meta = json.load(open(os.path.join(D, "index.json")))
    prm = dict(np.load(os.path.join(D, "params.npz")))
    prm["emb"] = np.memmap(os.path.join(D, "emb.f32"), np.float32, "r").reshape(-1, C.H)
    tmax = int(prm["tmax"])
    sp = SPBPE(os.path.join(D, "tokenizer.json"))

    # CMA は断片化のせいで、同じ構成でも取れたり取れなかったりする（2026-09-30 に 150 MiB で
    # 1回落ち、直後の2回は 179 MiB 取れた）。落ちたら返して、キャッシュを捨ててやり直す
    for attempt in range(3):
        subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True)
        time.sleep(3)
        t0 = time.time()
        ol = Overlay(BIT); time.sleep(1)
        try:
            be = FpgaBackend(ol, meta, tmax)
            break
        except CmaShort as e:
            print(f"{e} -> retrying ({attempt+1}/3)", flush=True)
    else:
        sys.exit("Could not allocate CMA in 3 attempts. Check that /proc/meminfo shows "
                 "CmaTotal: 262144 kB (cma=256M) and that nothing else (e.g. a Jupyter notebook) "
                 "holds CMA buffers.")
    print(f"Loaded in {time.time()-t0:.1f} s / {be.cma/1024/1024:.1f} MiB resident in CMA"
          f" (weights {meta['total_w']/1024/1024:.1f} + KV etc.) / Tmax {tmax}", flush=True)

    def fresh():
        be.curT = -1                        # 次の step で長さ欄と完了印を書き直す
        be.t_g = be.t_a = 0.0; be.n_g = be.n_a = 0

    def gen(ids, n, rec=None, show=True, keep=False):
        fresh()
        bk = C.Recorder(be) if rec else be
        r = C.Runner(prm, bk)
        out = list(ids); lgs = []; tt = []; comp = []
        for pos in range(len(ids) + n - 1):
            g0, a0 = be.t_g, be.t_a
            t1 = time.perf_counter()
            lg = r.step(out[pos], pos, logits=keep)
            tt.append(time.perf_counter() - t1)
            comp.append((pos + 1, tt[-1], be.t_g - g0, be.t_a - a0))   # T, 全体, 行列積, attention
            if keep: lgs.append(lg.copy())
            if pos >= len(ids) - 1:
                out.append(r.top)
                if show:
                    print(sp.decode(out[len(ids):]).split("\n")[-1][-60:].rjust(60)
                          + f"   {tt[-1]*1000:6.1f} ms", end="\r", flush=True)
        be.check()
        if show: print()
        gen.comp = np.array(comp)
        return out, np.array(lgs), tt, (bk.log if rec else None)

    if mode == "gen":
        text = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
        ids = sp.encode(text)
        print(f"Prompt {text!r} ({len(ids)} tokens) -> generating {n} tokens\n")
        out, _, tt, _ = gen(ids, n)
        print(f"\n{sp.decode(out)}\n")
        report(be, tt, len(ids))
        if os.environ.get("TT"): np.save(os.environ["TT"], gen.comp)   # 列: T, 全体, 行列積, attention（秒）
        return

    # ---- eval: Mac の基準と同じことをして、全部書き出す ----
    ref = np.load(os.path.join(D, "ref_in.npz"))
    save = {}
    ids = [int(v) for v in ref["ppl_ids"]]
    fresh(); r = C.Runner(prm, be); nll, lgs, tt = [], [], []
    for pos in range(len(ids) - 1):
        t1 = time.perf_counter()
        lg = r.step(ids[pos], pos).astype(np.float64)
        tt.append(time.perf_counter() - t1)
        lse = lg.max() + np.log(np.exp(lg - lg.max()).sum())
        nll.append(lse - lg[ids[pos+1]]); lgs.append(lg.astype(np.float32))
    be.check()
    save["ppl"] = float(np.exp(np.mean(nll))); save["ppl_logits"] = np.array(lgs)
    print(f"perplexity {save['ppl']:.2f} ({len(ids)} tokens, one at a time through the PL)")
    report(be, tt, 0)
    for i in range(2):
        pid = [int(v) for v in ref[f"gen{i}_ids"][:int(ref[f"gen{i}_plen"])]]
        n = len(ref[f"gen{i}_ids"]) - len(pid)
        out, lg, tt, log = gen(pid, n, rec=(i == 0), show=False, keep=True)
        print(f"  {sp.decode(pid)!r} -> {sp.decode(out[len(pid):])!r}")
        save[f"gen{i}_ids"] = np.array(out); save[f"gen{i}_logits"] = lg
        if log is not None:
            lo = np.empty(len(log), dtype=object)
            for k, e in enumerate(log): lo[k] = e
            np.savez(os.path.join(D, "dump_gen0.npz"), log=lo)
            print(f"    recorded {len(log):,} PL calls to dump_gen0.npz")
    np.savez(os.path.join(D, "board_out.npz"), **save)
    print("wrote board_out.npz and dump_gen0.npz")


def report(be, tt, nprompt):
    tt = np.array(tt) * 1000
    ng, na = be.n_g / len(tt), be.n_a / len(tt)
    g, a = be.t_g*1000/len(tt), be.t_a*1000/len(tt)
    m = np.median(tt)
    print(f"Per token (median of {len(tt)}): {m:.1f} ms -> {1000/m:.2f} tok/s"
          f" (mean {tt.mean():.1f} ms)")
    print(f"  matmul   {g:7.2f} ms ({ng:.0f} DMA transfers, incl. writing x and reading results)")
    print(f"  attention{a:7.2f} ms ({na:.0f} DMA transfers, incl. appending K/V)")
    print(f"  CPU      {m-g-a:7.2f} ms (RMSNorm, RoPE, quantization, SiLU, residuals, embedding)")
    print(f"  min {tt.min():.1f} / max {tt.max():.1f} ms")
    if len(tt) >= 100:
        print("  by context length (median of the 16 tokens around T):")
        for T in (16, 64, 128, 256, 384, 512):
            i = T - 1 - nprompt + 1                      # tt[i] は位置 nprompt-1+i のトークン
            if 8 <= i < len(tt) - 8:
                m = np.median(tt[i-8:i+8])
                print(f"    T={T:4d}: {m:6.1f} ms -> {1000/m:5.2f} tok/s")


if __name__ == "__main__":
    main()
