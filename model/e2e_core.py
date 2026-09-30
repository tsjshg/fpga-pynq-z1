#!/usr/bin/env python3
"""通しで動かすときの CPU 側。**Mac でもボードでも同じこのファイルを使う。**

1トークンの流れ（P=1・層ごと）:

    x ─ rms ─ 量子化 ─[PL 行列積 qkv]─ 逆量子化 ─ RoPE ─ q,K,V を INT8 に
      ─[PL attention]─ 割り算 ─ rms ─ 量子化 ─[PL 行列積 o]─ 逆量子化 ─ 残差
      ─ rms ─ 量子化 ─[PL 行列積 gate/up]─ 逆量子化 ─ silu·up ─ rms ─ 量子化
      ─[PL 行列積 down]─ 逆量子化 ─ 残差
    最後に rms ─ 量子化 ─[PL 行列積 lm_head（三値）]─ 逆量子化 → logits

[PL …] の部分だけを「裏方（backend）」に任せる。裏方は2つある。
  NumpyBackend … Mac で整数のまま厳密に計算する（照合相手）
  FpgaBackend  … ボードで DMA を叩く（pynq/bitnet_run.py）
それ以外の浮動小数の処理は **e2e_cpu.c の同じ関数**が両方で走る
（numpy だとボードで 305 ms/トークンかかったため C にした。結果が揃うように焼き方を固定）。

**KV の尺度は層・ヘッドごとに固定**（校正文で測った max|K|, max|V|）。
bitnet_np.attn_circuit(mode=2) は「全位置の max」で尺度を決めていたが、それは
未来の K を見ていることになり、しかも T が伸びるたびに過去の INT8 を全部
量子化し直すことになる。KV キャッシュを INT8 のまま積み増すには尺度を固定するしかない。
"""
import math
import numpy as np

H, FF, NL, NH, HD = 1536, 4096, 24, 16, 96
WPB = 40                    # 1ビート = 三値 40 個
SHM, TAU = 24, 8.0          # 回路のスコア倍率の固定シフトと、指数表の温度
EROM = np.array([max(0, min(255, int(round(255.0*math.exp(-d/TAU))))) for d in range(256)],
                dtype=np.int32)
POW = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
# 1バイト → 三値5個（0..242 を 3進に）。243..255 は使わない
TRIT = np.array([[((b // 3**d) % 3) - 1 for d in range(5)] for b in range(256)], dtype=np.int8)


def pack(T):
    """三値 (rows, L) → uint8 (rows, L/5)。回路が読む形。"""
    rows, L = T.shape
    u = (T.astype(np.uint16) + 1).reshape(rows, L//5, 5)
    return (u * POW).sum(2, dtype=np.uint16).astype(np.uint8)


def unpack(P, L):
    return TRIT[P].reshape(P.shape[0], -1)[:, :L]


def _lib():
    """e2e_cpu.c を焼いたもの。無ければその場で焼く（Mac は clang、ボードは gcc）。"""
    import ctypes, os, subprocess, sys
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "libe2e.dylib" if sys.platform == "darwin" else "libe2e.so")
    src = os.path.join(here, "e2e_cpu.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        arch = [] if sys.platform == "darwin" else ["-mfpu=neon"]
        subprocess.check_call(["cc", "-O2", *arch, "-ffp-contract=off", "-fno-fast-math",
                               "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so)
    P, F, I, Dd = ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_double
    for n, at in dict(rmsq=[P, P, I, F, P, P], attn_post=[P, P, P, F, P, P],
                      resid_rmsq=[P, P, F, F, P, F, P, P],
                      mlp_post=[P, F, F, F, P, F, P, P]).items():
        getattr(L, n).argtypes = at; getattr(L, n).restype = F
    L.qkv_post.argtypes = [P, F, F, F, F, P, P, P, P, P, P, P, P, P]
    L.head_post.argtypes = [P, F, F, I, P]; L.head_post.restype = I
    return L


class Runner:
    """1トークンずつ進める。backend が PL（または厳密な代役）。浮動小数の小物は e2e_cpu.c。"""

    def __init__(self, prm, backend):
        self.L = _lib()
        self.be = backend
        self.eps = float(prm["eps"])
        self.emb = prm["emb"]                                     # memmap でもよい
        c = lambda a, t: np.ascontiguousarray(a, dtype=t)
        self.a = dict(n_in=c(prm["n_in"], np.float32), n_attn=c(prm["n_attn"], np.float32),
                      n_post=c(prm["n_post"], np.float32), n_ffn=c(prm["n_ffn"], np.float32),
                      n_out=c(prm["n_out"], np.float32),
                      cos=c(prm["cos"], np.float32), sin=c(prm["sin"], np.float32),
                      ik=c(127.0 / prm["kmax"], np.float32), iv=c(127.0 / prm["vmax"], np.float32),
                      sk=c(prm["kmax"].astype(np.float64) / 127.0, np.float64),
                      sv=c(prm["vmax"].astype(np.float64) / 127.0, np.float64))
        ad = lambda k, i: self.a[k][i].ctypes.data
        self.pl = [dict((k, ad(k, l)) for k in ("n_in", "n_attn", "n_post", "n_ffn",
                                                "ik", "iv", "sk", "sv")) for l in range(NL)]
        self.pcos = [ad("cos", t) for t in range(len(self.a["cos"]))]
        self.psin = [ad("sin", t) for t in range(len(self.a["sin"]))]
        self.pout = self.a["n_out"].ctypes.data
        self.g = [[float(v) for v in r] for r in prm["gamma"]]
        self.gh = float(prm["g_head"])
        # 作業領域（固定番地）
        self.x = np.zeros(H, np.float32); self.h = np.zeros(FF, np.float32)
        self.xq = np.zeros(FF, np.int8)
        self.kq = np.zeros((NH, HD), np.int8); self.vq = np.zeros((NH, HD), np.int8)
        self.qi = np.zeros((NH, HD), np.int8); self.mm = np.zeros(NH, np.int64)
        self.acc = np.zeros(40000, np.int32)                  # 行列積の結果（lm_head の 32002 語が最大）
        self.ao = np.zeros((NH, HD+1), np.int32)              # attention の結果
        self.lg = np.zeros(len(self.emb), np.float32)
        # ndarray.ctypes.data はボードで 1回 37 µs かかるので、番地は最初に1回だけ取る
        self.px, self.ph, self.pxq = self.x.ctypes.data, self.h.ctypes.data, self.xq.ctypes.data
        self.pkq, self.pvq = self.kq.ctypes.data, self.vq.ctypes.data
        self.pqi, self.pmm = self.qi.ctypes.data, self.mm.ctypes.data
        self.pacc, self.pao, self.plg = self.acc.ctypes.data, self.ao.ctypes.data, self.lg.ctypes.data
        self.top = -1                                         # 直前の step の argmax
        if hasattr(backend, "bind"):
            backend.bind(self)                                # 裏方が作業領域の番地を覚える

    def step(self, tok, pos, logits=True):
        """1トークン進めて logits を返す（self.lg を使い回す。self.top が argmax）。
        logits=False なら argmax だけ求める（lm_head の後始末が 0.4 ms 減る）。"""
        L, be, eps, xq, acc, ao = self.L, self.be, self.eps, self.xq, self.acc, self.ao
        px, ph, pxq, pa = self.px, self.ph, self.pxq, self.pacc
        self.x[:] = self.emb[tok]
        s = L.rmsq(px, self.pl[0]["n_in"], H, eps, ph, pxq)
        for l in range(NL):
            pl = self.pl[l]
            gq, gk, gv, go, gg, gu, gd = self.g[l]
            be.gemv(4*l, xq[:H], acc)
            L.qkv_post(pa, gq, gk, gv, s, self.pcos[pos], self.psin[pos], pl["ik"], pl["iv"],
                       pl["sk"], self.pkq, self.pvq, self.pqi, self.pmm)
            be.attn(l, pos, self.kq, self.vq, self.qi, self.mm, ao)
            s = L.attn_post(self.pao, pl["sv"], pl["n_attn"], eps, ph, pxq)
            be.gemv(4*l+1, xq[:H], acc)
            s = L.resid_rmsq(px, pa, go, s, pl["n_post"], eps, ph, pxq)
            be.gemv(4*l+2, xq[:H], acc)
            s = L.mlp_post(pa, gg, gu, s, pl["n_ffn"], eps, ph, pxq)
            be.gemv(4*l+3, xq[:FF], acc)
            nxt = self.pl[l+1]["n_in"] if l+1 < NL else self.pout
            s = L.resid_rmsq(px, pa, gd, s, nxt, eps, ph, pxq)
        n = be.gemv(4*NL, xq[:H], acc)
        self.top = L.head_post(pa, self.gh, s, n, self.plg if logits else None)
        return self.lg[:n] if logits else None


class NumpyBackend:
    """PL の代役。整数で厳密に計算する（三値×INT8 の和は float32 で厳密）。"""

    def __init__(self, blocks, Tmax):
        self.W = [(T.astype(np.float32), ind) for T, ind in blocks]
        self.K = np.zeros((NL, NH, Tmax, HD), np.int8)
        self.V = np.zeros((NL, NH, Tmax, HD), np.int8)

    def gemv(self, bi, xq, out):
        """out の先頭に結果を書き、行数を返す。"""
        W, ind = self.W[bi]
        xp = np.zeros(W.shape[1], np.float32); xp[:ind] = xq
        out[:W.shape[0]] = W @ xp
        return W.shape[0]

    def attn(self, l, pos, kq, vq, qi, mm, out):
        self.K[l, :, pos] = kq; self.V[l, :, pos] = vq
        T = pos + 1
        K = self.K[l, :, :T].astype(np.int64); V = self.V[l, :, :T].astype(np.int64)
        si = np.einsum("htd,hd->ht", K, qi.astype(np.int64))
        s8 = np.clip((si * mm[:, None]) >> SHM, -128, 127)
        e = EROM[s8.max(1, keepdims=True) - s8].astype(np.int64)
        out[:, :HD] = np.einsum("ht,htd->hd", e, V)         # 回路と同じ形: 96 語 + 分母
        out[:, HD] = e.sum(1)


class Recorder:
    """裏方への呼び出しを全部書き留める。ボードで記録 → Mac で整数を照合する。"""

    def __init__(self, be):
        self.be, self.log = be, []

    def bind(self, r):
        if hasattr(self.be, "bind"): self.be.bind(r)

    def gemv(self, bi, xq, out):
        n = self.be.gemv(bi, xq, out)
        self.log.append(("g", bi, xq.copy(), out[:n].copy()))
        return n

    def attn(self, l, pos, kq, vq, qi, mm, out):
        self.be.attn(l, pos, kq, vq, qi, mm, out)
        self.log.append(("a", l, pos, kq.copy(), vq.copy(), qi.copy(), mm.copy(), out.copy()))
