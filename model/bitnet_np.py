#!/usr/bin/env python3
"""1bitLLM/bitnet_b1_58-large を numpy だけで動かす。

**三値で学習済み**のモデル。SmolLM2 を後から丸めたのとは違い、
リポジトリの utils_quant.py と同じ手順で量子化すれば学習時と同じ計算になる。

  重み: s = 1/mean(|W|)        W̃ = clamp(round(W·s), -1, 1)      … テンソル単位
  活性: s = 127/max(|x|)       x̃ = clamp(round(x·s), -128, 127)  … 行ごと
  （どちらも clamp(min=1e-5)）

構造は LLaMA に RMSNorm を2つ足したもの（modeling_bitnet.py より）:
  attention: ... → inner_attn_ln → o_proj
  MLP      : silu(gate)*up → ffn_layernorm → down_proj
**lm_head は BitLinear ではなく普通の Linear**（fp16 のまま。埋め込みと共有）。

三値×INT8 の総和は最大 4096×127 = 520,192 で float32 の整数精度（16,777,216）に
収まるので、**行列積は float32 で計算しても厳密**。int32 だと BLAS が効かず桁違いに遅い。
"""
import json
import numpy as np
from st_read import SafeTensors
from smollm2_np import silu


def wquant(W):
    """テンソル単位の γ。W̃ (int8) と γ を返す。y = (W̃ @ x̃) · γ · (1/sx)"""
    g = max(float(np.abs(W).mean()), 1e-5)
    return np.clip(np.rint(W / g), -1, 1).astype(np.int8), np.float32(g)


# ---- 回路（axis_attn 系）と同じ整数手順の attention ----
# 段階5b のコアは次の形で計算する。TAU と SHIFT は回路に焼かれた定数。
#   s_int = q_int · K_int          （int32）
#   s8    = clamp(s_int >> SHIFT)  （int8。オンチップに貯める）
#   e     = EROM[max(s8) - s8]     （256語の表引き。EROM[d]=round(255·exp(-d/TAU))）
#   out   = (e @ V_int) / Σe
# K と V は **ヘッドごとに1つの尺度**で INT8 にする必要がある。
# 位置ごとの尺度にすると t 間でスコアが比較できなくなる。
import math as _math
SHIFT_C, TAU_C = 8, 8.0
SHM_C = 24                  # 回路のスコア倍率の固定シフト
TRACE_ATTN = None           # list を入れると、ヘッドごとの INT8 入力と倍率を記録する
EROM_C = np.array([max(0, min(255, int(round(255.0*_math.exp(-d/TAU_C)))))
                   for d in range(256)], dtype=np.int32)


def _q8(a):
    """配列全体を1つの尺度で INT8 に。尺度はスカラーで返す。"""
    m = max(float(np.abs(a).max()), 1e-9)
    return np.clip(np.rint(a*(127.0/m)), -127, 127).astype(np.int32), m/127.0


def attn_circuit(q, K, V, mask, mode=1):
    """q (NH,P,HD) / K,V (NH,T,HD) float32、mask (P,T) True=見ない。→ (NH,P,HD)

    mode=1: 段階5b の回路そのまま。s8 = clamp(s_int >> SHIFT)。
            **q と K の量子化尺度を無視する**ので指数の温度が層・ヘッドごとにずれる。
    mode=2: 見出しに倍率 m を持たせる案。s8 = clamp(round(s_int · m))、
            m = TAU·sq·sk/√HD。こうすると d = TAU·(max−s) が float の単位で揃い、
            表引きが厳密に exp(−(max−s)) になる。回路側の追加は掛け算器1個。
    """
    NH, P, HD = q.shape
    out = np.zeros((NH, P, HD), np.float32)
    rs = 1.0/np.sqrt(HD)
    for h in range(NH):
        Ki, sk = _q8(K[h])                      # ヘッド単位の尺度（全 t・全 d で1つ）
        Vi, sv = _q8(V[h])
        for pp in range(P):
            qi, sq = _q8(q[h, pp])
            si = (Ki @ qi).astype(np.int64)     # s_int
            if mode == 2:
                mm = TAU_C * sq * sk * rs
                s8 = np.clip(np.rint(si * mm), -128, 127).astype(np.int32)
                if TRACE_ATTN is not None:
                    # 回路が実際に受け取る形（m は 18bit 固定小数・シフト SHM_C）で残す
                    mi = int(round(mm * (1 << SHM_C)))
                    TRACE_ATTN.append(dict(h=h, qi=qi.astype(np.int8), Ki=Ki.astype(np.int8),
                                           Vi=Vi.astype(np.int8), m=mi, sv=float(sv),
                                           sat=(mi > 262143 or mi < 1)))
            else:
                s8 = np.clip(si >> SHIFT_C, -128, 127).astype(np.int32)
            s8 = np.where(mask[pp], -128, s8)
            mx = int(s8.max())
            e  = EROM_C[np.clip(mx - s8, 0, 255)]
            e  = np.where(mask[pp], 0, e)
            den = int(e.sum())
            if den == 0: den = 1
            out[h, pp] = (e @ Vi).astype(np.float32) * (sv / den)
    return out


def aquant(x):
    m = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-5)
    s = (127.0 / m).astype(np.float32)
    return np.clip(np.rint(x * s), -128, 127).astype(np.int8), s


class BitNet:
    MATS = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
            "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]

    def __init__(self, path, cfg, quant=True, head_ternary=False, circ_attn=False):
        self.q = quant
        self.ht = head_ternary
        self.ca = circ_attn        # True なら attention を回路と同じ整数手順で
        self.H = cfg["hidden_size"]; self.FF = cfg["intermediate_size"]
        self.NL = cfg["num_hidden_layers"]; self.NH = cfg["num_attention_heads"]
        self.NKV = cfg["num_key_value_heads"]; self.HD = self.H // self.NH
        self.eps = cfg["rms_norm_eps"]; self.theta = float(cfg["rope_theta"])
        st = SafeTensors(path)
        names = set(st.keys())
        self.emb = st.get("model.embed_tokens.weight").astype(np.float32)
        self.nrm = st.get("model.norm.weight").astype(np.float32)
        self.head = st.get("lm_head.weight").astype(np.float32) if "lm_head.weight" in names else self.emb
        # lm_head は本来 BitLinear ではない（fp16 のまま）。ただしこのボードでは
        # 98 MB を毎トークン読む余裕が無いので、三値に落とす選択肢を用意する。
        # 実測: perplexity 17.90 → 35.59。文章は壊れない。
        if head_ternary:
            self.head = wquant(self.head)
        self.ly = []
        for l in range(self.NL):
            d = {}
            for n in ["input_layernorm","post_attention_layernorm",
                      "self_attn.inner_attn_ln","mlp.ffn_layernorm"]:
                k = f"model.layers.{l}.{n}.weight"
                d[n] = st.get(k).astype(np.float32) if k in names else None
            for m in self.MATS:
                W = st.get(f"model.layers.{l}.{m}.weight").astype(np.float32)
                d[m] = wquant(W) if quant else W
            self.ly.append(d)
        st.close()
        inv = 1.0 / (self.theta ** (np.arange(0, self.HD, 2, np.float32) / self.HD))
        f = np.outer(np.arange(cfg.get("max_position_embeddings", 2048), dtype=np.float32), inv)
        e = np.concatenate([f, f], -1)
        self.cos, self.sin = np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)

    def rope(self, x, cos, sin):
        h = self.HD // 2
        rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
        return x * cos + rot * sin

    def rms(self, x, w):
        return (x / np.sqrt((x**2).mean(-1, keepdims=True) + self.eps)) * w

    def bl(self, W, x, trace=None, tag=None):
        """BitLinear。三値と INT8 の積を float32 で厳密に取る。"""
        if not self.q:
            return x @ W.T
        T, g = W
        xq, s = aquant(x)
        acc = xq.astype(np.float32) @ T.astype(np.float32).T      # 厳密な整数
        # 1トークンぶん（P=1）のときだけ記録する。FPGA に渡すのはこの形。
        if trace is not None and tag is not None:
            if xq.ndim == 1:
                trace.append((tag, xq.copy(), acc.astype(np.int32).copy()))
            elif xq.shape[0] == 1:
                trace.append((tag, xq[0].copy(), acc[0].astype(np.int32).copy()))
        return acc * (g / s)

    def forward(self, ids, cache=None, pos0=0, trace=None):
        HD, NH, NKV = self.HD, self.NH, self.NKV
        x = self.emb[np.asarray(ids)]
        P = x.shape[0]
        cos, sin = self.cos[pos0:pos0+P], self.sin[pos0:pos0+P]
        for l, d in enumerate(self.ly):
            h = self.rms(x, d["input_layernorm"])
            q = self.bl(d["self_attn.q_proj"], h, trace, f"L{l}.q")
            k = self.bl(d["self_attn.k_proj"], h, trace, f"L{l}.k")
            v = self.bl(d["self_attn.v_proj"], h, trace, f"L{l}.v")
            q = q.reshape(P, NH, HD).transpose(1, 0, 2)
            k = k.reshape(P, NKV, HD).transpose(1, 0, 2)
            v = v.reshape(P, NKV, HD).transpose(1, 0, 2)
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            if cache is not None:
                if len(cache) <= l: cache.append([k, v])
                else:
                    cache[l][0] = np.concatenate([cache[l][0], k], 1)
                    cache[l][1] = np.concatenate([cache[l][1], v], 1)
                K, V = cache[l]
            else:
                K, V = k, v
            T = K.shape[1]
            Kr, Vr = np.repeat(K, NH//NKV, 0), np.repeat(V, NH//NKV, 0)
            m = np.arange(T)[None, :] > (pos0 + np.arange(P))[:, None]
            if self.ca:
                o = attn_circuit(q, Kr, Vr, m, mode=self.ca).transpose(1, 0, 2).reshape(P, NH*HD)
            else:
                s = (q @ Kr.transpose(0, 2, 1)) / np.sqrt(HD)
                s = np.where(m[None], -np.inf, s)
                s = s - s.max(-1, keepdims=True)
                e = np.exp(s); a = e / e.sum(-1, keepdims=True)
                o = (a @ Vr).transpose(1, 0, 2).reshape(P, NH*HD)
            o = self.rms(o, d["self_attn.inner_attn_ln"])          # ★ BitNet 固有
            x = x + self.bl(d["self_attn.o_proj"], o, trace, f"L{l}.o")

            h2 = self.rms(x, d["post_attention_layernorm"])
            g_ = self.bl(d["mlp.gate_proj"], h2, trace, f"L{l}.gate")
            u_ = self.bl(d["mlp.up_proj"], h2, trace, f"L{l}.up")
            mm = silu(g_) * u_
            mm = self.rms(mm, d["mlp.ffn_layernorm"])              # ★ BitNet 固有
            x = x + self.bl(d["mlp.down_proj"], mm, trace, f"L{l}.down")
        x = self.rms(x, self.nrm)
        if self.ht:
            return self.bl(self.head, x, trace, "head")
        return x @ self.head.T                                     # lm_head は fp16 のまま


def ppl(model, ids):
    lg = model.forward(ids[:-1], cache=[], pos0=0)
    lg = lg - lg.max(-1, keepdims=True)
    lse = np.log(np.exp(lg).sum(-1))
    t = np.asarray(ids[1:])
    return float(np.exp((-(lg[np.arange(len(t)), t] - lse)).mean()))


if __name__ == "__main__":
    import sys, time
    from huggingface_hub import hf_hub_download
    from bpe import SPBPE
    from ppl import TEXT
    R = "1bitLLM/bitnet_b1_58-large"
    cfg = json.load(open(hf_hub_download(R, "config.json")))
    sp  = SPBPE(hf_hub_download(R, "tokenizer.json"))
    mp  = hf_hub_download(R, "model.safetensors")
    ids = sp.encode(TEXT)
    print(f"{R}")
    print(f"  hidden {cfg['hidden_size']} / {cfg['num_hidden_layers']}層 / "
          f"head_dim {cfg['hidden_size']//cfg['num_attention_heads']} / "
          f"q {cfg['num_attention_heads']}ヘッド : kv {cfg['num_key_value_heads']}ヘッド / 語彙 {cfg['vocab_size']}")
    print(f"  評価テキスト {len(ids)} トークン（SmolLM2 の評価と同じ文章）\n")

    t0 = time.time(); m = BitNet(mp, cfg, quant=True)
    print(f"  読み込み {time.time()-t0:.0f} 秒")
    t0 = time.time(); p = ppl(m, ids)
    print(f"  ★ 三値＋INT8（学習時と同じ計算）: perplexity {p:8.2f}   （{time.time()-t0:.0f} 秒）\n")
    for prompt in ["The capital of France is", "Once upon a time, there was a little"]:
        i2 = sp.encode(prompt); cache = []; out = list(i2)
        lg = m.forward(i2, cache=cache, pos0=0)
        for _ in range(14):
            n = int(lg[-1].argmax()); out.append(n)
            lg = m.forward([n], cache=cache, pos0=len(out)-1)
        print(f"  入力: {prompt!r}")
        print(f"  続き: {sp.decode(out[len(i2):])!r}")
    del m

    if "--full" in sys.argv:
        m = BitNet(mp, cfg, quant=False)
        print(f"\n  参考・量子化なし（影の重み fp32）: perplexity {ppl(m, ids):8.2f}")
