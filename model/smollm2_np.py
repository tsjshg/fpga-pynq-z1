#!/usr/bin/env python3
"""SmolLM2-135M の推論を numpy だけで書く。

目的は3つ:
  ① FPGA の照合相手（golden model）になる
  ② 三値化で品質がどれだけ落ちるかを測る
  ③ 本物の活性値 x を取り出して回路に流す

torch は使わない。config.json のとおり:
  hidden 576 / 30層 / q 9ヘッド / kv 3ヘッド / head_dim 64 / FFN 1536
  RMSNorm eps=1e-5 / RoPE theta=100000 / rope_interleaved=false / 重み共有あり
"""
import numpy as np
from st_read import SafeTensors
from ternarize import ternarize

H, NL, NH, NKV, HD, FF, VOC = 576, 30, 9, 3, 64, 1536, 49152
EPS, THETA = 1e-5, 100000.0
MATS = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
        "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]


def rmsnorm(x, w):
    return (x / np.sqrt((x.astype(np.float32)**2).mean(-1, keepdims=True) + EPS)) * w


def silu(x):
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def rope_tables(maxpos):
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD))   # (HD/2,)
    f = np.outer(np.arange(maxpos, dtype=np.float32), inv)               # (P, HD/2)
    e = np.concatenate([f, f], axis=-1)                                   # (P, HD)
    return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)


def rope(x, cos, sin):
    """x: (..., P, HD)。rope_interleaved=false の HF 方式（前後半を入れ替えて回す）。"""
    h = HD // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return x * cos + rot * sin


def quant8(x):
    """BitNet 方式の活性量子化。行ごとの絶対値最大で INT8 に落とす。"""
    s = np.abs(x).max(axis=-1, keepdims=True) / 127.0
    s = np.where(s == 0, 1e-8, s).astype(np.float32)
    return np.clip(np.rint(x / s), -127, 127).astype(np.int8), s


class SmolLM2:
    def __init__(self, path, ternary=False):
        self.ternary = ternary
        st = SafeTensors(path)
        self.emb  = st.get("model.embed_tokens.weight").astype(np.float32)   # (V,H)
        self.nrm  = st.get("model.norm.weight").astype(np.float32)
        self.ly = []
        for l in range(NL):
            d = {"ln1": st.get(f"model.layers.{l}.input_layernorm.weight").astype(np.float32),
                 "ln2": st.get(f"model.layers.{l}.post_attention_layernorm.weight").astype(np.float32)}
            for m in MATS:
                W = st.get(f"model.layers.{l}.{m}.weight").astype(np.float32)
                d[m] = ternarize(W, per_row=True) if ternary else W
            self.ly.append(d)
        # lm_head は embed_tokens と結ばれている
        self.head = ternarize(self.emb, per_row=True) if ternary else self.emb
        st.close()
        self.cos, self.sin = rope_tables(4096)

    # ---- 行列積。三値のときは INT8 の活性で int32 を作り、あとから尺度を掛ける ----
    def lin(self, W, x, tag=None, trace=None):
        if not self.ternary:
            return x @ W.T
        T, g = W
        xq, s = quant8(x)
        acc = (T.astype(np.int32) @ xq.astype(np.int32).T).T if x.ndim > 1 else \
              T.astype(np.int32) @ xq.astype(np.int32)
        if trace is not None and tag is not None and x.ndim == 1:
            trace.append((tag, xq.copy(), acc.astype(np.int32).copy()))
        return acc.astype(np.float32) * g[:, 0] * s

    def forward(self, ids, cache=None, pos0=0, trace=None):
        """ids: (P,) の int 列。cache は層ごとの [K, V] のリスト（増やしていく）。"""
        x = self.emb[np.asarray(ids)]                    # (P,H)
        P = x.shape[0]
        one = (P == 1)
        cos = self.cos[pos0:pos0+P]; sin = self.sin[pos0:pos0+P]

        for l, d in enumerate(self.ly):
            h = rmsnorm(x, d["ln1"])
            hv = h[0] if one else h
            q = self.lin(d["self_attn.q_proj"], hv, f"L{l}.q", trace)
            k = self.lin(d["self_attn.k_proj"], hv, f"L{l}.k", trace)
            v = self.lin(d["self_attn.v_proj"], hv, f"L{l}.v", trace)
            q = q.reshape(P, NH, HD).transpose(1, 0, 2)
            k = k.reshape(P, NKV, HD).transpose(1, 0, 2)
            v = v.reshape(P, NKV, HD).transpose(1, 0, 2)
            q = rope(q, cos, sin); k = rope(k, cos, sin)

            if cache is not None:
                if len(cache) <= l: cache.append([k, v])
                else:
                    cache[l][0] = np.concatenate([cache[l][0], k], axis=1)
                    cache[l][1] = np.concatenate([cache[l][1], v], axis=1)
                K, V = cache[l]
            else:
                K, V = k, v
            T = K.shape[1]

            Kr = np.repeat(K, NH // NKV, axis=0)          # (NH,T,HD)
            Vr = np.repeat(V, NH // NKV, axis=0)
            s = (q @ Kr.transpose(0, 2, 1)) / np.sqrt(HD)  # (NH,P,T)
            mask = np.arange(T)[None, :] > (pos0 + np.arange(P))[:, None]
            s = np.where(mask[None], -np.inf, s)
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s); a = e / e.sum(-1, keepdims=True)
            o = (a @ Vr).transpose(1, 0, 2).reshape(P, NH*HD)

            ov = o[0] if one else o
            x = x + (self.lin(d["self_attn.o_proj"], ov, f"L{l}.o", trace)[None] if one
                     else self.lin(d["self_attn.o_proj"], ov))

            h2 = rmsnorm(x, d["ln2"]); h2v = h2[0] if one else h2
            gt = self.lin(d["mlp.gate_proj"], h2v, f"L{l}.gate", trace)
            up = self.lin(d["mlp.up_proj"], h2v, f"L{l}.up", trace)
            act = silu(gt) * up
            x = x + (self.lin(d["mlp.down_proj"], act, f"L{l}.down", trace)[None] if one
                     else self.lin(d["mlp.down_proj"], act))

        x = rmsnorm(x, self.nrm)
        xv = x[0] if one else x
        lg = self.lin(self.head, xv, "head", trace)
        return lg[None] if one else lg


if __name__ == "__main__":
    import sys, time
    from huggingface_hub import hf_hub_download
    from bpe import BPE
    tern = "--ternary" in sys.argv
    mp = hf_hub_download("HuggingFaceTB/SmolLM2-135M", "model.safetensors")
    bpe = BPE(hf_hub_download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json"))
    print(f"重みを読み込み中（{'三値' if tern else 'bf16→fp32'}）…", flush=True)
    t0 = time.time(); m = SmolLM2(mp, ternary=tern); print(f"  {time.time()-t0:.1f} 秒")

    for prompt in ["The capital of France is", "Once upon a time, there was a little"]:
        ids = bpe.encode(prompt); cache = []
        lg = m.forward(ids, cache=cache, pos0=0)
        out = list(ids)
        for i in range(12):
            nxt = int(lg[-1].argmax())
            out.append(nxt)
            lg = m.forward([nxt], cache=cache, pos0=len(out)-1)
        print(f"\n  入力: {prompt!r}")
        print(f"  続き: {bpe.decode(out[len(ids):])!r}")
