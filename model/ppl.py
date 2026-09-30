#!/usr/bin/env python3
"""三値化で品質がどれだけ落ちるかを perplexity で測る。

比較するもの:
  fp32        そのまま
  三値・全部  210 個の行列 + lm_head を全部三値に
  三値・頭は別 lm_head（= embed_tokens）だけ fp32 で残す
              （BitNet の論文も埋め込みと出力層は高精度で残している）
"""
import sys, numpy as np
from huggingface_hub import hf_hub_download
from bpe import BPE
import smollm2_np as M

TEXT = ("A river flows from higher ground toward the sea. Along the way it collects water "
        "from smaller streams, and the volume it carries grows. When the land becomes flat, "
        "the river slows down and drops the sand and gravel it was carrying. Over many years "
        "this material builds up and forms new land at the mouth of the river. Farmers have "
        "long settled on such land because the soil is deep and easy to work. The same process "
        "also makes the channel shallow, so boats must follow marked routes. Engineers study "
        "the shape of the bed and decide where to dig, and how often the work must be repeated.")


def ppl(model, ids):
    cache = []
    lg = model.forward(ids[:-1], cache=cache, pos0=0)          # (P,V)
    lg = lg - lg.max(-1, keepdims=True)
    lse = np.log(np.exp(lg).sum(-1))
    tgt = np.asarray(ids[1:])
    nll = -(lg[np.arange(len(tgt)), tgt] - lse)
    return float(np.exp(nll.mean())), len(tgt)


if __name__ == "__main__":
    mp  = hf_hub_download("HuggingFaceTB/SmolLM2-135M", "model.safetensors")
    bpe = BPE(hf_hub_download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json"))
    ids = bpe.encode(TEXT)
    print(f"評価テキスト: {len(ids)} トークン\n")

    rows = []
    m = M.SmolLM2(mp, ternary=False); p, n = ppl(m, ids)
    rows.append(("fp32（そのまま）", p)); del m
    print(f"  fp32                : perplexity {p:8.2f}")

    m = M.SmolLM2(mp, ternary=True)
    p, _ = ppl(m, ids); rows.append(("三値・全部", p))
    print(f"  三値・全部          : perplexity {p:8.2f}")

    # lm_head だけ fp32 に戻す
    from st_read import SafeTensors
    st = SafeTensors(mp); m.head = st.get("model.embed_tokens.weight").astype(np.float32); st.close()
    _lin = m.lin
    def lin2(W, x, tag=None, trace=None):
        if isinstance(W, np.ndarray):                 # fp32 のまま残した行列
            return x @ W.T
        return _lin(W, x, tag, trace)
    m.lin = lin2
    p, _ = ppl(m, ids); rows.append(("三値・lm_head は fp32", p))
    print(f"  三値・頭だけ fp32   : perplexity {p:8.2f}")

    base = rows[0][1]
    print(f"\n{'条件':26s} {'perplexity':>12s} {'fp32 比':>10s}")
    print("-"*52)
    for n_, p_ in rows:
        print(f"{n_:26s} {p_:12.2f} {p_/base:9.1f}倍")
    print(f"\n参考: 語彙 {49152} 個をでたらめに選ぶと perplexity は {49152}")
