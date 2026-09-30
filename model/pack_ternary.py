#!/usr/bin/env python3
"""SmolLM2-135M を三値化して、axis_tmac が食える形に詰める。

回路の形式（rtl/gen_tmac.py）:
    [ x: 640 バイト (INT8) ][ 重み: 128 バイト/行 × R 行 ]
    1行 = 三値ちょうど 640 個。バイト v = Σ (w_d+1)·3^d  (d=0..4)
    出力は 1行につき int32 ひとつ。

SmolLM2 の入力次元は 576 と 1536 で、どちらも 640 の倍数ではない。
    576  → 0 を 64 個足して 640（1 塊・詰め物 10.0%）
    1536 → 640×3 = 1920 に足す（3 塊・詰め物 25.0%）
        1行を3つに割って流し、int32 の部分和を CPU で足す。

出力:
    out/smollm2_tern.bin    詰めた重み（行列を順に連結）
    out/smollm2_scale.bin   行ごとの尺度 γ（float32）
    out/smollm2_tern.json   目録（名前・形・位置・塊数）
"""
import json, os, sys
import numpy as np
from huggingface_hub import hf_hub_download
from st_read import SafeTensors
from ternarize import ternarize

L, BPR = 640, 128
POW = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
NL  = 30
MATS = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
        "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]

INCLUDE_HEAD = "--no-head" not in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def pack_rows(T):
    """三値 (out, in) を詰める。in は 640 の倍数に 0 で伸ばす。→ uint8 (out, chunks*128)"""
    out, ind = T.shape
    chunks = (ind + L - 1) // L
    if chunks * L != ind:
        T = np.concatenate([T, np.zeros((out, chunks*L - ind), np.int8)], axis=1)
    u = (T.astype(np.uint16) + 1).reshape(out, chunks*BPR, 5)
    return (u * POW).sum(axis=2, dtype=np.uint16).astype(np.uint8), chunks


def main():
    st = SafeTensors(hf_hub_download("HuggingFaceTB/SmolLM2-135M", "model.safetensors"))
    fw = open(os.path.join(OUT, "smollm2_tern.bin"), "wb")
    fs = open(os.path.join(OUT, "smollm2_scale.bin"), "wb")
    man, woff, soff, real, sent = [], 0, 0, 0, 0

    names = [f"model.layers.{l}.{m}.weight" for l in range(NL) for m in MATS]
    if INCLUDE_HEAD:
        names.append("model.embed_tokens.weight")     # lm_head と結ばれている

    for name in names:
        W = st.get(name).astype(np.float32)
        T, g = ternarize(W, per_row=True)             # γ は行ごと
        packed, chunks = pack_rows(T)
        fw.write(packed.tobytes())
        fs.write(g.reshape(-1).astype(np.float32).tobytes())
        man.append(dict(name=name, out=int(W.shape[0]), inn=int(W.shape[1]),
                        chunks=int(chunks), woff=woff, wlen=int(packed.nbytes),
                        soff=soff, zero=float((T == 0).mean())))
        woff += packed.nbytes
        soff += W.shape[0]*4
        real += W.size
        sent += W.shape[0]*chunks*L
        print(f"\r  {len(man):3d}/{len(names)}  {name[:52]:52s}", end="", flush=True)
    fw.close(); fs.close(); st.close()

    meta = dict(L=L, BPR=BPR, n_layers=NL, mats=MATS, include_head=INCLUDE_HEAD,
                real_weights=int(real), sent_weights=int(sent), items=man)
    with open(os.path.join(OUT, "smollm2_tern.json"), "w") as f:
        json.dump(meta, f)

    print(f"\n\n行列 {len(man)} 個")
    print(f"  本物の重み      : {real/1e6:8.2f} M")
    print(f"  流す重み        : {sent/1e6:8.2f} M  （詰め物 {100*(sent/real-1):.1f}%）")
    print(f"  詰めた大きさ    : {woff/1024/1024:8.2f} MB")
    print(f"  尺度 γ          : {soff/1024:8.1f} KB（行ごと float32）")
    print(f"  1トークンの転送 : {len(man)} 回（x を共有するものをまとめれば {4*NL + (1 if INCLUDE_HEAD else 0)} 回）")
    for bw in (1.894,):
        print(f"  帯域だけなら    : {woff/bw/1e9*1000:8.2f} ms/トークン  （段階4 の実測 {bw} GB/s）")


if __name__ == "__main__":
    main()
