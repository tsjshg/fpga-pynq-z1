#!/usr/bin/env python3
"""通しで動かすための一式を書き出す（段階11）。

出力（model/e2e/）:
  g{p}_{i:02d}.bin   行列積の重み。組ごとに [見出し][x の空き][行…]。x は実行時にボードが書く
  index.json         組の目録（どのバッファの何バイト目か・行数・γ の区切り）
  params.npz         正規化の重み・γ・KV の固定尺度・RoPE 表
  emb.f32            埋め込み（32002 × 1536 の float32。ボードでは memmap で1行ずつ読む）
  tokenizer.json     モデルの tokenizer（ボードの bpe.py が読む）

段階10 までの make_bitnet_job.py との違い:
  ・x は入れない（実行時に書く）ので、活性値のための順伝播が要らない
  ・期待値も書かない（照合は e2e_eval.py が Mac で整数のまま行う）
  ・**組ごとに1転送**になるので、組の先頭が 8 バイト境界に乗っていることだけが条件

KV の尺度は校正文（評価文とは別の文章）で測った max|K|, max|V| に固定する。

使い方: python3 pack_e2e.py [Tmax] [本数]
"""
import json, os, shutil, sys
import numpy as np
from huggingface_hub import hf_hub_download
from bpe import SPBPE
from bitnet_np import BitNet
import e2e_core as C

REPO  = "1bitLLM/bitnet_b1_58-large"
TMAX  = int(sys.argv[1]) if len(sys.argv) > 1 else 512
NPORT = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CAP   = 8*1024*1024
OUT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e")
os.makedirs(OUT, exist_ok=True)

# 校正用。評価文（ppl.TEXT・川の話）とは別の文章にする
CALIB = ("The library opened early on Saturday, and a line of students had already formed "
         "outside the door. Most of them were preparing for the final exams, which would begin "
         "the following week. Inside, the tables near the windows filled up first. A librarian "
         "walked between the shelves, returning books that had been left on carts overnight. "
         "In the corner, two friends argued quietly about a math problem, each convinced that "
         "the other had made a mistake. The capital of France is Paris, and the capital of Japan "
         "is Tokyo. Once upon a time, there was a little girl who lived near the forest with her "
         "grandmother, and every morning she walked to the village to buy bread.")


def main():
    cfg = json.load(open(hf_hub_download(REPO, "config.json")))
    tok = hf_hub_download(REPO, "tokenizer.json")
    sp  = SPBPE(tok)
    shutil.copyfile(tok, os.path.join(OUT, "tokenizer.json"))     # ボードでトークン化に使う
    print("Loading the model and ternarizing weights...", flush=True)
    m = BitNet(hf_hub_download(REPO, "model.safetensors"), cfg, quant=True, head_ternary=True)

    # ---- KV の尺度を校正する（attention は float のまま・本来の計算で） ----
    ids = sp.encode(CALIB); cache = []
    m.forward(ids, cache=cache, pos0=0)
    kmax = np.array([[np.abs(cache[l][0][h]).max() for h in range(C.NH)] for l in range(C.NL)], np.float32)
    vmax = np.array([[np.abs(cache[l][1][h]).max() for h in range(C.NH)] for l in range(C.NL)], np.float32)
    print(f"  KV scales fixed from a {len(ids)}-token calibration text"
          f" (max|K| {kmax.min():.2f}..{kmax.max():.2f} / max|V| {vmax.min():.3f}..{vmax.max():.3f})")

    # ---- 行列積の組 ----
    mats = []
    for l in range(C.NL):
        d = m.ly[l]
        T = lambda n: d[n][0]
        mats.append((f"L{l}.qkv", np.concatenate([T("self_attn.q_proj"), T("self_attn.k_proj"),
                                                   T("self_attn.v_proj")], 0)))
        mats.append((f"L{l}.o", T("self_attn.o_proj")))
        mats.append((f"L{l}.gu", np.concatenate([T("mlp.gate_proj"), T("mlp.up_proj")], 0)))
        mats.append((f"L{l}.down", T("mlp.down_proj")))
    mats.append(("head", m.head[0]))

    cur  = [bytearray() for _ in range(NPORT)]
    bufs = [[] for _ in range(NPORT)]
    eoff = [0]*NPORT
    blocks = []

    def flush():
        if not any(cur): return
        i = len(bufs[0])
        for p in range(NPORT):
            with open(os.path.join(OUT, f"g{p}_{i:02d}.bin"), "wb") as f: f.write(cur[p])
            bufs[p].append(dict(file=f"g{p}_{i:02d}.bin", nbytes=len(cur[p])))
            cur[p] = bytearray()

    for tag, T in mats:
        rows, ind = T.shape
        bpr = (ind + C.WPB - 1)//C.WPB
        L = bpr*C.WPB
        assert bpr <= 128
        Tp = T if L == ind else np.concatenate([T, np.zeros((rows, L-ind), np.int8)], 1)
        cuts = [rows*p//NPORT for p in range(NPORT)] + [rows]
        blks = []
        for p in range(NPORT):
            r0, r1 = cuts[p], cuts[p+1]
            hdr = ((r1-r0) & 0xFFFFFFFF) | ((bpr & 0xFFFF) << 32)
            blks.append(hdr.to_bytes(8, "little") + bytes(L) + C.pack(Tp[r0:r1]).tobytes())
        assert all(len(b) <= CAP for b in blks)
        if any(len(cur[p]) + len(blks[p]) > CAP for p in range(NPORT)):
            flush()
        rec = dict(tag=tag, rows=rows, ind=ind, bpr=bpr, per=[])
        for p in range(NPORT):
            r0, r1 = cuts[p], cuts[p+1]
            rec["per"].append(dict(buf=len(bufs[0]), off=len(cur[p]), nbytes=len(blks[p]),
                                   rows=r1-r0, eoff=eoff[p]))
            cur[p] += blks[p]; eoff[p] += r1-r0
        blocks.append(rec)
        print(f"\r  packed {len(blocks)}/{len(mats)} matrices", end="", flush=True)
    flush()

    tot = sum(b["nbytes"] for p in range(NPORT) for b in bufs[p])
    meta = dict(model=REPO, n_ports=NPORT, tmax=TMAX, wpb=C.WPB, total_w=tot,
                n_bufs=len(bufs[0]), bufs=bufs, rows_per_port=eoff, blocks=blocks)
    json.dump(meta, open(os.path.join(OUT, "index.json"), "w"))

    g = np.array([[m.ly[l][n][1] for n in BitNet.MATS] for l in range(C.NL)], np.float32)
    ly = m.ly
    np.savez(os.path.join(OUT, "params.npz"),
             n_in=np.stack([ly[l]["input_layernorm"] for l in range(C.NL)]),
             n_attn=np.stack([ly[l]["self_attn.inner_attn_ln"] for l in range(C.NL)]),
             n_post=np.stack([ly[l]["post_attention_layernorm"] for l in range(C.NL)]),
             n_ffn=np.stack([ly[l]["mlp.ffn_layernorm"] for l in range(C.NL)]),
             n_out=m.nrm, gamma=g, g_head=np.float32(m.head[1]),
             kmax=kmax, vmax=vmax, cos=m.cos[:TMAX].copy(), sin=m.sin[:TMAX].copy(),
             eps=np.float32(m.eps), tmax=np.int32(TMAX))
    m.emb.astype(np.float32).tofile(os.path.join(OUT, "emb.f32"))
    print(f"\n\n{len(blocks)} matrix groups (4 per layer x {C.NL} + lm_head) / "
          f"{len(bufs[0])} buffers x {NPORT} ports")
    print(f"  weights {tot/1024/1024:.1f} MiB / {sum(eoff):,} output words per token")
    kv = C.NL*C.NH*(2*TMAX*C.HD + 8 + C.HD)
    print(f"  KV (Tmax={TMAX}) {kv/1024/1024:.1f} MiB -> {(tot+kv)/1024/1024:.1f} MiB of CMA in total")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
