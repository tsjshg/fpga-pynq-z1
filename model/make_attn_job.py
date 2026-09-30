#!/usr/bin/env python3
"""bitnet_b1_58-large の1トークンぶんの attention を、回路の形で書き出す。

回路 (axis_attnv, head_dim 96 / GQA なし):
    [ 見出し 8B: [31:0]=T, [49:32]=m ][ q: 96 B ][ K: T×96 B ][ V: T×96 B ] × 組
    → 出力 96 語 + 分母 1 語 = 97 語

1トークンで 24層 × 16ヘッド = 384 組。**1層の 16 ヘッドは互いに独立**なので、
ヘッドを半分ずつ両ポートに振り分けられる（層の順序依存は壊れない）。

m = TAU·sq·sk/√HD を 2^24 倍した 18bit 整数。これを見出しに載せることで
指数の温度が層・ヘッドごとに正しくなる（numpy で perplexity 58.25 → 35.90）。

使い方: python3 make_attn_job.py [文脈長T] [本数] [出力先]
"""
import json, os, sys
import numpy as np
from huggingface_hub import hf_hub_download
from bpe import SPBPE
from ppl import TEXT
import bitnet_np as B

HD, CAP = 96, 8*1024*1024      # DMA の転送長カウンタを 23bit にしたので 8 MiB 以下
REPO = "1bitLLM/bitnet_b1_58-large"
T     = int(sys.argv[1]) if len(sys.argv) > 1 else 512
NPORT = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OUT   = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "atjob")
os.makedirs(OUT, exist_ok=True)


def main():
    cfg = json.load(open(hf_hub_download(REPO, "config.json")))
    sp  = SPBPE(hf_hub_download(REPO, "tokenizer.json"))
    print("三値の重みを用意中…", flush=True)
    m = B.BitNet(hf_hub_download(REPO, "model.safetensors"), cfg,
                 quant=True, head_ternary=True, circ_attn=2)

    ids = sp.encode(TEXT)
    while len(ids) < T + 1:                    # T 位置ぶん埋めるため文章を繰り返す
        ids = ids + sp.encode(TEXT, bos=False)
    ids = ids[:T]
    print(f"  前半 {len(ids)} トークンを流して KV を作る（文章を繰り返して埋めた）", flush=True)
    cache = []
    m.forward(ids, cache=cache, pos0=0)

    B.TRACE_ATTN = tr = []
    m.forward([ids[-1]], cache=cache, pos0=len(ids))
    print(f"  attention の組を {len(tr)} 個取り出した（24層 × 16ヘッド）")
    nsat = sum(1 for e in tr if e["sat"])
    ms = [e["m"] for e in tr]
    print(f"  倍率 m: {min(ms)} 〜 {max(ms)}（18bit の上限 262143）"
          f"  はみ出し {nsat} 個")

    # 1層 = 16 ヘッド。前半 8 を ポート0、後半 8 を ポート1 に
    per_layer = len(tr)//cfg["num_hidden_layers"]
    cur  = [bytearray() for _ in range(NPORT)]
    rows_= [0]*NPORT
    bufs = [[] for _ in range(NPORT)]
    fe   = [open(os.path.join(OUT, f"expect{p}.bin"), "wb") for p in range(NPORT)]
    eoff = [0]*NPORT
    blocks = []

    def flush():
        if not any(cur): return
        i = len(bufs[0])
        for p in range(NPORT):
            with open(os.path.join(OUT, f"a{p}_{i:02d}.bin"), "wb") as f: f.write(cur[p])
            bufs[p].append(dict(file=f"a{p}_{i:02d}.bin", nbytes=len(cur[p]), rows=rows_[p]))
            cur[p] = bytearray(); rows_[p] = 0

    for gi, e in enumerate(tr):
        p = (e["h"] * NPORT) // per_layer      # ヘッド番号で振り分ける
        Tn = e["Ki"].shape[0]
        blk = ((Tn & 0xFFFFFFFF) | ((e["m"] & 0x3FFFF) << 32)).to_bytes(8, "little")
        blk += e["qi"].tobytes() + e["Ki"].tobytes() + e["Vi"].tobytes()
        assert len(blk) == 8 + HD + 2*Tn*HD
        if len(cur[p]) + len(blk) > CAP:
            flush()
        # 期待値（回路と同じ手順）
        si = e["Ki"].astype(np.int64) @ e["qi"].astype(np.int64)
        s8 = np.clip(np.array([(int(v)*e["m"]) >> B.SHM_C for v in si]), -128, 127)
        ee = B.EROM_C[np.clip(int(s8.max()) - s8, 0, 255)].astype(np.int64)
        o  = ee @ e["Vi"].astype(np.int64)
        fe[p].write(np.concatenate([o, [ee.sum()]]).astype(np.int32).tobytes())
        blocks.append(dict(g=gi, h=int(e["h"]), port=p, off=len(cur[p]),
                           rows=HD+1, eoff=eoff[p], m=e["m"]))
        cur[p] += blk; rows_[p] += HD+1; eoff[p] += (HD+1)*4
        if gi % 32 == 0: print(f"\r  組 {gi+1}/{len(tr)}", end="", flush=True)
    flush()
    for f in fe: f.close()

    tot = sum(sum(b["nbytes"] for b in bufs[p]) for p in range(NPORT))
    meta = dict(model=REPO, T=T, n_ports=NPORT, n_blocks=len(blocks),
                n_bufs=len(bufs[0]), total_w=tot, hd=HD, shm=B.SHM_C,
                ports=[dict(bufs=bufs[p], expect=f"expect{p}.bin",
                            total_w=sum(b["nbytes"] for b in bufs[p]),
                            total_rows=eoff[p]//4) for p in range(NPORT)],
                blocks=blocks)
    json.dump(meta, open(os.path.join(OUT, "index.json"), "w"))
    print(f"\n\n組 {len(blocks)} / ポート {NPORT} 本 × バッファ {len(bufs[0])} 本"
          f" = 往復 {len(bufs[0])} 回")
    print(f"  合計 {tot/1024/1024:.1f} MiB / 出力 {eoff[0]//4*NPORT:,} 語")
    for bw in (1.927,):
        print(f"  帯域だけなら {tot/bw/1e9*1000:.1f} ms（HP2本・段階8 の実測 {bw} GB/s）")
    print(f"  CPU 実測は 96.22 ms（T=512）")


if __name__ == "__main__":
    main()
