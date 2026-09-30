#!/usr/bin/env python3
"""1トークンぶんの行列積を「FPGA に流す仕事」として書き出す。

回路 (axis_tmac, L=640) の形式は 1 転送につき:
    [ x: 640 バイト ][ 重み: 128 バイト × R 行 ]  →  int32 × R 行
x は転送ごとに1つなので、**同じ x を使う行列はまとめて1転送にできる**。

1層の内訳（6 転送）:
    qkv   960 行  x = RMSNorm(残差, ln1)
    o     576 行  x = attention の出力
    gu   3072 行  x = RMSNorm(残差, ln2)
    down  576 行 × 3  x = SiLU(gate)·up（1536 → 640×3 に割る。部分和を CPU で足す）
→ 30 層で 180 転送 + lm_head 1 転送 = 181 転送

x は本物の活性値。実際に SmolLM2 を三値で1ステップ走らせて取り出す。
期待値も同じ三値重みから作るので、**PL と numpy を厳密に突き合わせられる**。
"""
import json, os, sys
import numpy as np
from huggingface_hub import hf_hub_download
from bpe import BPE
import smollm2_np as M

L, BPR = 640, 128
POW = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
PROMPT = "The capital of France is"


def pack(T):
    """三値 (rows, 640) → uint8 (rows, 128)"""
    assert T.shape[1] == L
    u = (T.astype(np.uint16) + 1).reshape(T.shape[0], BPR, 5)
    return (u * POW).sum(axis=2, dtype=np.uint16).astype(np.uint8)


def unpack(P):
    """詰めたバイトを三値へ戻す。詰め方の自己検査に使う。"""
    v = P.astype(np.int32)[:, :, None] // np.array([1, 3, 9, 27, 81]) % 3 - 1
    return v.reshape(P.shape[0], -1).astype(np.int8)


def padcols(A, n):
    return A if A.shape[1] == n else np.concatenate(
        [A, np.zeros((A.shape[0], n - A.shape[1]), A.dtype)], axis=1)


def main():
    mp  = hf_hub_download("HuggingFaceTB/SmolLM2-135M", "model.safetensors")
    bpe = BPE(hf_hub_download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json"))
    print("三値の重みを用意中…", flush=True)
    m = M.SmolLM2(mp, ternary=True)

    ids = bpe.encode(PROMPT)
    cache = []
    m.forward(ids, cache=cache, pos0=0)            # 前半（ここは記録しない）
    tr = []
    m.forward([ids[-1]], cache=cache, pos0=len(ids), trace=tr)   # 1ステップだけ記録
    X = {t: xq for t, xq, _ in tr}
    print(f"  活性値を {len(X)} 本取り出した（入力 {PROMPT!r}）")

    fw = open(os.path.join(OUT, "job_weights.bin"), "wb")
    fe = open(os.path.join(OUT, "job_expect.bin"), "wb")
    blocks, woff, eoff, bad = [], 0, 0, 0

    def emit(tag, xq, T):
        """1転送ぶんを書き出す。T は (rows, in)。in が 640 を超えるなら塊に割る。"""
        nonlocal woff, eoff, bad
        rows, ind = T.shape
        ch = (ind + L - 1) // L
        Tp = padcols(T, ch*L)
        xp = np.concatenate([xq, np.zeros(ch*L - len(xq), np.int8)])
        for c in range(ch):
            Tc = Tp[:, c*L:(c+1)*L]
            xc = xp[c*L:(c+1)*L]
            pk = pack(Tc)
            if not np.array_equal(unpack(pk), Tc):     # 詰め方の自己検査
                bad += 1
            exp = (Tc.astype(np.int32) @ xc.astype(np.int32)).astype(np.int32)
            fw.write(xc.tobytes()); fw.write(pk.tobytes())
            fe.write(exp.tobytes())
            blocks.append(dict(tag=f"{tag}.c{c}" if ch > 1 else tag,
                               woff=woff, wlen=L + rows*BPR, rows=rows, eoff=eoff))
            woff += L + rows*BPR
            eoff += rows*4

    for l in range(M.NL):
        d = m.ly[l]
        Tq, Tk, Tv = (d[f"self_attn.{n}_proj"][0] for n in "qkv")
        emit(f"L{l}.qkv", X[f"L{l}.q"], np.concatenate([Tq, Tk, Tv], axis=0))
        emit(f"L{l}.o",   X[f"L{l}.o"], d["self_attn.o_proj"][0])
        emit(f"L{l}.gu",  X[f"L{l}.gate"],
             np.concatenate([d["mlp.gate_proj"][0], d["mlp.up_proj"][0]], axis=0))
        emit(f"L{l}.down", X[f"L{l}.down"], d["mlp.down_proj"][0])
        print(f"\r  層 {l+1}/{M.NL}", end="", flush=True)
    emit("head", X["head"], m.head[0])

    fw.close(); fe.close()
    meta = dict(L=L, BPR=BPR, prompt=PROMPT, n_blocks=len(blocks),
                total_w=woff, total_rows=eoff//4, blocks=blocks)
    json.dump(meta, open(os.path.join(OUT, "job_index.json"), "w"))

    print(f"\n\n転送 {len(blocks)} 回 / 重み {woff/1024/1024:.2f} MB / 出力 {eoff//4:,} 語")
    print(f"詰め方の自己検査: {'全部一致' if bad == 0 else f'★{bad} 塊が不一致'}")
    for bw in (1.894,):
        print(f"帯域だけなら {woff/bw/1e9*1000:.2f} ms  ／  転送の固定費 0.7ms×{len(blocks)} = "
              f"{0.7*len(blocks):.0f} ms")


if __name__ == "__main__":
    main()
