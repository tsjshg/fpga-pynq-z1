#!/usr/bin/env python3
"""bitnet_b1_58-large の1トークンぶんの行列積を、見出し付き形式で書き出す。

回路 (axis_tmacv, 行長可変):
    [ 見出し 8B: [31:0]=R 行数, [47:32]=BPR 1行のビート数 ]
    [ x: BPR*40 B ][ 重み: BPR*8 B × R 行 ] × 何組でも → TLAST
  1ビート = 三値 40 個。1行 = BPR*40 重み。BPR は 1..128（＝重み 5120 個まで）

**HP を何本使うか**を引数で選ぶ。2本にするときは
**各組の「行」を半分ずつ両ポートに振り分ける**。
  ・x は両方に複製する（1組 640 B × 本数。全体で 0.5 MB 程度・無視できる）
  ・行を割るだけなので、層どうしの順序依存は壊れない
    （バッファをポートに振り分ける方式だと、後ろの層を先に流すことになって嘘になる）

組の作り方（x を共有するものはまとめる）。**行長が可変になったので塊に割らなくてよい**:
  入力 1536 → BPR 39（L=1560・詰め物 1.6%）  入力 4096 → BPR 103（L=4120・0.6%）
  1層 = qkv + o + gate/up + down = 4 組。24層で 96 組 + lm_head = 97 組
  （L=640 固定だと 387 組で、しかも部分和を CPU で足す必要があった）

CMA の実用上限が 192 MB（16 MB 刻み）なので 16 MB ごとに切る。
両ポートの切れ目は揃える（同じ添字のバッファを同時に走らせるため）。

使い方: python3 make_bitnet_job.py [本数] [出力先]
"""
import json, os, sys
import numpy as np
from huggingface_hub import hf_hub_download
from bpe import SPBPE
from bitnet_np import BitNet

WPB, BPRMAX, CAP = 40, 128, 8*1024*1024   # DMA の転送長カウンタ 23bit に合わせて 8 MiB
LMAX = WPB*BPRMAX            # 1組で扱える最大の入力次元 = 5120
POW = np.array([1,3,9,27,81], dtype=np.uint16)
REPO = "1bitLLM/bitnet_b1_58-large"
PROMPT = "The capital of France is"
NPORT = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"bnjob{NPORT}")
os.makedirs(OUT, exist_ok=True)


def pack(T):
    """三値 (rows, L) を詰める。L は 5 の倍数。→ uint8 (rows, L/5)"""
    rows, L = T.shape
    u = (T.astype(np.uint16) + 1).reshape(rows, L//5, 5)
    return (u * POW).sum(2, dtype=np.uint16).astype(np.uint8)


def main():
    cfg = json.load(open(hf_hub_download(REPO, "config.json")))
    sp  = SPBPE(hf_hub_download(REPO, "tokenizer.json"))
    print(f"三値の重みを用意中…（HP {NPORT} 本ぶんに割る）", flush=True)
    m = BitNet(hf_hub_download(REPO, "model.safetensors"), cfg,
               quant=True, head_ternary=True)

    ids = sp.encode(PROMPT); cache = []
    m.forward(ids, cache=cache, pos0=0)
    tr = []
    m.forward([ids[-1]], cache=cache, pos0=len(ids), trace=tr)
    X = {t: xq for t, xq, _ in tr}
    print(f"  活性値を {len(X)} 本取り出した（入力 {PROMPT!r}）")

    cur   = [bytearray() for _ in range(NPORT)]
    rows_ = [0]*NPORT
    bufs  = [[] for _ in range(NPORT)]
    fe    = [open(os.path.join(OUT, f"expect{p}.bin"), "wb") for p in range(NPORT)]
    eoff  = [0]*NPORT
    blocks = []

    def flush():
        """全ポートまとめて書き出す。切れ目を揃えるのが肝。"""
        if not any(cur): return
        i = len(bufs[0])
        for p in range(NPORT):
            with open(os.path.join(OUT, f"w{p}_{i:02d}.bin"), "wb") as f:
                f.write(cur[p])
            bufs[p].append(dict(file=f"w{p}_{i:02d}.bin", nbytes=len(cur[p]), rows=rows_[p]))
            cur[p] = bytearray(); rows_[p] = 0

    def emit(tag, xq, T):
        """1つの行列を組にする。行長は 40 の倍数に切り上げるだけ（塊割りは不要）。"""
        rows, ind = T.shape
        nch = (ind + LMAX - 1)//LMAX          # 5120 を超える入力だけ塊に割る
        for c in range(nch):
            sub = T[:, c*LMAX:(c+1)*LMAX]
            xs  = xq[c*LMAX:(c+1)*LMAX]
            w   = sub.shape[1]
            bpr = (w + WPB - 1)//WPB           # 40 の倍数に切り上げ
            L   = bpr*WPB
            Tc  = sub if L == w else np.concatenate([sub, np.zeros((rows, L-w), np.int8)], 1)
            xc  = xs  if L == len(xs) else np.concatenate([xs, np.zeros(L-len(xs), np.int8)])
            cuts = [rows*p//NPORT for p in range(NPORT)] + [rows]
            blks = []
            for p in range(NPORT):
                r0, r1 = cuts[p], cuts[p+1]
                hdr = ((r1-r0) & 0xFFFFFFFF) | ((bpr & 0xFFFF) << 32)
                blks.append(hdr.to_bytes(8, "little") + xc.tobytes()
                            + pack(Tc[r0:r1]).tobytes())
            if any(len(cur[p]) + len(blks[p]) > CAP for p in range(NPORT)):
                flush()
            rec = dict(tag=f"{tag}.c{c}" if nch > 1 else tag, buf=len(bufs[0]),
                       rows=rows, bpr=bpr, per=[])
            for p in range(NPORT):
                r0, r1 = cuts[p], cuts[p+1]
                fe[p].write((Tc[r0:r1].astype(np.int32) @ xc.astype(np.int32))
                            .astype(np.int32).tobytes())
                rec["per"].append(dict(off=len(cur[p]), rows=r1-r0, eoff=eoff[p]))
                cur[p] += blks[p]; rows_[p] += r1-r0; eoff[p] += (r1-r0)*4
            blocks.append(rec)

    for l in range(m.NL):
        d = m.ly[l]
        Tq, Tk, Tv = (d[f"self_attn.{n}_proj"][0] for n in "qkv")
        emit(f"L{l}.qkv",  X[f"L{l}.q"],    np.concatenate([Tq, Tk, Tv], 0))
        emit(f"L{l}.o",    X[f"L{l}.o"],    d["self_attn.o_proj"][0])
        emit(f"L{l}.gu",   X[f"L{l}.gate"],
             np.concatenate([d["mlp.gate_proj"][0], d["mlp.up_proj"][0]], 0))
        emit(f"L{l}.down", X[f"L{l}.down"], d["mlp.down_proj"][0])
        print(f"\r  層 {l+1}/{m.NL}  バッファ {len(bufs[0])}", end="", flush=True)
    emit("head", X["head"], m.head[0])
    flush()
    for f in fe: f.close()

    tot = sum(sum(b["nbytes"] for b in bufs[p]) for p in range(NPORT))
    meta = dict(wpb=WPB, bprmax=BPRMAX, model=REPO, prompt=PROMPT, n_ports=NPORT,
                n_blocks=len(blocks), n_bufs=len(bufs[0]), total_w=tot,
                ports=[dict(bufs=bufs[p], expect=f"expect{p}.bin",
                            total_w=sum(b["nbytes"] for b in bufs[p]),
                            total_rows=eoff[p]//4) for p in range(NPORT)],
                blocks=blocks)
    json.dump(meta, open(os.path.join(OUT, "index.json"), "w"))
    print(f"\n\n組 {len(blocks)} / ポート {NPORT} 本 × バッファ {len(bufs[0])} 本"
          f" = 転送 {NPORT*len(bufs[0])} 回（同時に走るので往復は {len(bufs[0])} 回）")
    print(f"  合計 {tot/1024/1024:.1f} MB（x の複製ぶん含む）")
    for p in range(NPORT):
        print(f"    ポート{p}: {meta['ports'][p]['total_w']/1024/1024:6.1f} MB / "
              f"{meta['ports'][p]['total_rows']:,} 語")
    bw = 1.130 if NPORT == 1 else 1.844
    print(f"  帯域だけなら {tot/bw/1e9*1000:.1f} ms（{NPORT}本で {bw} GB/s・段階7 の実測）")
    bprs = sorted({b["bpr"] for b in blocks})
    print(f"  使った行長: " + " / ".join(f"BPR {b}（重み {b*WPB}）" for b in bprs))


if __name__ == "__main__":
    main()
