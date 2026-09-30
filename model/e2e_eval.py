#!/usr/bin/env python3
"""通しの計算を Mac で走らせる（PL の代わりに整数で厳密に計算する）。

  python3 e2e_eval.py ref                   perplexity と生成。基準を ref.npz に、ボードに渡す入力を ref_in.npz に残す
  python3 e2e_eval.py compare BOARD_OUT.npz 基準とボードの logits・生成トークンがビット単位で一致するか
  python3 e2e_eval.py verify DUMP.npz       ボードが記録した PL の入出力を、整数のまま全部照合する

e2e_cpu.c は NEON で書いてあるので、これは ARM64 の機械（Apple Silicon の Mac など）でしか動かない。

重みは pack_e2e.py が書き出した**ボードと同じファイル**から読み戻す（詰め方の検査も兼ねる）。
"""
import json, os, sys, time
import numpy as np
from huggingface_hub import hf_hub_download
from bpe import SPBPE
from ppl import TEXT
import e2e_core as C

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e")
REPO = "1bitLLM/bitnet_b1_58-large"
PROMPTS = ["The capital of France is", "Once upon a time, there was a little"]
NGEN = 24


def load_blocks():
    """ボードに送るのと同じ .bin から三値行列を復元する。"""
    meta = json.load(open(os.path.join(D, "index.json")))
    raw = [[np.fromfile(os.path.join(D, b["file"]), np.uint8) for b in meta["bufs"][p]]
           for p in range(meta["n_ports"])]
    blocks = []
    for b in meta["blocks"]:
        parts = []
        for p, q in enumerate(b["per"]):
            a = raw[p][q["buf"]][q["off"]:q["off"]+q["nbytes"]]
            hdr = int.from_bytes(a[:8].tobytes(), "little")
            assert hdr & 0xFFFFFFFF == q["rows"] and (hdr >> 32) & 0xFFFF == b["bpr"]
            L = b["bpr"]*C.WPB
            assert not a[8:8+L].any()                      # x の空きは 0
            w = a[8+L:].reshape(q["rows"], b["bpr"]*8)
            parts.append(C.unpack(w, L))
        blocks.append((np.concatenate(parts, 0), b["ind"]))
    return meta, blocks


def params():
    p = dict(np.load(os.path.join(D, "params.npz")))
    p["emb"] = np.memmap(os.path.join(D, "emb.f32"), np.float32, "r").reshape(-1, C.H)
    return p


def run_ppl(prm, blocks, ids):
    be = C.NumpyBackend(blocks, int(prm["tmax"]))
    r = C.Runner(prm, be)
    nll, lgs = [], []
    for pos in range(len(ids) - 1):
        lg = r.step(ids[pos], pos).astype(np.float64)       # 写しを取る（step は同じ領域を使い回す）
        lse = lg.max() + np.log(np.exp(lg - lg.max()).sum())
        nll.append(lse - lg[ids[pos+1]]); lgs.append(lg.astype(np.float32))
    return float(np.exp(np.mean(nll))), np.array(lgs)


def run_gen(prm, blocks, ids, n):
    be = C.NumpyBackend(blocks, int(prm["tmax"]))
    r = C.Runner(prm, be)
    out = list(ids); lgs = []
    for pos in range(len(ids) + n - 1):
        lg = r.step(out[pos], pos); lgs.append(lg.copy())
        if pos >= len(ids) - 1:
            out.append(r.top)
    return out, np.array(lgs)


def ref():
    sp = SPBPE(hf_hub_download(REPO, "tokenizer.json"))
    prm = params()
    t0 = time.time(); meta, blocks = load_blocks()
    print(f"rebuilt {len(blocks)} matrix groups from the same .bin files the board uses ({time.time()-t0:.0f} s)")
    ids = sp.encode(TEXT)
    t0 = time.time(); p, lg_ppl = run_ppl(prm, blocks, ids)
    print(f"perplexity {p:.2f} ({len(ids)} tokens, one at a time, {time.time()-t0:.0f} s)")
    save = dict(ppl=p, ppl_ids=np.array(ids), ppl_logits=lg_ppl)
    for i, pr in enumerate(PROMPTS):
        pid = sp.encode(pr)
        out, lg = run_gen(prm, blocks, pid, NGEN)
        print(f"  {pr!r} -> {sp.decode(out[len(pid):])!r}")
        save[f"gen{i}_ids"] = np.array(out); save[f"gen{i}_logits"] = lg
        save[f"gen{i}_plen"] = len(pid)
    np.savez(os.path.join(D, "ref.npz"), **save)
    # ボードの eval が読む入力（logits を除いた軽いもの）
    np.savez(os.path.join(D, "ref_in.npz"), **{k: v for k, v in save.items()
                                              if k.endswith("_ids") or k.endswith("_plen")})
    print(f"wrote {D}/ref.npz and ref_in.npz")


def compare(path):
    """ボードの board_out.npz と ref.npz を突き合わせる。浮動小数まで含めてビット単位で比べる。"""
    ref, got = np.load(os.path.join(D, "ref.npz")), np.load(path)
    ok = True
    for k in ("ppl_logits", "gen0_ids", "gen0_logits", "gen1_ids", "gen1_logits"):
        a, b = ref[k], got[k]
        same = a.shape == b.shape and np.array_equal(a.view(np.uint32) if a.dtype == np.float32 else a,
                                                     b.view(np.uint32) if b.dtype == np.float32 else b)
        ok &= same
        print(f"  {k:12s} {str(a.shape):14s} {'bit-identical' if same else 'DIFFERENT'}")
    print(f"perplexity: host {float(ref['ppl']):.6f} / board {float(got['ppl']):.6f}")
    print("all logits and tokens match bit for bit" if ok else "MISMATCH")
    return ok


def verify(path):
    """ボードの記録を整数で照合する。KV も記録から積み直すので、状態ごと検査になる。"""
    prm = params()
    meta, blocks = load_blocks()
    z = np.load(path, allow_pickle=True)
    log = list(z["log"])
    be = C.NumpyBackend(blocks, int(prm["tmax"]))
    ng = na = bad = 0; words = 0
    for e in log:
        if e[0] == "g":
            _, bi, xq, got = e
            buf = np.zeros(40000, np.int32); n = be.gemv(bi, xq, buf); want = buf[:n]
            ng += 1; words += want.size
            if not np.array_equal(want, got):
                bad += 1; print(f"  * matmul {meta['blocks'][bi]['tag']} mismatch "
                                f"{int((want != got).sum())}/{want.size} words")
        else:
            _, l, pos, kq, vq, qi, mm, got = e
            want = np.zeros((C.NH, C.HD+1), np.int32); be.attn(l, pos, kq, vq, qi, mm, want)
            na += 1; words += want.size
            if not np.array_equal(want, got):
                bad += 1; print(f"  * attention layer {l} position {pos} mismatch")
    print(f"checked {ng:,} matmul + {na:,} attention calls = {words:,} words -> "
          f"{'all match' if bad == 0 else f'{bad} calls MISMATCH'}")
    return bad == 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("ref", "compare", "verify"):
        sys.exit("usage: e2e_eval.py ref | compare BOARD_OUT.npz | verify DUMP.npz")
    if sys.argv[1] == "ref": ref()
    elif sys.argv[1] == "compare": sys.exit(0 if compare(sys.argv[2]) else 1)
    else: sys.exit(0 if verify(sys.argv[2]) else 1)
