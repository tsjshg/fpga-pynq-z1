#!/usr/bin/env python3
"""SmolLM2 の BPE トークナイザ（ASCII 用の最小実装）。

tokenizer.json は GPT-2 系のバイトレベル BPE。
  pre_tokenizer = Sequence[ Digits(individual_digits), ByteLevel(use_regex) ]

本来の分割正規表現は \\p{L} などを使うが、Python 標準の re には無い。
`regex` を入れないために **ASCII 前提の近似**にしてある。
英語の短い文なら本家と一致する。日本語や記号の多い文では一致しない。
"""
import json, re
from functools import lru_cache

PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


@lru_cache(maxsize=1)
def byte_encoder():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class BPE:
    def __init__(self, tokenizer_json):
        t = json.load(open(tokenizer_json))
        self.vocab = t["model"]["vocab"]
        self.inv = {v: k for k, v in self.vocab.items()}
        merges = t["model"]["merges"]
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.b2u = byte_encoder()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.added = {a["content"]: a["id"] for a in t.get("added_tokens", [])}

    def _bpe(self, word):
        toks = list(word)
        while len(toks) > 1:
            best, bi = None, None
            for i in range(len(toks)-1):
                r = self.ranks.get((toks[i], toks[i+1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            toks[bi:bi+2] = [toks[bi] + toks[bi+1]]
        return toks

    def encode(self, text):
        ids = []
        for piece in PAT.findall(text):
            # Digits(individual_digits) は数字を1文字ずつに割る
            parts = re.findall(r"[0-9]|[^0-9]+", piece) if any(c.isdigit() for c in piece) else [piece]
            for p in parts:
                w = "".join(self.b2u[b] for b in p.encode("utf-8"))
                for tok in self._bpe(w):
                    if tok not in self.vocab:
                        raise KeyError(f"語彙に無い: {tok!r}")
                    ids.append(self.vocab[tok])
        return ids

    def decode(self, ids):
        s = "".join(self.inv[i] for i in ids)
        return bytes(self.u2b[c] for c in s if c in self.u2b).decode("utf-8", errors="replace")


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    bpe = BPE(hf_hub_download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json"))
    for s in ["The capital of France is", "Hello world!", "def add(a, b):\n    return"]:
        ids = bpe.encode(s)
        back = bpe.decode(ids)
        ok = "○" if back == s else "×"
        print(f"{ok} {s!r}\n   → {ids}\n   → {back!r}")


class SPBPE:
    """LLaMA 系（SentencePiece を HF 形式にしたもの）のトークナイザ。

    tokenizer.json の指示どおり:
      normalizer    = Prepend("▁") → Replace(" " → "▁")
      pre_tokenizer = なし（文字列全体に BPE をかける）
      byte_fallback = 語彙に無い文字は <0xNN> に落とす
    """
    def __init__(self, tokenizer_json):
        t = json.load(open(tokenizer_json))
        self.vocab = t["model"]["vocab"]
        self.inv = {v: k for k, v in self.vocab.items()}
        mg = t["model"]["merges"]
        mg = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in mg]
        self.ranks = {m: i for i, m in enumerate(mg)}
        for a in t.get("added_tokens", []):
            self.vocab.setdefault(a["content"], a["id"]); self.inv[a["id"]] = a["content"]

    def _chars(self, s):
        out = []
        for ch in s:
            if ch in self.vocab:
                out.append(ch)
            else:                                   # byte fallback
                out += [f"<0x{b:02X}>" for b in ch.encode("utf-8")]
        return out

    def encode(self, text, bos=True):
        s = ("▁" + text).replace(" ", "▁")
        toks = self._chars(s)
        while len(toks) > 1:
            best, bi = None, None
            for i in range(len(toks)-1):
                r = self.ranks.get((toks[i], toks[i+1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            toks[bi:bi+2] = [toks[bi] + toks[bi+1]]
        ids = [self.vocab[t] for t in toks]
        return ([1] + ids) if bos else ids          # <s> = 1

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            t = self.inv.get(int(i), "")
            if t.startswith("<0x") and t.endswith(">"):
                out.append(int(t[3:5], 16))
            elif t in ("<s>", "</s>", "<unk>"):
                continue
            else:
                out += t.replace("▁", " ").encode("utf-8")
        return out.decode("utf-8", errors="replace").lstrip()
