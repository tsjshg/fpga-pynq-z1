#!/usr/bin/env python3
"""safetensors を numpy だけで読む。

torch も safetensors も入れずに済ませるための最小実装。
この numpy 配列が、そのまま FPGA の照合相手（golden model）になる。

書式:
  [8B: ヘッダ長 N (u64 LE)][N B: JSON][残り: テンソル本体]
  JSON は {名前: {dtype, shape, data_offsets:[a,b]}, "__metadata__": {...}}
  data_offsets はヘッダ末尾からの相対位置。
"""
import json, mmap
import numpy as np

# bfloat16 は numpy に無い。上位16bit なので 16bit 左シフトして float32 と見る。
_DT = {"F32": np.float32, "F16": np.float16, "I64": np.int64,
       "I32": np.int32, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


class SafeTensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = int.from_bytes(self.mm[:8], "little")
        self.hdr = json.loads(self.mm[8:8+n])
        self.base = 8 + n
        self.meta = self.hdr.pop("__metadata__", {})

    def keys(self):
        return list(self.hdr.keys())

    def info(self, name):
        h = self.hdr[name]
        return h["dtype"], tuple(h["shape"])

    def get(self, name):
        """float32 の numpy 配列で返す（bf16 は展開する）。"""
        h = self.hdr[name]
        a, b = h["data_offsets"]
        buf = self.mm[self.base + a: self.base + b]
        shape = tuple(h["shape"])
        if h["dtype"] == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        dt = _DT.get(h["dtype"])
        if dt is None:
            raise ValueError(f"未対応の dtype: {h['dtype']}")
        return np.frombuffer(buf, dtype=dt).reshape(shape)

    def close(self):
        self.mm.close(); self.f.close()


if __name__ == "__main__":
    import sys
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("HuggingFaceTB/SmolLM2-135M", "model.safetensors")
    st = SafeTensors(p)
    tot = 0
    kinds = {}
    for k in st.keys():
        dt, sh = st.info(k)
        n = int(np.prod(sh)); tot += n
        # 層番号を伏せて種類ごとにまとめる
        kind = k.replace(".".join(k.split(".")[:3]), "model.layers.N") if ".layers." in k else k
        kinds.setdefault(kind, [0, sh, dt])
        kinds[kind][0] += 1
    print(f"テンソル {len(st.keys())} 個 / 合計 {tot/1e6:.2f} M パラメータ\n")
    for kind, (cnt, sh, dt) in kinds.items():
        print(f"  {kind:44s} ×{cnt:3d}  {str(sh):14s} {dt}")
    print(f"\nメタデータ: {st.meta}")
