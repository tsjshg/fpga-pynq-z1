#!/usr/bin/env python3
"""axis_attn2 の xsim 用の刺激と期待値を作る。

実機に持って行く前にここで潰す。5a では実機で3回誤診して丸一日使ったので、
**論理の誤りはシミュレータで、タイミングと帯域だけ実機で**という切り分けにする。

出力:
  sim/tb_in.hex   入力ビート（64bit hex、1行1ビート）
  sim/tb_exp.hex  期待値（32bit hex、1行1語）
  sim/tb_cfg.vh   ビート数・語数の define
"""
import math, sys
import numpy as np

HD, NQ, BPB, SHIFT, TAU = 64, 3, 8, 8, 8.0
BPR   = HD//BPB
NOUT  = NQ*HD
NOTOT = NOUT + NQ

EROM = np.array([max(0, min(255, int(round(255.0*math.exp(-d/TAU))))) for d in range(256)],
                dtype=np.int32)

# 小さい T で、しかも端（T=1 の行や最大値が先頭/末尾に来る場合）を踏む組み合わせ
TS = [1, 2, 8, 5, 16, 3, 64, 12]
rng = np.random.default_rng(20260918)

beats, exps = [], []
dmax_seen = 0
for gi, T in enumerate(TS):
    q = rng.integers(-127, 128, size=(NQ, HD), dtype=np.int8)
    # K の行ごとに大きさを振る。そうしないと score が固まって指数表の端しか引かない。
    scale = rng.uniform(0.05, 1.0, size=(T, 1))
    K = np.clip(np.round(rng.integers(-127, 128, size=(T, HD)) * scale), -127, 127).astype(np.int8)
    V = rng.integers(-127, 128, size=(T, HD), dtype=np.int8)

    # --- 回路と同じ手順 ---
    s   = q.astype(np.int32) @ K.astype(np.int32).T          # (NQ,T)
    s8  = np.clip(s >> SHIFT, -128, 127).astype(np.int32)    # 算術シフト
    mx  = s8.max(axis=1)                                     # (NQ,)
    d   = (mx[:, None] - s8).astype(np.int32)                # 0..255
    dmax_seen = max(dmax_seen, int(d.max()))
    e   = EROM[d]                                            # (NQ,T) 0..255
    o   = e @ V.astype(np.int32)                             # (NQ,HD)
    den = e.sum(axis=1)                                      # (NQ,)

    exps.extend(int(x) for x in o.reshape(-1))
    exps.extend(int(x) for x in den)

    # --- 入力ビート ---
    buf = bytearray()
    buf += int(T).to_bytes(8, "little")
    buf += q.reshape(-1).tobytes()
    buf += K.reshape(-1).tobytes()
    buf += V.reshape(-1).tobytes()
    assert len(buf) == 8 + NQ*HD + 2*T*HD
    for i in range(0, len(buf), 8):
        beats.append(int.from_bytes(buf[i:i+8], "little"))

with open("sim/tb_in.hex", "w") as f:
    for b in beats:
        f.write(f"{b:016x}\n")
with open("sim/tb_exp.hex", "w") as f:
    for v in exps:
        f.write(f"{v & 0xFFFFFFFF:08x}\n")
with open("sim/tb_cfg.vh", "w") as f:
    f.write(f"`define NIN  {len(beats)}\n")
    f.write(f"`define NEXP {len(exps)}\n")

print(f"グループ {len(TS)} 個 T={TS}")
print(f"  入力 {len(beats)} ビート / 期待値 {len(exps)} 語（{len(TS)}×{NOTOT}）")
print(f"  指数表を引いた最大の距離 d = {dmax_seen}（表の非零は {int((EROM>0).sum())} 語まで）")
print(f"  期待値の範囲 {min(exps)} 〜 {max(exps)}")
