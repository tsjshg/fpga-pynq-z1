#!/usr/bin/env python3
"""axis_tmacv（行長可変）の xsim 用の刺激と期待値。

行長が組ごとに変わること、端（BPR=1 と BPR=128）でも壊れないことを確かめる。
"""
import numpy as np
WPB = 40
POW = np.array([1,3,9,27,81], dtype=np.uint16)
# (行数 R, 1行のビート数 BPR)。BPR は 1..128
CASES = [(3,15), (1,39), (5,103), (2,1), (4,128), (7,39), (1,1)]
rng = np.random.default_rng(271828)

beats, exps = [], []
for R, BPR in CASES:
    L = BPR*WPB
    x = rng.integers(-127, 128, size=L, dtype=np.int8)
    W = rng.integers(-1, 2, size=(R, L), dtype=np.int8)
    pk = ((W+1).astype(np.uint16).reshape(R, L//5, 5) * POW).sum(2, dtype=np.uint16).astype(np.uint8)
    hdr = (R & 0xFFFFFFFF) | ((BPR & 0xFFFF) << 32)
    buf = bytearray()
    buf += hdr.to_bytes(8, "little")
    buf += x.tobytes()                       # BPR*8 バイト
    buf += pk.tobytes()                      # R × BPR*8 バイト
    assert len(buf) == 8 + BPR*WPB + R*BPR*8   # x は重み1個に1バイト＝BPR*40
    for i in range(0, len(buf), 8):
        beats.append(int.from_bytes(buf[i:i+8], "little"))
    exps.extend(int(v) for v in (W.astype(np.int32) @ x.astype(np.int32)))

open("sim/tbv_in.hex","w").write("".join(f"{b:016x}\n" for b in beats))
open("sim/tbv_exp.hex","w").write("".join(f"{v & 0xFFFFFFFF:08x}\n" for v in exps))
open("sim/tbv_cfg.vh","w").write(f"`define NIN  {len(beats)}\n`define NEXP {len(exps)}\n")
print(f"組 {len(CASES)} 個 (R,BPR)={CASES}")
print(f"  入力 {len(beats)} ビート / 期待値 {len(exps)} 語 / 範囲 {min(exps)}〜{max(exps)}")
