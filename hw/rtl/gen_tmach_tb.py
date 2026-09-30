#!/usr/bin/env python3
"""axis_tmach（見出し付き三値コア）の xsim 用の刺激と期待値。

見出しで組を区切れているか、組ごとに x が入れ替わっているかを確かめる。
R=1 のような端も踏ませる。
"""
import numpy as np
L, BPR = 640, 128
POW = np.array([1,3,9,27,81], dtype=np.uint16)
RS = [3, 1, 17, 8, 2, 64]          # 組ごとの行数
rng = np.random.default_rng(31415)

beats, exps = [], []
for R in RS:
    x = rng.integers(-127, 128, size=L, dtype=np.int8)
    W = rng.integers(-1, 2, size=(R, L), dtype=np.int8)
    pk = ((W+1).astype(np.uint16).reshape(R, BPR, 5) * POW).sum(2, dtype=np.uint16).astype(np.uint8)
    buf = bytearray()
    buf += int(R).to_bytes(8, "little")          # 見出し
    buf += x.tobytes()                            # x（640 B = 80 ビート）
    buf += pk.tobytes()                           # 重み（R × 128 B）
    assert len(buf) == 8 + L + R*BPR
    for i in range(0, len(buf), 8):
        beats.append(int.from_bytes(buf[i:i+8], "little"))
    exps.extend(int(v) for v in (W.astype(np.int32) @ x.astype(np.int32)))

open("sim/tbh_in.hex","w").write("".join(f"{b:016x}\n" for b in beats))
open("sim/tbh_exp.hex","w").write("".join(f"{v & 0xFFFFFFFF:08x}\n" for v in exps))
open("sim/tbh_cfg.vh","w").write(f"`define NIN  {len(beats)}\n`define NEXP {len(exps)}\n")
print(f"組 {len(RS)} 個 R={RS} → 入力 {len(beats)} ビート / 期待値 {len(exps)} 語")
print(f"  期待値の範囲 {min(exps)} 〜 {max(exps)}")
