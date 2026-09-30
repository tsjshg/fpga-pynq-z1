#!/usr/bin/env python3
"""axis_attnv の xsim 用の刺激と期待値。

回路と同じ整数手順をそのまま numpy で書く（これが golden model）:
    s_int = q·K[t]                        int32（|s| < 2^21）
    s8    = clamp((s_int·m) >> 24)        算術シフト。m は 18bit
    e     = EROM[max(s8) - s8]
    out[d]= Σ_t e[t]·V[t][d]  /  den = Σ e[t]
出力の順は out[0..95] のあと den。
"""
import math
import numpy as np

HD, BPB, SHM, TAU = 96, 8, 24, 8.0
BPR = HD//BPB
EROM = np.array([max(0, min(255, int(round(255.0*math.exp(-d/TAU))))) for d in range(256)],
                dtype=np.int64)
# (T, m)。m は 18bit。小さい m / 大きい m / 実測の中央付近 を混ぜる
CASES = [(3, 2003), (1, 52633), (8, 12000), (16, 262143), (2, 1), (12, 7669)]
rng = np.random.default_rng(999331)

beats, exps, sat = [], [], 0
for T, m in CASES:
    q = rng.integers(-127, 128, size=HD, dtype=np.int8)
    K = rng.integers(-127, 128, size=(T, HD), dtype=np.int8)
    V = rng.integers(-127, 128, size=(T, HD), dtype=np.int8)

    si = K.astype(np.int64) @ q.astype(np.int64)          # (T,)
    assert np.abs(si).max() < 2**21, "s が 22bit を超えた"
    s8 = np.array([(int(v)*m) >> SHM for v in si], dtype=np.int64)  # Python の >> は床関数
    sat += int(((s8 > 127) | (s8 < -128)).sum())
    s8 = np.clip(s8, -128, 127)
    mxv = int(s8.max())
    e = EROM[np.clip(mxv - s8, 0, 255)]
    o = (e @ V.astype(np.int64)).astype(np.int64)          # (HD,)
    den = int(e.sum())

    buf = bytearray()
    buf += ((T & 0xFFFFFFFF) | ((m & 0x3FFFF) << 32)).to_bytes(8, "little")
    buf += q.tobytes()
    buf += K.tobytes()
    buf += V.tobytes()
    assert len(buf) == 8 + HD + 2*T*HD
    for i in range(0, len(buf), 8):
        beats.append(int.from_bytes(buf[i:i+8], "little"))
    exps.extend(int(x) for x in o); exps.append(den)

open("sim/tba_in.hex","w").write("".join(f"{b:016x}\n" for b in beats))
open("sim/tba_exp.hex","w").write("".join(f"{v & 0xFFFFFFFF:08x}\n" for v in exps))
open("sim/tba_cfg.vh","w").write(f"`define NIN  {len(beats)}\n`define NEXP {len(exps)}\n")
print(f"組 {len(CASES)} 個 (T,m)={CASES}")
print(f"  入力 {len(beats)} ビート / 期待値 {len(exps)} 語（{len(CASES)}×{HD+1}）")
print(f"  clamp が効いた回数 {sat} / 期待値の範囲 {min(exps)}〜{max(exps)}")
