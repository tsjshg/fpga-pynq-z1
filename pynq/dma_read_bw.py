#!/usr/bin/env python3
"""PYNQ-Z1: PL が DDR から「読むだけ」の帯域を測る。

デザイン: PS7 -HP0- SmartConnect - AXI DMA -MM2S-> axis_sink
シンクは受け取ったビートを捨てるだけなので、DDR への書き戻しが無い。
つまりここで出る値がそのまま DDR から見たトラフィックであり、
「重みを毎秒何バイト流し込めるか」の上限になる。

比べるべき天井は2つある:
  経路の上限 = 8 B (64bit AXIS) × FCLK      … PL 側の道幅
  DDR の上限 = 2.1 GB/s                      … DDR3 16bit @1050MT/s
どちらに張り付いているかで、次に何を変えるべきかが決まる。

使い方: python3 dma_read_bw.py <bitstream> [MB] [繰り返し回数]
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

BIT = sys.argv[1] if len(sys.argv) > 1 else "/opt/fpga-bench/sink100.bit"
MB  = int(sys.argv[2]) if len(sys.argv) > 2 else 32
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
N   = MB * 1024 * 1024

DDR_PEAK = 2.1   # GB/s  DDR3 16bit @525MHz(=1050MT/s)

print(f"ビットストリーム: {BIT}")
ol = Overlay(BIT)
print("  ロード成功  IP:", list(ol.ip_dict.keys()))

# FCLK は .hwh の指定にしたがって Overlay が設定する。実測の解釈に必要なので読み戻す。
try:
    from pynq.ps import Clocks
    fclk = Clocks.fclk0_mhz
except Exception as e:                      # 取れなくても測定自体は続行する
    fclk = None
    print(f"  （FCLK の読み出しに失敗: {e}）")

if fclk:
    path_peak = 8 * fclk * 1e6 / 1e9        # 64bit ストリーム = 8 B/クロック
    print(f"  FCLK_CLK0 = {fclk:.2f} MHz → 経路の上限 {path_peak:.2f} GB/s")
else:
    path_peak = None

dma = ol.axi_dma_0
# recvchannel の有無では判定できない（PYNQ は S2MM が無くても属性を持つ）。
# .hwh に焼かれている生成パラメータを直接見る。
par = ol.ip_dict["axi_dma_0"]["parameters"]
if par.get("C_INCLUDE_MM2S") != "1":
    sys.exit("★ このビットストリームには MM2S がありません")
if par.get("C_INCLUDE_S2MM") == "1":
    print("  注意: S2MM が有効なままです。読み専用の測定になっていません")
else:
    print("  S2MM 無し = 書き戻しトラフィック無しを確認")

print(f"\nバッファ {MB} MB を確保します（連続物理メモリ）")
src = allocate(shape=(N,), dtype=np.uint8)
print(f"  物理アドレス = 0x{src.physical_address:08x}")
rng = np.random.default_rng(12345)
src[:] = rng.integers(0, 256, size=N, dtype=np.uint8)
src.flush()

print(f"\n{REP} 回計測します（1回あたり {MB} MB を読み出し）")
times = []
for i in range(REP):
    t0 = time.perf_counter()
    dma.sendchannel.transfer(src)
    dma.sendchannel.wait()
    dt = time.perf_counter() - t0
    times.append(dt)
    bw = N / dt / 1e9
    print(f"  {i+1}回目  {dt*1000:7.2f} ms   読み出し {bw:5.2f} GB/s")

t  = min(times)
bw = N / t / 1e9

print("\n=== 結果 ===")
print(f"  最速          : {t*1000:.2f} ms")
print(f"  PL の読み出し : {bw:.2f} GB/s  ← これがそのまま DDR トラフィック")
if path_peak:
    print(f"  経路の上限比  : {100*bw/path_peak:5.1f}%  (8B × {fclk:.2f}MHz = {path_peak:.2f} GB/s)")
print(f"  DDR の上限比  : {100*bw/DDR_PEAK:5.1f}%  ({DDR_PEAK} GB/s)")

if path_peak and bw / path_peak > 0.90:
    print("\n  → 道幅に張り付いている。DDR にはまだ余裕がある。FCLK を上げるべき")
elif bw / DDR_PEAK > 0.80:
    print("\n  → DDR 側の壁に到達。これがこのボードの真の上限")
else:
    print("\n  → どちらの天井にも届いていない。DMA かバースト長を疑う")

# INT8 なら 1 バイト = 重み1個 = 掛け算1回。三値なら 1 バイトに約5個入る。
print(f"\n  重みの流量に換算すると:")
print(f"    INT8 (1.0 B/重み)      {bw:6.2f} G重み/s")
print(f"    三値 (約0.2 B/重み)    {bw/0.2:6.2f} G重み/s")

src.freebuffer()
