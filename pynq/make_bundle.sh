#!/bin/bash
# ボードへ送る一式を dist/bitnet-pynq/ にまとめる（PC 側で実行）。
#   pynq/make_bundle.sh            → dist/bitnet-pynq/（約 430 MB）
#   scp -r dist/bitnet-pynq xilinx@<board>:~/
# 先に python3 model/pack_e2e.py で model/e2e/ を作っておくこと。
# model/e2e/ref_in.npz があれば入れる（ボードの ./run.sh eval が使う）。
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
E=$R/model/e2e
O=$R/dist/bitnet-pynq

for f in index.json params.npz emb.f32 tokenizer.json; do
  [ -f "$E/$f" ] || { echo "missing $E/$f -- run: python3 model/pack_e2e.py" >&2; exit 1; }
done
[ -f "$R/hw/out/combo3_125.bit" ] || { echo "missing hw/out/combo3_125.bit" >&2; exit 1; }

rm -rf "$O"; mkdir -p "$O"
cp "$R"/pynq/bitnet_run.py "$R"/pynq/dma_glue.c "$R"/pynq/run.sh "$O"/
cp "$R"/model/e2e_core.py "$R"/model/e2e_cpu.c "$R"/model/bpe.py "$O"/
cp "$R"/hw/out/combo3_125.bit "$R"/hw/out/combo3_125.hwh "$O"/
cp "$E"/g*_*.bin "$E"/index.json "$E"/params.npz "$E"/emb.f32 "$E"/tokenizer.json "$O"/
[ -f "$E/ref_in.npz" ] && cp "$E/ref_in.npz" "$O"/
chmod +x "$O/run.sh"
echo "wrote $O ($(du -sh "$O" | cut -f1))"
