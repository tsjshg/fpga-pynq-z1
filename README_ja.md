# PYNQ-Z1 で動く BitNet b1.58

[English](README.md)

趣味向けの Zynq-7020 ボード **PYNQ-Z1** の上で、小さな言語モデルに文章を書かせます。
行列積と attention は **FPGA のファブリックに手で書いた回路**で計算し、残りは ARM の 2 コアが受け持ちます。

```
$ ./run.sh gen "The capital of France is" 24
The capital of France is the city of Paris. The city has been the home of many people,
from the ancient Greeks and the Romans

Per token (median of 29): 124.3 ms -> 8.04 tok/s
```

- モデル: [`1bitLLM/bitnet_b1_58-large`](https://huggingface.co/1bitLLM/bitnet_b1_58-large)
  （0.7 B パラメータ。重みを {−1, 0, +1} の三値で学習したもの）
- 速さ（実測）: 文脈が短いときで**約 8 tok/s**、**文脈 512 トークンで 7.3 tok/s**
- ボードの logits は、PC 側の基準実装と**ビット単位で一致**します。手元で確かめる手順は
  [ビット単位の一致を確かめる](#任意ビット単位の一致を確かめる)にあります
- ビルド済みのビットストリームが入っているので、**Vivado がなくても試せます**

目的は速い LLM 実行環境を作ることではありません（それなら GPU を買えば済みます）。
回路、DMA まわり、量子化、ホストとボードの分担まで、全部を自分で作ることが目的です。

## 開発記録

ここまでの経緯を、最初の DDR の帯域の測定から通しの生成まで、図入りの 5 回分にまとめてあります。
途中の誤りと後からの訂正もそのまま残してあります（日本語・英語）。

**Web で読む: <https://tsjshg.github.io/fpga-pynq-z1/>**（元のファイルは [`docs/`](docs/)）

## しくみ

```
                     PYNQ-Z1 (XC7Z020, DDR3 512 MB)
 ┌──────────── PS: Cortex-A9 x2 @ 650 MHz ────────────────────┐
 │ トークン化・埋め込み・RMSNorm・RoPE・SiLU・残差・          │
 │ INT8 への量子化（C + NEON、e2e_cpu.c）                     │
 └──────┬─────────────────────────────────────────────▲───────┘
        │ DMA の記述子・レジスタ（C、dma_glue.c）     │ 結果
 ┌──────▼──────────────── PL @ 125 MHz ───────────────┴───────┐
 │ 三値の行列ベクトル積 x2（axis_tmacv）1クロック 80 重み、   │
 │                                     LUT のみ・DSP 0 個     │
 │ attention + softmax x2（axis_attnv）掛け算器 17 個         │
 │ AXI DMA x4（単純転送 2・SG 2）-> HP0 + HP2                 │
 └──────────────────────────┬─────────────────────────────────┘
                            │ 約 2.0 GB/s（実測した DDR の上限）
                  重み 141 MiB + INT8 の KV キャッシュ 36 MiB（CMA 上）
```

- **律速は演算ではなくメモリ帯域です。** INT8 なら重み 1 バイトにつき掛け算 1 回なので、
  1 秒に何バイト読めるかがそのまま速さの上限になります。三値の重みは 1 バイトに 5 個詰められ
  （3⁵ = 243 ≤ 256）、掛け算は「足す／何もしない／引く」になります。これは LUT で組めるので、
  行列ベクトル積の回路は **DSP を 1 個も使いません**。
- どちらの回路も、**実測した DDR の上限の 93〜96 %**（HP0 + HP2 で約 2.0 GB/s）で読み続けます。
- 1 トークンは DMA 転送 121 回です。内訳は、行長可変の行列ベクトル積が 97 回、attention が 24 回
  （層ごとに 1 回）。attention 側の DMA は SG（散布収集）なので、伸びていく KV キャッシュを
  動かしたり詰め直したりしません。毎トークン書き換えるのは記述子の長さ欄だけです。
- KV キャッシュは INT8 で持ちます。尺度は層・ヘッドごとに、校正用の文章であらかじめ固定しています。
- CPU 側は NEON の組み込み関数を使った C です。同じソースがボード（ARMv7）と Apple Silicon の
  Mac（AArch64）で**ビット単位で同じ結果**を出すように書いています。そのために、積和の融合をさせない、
  和の順序を固定する、exp と丸めを自前で書く、といった制約を置いています。

文脈 512 トークンでの 1 トークンの内訳（実測）: 行列ベクトル積 87.4 ms + attention 22.0 ms +
CPU 26.7 ms = 136 ms。

資源: LUT 21,353（40 %）/ FF 32,483（31 %）/ BRAM 41 / DSP 38、125 MHz で WNS +0.514 ns。

**品質について:** このモデルの `lm_head` は三値ではありません（fp16 で 98 MB）。
KV キャッシュと一緒に CMA に収めるため、ここでは `lm_head` も三値にしています。
そのぶん評価用の文章での perplexity は 17.9（元の fp16 の `lm_head`）から 35.7 に上がりますが、
上の例のとおり文章は崩れません。

## 必要なもの

| | |
|---|---|
| ボード | PYNQ-Z1 と **PYNQ v3.1** の SD イメージ（確認済み: pynq 3.1.1・kernel 6.6.10-xilinx-v2024.1）。ネットワークにつながっていること |
| PC | Python 3 と `numpy`・`huggingface_hub`。空きメモリ約 3 GB（実測の最大 2.9 GB）、ディスク約 4 GB（モデルのダウンロード 2.7 GB + 出力 0.4 GB） |
| Vivado | **不要**。ビットストリームを作り直すときだけ要ります |

PC 側の手順は numpy だけなので OS を選ばないはずですが、確認したのは macOS（Apple Silicon）だけです。
任意の「ビット単位の一致の確認」だけは ARM64 の PC が要ります（後述）。

## 手順

### 1. ボードの準備（1 回だけ）: CMA を 256 MB にする

重みと KV キャッシュで、物理的に連続したメモリ（CMA）が 179 MiB 要ります。
PYNQ のイメージの既定は 128 MB です。`xilinx` でボードにログインし（既定のパスワードは `xilinx`）、
起動設定を確認します。

```bash
cat /boot/uEnv.txt
```

`bootargs=` の行が**なければ**、今のカーネル引数に `cma=256M` を足した行を作ります。

```bash
echo "bootargs=$(tr -d '\0' < /proc/device-tree/chosen/bootargs) cma=256M" | sudo tee -a /boot/uEnv.txt
```

`bootargs=` の行がすでにあるなら、代わりにエディタでその行の末尾に ` cma=256M` を足してください。
そのあと再起動します。

```bash
sudo reboot
```

再起動したら確認します。

```bash
grep CmaTotal /proc/meminfo
```

`CmaTotal: 262144 kB` と出れば成功です。

> 起動しなくなったら、SD カードを PC に挿してください。起動用のパーティションは FAT なので
> どの OS でも開けます。`uEnv.txt` に足した行を消せば元に戻ります。

### 2. PC でモデルのファイルを作る

```bash
git clone https://github.com/tsjshg/fpga-pynq-z1.git
cd fpga-pynq-z1
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 model/pack_e2e.py
```

Hugging Face からモデルを落とし（初回だけ 2.7 GB）、三値にして、回路が読む形に詰めます。
結果は `model/e2e/` に出ます。ダウンロード済みなら 15 秒ほどです。
続けて、ボードに要るものを 1 つのフォルダにまとめます。

```bash
pynq/make_bundle.sh
```

`dist/bitnet-pynq/`（336 MB）ができます。中身は、詰めた重み・tokenizer・ビットストリーム・ボード側のスクリプトです。

### 3. ボードへ送る

```bash
scp -r dist/bitnet-pynq xilinx@<ボードの IP>:~/
```

### 4. 動かす

ボードの上で:

```bash
cd ~/bitnet-pynq
./run.sh gen "The capital of France is" 24
```

引数は、書き出しの文章と、生成するトークン数です。書き出しと生成を合わせて 512 トークンに
収めてください。ビットストリームの書き込みと DMA の操作には root が要るので、`run.sh` は
sudo のパスワードを聞きます。初回は、ボードの `gcc` で小さな C のライブラリを 2 つ作ります。
読み込みに 20 秒ほどかかったあと、1 トークンずつ所要時間つきで表示され、最後に時間の内訳が出ます。

## 任意：ビット単位の一致を確かめる

ボードと PC は同じ C のコードを走らせます。PC 側では FPGA の代わりに、整数で厳密に計算する
模型を使います。なので両者はビット単位で一致するはずです。`model/e2e_cpu.c` が NEON の組み込み関数で
書いてあるため、この確認には **ARM64 の PC**（Apple Silicon の Mac など）が要ります。

1. PC で基準を計算します（15 秒ほど）。そのあと、ボードに要る入力を含めて一式を作り直して送ります。

   ```bash
   python3 model/e2e_eval.py ref
   pynq/make_bundle.sh
   scp -r dist/bitnet-pynq xilinx@<ボードの IP>:~/
   ```

2. ボードで同じ文章と書き出しを流します（1 分ほど）。

   ```bash
   cd ~/bitnet-pynq
   ./run.sh eval
   ```

3. PC に結果を持ち帰って比べます。

   ```bash
   scp xilinx@<ボードの IP>:bitnet-pynq/board_out.npz xilinx@<ボードの IP>:bitnet-pynq/dump_gen0.npz .
   python3 model/e2e_eval.py compare board_out.npz
   python3 model/e2e_eval.py verify dump_gen0.npz
   ```

   `compare` は 191 トークンぶん（perplexity の文章と生成 2 本）の logits を全部比べます。
   `verify` は生成 1 本ぶんの FPGA 呼び出し 3,509 回（出力 1,305 万語）を、KV キャッシュの状態も
   含めて整数で計算し直して照合します。次のように出れば一致です。

   ```
   all logits and tokens match bit for bit
   checked 2,813 matmul + 696 attention calls = 13,055,162 words -> all match
   ```

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| `This must run as root …` | `python3 bitnet_run.py …` ではなく `./run.sh …` で起動する |
| `ModuleNotFoundError: No module named 'pynq'` | `sudo python3 …` で起動している。PYNQ の venv はログインシェルでしか有効にならない。`run.sh` がそこを面倒みる |
| `Could not allocate CMA in 3 attempts` | `CmaTotal` が 262144 kB か確かめる（手順 1）。PYNQ のバッファを握っている Jupyter のノートブックを閉じてやり直す。断片化は再起動で解消する |
| `CMA allocation failed …` のあと `retrying` | 正常な動作。CMA は使ううちに断片化するので、取った分を返して 3 回までやり直す |

## ハードウェアを作り直す（任意）

Vivado 2024.1 が要ります（XC7Z020 は無償版で扱えます）。すべて `hw/` でバッチ実行します。

```bash
cd hw
vivado -mode batch -source build_combo3.tcl -tclargs bd 125
vivado -mode batch -source build_combo3.tcl -tclargs all 125
```

1 つ目はブロックデザインを組んで検証するだけなので短時間で終わります。2 つ目は合成からビットストリームまで通し、
`hw/out/combo3_125.{bit,hwh}` を書き出します。

2 つの回路の Verilog は **Python で生成**しています。`hw/rtl/axis_*.v` は直接編集せず、生成スクリプトを
直して `hw/` で実行し直してください（例: `python3 rtl/gen_tmacv.py`、`python3 rtl/gen_attnv.py`）。
xsim 用のテストベンチは `hw/sim/` にあり、刺激と期待値は `hw/rtl/gen_*_tb.py` が作ります。

Apple Silicon の Mac では、[vivado-on-silicon-mac](https://github.com/ichi4096/vivado-on-silicon-mac) で
Vivado を Docker の中で動かせます。`hw/vivado.sh` は、そのインストール先と `hw/` の両方を載せたコンテナを立てて
ビルドを流します（例: `hw/vivado.sh build build_combo3.tcl bd 125`）。

## ファイルの構成

```
model/pack_e2e.py        PC: モデルを落として三値にし、回路の形に詰める → model/e2e/
model/e2e_core.py        1 トークンずつ進める本体。PC とボードで共通。FPGA の代わりを整数で厳密に計算する模型も入っている
model/e2e_cpu.c          CPU 側の処理（C + NEON）。ARMv7 と AArch64 でビット単位で同じ結果
model/e2e_eval.py        PC: 基準の計算、ボードとの比較（compare）と照合（verify）
model/bitnet_np.py       numpy だけで書いたモデルの基準実装
model/bpe.py, st_read.py numpy だけの tokenizer と safetensors の読み込み（torch / transformers 不要）
pynq/bitnet_run.py       ボード: FPGA でモデルを動かす（gen / eval）
pynq/dma_glue.c          ボード: DMA の起動・待ち・写しを C で
pynq/run.sh              ボード: bitnet_run.py を root・ログインシェルで起動し直す
pynq/make_bundle.sh      PC: ボードに要るものを dist/bitnet-pynq/ にまとめる
hw/build_combo3.tcl      ここで使う回路（行列ベクトル積 2 + attention 2、125 MHz）
hw/rtl/gen_tmacv.py      三値の行列ベクトル積の回路を生成する
hw/rtl/gen_attnv.py      attention + softmax の回路を生成する
hw/out/combo3_125.*      ビルド済みのビットストリームとハードウェア情報
```

残りは、ここに至るまでの実験を 1 段ずつ残したものです。DDR の帯域の測定、INT8 の積和アレイ、
最初の三値の回路、SmolLM2 向けの attention などで、`bench/`、ほかの `pynq/dma_*.py` と
`hw/build_*.tcl`、`hw/rtl/` のほかの生成スクリプトがそれにあたります。これらのスクリプトは、
第 1 引数にビットストリームのパスを取ります。
各段階を図入りでまとめた記録は [`docs/`](docs/) にあり、<https://tsjshg.github.io/fpga-pynq-z1/> で公開しています。

## ライセンス

MIT（[LICENSE](LICENSE)）。モデルの重みは含みません。`pack_e2e.py` が Hugging Face から落とし、
重みにはそのモデル自身のライセンスが適用されます。`hw/ps7_common.tcl` の PS7 の設定は、
[Xilinx/PYNQ](https://github.com/Xilinx/PYNQ) の PYNQ-Z1 base overlay から取ったものです（BSD 3-Clause）。
