# BitNet b1.58 on a PYNQ-Z1

[日本語版はこちら (Japanese)](README_ja.md)

A small language model generating text on a **PYNQ-Z1**, a hobbyist Zynq-7020 board. The matrix
multiplications and attention run on **hand-written circuits in the FPGA fabric**; the two ARM
cores handle the rest.

```
$ ./run.sh gen "The capital of France is" 24
The capital of France is the city of Paris. The city has been the home of many people,
from the ancient Greeks and the Romans

Per token (median of 29): 124.3 ms -> 8.04 tok/s
```

- Model: [`1bitLLM/bitnet_b1_58-large`](https://huggingface.co/1bitLLM/bitnet_b1_58-large)
  (0.7 B parameters, trained with ternary weights {−1, 0, +1})
- **About 8 tok/s** with a short context and **7.3 tok/s at 512 tokens of context** (measured)
- The board's logits match the host's reference implementation **bit for bit**. You can check
  this yourself; see [Verify bit-exactness](#optional-verify-bit-exactness)
- A prebuilt bitstream is included, so **you don't need Vivado** to try it

Building a fast LLM box is not the goal (a GPU would win easily). The point is to build the whole
path yourself: RTL, DMA plumbing, quantization and the host–board split.

## Development journal

How this got built, one stage at a time, with the wrong turns and later corrections left in:
[**docs/**](docs/). There are five illustrated entries, from the first DDR bandwidth probe to
end-to-end generation, in English and Japanese.

## How it works

```
                     PYNQ-Z1 (XC7Z020, 512 MB DDR3)
 ┌──────────────── PS: 2x Cortex-A9 @ 650 MHz ────────────────┐
 │ tokenizer, embedding, RMSNorm, RoPE, SiLU, residuals,      │
 │ INT8 quantization  (C + NEON, e2e_cpu.c)                   │
 └──────┬─────────────────────────────────────────────▲───────┘
        │ DMA descriptors / registers (C, dma_glue.c) │ results
 ┌──────▼──────────────── PL @ 125 MHz ───────────────┴───────┐
 │ 2x ternary mat-vec engine (axis_tmacv)  80 weights/clock,  │
 │                                         LUTs only, 0 DSPs  │
 │ 2x attention + softmax core (axis_attnv) 17 multipliers    │
 │ 4x AXI DMA (2 simple, 2 scatter-gather)  -> HP0 + HP2      │
 └──────────────────────────┬─────────────────────────────────┘
                            │ ~2.0 GB/s (measured DDR ceiling)
                  weights 141 MiB + INT8 KV cache 36 MiB (in CMA)
```

- **Memory bandwidth is the bottleneck, not arithmetic.** With INT8, every weight byte costs one
  multiply, so bytes per second sets the speed limit. Ternary weights pack 5 to a byte
  (3⁵ = 243 ≤ 256), and a multiply becomes add / skip / subtract. That fits in LUTs, and the
  mat-vec engine uses **no DSP slices at all**.
- Both cores stream at **93–96 % of the DDR's measured ceiling** (~2.0 GB/s through HP0 + HP2).
- One token is 121 DMA transfers: 97 for mat-vec groups with variable row length, 24 for attention
  (one per layer). The attention DMAs use scatter-gather, so the growing KV cache is never moved
  or repacked. Only descriptor lengths are rewritten each token.
- The KV cache is kept as INT8 with per-layer, per-head scales fixed ahead of time from a
  calibration text.
- The CPU side is C with NEON intrinsics. It's written so the same source gives **bit-identical
  results** on the board (ARMv7) and on an Apple Silicon Mac (AArch64): no FMA contraction, fixed
  summation order, and hand-written exp and rounding.

Where the time goes at 512 tokens of context (measured): mat-vec 87.4 ms + attention 22.0 ms +
CPU 26.7 ms = 136 ms/token.

Resources: LUT 21,353 (40 %) / FF 32,483 (31 %) / BRAM 41 / DSP 38, WNS +0.514 ns at 125 MHz.

**Quality note:** `lm_head` in this model is not ternary (fp16, 98 MB). To fit everything into CMA
alongside the KV cache, it is ternarized here too. That raises perplexity on the test text from
17.9 (with the original fp16 `lm_head`) to 35.7. Text stays fluent, as the sample above shows.

## What you need

| | |
|---|---|
| Board | PYNQ-Z1 with the **PYNQ v3.1** SD image (tested: pynq 3.1.1, kernel 6.6.10-xilinx-v2024.1) and network access |
| Host PC | Python 3 with `numpy` and `huggingface_hub`, about 3 GB of free RAM (peak 2.9 GB measured), and about 4 GB of disk (2.7 GB model download + 0.4 GB output) |
| Vivado | **Not needed.** Only needed if you want to rebuild the bitstream |

The host steps are plain numpy and should work on any OS. They were tested on macOS (Apple
Silicon). The optional bit-exact check needs an ARM64 host (see below).

## Quick start

### 1. Prepare the board (once): enlarge CMA to 256 MB

The weights and KV cache need 179 MiB of physically contiguous memory (CMA). The PYNQ image's
default is 128 MB. Log in to the board as `xilinx` (default password `xilinx`) and look at the
boot config:

```bash
cat /boot/uEnv.txt
```

If there is **no** `bootargs=` line, create one from the current kernel arguments, adding `cma=256M`:

```bash
echo "bootargs=$(tr -d '\0' < /proc/device-tree/chosen/bootargs) cma=256M" | sudo tee -a /boot/uEnv.txt
```

If a `bootargs=` line already exists, append ` cma=256M` to the end of that line with an editor
instead. Then reboot:

```bash
sudo reboot
```

After the reboot, check:

```bash
grep CmaTotal /proc/meminfo
```

It should say `CmaTotal: 262144 kB`.

> If the board no longer boots, put the SD card in a PC. The boot partition is FAT, so any OS can
> open it. Remove the line you added to `uEnv.txt`.

### 2. Build the model files on the host

```bash
git clone https://github.com/tsjshg/fpga-pynq-z1.git
cd fpga-pynq-z1
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 model/pack_e2e.py
```

This downloads the model from Hugging Face (2.7 GB, first time only), ternarizes it and packs it
into the layout the circuits read. The result goes to `model/e2e/`, which takes about 15 s once
the model is downloaded. Then gather everything the board needs into one folder:

```bash
pynq/make_bundle.sh
```

This writes `dist/bitnet-pynq/` (336 MB): the packed weights, tokenizer, bitstream and the board
scripts.

### 3. Copy it to the board

```bash
scp -r dist/bitnet-pynq xilinx@<board-ip>:~/
```

### 4. Run

On the board:

```bash
cd ~/bitnet-pynq
./run.sh gen "The capital of France is" 24
```

The arguments are the prompt and the number of tokens to generate. The prompt plus the generated
tokens must fit in 512 tokens. `run.sh` asks for your sudo password, because loading a bitstream
and driving the DMAs need root. On the first run it also compiles two small C libraries with the
board's `gcc`. Loading takes about 20 s, then tokens appear one by one with their latency, and a
timing breakdown is printed at the end.

## Optional: verify bit-exactness

The board and the host run the same C code, and the host replaces the FPGA with an exact integer
model. The two should agree bit for bit. This check needs an **ARM64 host** such as an Apple
Silicon Mac, because `model/e2e_cpu.c` uses NEON intrinsics.

1. On the host, compute the reference (about 15 s), then rebuild the bundle so it includes the
   inputs the board needs:

   ```bash
   python3 model/e2e_eval.py ref
   pynq/make_bundle.sh
   scp -r dist/bitnet-pynq xilinx@<board-ip>:~/
   ```

2. On the board, run the same text and prompts. About a minute:

   ```bash
   cd ~/bitnet-pynq
   ./run.sh eval
   ```

3. Back on the host, fetch the results and compare:

   ```bash
   scp xilinx@<board-ip>:bitnet-pynq/board_out.npz xilinx@<board-ip>:bitnet-pynq/dump_gen0.npz .
   python3 model/e2e_eval.py compare board_out.npz
   python3 model/e2e_eval.py verify dump_gen0.npz
   ```

   `compare` checks every logit of 191 tokens (perplexity text plus two generations).
   `verify` replays all 3,509 FPGA calls of one generation (13 M output words) in exact integer
   arithmetic, including the KV cache state. Expected output:

   ```
   all logits and tokens match bit for bit
   checked 2,813 matmul + 696 attention calls = 13,055,162 words -> all match
   ```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `This must run as root …` | Start it with `./run.sh …`, not `python3 bitnet_run.py …` |
| `ModuleNotFoundError: No module named 'pynq'` | You ran `sudo python3 …`. PYNQ's venv is only set up in a login shell. `run.sh` handles this for you |
| `Could not allocate CMA in 3 attempts` | Check that `CmaTotal` is 262144 kB (step 1). Close Jupyter notebooks that hold PYNQ buffers, then retry. Rebooting clears fragmentation |
| `CMA allocation failed …` then `retrying` | Normal: CMA fragments over time, and the script frees what it got and retries up to 3 times |

## Rebuilding the hardware (optional)

You need Vivado 2024.1 (the free edition covers the XC7Z020). Everything runs in batch mode from
`hw/`:

```bash
cd hw
vivado -mode batch -source build_combo3.tcl -tclargs bd 125
vivado -mode batch -source build_combo3.tcl -tclargs all 125
```

The first command only builds and validates the block design, which is quick. The second runs
synthesis through to the bitstream and writes `hw/out/combo3_125.{bit,hwh}`.

The Verilog for both cores is **generated by Python**. Don't edit `hw/rtl/axis_*.v` by hand;
edit the generator and re-run it from `hw/`, e.g. `python3 rtl/gen_tmacv.py` or
`python3 rtl/gen_attnv.py`. Testbenches for xsim are in `hw/sim/`. Their stimulus and expected
values come from `hw/rtl/gen_*_tb.py`.

On an Apple Silicon Mac, Vivado can run in Docker via
[vivado-on-silicon-mac](https://github.com/ichi4096/vivado-on-silicon-mac). `hw/vivado.sh`
starts a container that mounts both that installation and `hw/`, and then runs the build (for
example `hw/vivado.sh build build_combo3.tcl bd 125`).

## Repository layout

```
model/pack_e2e.py        host: download, ternarize and pack the model -> model/e2e/
model/e2e_core.py        the token loop, shared by host and board; exact integer stand-in for the FPGA
model/e2e_cpu.c          CPU-side kernels (C + NEON); bit-identical on ARMv7 and AArch64
model/e2e_eval.py        host: reference run, compare and verify against the board
model/bitnet_np.py       numpy-only reference implementation of the model
model/bpe.py, st_read.py numpy-only tokenizer and safetensors reader (no torch/transformers)
pynq/bitnet_run.py       board: runs the model on the FPGA (gen / eval)
pynq/dma_glue.c          board: DMA start/wait/copy in C
pynq/run.sh              board: relaunches bitnet_run.py as root in a login shell
pynq/make_bundle.sh      host: gathers what the board needs into dist/bitnet-pynq/
hw/build_combo3.tcl      the design used here (2 mat-vec + 2 attention cores, 125 MHz)
hw/rtl/gen_tmacv.py      generator for the ternary mat-vec engine
hw/rtl/gen_attnv.py      generator for the attention + softmax core
hw/out/combo3_125.*      prebuilt bitstream and hardware handoff
```

The rest records the steps that led here, one experiment at a time: DDR bandwidth probes, an INT8
MAC array, the first ternary engine, attention for SmolLM2, and so on. That includes
`bench/`, the other `pynq/dma_*.py` and `hw/build_*.tcl`, and the other generators in `hw/rtl/`.
Those scripts take the bitstream path as their first argument.
The illustrated write-ups of every stage are in [`docs/`](docs/).

## License

MIT (see [LICENSE](LICENSE)). The model weights are not included. `pack_e2e.py` downloads them
from Hugging Face, and they are covered by that model's own license. The PS7 configuration in
`hw/ps7_common.tcl` is taken from the PYNQ-Z1 base overlay of
[Xilinx/PYNQ](https://github.com/Xilinx/PYNQ) (BSD 3-Clause).
