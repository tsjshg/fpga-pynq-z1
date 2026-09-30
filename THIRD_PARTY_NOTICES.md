# Third-party notices

The MIT license in [LICENSE](LICENSE) covers the code and documents written for this repository.
The items below are not covered by it.

## PYNQ-Z1 PS7 configuration (Xilinx/PYNQ, BSD 3-Clause)

`hw/ps7_common.tcl` and `hw/build.tcl` contain the Zynq PS7 configuration properties of the
PYNQ-Z1 base overlay, taken unchanged from
[`boards/Pynq-Z1/base/base.tcl`](https://github.com/Xilinx/PYNQ/blob/v3.1/boards/Pynq-Z1/base/base.tcl)
in [Xilinx/PYNQ](https://github.com/Xilinx/PYNQ). They are used under the following license:

```
Copyright (c) 2016-2021, Xilinx, Inc.
SPDX-License-Identifier: BSD-3-Clause

BSD 3-Clause License

Copyright (c) 2018, Xilinx
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Bitstream and other Vivado outputs (AMD/Xilinx)

Vivado 2024.1 generated the files in `hw/out/`: the bitstream `combo3_125.bit`, the hardware
handoff files `*.hwh` and the utilization reports `*_util.txt`. The bitstream contains AMD
(Xilinx) IP cores: the Zynq-7000 processing system, AXI DMA, AXI SmartConnect, AXI Interconnect,
AXI4-Stream Data FIFO and Processor System Reset. The circuits written for this project are also in it.
The AMD/Xilinx parts of these files remain subject to AMD's license terms for Vivado and its IP.
They are provided only for use on the PYNQ-Z1 (Zynq XC7Z020), and they are not covered by the
MIT license. The circuits written for this project are MIT-licensed as source in `hw/rtl/`.

## Model weights

No model weights or tokenizer files are included. `model/pack_e2e.py` downloads
[`1bitLLM/bitnet_b1_58-large`](https://huggingface.co/1bitLLM/bitnet_b1_58-large) (MIT, per its
model card) from Hugging Face. The earlier-stage scripts download
[`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (Apache-2.0).
Their own licenses apply to the weights and anything derived from them, including the packed
files that `pack_e2e.py` writes.
