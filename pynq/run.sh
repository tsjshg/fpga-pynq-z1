#!/bin/bash
# bitnet_run.py を root・ログインシェルで起動し直す（ボード用）。
#   ./run.sh gen "The capital of France is" 24
#   ./run.sh eval
# PYNQ の venv と XILINX_XRT は /etc/profile.d でしか入らないので、sudo python3 … では動かない。
# 引数は printf %q で bash 向けに引用し直すので、空白や引用符を含む文章もそのまま渡せる。
cd "$(dirname "$0")" || exit 1
exec sudo bash -lc "cd $(printf %q "$PWD") && exec python3 bitnet_run.py $(printf '%q ' "$@")"
