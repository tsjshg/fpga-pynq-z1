#!/bin/zsh
# hw/ を Vivado（Docker + Rosetta 2）で焼くための入口。
#
# Vivado 本体は vivado-on-silicon-mac-main（44 GB）に入ったまま動かさない。
# そのディレクトリを /home/user に、このリポジトリの hw/ を /hw に載せたコンテナを
# VNC なしで立て、batch で叩く。VNC つきの vivado_container とは別名なので同時に動いてもよい。
#
#   hw/vivado.sh start                          コンテナを立てる（数秒）
#   hw/vivado.sh build build_combo3.tcl bd 125  /hw で vivado -mode batch（bd を先に通すこと）
#   hw/vivado.sh sh 'python3 rtl/gen_attnv.py'  /hw で任意のコマンド（xsim など）
#   hw/vivado.sh stop
#
# Vivado 本体の場所は VIVADO_MAC で変えられる。

VIVADO_MAC=${VIVADO_MAC:-$HOME/tools/vivado-on-silicon-mac-main}
NAME=vivado_pynq
HW=$(cd "$(dirname "$0")" && pwd)
PRE='export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1 /lib/x86_64-linux-gnu/libselinux.so.1 /lib/x86_64-linux-gnu/libz.so.1 /lib/x86_64-linux-gnu/libgdk-x11-2.0.so.0"; source /home/user/Xilinx/Vivado/2024.1/settings64.sh; cd /hw'

case "$1" in
  start)
    [[ -d $VIVADO_MAC/Xilinx/Vivado/2024.1 ]] || { echo "Vivado が見つからない: $VIVADO_MAC" >&2; exit 1; }
    docker ps --format '{{.Names}}' | grep -qx $NAME && { echo "$NAME は起動済み"; exit 0; }
    docker run -d --init --rm --name $NAME \
      --mount type=bind,source="$VIVADO_MAC",target=/home/user \
      --mount type=bind,source="$HW",target=/hw \
      --platform linux/amd64 x64-linux sleep infinity ;;
  build)
    shift; tcl=$1; shift
    docker exec -u user $NAME bash -lc "$PRE && vivado -mode batch -nojournal -notrace -source $tcl -tclargs $*" ;;
  sh)
    shift
    docker exec -u user $NAME bash -lc "$PRE && $*" ;;
  stop)
    docker kill $NAME ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
