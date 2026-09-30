#!/usr/bin/env bash
# Einmal-Verifikation der x86_64-linux vec0/lembed0-Extensions (Docker-Deploy-Ziel)
# gegen mORMots STATISCHES SQLite (nicht die Distro-CLI). Baut rag-spike nativ fuer
# x86_64-linux mit FPC und laeuft es mit den x86_64-linux-Extensions. Gedacht fuer
# einen amd64-Container (QEMU) auf einem aarch64-Host. Output in /tmp (kein Eingriff
# in den aarch64-Unit-Cache unter bin/fpc).
set -euo pipefail

ROOT="${ROOT:-/work}"
MORMOT2="$ROOT/shared/delphi/libs/_git_Synopse2"
SRC="$MORMOT2/src"
STATIC="$MORMOT2/static"
LAI="${LAI_ROOT:-$ROOT/shared/delphi/landrixai}"
# Compiler fest auf die Baseline (setzt $FPC/$FPC_VERSION, Override: FPC=<compiler>).
source "$LAI/scripts/fpc-env.sh"
LIB_SRC="$LAI/src"
DEMO_SRC="$LAI/demos"
VENDOR="$LAI/vendor"
ARCH="x86_64"
OUT="/tmp/x64-ext"
UNIT_OUT="$OUT/lib/$FPC_VERSION"
mkdir -p "$UNIT_OUT"

UNITS="$SRC/app;$SRC/core;$SRC/crypt;$SRC/db;$SRC/lib;$SRC/net;$SRC/orm;$SRC/rest;$SRC/soa;$SRC/script;$SRC/misc;$SRC/tools/mget"
INCLUDES="$SRC;$SRC/core;$SRC/net"

echo ">>> fpc $FPC_VERSION ($FPC) target=$ARCH-linux"
"$FPC" -MDelphi -Sci -Ci -O2 -Tlinux -P"$ARCH" \
  -Fi"$INCLUDES;$VENDOR" \
  -Fu"$LIB_SRC;$DEMO_SRC;$VENDOR;$UNITS" \
  -Fl"$STATIC/$ARCH-linux" \
  -FU"$UNIT_OUT" \
  -FE"$OUT" \
  -o"$OUT/rag-spike" \
  "$DEMO_SRC/rag/rag-spike.lpr" > "$OUT/build.log" 2>&1 \
  || { echo "FAIL(compile)"; tail -25 "$OUT/build.log"; exit 1; }
echo "OK build: $OUT/rag-spike"

export SQLITE_EXT_DIR="$VENDOR/sqlite-ext/x86_64-linux"
export LEMBED_MODEL="$VENDOR/models/all-MiniLM-L6-v2.e4ce9877.f16.gguf"
export LEMBED_DIM=384
export LD_LIBRARY_PATH="$SQLITE_EXT_DIR"
echo ">>> run rag-spike (x86_64, mORMot static SQLite + x86_64 vec0/lembed0)"
"$OUT/rag-spike"
