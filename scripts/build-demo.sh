#!/usr/bin/env bash
set -euo pipefail

# Baut eine (oder alle) LandrixAI-MCP-Demo(s) unter Linux/FPC (i.d.R. ueber WSL).
# Teilt den mORMot-Unit-Cache (bin/fpc/lib) mit run-fpc-tests.sh -> Folgebuilds
# sind schnell.
#
#   build-demo.sh                         # alle Demos (examples + 4 Transporte)
#   build-demo.sh mcp.examples.dpr        # eine Demo (Pfad relativ zu demos/)
#   build-demo.sh stdio/demo.mcp.stdio.dpr
#   VERBOSE=1 build-demo.sh ...           # vollen Compilerlog zeigen

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MORMOT2="$ROOT/shared/delphi/libs/_git_Synopse2"
SRC="$MORMOT2/src"
STATIC="$MORMOT2/static"
LIB_SRC="$ROOT/shared/delphi/landrixai/src"
DEMO_SRC="$ROOT/shared/delphi/landrixai/demos"
OUT="$ROOT/shared/delphi/landrixai/bin/fpc"
DEMO_OUT="$OUT/demos"
UNIT_OUT="$OUT/lib"      # geteilt mit run-fpc-tests.sh (mORMot-Cache)
TARGET="${TARGET:-linux}"
ARCH="${ARCH:-$(fpc -iTP)}"

UNITS="$SRC/app;$SRC/core;$SRC/crypt;$SRC/db;$SRC/lib;$SRC/net;$SRC/orm;$SRC/rest;$SRC/soa;$SRC/script;$SRC/misc;$SRC/tools/mget"
INCLUDES="$SRC;$SRC/core;$SRC/net"

mkdir -p "$UNIT_OUT" "$DEMO_OUT"

if [ "$#" -ge 1 ] && [ "$1" != "all" ]; then
  DEMOS=("$1")
else
  DEMOS=(mcp.examples.dpr stdio/demo.mcp.stdio.dpr http/demo.mcp.http.dpr \
         sse/demo.mcp.sse.dpr streamable/demo.mcp.streamable.dpr)
fi

build_one() {
  local dpr="$1"
  local name; name="$(basename "$dpr" .dpr)"
  local log="$DEMO_OUT/$name.buildlog"
  echo ">>> demo $name ($ARCH-$TARGET)"
  set +e
  fpc -MDelphi -Sci -Ci -O2 -g -gl -gw2 \
    -T"$TARGET" -P"$ARCH" \
    -Fi"$INCLUDES;$ROOT/shared/delphi/landrixai/vendor" \
    -Fu"$LIB_SRC;$DEMO_SRC;$ROOT/shared/delphi/landrixai/vendor;$UNITS" \
    -Fl"$STATIC/$ARCH-$TARGET" \
    -FU"$UNIT_OUT" \
    -FE"$DEMO_OUT" \
    -o"$DEMO_OUT/$name" \
    "$DEMO_SRC/$dpr" > "$log" 2>&1
  local rc=$?
  set -e
  if [ "${VERBOSE:-0}" = "1" ]; then
    cat "$log"
  else
    grep -E '(Warning|Error|Fatal):' "$log" | grep -vE '^mormot\..*Warning:' || true
  fi
  if [ $rc -ne 0 ]; then
    echo "FAIL(compile): $name (rc=$rc) - $log" >&2
    tail -20 "$log" >&2
    return $rc
  fi
  echo "OK: $DEMO_OUT/$name"
}

fail=0
for d in "${DEMOS[@]}"; do
  build_one "$d" || fail=1
done
[ "$fail" -eq 0 ] && echo "ALL DEMOS OK" || { echo "SOME DEMOS FAILED" >&2; exit 1; }
