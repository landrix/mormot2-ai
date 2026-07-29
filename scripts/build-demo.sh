#!/usr/bin/env bash
set -euo pipefail

# Baut eine (oder alle) LandrixAI-MCP-Demo(s) unter Linux/FPC (i.d.R. ueber WSL).
# Teilt den mORMot-Unit-Cache (bin/fpc/lib) mit run-fpc-tests.sh -> Folgebuilds
# sind schnell.
#
#   build-demo.sh                         # ALLE Demos (MCP-Transporte + LLM + RAG)
#   build-demo.sh mcp.examples.dpr        # eine Demo (Pfad relativ zu demos/)
#   build-demo.sh stdio/demo.mcp.stdio.dpr
#   VERBOSE=1 build-demo.sh ...           # vollen Compilerlog zeigen

# Wurzel dieser Extension (scripts/..) - unabhaengig vom Einbindungsort.
LAI="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Konsumenten-Repo: im Submodul-Fall liegt die Extension unter
# <konsument>/shared/delphi/landrixai, also drei Ebenen ueber LAI. Nur noch
# fuer den mORMot2-Default unten relevant.
ROOT="$(cd "$LAI/../../.." && pwd)"
# mORMot2-Quellbaum: per MORMOT2_ROOT frei setzbar (Standalone-Klon dieser
# Extension); ohne Override gilt der Submodul-Pfad im Konsumenten-Repo.
MORMOT2="${MORMOT2_ROOT:-$ROOT/shared/delphi/libs/_git_Synopse2}"
SRC="$MORMOT2/src"
STATIC="$MORMOT2/static"
LIB_SRC="$LAI/src"
DEMO_SRC="$LAI/demos"
OUT="$LAI/bin/fpc"
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
  # WICHTIG: hier ALLE Demos auflisten, sonst meldet der "all"-Build faelschlich
  # "ALL DEMOS OK", obwohl LLM-/RAG-Demos nie gebaut (und evtl. kaputt) sind.
  DEMOS=(mcp.examples.dpr stdio/demo.mcp.stdio.dpr http/demo.mcp.http.dpr \
         sse/demo.mcp.sse.dpr streamable/demo.mcp.streamable.dpr \
         llm/llm-chat.dpr llm/llm-agent.dpr llm/llm-structured.dpr \
         llm/llm-embed.dpr llm/llm-vision.dpr llm/llm-anthropic.dpr \
         rag/rag-spike.dpr rag/rag-chat.dpr rag/rag-agent.dpr)
fi

build_one() {
  local dpr="$1"
  local name; name="$(basename "$dpr" .dpr)"
  local log="$DEMO_OUT/$name.buildlog"
  echo ">>> demo $name ($ARCH-$TARGET)"
  set +e
  fpc -MDelphi -Sci -Ci -O2 -g -gl -gw2 \
    -T"$TARGET" -P"$ARCH" \
    -Fi"$INCLUDES;$LAI/vendor" \
    -Fu"$LIB_SRC;$DEMO_SRC;$LAI/vendor;$UNITS" \
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
