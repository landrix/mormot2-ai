#!/usr/bin/env bash
set -euo pipefail

# Baut den code-nav MCP-Server (stdio) unter Linux/FPC (i.d.R. ueber WSL).
# Teilt den mORMot-Unit-Cache (bin/fpc/lib) mit run-fpc-tests.sh.
#
#   build-codenav.sh            # bauen
#   VERBOSE=1 build-codenav.sh  # vollen Compilerlog zeigen

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MORMOT2="$ROOT/shared/delphi/libs/_git_Synopse2"
SRC="$MORMOT2/src"
STATIC="$MORMOT2/static"
LIB_SRC="$ROOT/shared/delphi/landrixai/src"
CN_SRC="$ROOT/shared/delphi/landrixai/codenav"
DEP="$ROOT/shared/delphi/landrixai/vendor"   # mormot.ext.os (fork+pipe RunRedirect)
OUT="$ROOT/shared/delphi/landrixai/bin/fpc"
UNIT_OUT="$OUT/lib"
TARGET="${TARGET:-linux}"
ARCH="${ARCH:-$(fpc -iTP)}"

UNITS="$SRC/app;$SRC/core;$SRC/crypt;$SRC/db;$SRC/lib;$SRC/net;$SRC/orm;$SRC/rest;$SRC/soa;$SRC/script;$SRC/misc;$SRC/tools/mget"
INCLUDES="$SRC;$SRC/core;$SRC/net"

PROG="codenav.mcp"
BUILDLOG="$OUT/$PROG.buildlog"

mkdir -p "$UNIT_OUT" "$OUT"

echo ">>> $PROG ($ARCH-$TARGET)"
set +e
fpc -MDelphi -Sci -Ci -O2 -g -gl -gw2 \
  -T"$TARGET" -P"$ARCH" \
  -Fi"$INCLUDES;$DEP" \
  -Fu"$LIB_SRC;$CN_SRC;$DEP;$UNITS" \
  -Fl"$STATIC/$ARCH-$TARGET" \
  -FU"$UNIT_OUT" \
  -FE"$OUT" \
  -o"$OUT/$PROG" \
  "$CN_SRC/$PROG.lpr" > "$BUILDLOG" 2>&1
rc=$?
set -e

if [ "${VERBOSE:-0}" = "1" ]; then
  cat "$BUILDLOG"
else
  grep -E '(Warning|Error|Fatal):' "$BUILDLOG" | grep -vE '^mormot\..*Warning:' || true
fi

if [ $rc -ne 0 ]; then
  echo "FAIL(compile): $PROG (rc=$rc) - $BUILDLOG" >&2
  tail -25 "$BUILDLOG" >&2
  exit $rc
fi
echo "OK: $OUT/$PROG"
