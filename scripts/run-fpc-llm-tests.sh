#!/usr/bin/env bash
set -euo pipefail

# Baut + laeuft die mORMot-Testsuite des LLM-Clients (mormot.ai.llm.*) unter
# Linux/FPC, i. d. R. ueber WSL. Modelliert nach run-fpc-tests.sh (gleiche
# mORMot-Pfad-/Static-Verdrahtung); Runner ist mORMot-TSynTests (llm.tests.dpr).
#
#   run-fpc-llm-tests.sh            # bauen + ausfuehren
#   VERBOSE=1 run-fpc-llm-tests.sh  # vollen Compiler-/Testlog zeigen
#
# Logs: bin/fpc/llm.tests.buildlog (Compiler)
#       bin/fpc/llm.tests.log      (Testlauf)
# ExitCode <> 0 bei Compile- oder Testfehlern.

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
TEST_SRC="$LAI/tests"
OUT="$LAI/bin/fpc"
UNIT_OUT="$OUT/lib"
TARGET="${TARGET:-linux}"
ARCH="${ARCH:-$(fpc -iTP)}"

UNITS="$SRC/app;$SRC/core;$SRC/crypt;$SRC/db;$SRC/lib;$SRC/net;$SRC/orm;$SRC/rest;$SRC/soa;$SRC/script;$SRC/misc;$SRC/tools/mget"
INCLUDES="$SRC;$SRC/core;$SRC/net"

PROG="llm.tests"
BUILDLOG="$OUT/$PROG.buildlog"
TESTLOG="$OUT/$PROG.log"

mkdir -p "$UNIT_OUT"

echo ">>> $PROG ($ARCH-$TARGET)"
set +e
fpc -MDelphi -Sci -Ci -O2 -g -gl -gw2 \
  -T"$TARGET" -P"$ARCH" \
  -Fi"$INCLUDES" \
  -Fu"$LIB_SRC;$TEST_SRC;$UNITS" \
  -Fl"$STATIC/$ARCH-$TARGET" \
  -FU"$UNIT_OUT" \
  -FE"$OUT" \
  -o"$OUT/$PROG" \
  "$TEST_SRC/$PROG.dpr" > "$BUILDLOG" 2>&1
rc=$?
set -e

if [ "${VERBOSE:-0}" = "1" ]; then
  cat "$BUILDLOG"
else
  grep -E '(Warning|Error|Fatal):' "$BUILDLOG" | grep -vE '^mormot\..*Warning:' || true
fi

if [ $rc -ne 0 ]; then
  echo "FAIL(compile): $PROG (rc=$rc) - Details: $BUILDLOG" >&2
  tail -25 "$BUILDLOG" >&2
  exit $rc
fi

set +e
LD_LIBRARY_PATH="$OUT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$OUT/$PROG" > "$TESTLOG" 2>&1
run_rc=$?
set -e

if [ "${VERBOSE:-0}" = "1" ]; then
  cat "$TESTLOG"
fi

if [ "$run_rc" -eq 0 ] && grep -qiE 'Failed +0|0 +failed|Assertion' "$TESTLOG"; then
  echo "PASS: $PROG (rc=0). (Log: $TESTLOG)"
  grep -iE 'Tests passed|Assertion|Failed' "$TESTLOG" | tail -3 || true
  exit 0
fi

echo "FAIL: $PROG (run_rc=$run_rc). (Voller Log: $TESTLOG)" >&2
tail -40 "$TESTLOG" >&2 || true
exit 1
