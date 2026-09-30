#!/usr/bin/env bash
set -euo pipefail

# Baut + laeuft die mORMot-Testsuite des LLM-Clients (mormot.ai.llm.*) unter
# Linux/FPC, i. d. R. ueber WSL. Modelliert nach run-fpc-tests.sh (gleiche
# mORMot-Pfad-/Static-Verdrahtung); Runner ist mORMot-TSynTests (llm.tests.lpr).
#
#   run-fpc-llm-tests.sh            # bauen + ausfuehren
#   VERBOSE=1 run-fpc-llm-tests.sh  # vollen Compiler-/Testlog zeigen
#
# Logs: bin/fpc/llm.tests.buildlog (Compiler)
#       bin/fpc/llm.tests.log      (Testlauf)
# ExitCode <> 0 bei Compile- oder Testfehlern.

# Wurzel dieser Extension (scripts/..) - unabhaengig vom Einbindungsort.
LAI="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Compiler fest auf die Baseline (setzt $FPC/$FPC_VERSION, Override: FPC=<compiler>).
source "$LAI/scripts/fpc-env.sh"
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
UNIT_OUT="$OUT/lib/$FPC_VERSION"
TARGET="${TARGET:-linux}"
ARCH="${ARCH:-$("$FPC" -iTP)}"

# vec0 selbst finden. Die Extension liegt (gitignored) im Repo, aber der
# Vectorstore-Test ueberspringt sich stumm ohne SQLITE_EXT_DIR - und meldet
# dabei "1 assertion passed", sieht also aus wie ein Lauf. Dieselbe Klasse wie
# der fehlende Rebuild: ein Gate, das weniger prueft als es koennte, und das
# nicht sagt. Ein explizit gesetztes SQLITE_EXT_DIR gewinnt weiterhin.
if [ -z "${SQLITE_EXT_DIR:-}" ] && [ -d "$LAI/vendor/sqlite-ext/$ARCH-$TARGET" ]; then
  export SQLITE_EXT_DIR="$LAI/vendor/sqlite-ext/$ARCH-$TARGET"
  echo ">>> SQLITE_EXT_DIR=$SQLITE_EXT_DIR (vec0 gefunden)"
fi

UNITS="$SRC/app;$SRC/core;$SRC/crypt;$SRC/db;$SRC/lib;$SRC/net;$SRC/orm;$SRC/rest;$SRC/soa;$SRC/script;$SRC/misc;$SRC/tools/mget"
INCLUDES="$SRC;$SRC/core;$SRC/net"

PROG="llm.tests"
BUILDLOG="$OUT/$PROG.buildlog"
TESTLOG="$OUT/$PROG.log"

mkdir -p "$UNIT_OUT"

# Die eigenen Units IMMER neu uebersetzen. FPC erkennt eine geaenderte .pas
# neben einer bestehenden .ppu in $UNIT_OUT hier nicht zuverlaessig: eine
# Aenderung an src/mormot.ai.mcp.pas lief nachweislich gegen den alten Binary
# durch (der Test blieb gruen, obwohl der Fix ausgebaut war), erst eine
# Aenderung unter tests/ loeste den Rebuild aus. Ein Gate, das Quellaenderungen
# ignoriert, ist schlimmer als gar keins - es belegt Fixes, die nicht drin sind.
# Nur die eigenen Units wegwerfen: der mORMot2-Cache bleibt stehen, sonst
# kostet jeder Lauf Minuten statt Sekunden.
rm -f "$UNIT_OUT"/mormot.ai.*.ppu "$UNIT_OUT"/mormot.ai.*.o
rm -f "$UNIT_OUT"/test.*.ppu "$UNIT_OUT"/test.*.o

echo ">>> $PROG ($ARCH-$TARGET, FPC $FPC_VERSION)"
set +e
"$FPC" -MDelphi -Sci -Ci -O2 -g -gl -gw2 \
  -T"$TARGET" -P"$ARCH" \
  -Fi"$INCLUDES" \
  -Fu"$LIB_SRC;$TEST_SRC;$UNITS" \
  -Fl"$STATIC/$ARCH-$TARGET" \
  -FU"$UNIT_OUT" \
  -FE"$OUT" \
  -o"$OUT/$PROG" \
  "$TEST_SRC/$PROG.lpr" > "$BUILDLOG" 2>&1
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
