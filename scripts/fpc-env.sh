# shellcheck shell=bash
# Waehlt den FPC-Compiler fuer die Build-/Test-Skripte dieser Extension. Wird per
# `source` eingebunden und setzt $FPC (Aufruf) + $FPC_VERSION (Banner/Logs und der
# versionsgetrennte Unit-Cache lib/$FPC_VERSION der Build-Skripte).
#
# Warum nicht einfach `fpc` aus dem PATH: Auf Entwicklungsrechnern liegen oft mehrere
# FPC nebeneinander (Release, Release Candidate, Trunk). Welcher davon `fpc` heisst,
# entscheidet die Shell-Startup-Kette - ein Trunk-Default stellt den Build sonst still um.
#
# Baseline: FPC 3.2.4 (seit 2026-09-30; der Release Candidate, bis 3.2.4 final erscheint).
# Aufloesung:
#   1. $FPC gesetzt -> genau dieser Compiler, ohne Baseline-Pruefung (bewusste Gegenprobe,
#                     z. B. `FPC=fpc322 scripts/run-fpc-tests.sh`); FPC_EXPECT=<version>
#                     verlangt dabei trotzdem eine bestimmte Version
#   2. fpc324       -> Wrapper im PATH, falls vorhanden (Eigenbau mit isolierter Config)
#   3. fpc          -> PATH-Default (z. B. Distributions-FPC im Container)
# In Fall 2/3 muss die Version der Baseline entsprechen, sonst bricht der Build ab.

FPC_BASELINE="3.2.4"

if [ -n "${FPC:-}" ]; then
  FPC_CHECK=0
elif command -v fpc324 >/dev/null 2>&1; then
  FPC="fpc324"
  FPC_CHECK=1
# Also look in ~/.local/bin without a login shell (wsl -e bash ...), where the wrapper lives.
elif [ -x "$HOME/.local/bin/fpc324" ]; then
  FPC="$HOME/.local/bin/fpc324"
  FPC_CHECK=1
else
  FPC="fpc"
  FPC_CHECK=1
fi

if ! FPC_VERSION="$("$FPC" -iV 2>/dev/null)"; then
  echo "FAIL: FPC compiler '$FPC' not executable - set FPC=<compiler> or install FPC $FPC_BASELINE" >&2
  exit 127
fi

if [ -n "${FPC_EXPECT:-}" ] && [ "$FPC_VERSION" != "$FPC_EXPECT" ]; then
  echo "FAIL: '$FPC' is FPC $FPC_VERSION, FPC_EXPECT requires $FPC_EXPECT." >&2
  exit 1
fi

if [ "$FPC_CHECK" = "1" ] && [ "$FPC_VERSION" != "$FPC_BASELINE" ]; then
  echo "FAIL: '$FPC' is FPC $FPC_VERSION, baseline is $FPC_BASELINE." \
       "Override deliberately with FPC=<compiler>." >&2
  exit 1
fi
