# shellcheck shell=bash
# Waehlt den FPC-Compiler fuer die Build-/Test-Skripte dieser Extension. Wird per
# `source` eingebunden und setzt $FPC (Aufruf) + $FPC_VERSION (Banner/Logs).
#
# Warum nicht einfach `fpc` aus dem PATH: Auf Entwicklungsrechnern liegen oft mehrere
# FPC nebeneinander (Release, Release Candidate, Trunk). Welcher davon `fpc` heisst,
# entscheidet die Shell-Startup-Kette - ein Trunk-Default stellt den Build sonst still um.
#
# Baseline: FPC 3.2.2, bis FPC 3.2.4 final erscheint (dann FPC_BASELINE anheben).
# Aufloesung:
#   1. $FPC gesetzt -> genau dieser Compiler, OHNE Versionspruefung (bewusste Gegenprobe,
#                     z. B. `FPC=fpc324 scripts/run-fpc-tests.sh`)
#   2. fpc322       -> Wrapper im PATH, falls vorhanden (Release-FPC mit isolierter Config)
#   3. fpc          -> PATH-Default (z. B. Distributions-FPC im Container)
# In Fall 2/3 muss die Version der Baseline entsprechen, sonst bricht der Build ab.

FPC_BASELINE="3.2.2"

if [ -n "${FPC:-}" ]; then
  FPC_CHECK=0
elif command -v fpc322 >/dev/null 2>&1; then
  FPC="fpc322"
  FPC_CHECK=1
else
  FPC="fpc"
  FPC_CHECK=1
fi

if ! FPC_VERSION="$("$FPC" -iV 2>/dev/null)"; then
  echo "FAIL: FPC compiler '$FPC' not executable - set FPC=<compiler> or install FPC $FPC_BASELINE" >&2
  exit 127
fi

if [ "$FPC_CHECK" = "1" ] && [ "$FPC_VERSION" != "$FPC_BASELINE" ]; then
  echo "FAIL: '$FPC' is FPC $FPC_VERSION, baseline is $FPC_BASELINE." \
       "Override deliberately with FPC=<compiler>." >&2
  exit 1
fi
