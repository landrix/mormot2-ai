#!/usr/bin/env bash
# Launcher für den code-nav MCP-Server (stdio). Leitet die Repo-Wurzel aus dem
# eigenen Skript-Pfad ab (portabel, kein hartkodierter Pfad) und startet das
# FPC-Binary. Als MCP-command nutzen: `wsl -e bash <pfad>/codenav-mcp.sh`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CODENAV_ROOT="$(cd "$HERE/../../../.." && pwd)"   # codenav -> landrixai -> delphi -> shared -> repo
exec "$HERE/../bin/fpc/codenav.mcp"
