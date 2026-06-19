<#
  PowerShell-Wrapper für shared/delphi/landrixai/scripts/build-codenav.sh.

  Baut den code-nav MCP-Server (stdio) unter Linux/FPC über WSL (nativ für die
  Arch der WSL-Distro, i. d. R. aarch64 bzw. x86_64). Der Exit-Code des
  Bash-Skripts wird durchgereicht (0 = grün).

  Binary: shared/delphi/landrixai/bin/fpc/codenav.mcp (gitignored).

  Beispiele:
    .\build-codenav.ps1            # bauen
    .\build-codenav.ps1 -ShowAll   # vollen Compilerlog zeigen (VERBOSE=1)
#>
[CmdletBinding()]
param(
  # Zeigt den vollen Compilerlog (entspricht VERBOSE=1).
  [switch]$ShowAll
)

$ErrorActionPreference = 'Stop'

# Repo-Root (landrixai\scripts -> landrixai -> delphi -> shared -> repo) ermitteln
# und in WSL-Pfad wandeln.
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$wslRoot = (& wsl -e wslpath -u "$repoRoot" 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslRoot)) {
  throw 'WSL-Pfad konnte nicht ermittelt werden — ist WSL installiert und eine Default-Distro gesetzt?'
}
$wslRoot = $wslRoot.Trim()

$envPrefix = if ($ShowAll) { 'VERBOSE=1 ' } else { '' }

$inner = "cd '$wslRoot' && ${envPrefix}bash shared/delphi/landrixai/scripts/build-codenav.sh"
& wsl -e bash -lc $inner
exit $LASTEXITCODE
