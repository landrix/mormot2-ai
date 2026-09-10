<#
  Baut + laeuft die Testsuiten dieser Extension mit Delphi (dcc64, Win64).
  Pendant zu scripts/run-fpc-tests.sh und scripts/run-fpc-llm-tests.sh.

  FPC bleibt der Primaer-Compiler; Delphi wird unterstuetzt, soweit der Code es
  erlaubt. Baseline: Delphi 13 (Compiler-Version 37.0).

  Beispiele:
    .\scripts\run-delphi-tests.ps1                  # MCP- + LLM-Suite
    .\scripts\run-delphi-tests.ps1 -Suite mcp       # eine Suite
    .\scripts\run-delphi-tests.ps1 -ShowAll         # voller Compiler- + Testlog
    .\scripts\run-delphi-tests.ps1 -Dcc 'C:\...\dcc64.exe'   # expliziter Compiler, ohne Versionspruefung

  mORMot2: $env:MORMOT2_ROOT, sonst das Submodul-Layout des Konsumenten
  (<konsument>\shared\delphi\libs\_git_Synopse2, drei Ebenen ueber diesem Repo).
  Ausgabe: bin\delphi\ (gitignored). Die .lpr-Programme werden dort als .dpr
  abgelegt - dcc akzeptiert nur eine .dpr als Projektdatei.
  Auf ARM64-Windows laufen die x64-Testprogramme unter Emulation (Prism).
#>
[CmdletBinding()]
param(
  [ValidateSet('all', 'mcp', 'llm')]
  [string]$Suite = 'all',
  # Pfad zu dcc64.exe; gesetzt = ohne Versionspruefung (bewusste Gegenprobe)
  [string]$Dcc = $env:DCC64,
  [switch]$ShowAll
)

$ErrorActionPreference = 'Stop'
$baseline = '37.0'   # Delphi 13

$lai = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$mormot2 = if ($env:MORMOT2_ROOT) { $env:MORMOT2_ROOT } else { Join-Path $lai '..\..\..\shared\delphi\libs\_git_Synopse2' }
if (-not (Test-Path (Join-Path $mormot2 'src\mormot.defines.inc'))) {
  throw "mORMot2-Quellbaum nicht gefunden unter '$mormot2' - MORMOT2_ROOT setzen."
}
$mormot2 = (Resolve-Path $mormot2).Path

# --- Compiler: explizit (ohne Pruefung) oder das installierte Delphi 13 ---
if ([string]::IsNullOrWhiteSpace($Dcc)) {
  $bds = if ($env:BDS) { $env:BDS } else { Join-Path ${env:ProgramFiles(x86)} "Embarcadero\Studio\$baseline" }
  $Dcc = Join-Path $bds 'bin\dcc64.exe'
  if (-not (Test-Path $Dcc)) {
    throw "dcc64.exe nicht gefunden unter '$Dcc' - Delphi 13 installieren oder -Dcc angeben."
  }
  $banner = (& $Dcc --version 2>&1 | Out-String).Trim()
  if ($banner -notmatch "version $([regex]::Escape($baseline))") {
    throw "'$Dcc' ist nicht Delphi 13 (Compiler $baseline): $banner. Bewusst uebersteuern mit -Dcc."
  }
} else {
  if (-not (Test-Path $Dcc)) { throw "dcc64.exe nicht gefunden: '$Dcc'" }
  $bds = Split-Path -Parent (Split-Path -Parent $Dcc)
}
$compilerVersion = if ((& $Dcc --version 2>&1 | Out-String) -match 'version (\d+\.\d+)') { $Matches[1] } else { '?' }

$src = Join-Path $mormot2 'src'
$units = @((Join-Path $bds 'lib\win64\release'), $src) +
  @('core', 'lib', 'crypt', 'net', 'db', 'orm', 'rest', 'soa', 'app', 'misc', 'script', 'tools\mget' |
    ForEach-Object { Join-Path $src $_ }) +
  @((Join-Path $lai 'src'), (Join-Path $lai 'tests'), (Join-Path $lai 'vendor'))
$inc = @($src, (Join-Path $src 'core'), (Join-Path $src 'net'))
$out = Join-Path $lai 'bin\delphi'
$dcu = Join-Path $out 'dcu'
New-Item -ItemType Directory -Force -Path $dcu | Out-Null

# vec0 selbst finden (wie run-fpc-llm-tests.sh): ohne SQLITE_EXT_DIR ueberspringt
# sich der Vectorstore-Test stumm und sieht trotzdem wie ein Lauf aus
$ext = Join-Path $lai 'vendor\sqlite-ext\x86_64-win64'
if (-not $env:SQLITE_EXT_DIR -and (Test-Path $ext)) {
  $env:SQLITE_EXT_DIR = $ext
  Write-Host ">>> SQLITE_EXT_DIR=$ext (vec0 gefunden)"
}

$suites = if ($Suite -eq 'all') { @('mcp', 'llm') } else { @($Suite) }
$failed = 0
foreach ($s in $suites) {
  $prog = "$s.tests"
  $dpr = Join-Path $out "$prog.dpr"
  $buildLog = Join-Path $out "$prog.buildlog"
  $testLog = Join-Path $out "$prog.log"
  Copy-Item (Join-Path $lai "tests\$prog.lpr") $dpr -Force

  Write-Host ">>> $prog (x86_64-win64, Delphi $compilerVersion)"
  # -B: immer alles neu uebersetzen - dieselbe Lehre wie in run-fpc-tests.sh: ein
  # Gate, das einen veralteten Kompilat-Cache testet, belegt Fixes, die nicht drin
  # sind. -GD: Map-Datei, damit mORMot Exception-Adressen auf Unit/Zeile aufloest.
  & $Dcc -B -Q -GD "-U$($units -join ';')" "-I$($inc -join ';')" "-R$src" `
    "-O$(Join-Path $mormot2 'static\delphi')" '-NSSystem;Winapi;System.Win;Data;Data.Win;Xml' `
    "-NU$dcu" "-E$out" $dpr *> $buildLog
  $rc = $LASTEXITCODE

  if ($ShowAll) {
    Get-Content $buildLog
  } else {
    # Meldungen sind lokalisiert (englisch/deutsch). Fehler immer zeigen, Warnungen
    # nur aus den eigenen Quellen - die aus mORMot2 selbst sind nicht unsere.
    Select-String -Path $buildLog -Pattern '(Error|Fatal|Fehler|Schwerwiegend):' | ForEach-Object Line
    Select-String -Path $buildLog -Pattern '(Warning|Warnung):' |
      Where-Object { $_.Line -notmatch '\\mormot\.(core|net|db|orm|rest|crypt|lib|soa|app|misc|script|tools)\.' } |
      ForEach-Object Line
  }
  if ($rc -ne 0) {
    Write-Host "FAIL(compile): $prog (rc=$rc) - Details: $buildLog" -ForegroundColor Red
    Get-Content $buildLog -Tail 15
    $failed++
    continue
  }

  # mORMot-TSynTests: ohne Parameter -> Run; ExitCode spiegelt fehlgeschlagene Assertions
  $runOut = & (Join-Path $out "$prog.exe") 2>&1
  $runRc = $LASTEXITCODE
  $runOut | Set-Content -Path $testLog -Encoding utf8
  if ($ShowAll) { $runOut }
  $summary = $runOut | Select-String -Pattern 'Total assertions failed for all test suits' | Select-Object -Last 1
  if (($runRc -eq 0) -and $summary) {
    Write-Host "PASS: $prog - $($summary.Line.Trim()) (Log: $testLog)"
  } else {
    Write-Host "FAIL: $prog (rc=$runRc) - Voller Log: $testLog" -ForegroundColor Red
    $runOut | Select-String -Pattern '(^!|FAILED|Exception)' | Select-Object -First 30 | ForEach-Object Line
    $failed++
  }
}
if ($failed -gt 0) { exit 1 }
exit 0
