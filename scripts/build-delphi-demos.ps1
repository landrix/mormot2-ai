<#
  Baut die Demos dieser Extension mit Delphi: Delphi 13 (dcc64, Win64) oder mit
  -Delphi7 Delphi 7 (dcc32, Win32). Pendant zu scripts/build-demo.sh (FPC);
  dieselbe Demo-Liste.

  FPC ist Referenz-Compiler. Dieses Skript prueft, dass die Demos auch mit dem
  oberen und dem unteren Ende der Delphi-Breite uebersetzen - die Tests decken
  run-delphi-tests.ps1 und run-delphi7-tests.ps1 ab.

  Beispiele:
    .\scripts\build-delphi-demos.ps1                       # alle Demos, Delphi 13
    .\scripts\build-delphi-demos.ps1 -Demo infer\llm-bench.lpr
    .\scripts\build-delphi-demos.ps1 -ShowAll              # voller Compilerlog
    .\scripts\build-delphi-demos.ps1 -Dcc 'C:\...\dcc64.exe'   # expliziter Compiler, ohne Versionspruefung
    .\scripts\build-delphi-demos.ps1 -Delphi7 -FastMM4 'C:\lib\FastMM4'

  Delphi 7: -Delphi7Root / $env:DELPHI7 (Installationsordner mit Bin\DCC32.EXE
  und Lib) und -FastMM4 / $env:FASTMM4 (Ordner mit FastMM4.pas), wie in
  run-delphi7-tests.ps1.
  mORMot2: $env:MORMOT2_ROOT, sonst das Submodul-Layout des Konsumenten
  (<konsument>\shared\delphi\libs\_git_Synopse2, drei Ebenen ueber diesem Repo).
  Ausgabe: bin\delphi\demos\ bzw. bin\delphi7\demos\ (gitignored). Die
  .lpr-Programme werden dort als .dpr abgelegt - dcc akzeptiert nur eine .dpr
  als Projektdatei.
#>
[CmdletBinding()]
param(
  # eine Demo, Pfad relativ zu demos\ (z. B. infer\llm-bench.lpr); leer = alle
  [string]$Demo = '',
  # Pfad zu dcc64.exe; gesetzt = ohne Versionspruefung (bewusste Gegenprobe)
  [string]$Dcc = $env:DCC64,
  # Delphi 7 (Win32) statt Delphi 13
  [switch]$Delphi7,
  [string]$Delphi7Root = $env:DELPHI7,
  [string]$FastMM4 = $env:FASTMM4,
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
$src = Join-Path $mormot2 'src'
$mormotUnits = @($src) +
  @('core', 'lib', 'crypt', 'net', 'db', 'orm', 'rest', 'soa', 'app', 'misc', 'script', 'tools\mget' |
    ForEach-Object { Join-Path $src $_ })
$laiUnits = @((Join-Path $lai 'src'), (Join-Path $lai 'src\infer'), (Join-Path $lai 'demos'),
  (Join-Path $lai 'vendor'))
$inc = @($src, (Join-Path $src 'core'), (Join-Path $src 'net'), (Join-Path $lai 'vendor'))

if ($Delphi7) {
  # --- Delphi 7: wie run-delphi7-tests.ps1 ---
  if ([string]::IsNullOrWhiteSpace($Delphi7Root)) {
    # Ablageort auf dem Entwicklungsrechner; anderswo -Delphi7Root oder DELPHI7 setzen
    $Delphi7Root = Join-Path $lai '..\..\..\..\Borland\Delphi7'
  }
  $Dcc = Join-Path $Delphi7Root 'Bin\DCC32.EXE'
  if (-not (Test-Path $Dcc)) {
    throw "DCC32.EXE nicht gefunden unter '$Dcc' - -Delphi7Root oder DELPHI7 setzen."
  }
  $Delphi7Root = (Resolve-Path $Delphi7Root).Path
  $banner = (& $Dcc 2>&1 | Select-Object -First 1 | Out-String).Trim()
  if ($banner -notmatch 'Version 15\.0') {
    throw "'$Dcc' ist nicht Delphi 7 (Compiler 15.0): $banner"
  }
  if ([string]::IsNullOrWhiteSpace($FastMM4) -or
      -not (Test-Path (Join-Path $FastMM4 'FastMM4.pas'))) {
    throw "FastMM4.pas nicht gefunden ('$FastMM4') - -FastMM4 oder FASTMM4 setzen."
  }
  $FastMM4 = (Resolve-Path $FastMM4).Path
  $label = 'i386-win32, Delphi 7'
  $units = @((Join-Path $Delphi7Root 'Lib'), (Join-Path $Delphi7Root 'Lib\Obj'), $FastMM4) +
    $mormotUnits + $laiUnits
  $inc += @($FastMM4)
  $out = Join-Path $lai 'bin\delphi7\demos'
} else {
  # --- Delphi 13: explizit (ohne Pruefung) oder das installierte ---
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
  $label = "x86_64-win64, Delphi $compilerVersion"
  $units = @((Join-Path $bds 'lib\win64\release')) + $mormotUnits + $laiUnits
  $out = Join-Path $lai 'bin\delphi\demos'
}
$dcu = Join-Path $out 'dcu'
New-Item -ItemType Directory -Force -Path $dcu | Out-Null

# WICHTIG: hier ALLE Demos auflisten (wie build-demo.sh), sonst meldet der
# Gesamtlauf faelschlich Erfolg fuer Demos, die nie gebaut wurden
$demos = if ($Demo) { @($Demo) } else {
  @('mcp.examples.lpr', 'stdio\demo.mcp.stdio.lpr', 'http\demo.mcp.http.lpr',
    'jsonrpc\demo.mcp.jsonrpc.lpr', 'streamable\demo.mcp.streamable.lpr',
    'llm\llm-chat.lpr', 'llm\llm-agent.lpr', 'llm\llm-structured.lpr',
    'llm\llm-embed.lpr', 'llm\llm-vision.lpr', 'llm\llm-anthropic.lpr',
    'rag\rag-spike.lpr', 'rag\rag-chat.lpr', 'rag\rag-agent.lpr',
    'rag\embed-reference.lpr', 'rag\vector-bench.lpr', 'infer\embed-bench.lpr',
    'infer\llm-bench.lpr', 'llm\llm-quickstart.lpr')
}

$failed = 0
foreach ($d in $demos) {
  $lpr = Join-Path (Join-Path $lai 'demos') $d
  if (-not (Test-Path $lpr)) { throw "Demo nicht gefunden: '$lpr'" }
  $name = [IO.Path]::GetFileNameWithoutExtension($lpr)
  $dpr = Join-Path $out "$name.dpr"
  $buildLog = Join-Path $out "$name.buildlog"
  Copy-Item $lpr $dpr -Force
  # Delphi 7 nimmt den Programmnamen ohne den letzten Punkt-Teil
  # (demo.mcp.http -> demo.mcp.exe, mcp.examples -> mcp.exe): beide Namen vorher
  # entfernen, nie ein altes Programm melden
  $short = [IO.Path]::GetFileNameWithoutExtension($name)
  Remove-Item (Join-Path $out "$name.exe"), (Join-Path $out "$short.exe") -ErrorAction SilentlyContinue

  Write-Host ">>> demo $name ($label)"
  # dazu das Verzeichnis der Demo: FPC findet Units neben dem Programm von selbst,
  # dcc nicht (die .dpr liegt im Ausgabeordner)
  $demoUnits = @($units) + @(Split-Path -Parent $lpr)
  # -B: immer alles neu uebersetzen (kein veralteter Kompilat-Cache), -GD: Map-Datei
  if ($Delphi7) {
    # -N: Unit-Ausgabe (Delphi 7 kennt -NU noch nicht); Pfade auch als -R
    & $Dcc -B -Q -GD "-U$($demoUnits -join ';')" "-I$($inc -join ';')" "-R$($demoUnits -join ';')" `
      "-O$(Join-Path $mormot2 'static\delphi')" "-N$dcu" "-E$out" $dpr *> $buildLog
  } else {
    & $Dcc -B -Q -GD "-U$($demoUnits -join ';')" "-I$($inc -join ';')" "-R$src" `
      "-O$(Join-Path $mormot2 'static\delphi')" '-NSSystem;Winapi;System.Win;Data;Data.Win;Xml' `
      "-NU$dcu" "-E$out" $dpr *> $buildLog
  }
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
    Write-Host "FAIL(compile): $name (rc=$rc) - Details: $buildLog" -ForegroundColor Red
    Get-Content $buildLog -Tail 15
    $failed++
    continue
  }
  # unter dem vollen Namen ablegen: sonst ueberschreiben sich unter Delphi 7 die
  # Transport-Demos (demo.mcp.*) gegenseitig
  $exe = Join-Path $out "$name.exe"
  if (-not (Test-Path $exe) -and (Test-Path (Join-Path $out "$short.exe"))) {
    Move-Item (Join-Path $out "$short.exe") $exe -Force
  }
  if (-not (Test-Path $exe)) {
    Write-Host "FAIL: $name - kein Programm erzeugt ($exe)" -ForegroundColor Red
    $failed++
    continue
  }
  Write-Host "OK: $exe"
}
if ($failed -gt 0) {
  Write-Host "SOME DEMOS FAILED ($failed)" -ForegroundColor Red
  exit 1
}
Write-Host 'ALL DEMOS OK'
exit 0
