# code-nav MCP

Ein stdio-MCP-Server (auf `mormot.ai.mcp`), der Coding-Agenten Code-Navigation
bietet, damit sie das Repo durchsuchen, **ohne große Dateien ganz zu lesen** —
spart Token. Erster echter Anwendungsfall der `mormot.ai.*`-Extension.

## Tools

| Tool | Zweck | Sprachen | Backend |
|---|---|---|---|
| `get_outline(path)` | Code-Outline (Typen/Klassen/Member, kompakt) | **Pascal · TypeScript · Kotlin** | Pascal: eigener Scanner; sonst ctags |
| `find_definition(name)` | Symbol → `file:line [kind] signature` | **Pascal · TypeScript · Kotlin** | ctags |
| `search_text(pattern, glob)` | kompakte Volltextsuche (`file:line:text`, gedeckelt) | alle | grep |

`get_outline` dispatcht nach Dateiendung: Pascal (`.pas`/`.pp`/`.inc`/`.lpr`/`.dpr`)
nutzt einen eigenen Scanner (interface-Teil, dedup) — *weil* ctags' Pascal-Parser nur
`function`/`procedure` ohne Scope kennt; TS/Kotlin und andere nutzen ctags (Symbole
line-sortiert, Member via Scope eingerückt).

## Token-Nutzen

`get_outline` einer großen Unit spart **~8–60×** (z. B. `DbFederation.pas`
178 KB → 17 KB; Handler-Units bis ~59×). `find_definition`/`search_text` ersetzen
mehrfaches Grep+Read durch eine kompakte, präzise Antwort.

## Grenzen (bekannt)

Die Genauigkeit hängt am jeweiligen Parser — die Tools liefern einen *Outline/Index*,
keinen vollständigen AST. Wichtig: Ein leeres Ergebnis ist **keine Fehlermeldung**,
sondern eine normale Antwort — „nichts gefunden" ist also **kein Beweis für
Abwesenheit**. Damit ein Agent das nicht falsch versteht, hängen `find_definition`
und `search_text` im Leerfall einen Hinweis an (Blind Spots + indizierte Dirs +
Fallback auf grep/Read).

- **`get_outline` TS/Kotlin (ctags):** ctags ist kein vollständiger Parser. Bei
  manchen Konstrukten (z. B. Properties von TS-`interface`s, verschachtelte/anonyme
  Typen) werden **nicht immer alle Member** gelistet. Für die Outline-Übersicht
  i. d. R. ausreichend, aber nicht garantiert vollständig.
- **`get_outline` Pascal (eigener Scanner):** scannt nur den `interface`-Teil;
  verschachtelte Typen *innerhalb* einer Klasse können das erste `end;` vorzeitig als
  Body-Ende werten (im interface selten, in der Praxis bislang nicht aufgetreten).
- **`find_definition` Pascal:** ctags' Pascal-Parser kennt **nur `function`/
  `procedure`** — Pascal-**Klassen/Records/Interfaces** (`type X = class`) sind **nicht**
  als Definition auffindbar; nur Methoden/Funktionen. Für solche Typen `get_outline`
  oder `search_text` nutzen. (TS/Kotlin: vollständig, inkl. class/interface.)

Wenn echte AST-Treue oder cross-unit-Referenzen (`find_references`, geerbte
Definitionen) nötig werden, wäre **pasls** (CodeTools-LSP) der Weg — aktuell bewusst
nicht umgesetzt (Build-/Bridge-Overhead, für den Token-Spar-Zweck Overkill).

## Voraussetzung

`universal-ctags` muss installiert sein (`ctags --version` → „Universal Ctags").

## Bauen

```bash
# in WSL (nativ aarch64), aus dem Repo-Root:
bash shared/delphi/landrixai/scripts/build-codenav.sh
```
Binary: `shared/delphi/landrixai/bin/fpc/codenav.mcp` (gitignored).

## Einbindung (Claude Code)

Über das portable Launcher-Skript `codenav-mcp.sh` (leitet die Repo-Wurzel aus dem
eigenen Pfad ab — kein hartkodierter Pfad, keine env-Variable nötig). Registrieren:

```powershell
# in PowerShell ausführen (NICHT Git-Bash — das mangled /mnt-Pfade)
claude mcp add -s local landrix-codenav -- `
  wsl -e bash /mnt/d/Projekte/landrix-platform/shared/delphi/landrixai/codenav/codenav-mcp.sh
```
- Scope `local` = nur du, nur dieses Projekt (landet in `~/.claude.json`, nicht
  eingecheckt). Prüfen: `claude mcp get landrix-codenav` → `Status: ✓ Connected`.
- **Pro Maschine einmalig**: Registrierung (`~/.claude.json`) **und** Binary sind nicht
  in git — auf jedem Rechner separat `build-codenav.sh` (in dessen WSL, baut nativ für
  dessen Arch: aarch64 bzw. x86_64) **und** den `claude mcp add`-Befehl oben ausführen.
  Launcher + Befehl sind auf allen Rechnern identisch (laufen über den WSL-Pfad).
- Das Binary ist ein **aarch64-linux**-Build (WSL); Claude Code läuft auf Windows →
  `wsl`-Wrapper. Vorher bauen (`build-codenav.sh`). Aktiv nach Reload/neuer Session.
- `CODENAV_ROOT` setzt der Launcher selbst; die durchsuchten Quell-Dirs sind in
  `codenav.tools.pas` kuratiert (ohne vendored libs / node_modules / generated).

> **⚠️ Drive-Letter-Casing-Falle (Windows + VS-Code-Extension).** `~/.claude.json`
> schlüsselt local-scope-Server **case-sensitiv** pro Projektpfad. `claude mcp add` aus
> PowerShell registriert unter **groß** geschriebenem Laufwerk (`D:/…`, weil Node
> `process.cwd()` das Laufwerk großschreibt), die VS-Code-Extension/Agent-Session nutzt
> aber oft **klein** (`d:/…`). Folge: `claude mcp get` zeigt `✓ Connected` (CLI-cwd =
> groß), in der Session fehlen die Tools trotzdem (Session-cwd = klein). Nach dem
> `claude mcp add` prüfen, unter welchem Casing die Session läuft (in Claude Code:
> „Primary working directory: …") und den `mcpServers`-Eintrag ggf. zusätzlich unter den
> **kleingeschriebenen** Projekt-Key in `~/.claude.json` kopieren. Diagnose-Beleg: zwei
> Projekt-Keys, die sich nur im Drive-Letter unterscheiden. Danach VS Code komplett neu
> starten; eine bereits laufende (resumte) Session lädt den Tool-Index nicht nach.

## Implementierungs-Hinweis (wichtig)

Die externen Tools (ctags/grep) werden über **`mormot.ext.os`'s `RunRedirect`
(fork+pipe)** aufgerufen — mORMots eigenes popen-basiertes `RunRedirect`
(`mormot.core.os`) **hängt in dieser WSL-Umgebung** (auch im Main-Thread, auch bei
trivialen Befehlen). `mormot.ext.os` liegt unter `landrixai/vendor/` (adoptiert von
flydev, geteilt mit den Demos).

## Dateien

```
codenav/
  codenav.outline.pas   Pascal-interface-Outline (RTL)
  codenav.tools.pas     find_definition (ctags) + search_text (grep), via mormot.ext.os
  codenav.mcp.lpr       stdio-MCP-Server, registriert die 3 Tools
  outline.lpr           eigenständiges Outline-CLI (Prototyp/Debug)
```
