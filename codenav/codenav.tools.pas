unit codenav.tools;

// code-nav backend tools that shell out to ctags/grep (no shell: RunRedirect
// parses argv directly, so tool arguments cannot inject shell commands).
//   FindDefinition(name)         -> symbol -> file:line [kind] signature (multi-language)
//   SearchText(pattern, glob)    -> compact grep over the curated source dirs
// Repo root from env CODENAV_ROOT, else the current directory.

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base;

function CodeNavRoot: TFileName;
function FindDefinition(const aName: RawUtf8): RawUtf8;
function SearchText(const aPattern, aGlob: RawUtf8): RawUtf8;
// dispatch by file extension: Pascal -> own scanner, else ctags-based outline
function GetOutline(const aFilePath: string): RawUtf8;

implementation

uses
  classes,
  mormot.core.os,
  mormot.ext.os,   // fork+pipe RunRedirect overload — mORMot's popen-based one
                   // hangs in this WSL environment (even from the main thread)
  mormot.core.text,
  mormot.core.unicode,
  mormot.core.variants,
  mormot.core.json,
  codenav.outline;   // ExtractPascalOutline for the Pascal path

const
  // kuratierte Quell-Verzeichnisse (relativ zur Root) - inkl. der mORMot2-Lib
  // Synopse2; node_modules/generated/Binaer-Artefakte bleiben aussen vor
  // (siehe EXCLUDE_DIRS fuer schwere Unterordner innerhalb dieser Dirs)
  CODE_DIRS: array[0..6] of string = (
    'backend/src',
    'shared/delphi/landrixai/src',
    'shared/delphi/client',
    'shared/delphi/dto',
    'frontend-react/src',
    'frontend-kmp/shared/src',
    'shared/delphi/libs/_git_Synopse2/src');
  // schwere Binär-/Build-Unterordner, die innerhalb der indizierten Dirs liegen
  // (landrixai/bin, landrixai/vendor/{models,sqlite-ext}, landrixai/_eval,
  // backend/bin) - per Verzeichnis-Basename ausgeschlossen, sonst liefe der
  // Index ueber ~1 GB GGUF-Modelle/Build-Artefakte
  EXCLUDE_DIRS: array[0..3] of string = (
    'bin', 'models', 'sqlite-ext', '_eval');
  MAX_SEARCH_LINES = 80;

function CodeNavRoot: TFileName;
begin
  result := GetEnvironmentVariable('CODENAV_ROOT');
  if result = '' then
    result := GetCurrentDir;
  result := ExcludeTrailingPathDelimiter(result);
end;

// kommaseparierte Liste der indizierten Quell-Dirs — für die Hinweis-Texte im
// Leerfall, damit der Agent die Reichweite kennt (sonst liest „nichts gefunden"
// wie „existiert nicht", obwohl evtl. nur außerhalb dieser Dirs gesucht wurde)
function IndexedDirs: RawUtf8;
var
  i: integer;
begin
  result := '';
  for i := 0 to high(CODE_DIRS) do
  begin
    if i > 0 then
      result := result + ', ';
    result := result + StringToUtf8(CODE_DIRS[i]);
  end;
end;

// existierende Quell-Dirs als gequotete, leerzeichengetrennte Argumentliste
function DirArgs(const aRoot: TFileName): RawUtf8;
var
  i: integer;
  d: TFileName;
begin
  result := '';
  for i := 0 to high(CODE_DIRS) do
  begin
    d := aRoot + PathDelim + StringReplace(CODE_DIRS[i], '/', PathDelim, [rfReplaceAll]);
    if DirectoryExists(d) then
      result := result + ' "' + StringToUtf8(d) + '"';
  end;
end;

// Ausschluss-Flags fuer die externen Tools aus EXCLUDE_DIRS bauen.
// aFlag ist das toolspezifische Flag inkl. '=' (ctags: '--exclude=',
// grep: '--exclude-dir='); Basename-Match je Tool.
function ExcludeArgs(const aFlag: RawUtf8): RawUtf8;
var
  i: integer;
begin
  result := '';
  for i := 0 to high(EXCLUDE_DIRS) do
    result := result + ' ' + aFlag + StringToUtf8(EXCLUDE_DIRS[i]);
end;

// '/^... $/' -> '...'  (ctags-Pattern in lesbare Signatur)
function CleanPattern(const p: RawUtf8): RawUtf8;
begin
  result := p;
  if (length(result) >= 2) and (result[1] = '/') and (result[2] = '^') then
    delete(result, 1, 2);
  if (length(result) >= 2) and (result[length(result)] = '/') and
     (result[length(result) - 1] = '$') then
    SetLength(result, length(result) - 2);
  result := TrimU(result);
end;

function FindDefinition(const aName: RawUtf8): RawUtf8;
var
  root: TFileName;
  cmd, ctagsOut, prefilter: RawUtf8;
  sl: TStringList;
  i, hits: integer;
  line: RawUtf8;
  doc: TDocVariantData;
begin
  if aName = '' then
    exit('{"error":"missing symbol name"}');
  root := CodeNavRoot;
  cmd := 'ctags -R --output-format=json --fields=+Kn ' +
    '--languages=Pascal,TypeScript,Kotlin ' +
    '--exclude=node_modules --exclude=.git --exclude=build --exclude=generated' +
    ExcludeArgs('--exclude=') +
    ' -f -' + DirArgs(root);
  ctagsOut := RunRedirect(cmd, '');  // '' stdinput -> mormot.ext.os fork+pipe overload
  if ctagsOut = '' then
    exit(FormatUtf8('{"error":"ctags returned nothing (is ctags installed? root=%)"}',
      [StringToUtf8(root)]));

  prefilter := '"name": "' + aName + '"';
  result := 'definitions of "' + aName + '":'#10;
  hits := 0;
  sl := TStringList.Create;
  try
    sl.Text := Utf8ToString(ctagsOut);
    for i := 0 to sl.Count - 1 do
    begin
      line := StringToUtf8(sl[i]);
      if (line = '') or (PosEx(prefilter, line) = 0) then
        continue;
      doc.InitJson(line, JSON_FAST);
      if doc.U['name'] = aName then
      begin
        result := result + FormatUtf8('  %:%  [%]  %'#10,
          [doc.U['path'], doc.I['line'], doc.U['kind'],
           CleanPattern(doc.U['pattern'])]);
        inc(hits);
      end;
      doc.Clear;
    end;
  finally
    sl.Free;
  end;
  if hits = 0 then
    result := result +
      '  (no definition found -- this does NOT prove the symbol is absent)'#10 +
      '  - Pascal class/record/interface TYPE names are NOT indexed (ctags'' Pascal'#10 +
      '    parser knows only function/procedure); for a Pascal type use get_outline'#10 +
      '    or search_text instead.'#10 +
      '  - the match is exact + case-sensitive; only these languages are indexed:'#10 +
      '    Pascal, TypeScript, Kotlin.'#10 +
      '  - only these dirs are indexed: ' + IndexedDirs + #10 +
      '    for anything outside them, fall back to grep/Read.'#10;
  result := result + FormatUtf8('# % match(es)'#10, [hits]);
end;

function SearchText(const aPattern, aGlob: RawUtf8): RawUtf8;
var
  root: TFileName;
  cmd, grepOut: RawUtf8;
  sl: TStringList;
  i, shown: integer;
begin
  if aPattern = '' then
    exit('{"error":"missing pattern"}');
  root := CodeNavRoot;
  cmd := 'grep -rIn --color=never --exclude-dir=.git' +
    ExcludeArgs('--exclude-dir=');
  if aGlob <> '' then
    cmd := cmd + ' --include="' + aGlob + '"';
  // -e schuetzt Pattern, das mit '-' beginnt; Pattern als eigenes argv-Element
  cmd := cmd + ' -e "' + aPattern + '"' + DirArgs(root);
  grepOut := RunRedirect(cmd, '');  // '' stdinput -> mormot.ext.os fork+pipe overload

  if aGlob <> '' then
    result := FormatUtf8('search "%" (glob %):'#10, [aPattern, aGlob])
  else
    result := FormatUtf8('search "%":'#10, [aPattern]);
  sl := TStringList.Create;
  try
    sl.Text := Utf8ToString(grepOut);
    shown := 0;
    for i := 0 to sl.Count - 1 do
    begin
      if sl[i] = '' then
        continue;
      if shown >= MAX_SEARCH_LINES then
      begin
        result := result + FormatUtf8('  ... (% more lines truncated)'#10,
          [sl.Count - i]);
        break;
      end;
      result := result + '  ' + StringToUtf8(sl[i]) + #10;
      inc(shown);
    end;
    if shown = 0 then
      result := result +
        '  (no matches -- this does NOT prove absence)'#10 +
        '  - grep uses basic regex (BRE, no -E); escape/adjust the pattern if it'#10 +
        '    contains + ? | ( ) { } etc.'#10 +
        '  - only these dirs are indexed: ' + IndexedDirs + #10 +
        '    for anything outside them, fall back to grep/Read.'#10;
  finally
    sl.Free;
  end;
end;

function IsContainerKind(const k: RawUtf8): boolean;
begin
  result := (k = 'class') or (k = 'interface') or (k = 'struct') or
    (k = 'enum') or (k = 'object') or (k = 'namespace') or
    (k = 'trait') or (k = 'record') or (k = 'module');
end;

// Outline via ctags (TypeScript/Kotlin/...): one ctags pass over the single
// file; symbols sorted by line, members (scope is a container) indented.
function CtagsOutline(const aFilePath: string): RawUtf8;
var
  cmd, ctagsOut, nm, kind, scopeKind, sig, entry: RawUtf8;
  sl, items: TStringList;
  i: integer;
  ln: Int64;
  doc: TDocVariantData;
begin
  cmd := 'ctags --output-format=json --fields=+nKsS -f - "' +
    StringToUtf8(aFilePath) + '"';
  ctagsOut := RunRedirect(cmd, '');
  if ctagsOut = '' then
    exit('{"error":"ctags returned nothing for ' +
      StringToUtf8(ExtractFileName(aFilePath)) + '"}');
  result := '# ' + StringToUtf8(ExtractFileName(aFilePath)) +
    '  (outline via ctags)'#10;
  sl := TStringList.Create;
  items := TStringList.Create;
  try
    sl.Text := Utf8ToString(ctagsOut);
    for i := 0 to sl.Count - 1 do
    begin
      if sl[i] = '' then
        continue;
      doc.InitJson(StringToUtf8(sl[i]), JSON_FAST);
      if doc.U['_type'] = 'tag' then
      begin
        nm := doc.U['name'];
        kind := doc.U['kind'];
        ln := doc.I['line'];
        scopeKind := doc.U['scopeKind'];
        sig := CleanPattern(doc.U['pattern']);
        if IsContainerKind(scopeKind) then
          entry := '    [' + kind + '] ' + nm + '  ' + sig
        else
          entry := '[' + kind + '] ' + nm + '  ' + sig;
        // zero-padded line number as a sort key (TAB-separated from the entry)
        items.Add(Format('%.8d', [Integer(ln)]) + #9 + Utf8ToString(entry));
      end;
      doc.Clear;
    end;
    items.Sort;
    for i := 0 to items.Count - 1 do
      result := result + StringToUtf8(Copy(items[i],
        Pos(#9, items[i]) + 1, MaxInt)) + #10;
    result := result + FormatUtf8('# % symbols'#10, [items.Count]);
  finally
    items.Free;
    sl.Free;
  end;
end;

function GetOutline(const aFilePath: string): RawUtf8;
var
  ext: string;
begin
  ext := LowerCase(ExtractFileExt(aFilePath));
  if (ext = '.pas') or (ext = '.pp') or (ext = '.inc') or
     (ext = '.lpr') or (ext = '.dpr') then
    result := StringToUtf8(ExtractPascalOutline(aFilePath))
  else
    result := CtagsOutline(aFilePath);
end;

end.
