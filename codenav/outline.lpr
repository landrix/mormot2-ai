program outline;

// Prototyp fuer den code-nav-MCP: kompakter Pascal-Unit-Outline.
// Liest eine .pas-Unit und gibt NUR den interface-Teil strukturiert aus:
//   - Klassen/Records/Interfaces mit ihren Membern (Scope erhalten)
//   - Record-Felder
//   - top-level Typen + Routinen
// Da nur der interface-Teil gescannt wird, entfallen die implementation-
// Duplikate automatisch. Reine RTL, keine externen Abhaengigkeiten.
//
//   outline <unit.pas>

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, StrUtils;

// fuehrendes Bezeichner-Wort einer Zeile (vor '=' / Leerzeichen / ':')
function LeadIdent(const S: string): string;
var
  i: Integer;
begin
  Result := '';
  i := 1;
  while (i <= Length(S)) and (S[i] in ['A'..'Z', 'a'..'z', '0'..'9', '_']) do
  begin
    Result := Result + S[i];
    Inc(i);
  end;
end;

// faltet eine (ggf. mehrzeilige) Deklaration ab Zeile idx zu einer kompakten
// Signatur zusammen. Ende = erstes ';' AUSSERHALB von Klammern (Klammer-Balance),
// damit Parameter-';' am Zeilenende nicht falsch abschneiden. max 8 Zeilen.
// idx wird auf die letzte konsumierte Zeile gesetzt.
function CompactSig(L: TStringList; var idx: Integer): string;
var
  s: string;
  n, k, depth: Integer;
  done: Boolean;
begin
  Result := '';
  depth := 0;
  done := False;
  n := 0;
  while (idx < L.Count) and (n < 8) do
  begin
    s := Trim(L[idx]);
    k := Pos('//', s);
    if k > 0 then
      s := TrimRight(Copy(s, 1, k - 1));
    if Result = '' then
      Result := s
    else
      Result := Result + ' ' + s;
    for k := 1 to Length(s) do
      case s[k] of
        '(': Inc(depth);
        ')': Dec(depth);
        ';': if depth <= 0 then begin done := True; Break; end;
      end;
    Inc(n);
    if done then
      Break;
    Inc(idx);
  end;
end;

function IsMemberDecl(const lower: string): Boolean;
begin
  Result :=
    StartsStr('function ', lower) or StartsStr('procedure ', lower) or
    StartsStr('constructor ', lower) or StartsStr('destructor ', lower) or
    StartsStr('property ', lower) or
    StartsStr('class function ', lower) or StartsStr('class procedure ', lower) or
    StartsStr('class property ', lower);
end;

// Record-/Object-Feld:  Ident[, Ident...] : Typ ;   (keine Routine/Sektion)
function LooksLikeField(const line, lower: string): Boolean;
begin
  Result := (LeadIdent(line) <> '') and (Pos(':', line) > 0) and
    (not IsMemberDecl(lower)) and (not StartsStr('case ', lower)) and
    (lower <> 'end;') and (lower <> 'end');
end;

var
  L: TStringList;
  i, eqPos: Integer;
  line, lower, rhs, section: string;
  inIntf, inBody, bodyIsRecord: Boolean;
  nMembers, nFields, nTypes, nRoutines: Integer;

begin
  if ParamCount < 1 then
  begin
    writeln(StdErr, 'usage: outline <unit.pas>');
    Halt(2);
  end;

  L := TStringList.Create;
  try
    L.LoadFromFile(ParamStr(1));
    inIntf := False;
    inBody := False;
    bodyIsRecord := False;
    section := '';
    nMembers := 0; nFields := 0; nTypes := 0; nRoutines := 0;

    writeln('# ', ExtractFileName(ParamStr(1)), '  (interface outline)');
    writeln;

    i := 0;
    while i < L.Count do
    begin
      line := Trim(L[i]);
      lower := LowerCase(line);

      if not inIntf then
      begin
        if lower = 'interface' then
          inIntf := True;
        Inc(i); Continue;
      end;
      if lower = 'implementation' then
        Break;
      if (line = '') or StartsStr('//', line) then
      begin
        Inc(i); Continue;
      end;

      // ---- innerhalb einer Klasse/Record/Interface ----
      if inBody then
      begin
        if (lower = 'end;') or (lower = 'end') then
        begin
          inBody := False;
          bodyIsRecord := False;
          Inc(i); Continue;
        end;
        if (lower = 'private') or (lower = 'public') or (lower = 'protected') or
           (lower = 'published') or StartsStr('strict ', lower) then
        begin
          Inc(i); Continue;
        end;
        if IsMemberDecl(lower) then
        begin
          writeln('    ', CompactSig(L, i));
          Inc(nMembers);
        end
        else if bodyIsRecord and LooksLikeField(line, lower) then
        begin
          writeln('    ', CompactSig(L, i));
          Inc(nFields);
        end;
        Inc(i); Continue;
      end;

      // ---- Sektions-Marker ----
      if (lower = 'type') or (lower = 'const') or (lower = 'var') then
      begin
        section := lower;
        Inc(i); Continue;
      end;

      // ---- Klassen-/Record-/Interface-Definition?  Name = [packed] class|record|... ----
      eqPos := Pos('=', line);
      if eqPos > 0 then
      begin
        rhs := LowerCase(Trim(Copy(line, eqPos + 1, MaxInt)));
        if StartsStr('packed ', rhs) then
          rhs := Trim(Copy(rhs, 8, MaxInt));
        if (StartsStr('class', rhs) or StartsStr('record', rhs) or
            StartsStr('object', rhs) or StartsStr('interface', rhs)) and
           (not StartsStr('class of', rhs)) and (Pos(';', line) = 0) then
        begin
          writeln;
          writeln(line);
          inBody := True;
          bodyIsRecord := StartsStr('record', rhs) or StartsStr('object', rhs);
          Inc(nTypes);
          Inc(i); Continue;
        end;
        // top-level Typ-Alias (nur in type-Sektion; Konstanten ausgeblendet)
        if (section = 'type') and (LeadIdent(line) <> '') then
        begin
          writeln('  ', CompactSig(L, i));
          Inc(nTypes);
          Inc(i); Continue;
        end;
      end;

      // ---- top-level Routine im interface ----
      if StartsStr('function ', lower) or StartsStr('procedure ', lower) then
      begin
        writeln(CompactSig(L, i));
        Inc(nRoutines);
        Inc(i); Continue;
      end;

      Inc(i);
    end;

    writeln;
    writeln(Format('# %d types, %d members, %d record fields, %d top-level routines',
      [nTypes, nMembers, nFields, nRoutines]));
  finally
    L.Free;
  end;
end.
