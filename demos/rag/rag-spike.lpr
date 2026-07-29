// LandrixAI RAG feasibility spike - proves the local SQLite stack works under
// FPC/Linux: load sqlite-vec (vec0) + sqlite-lembed (lembed0) into mORMot's
// static SQLite, register a GGUF model, embed texts via lembed(), store them in a
// vec0 table and run a KNN semantic search - all in-process, no API.
//
//   env: SQLITE_EXT_DIR (dir with vec0.so/lembed0.so + their llama/ggml deps)
//        LEMBED_MODEL    (path to a .gguf embedding model)
//        LEMBED_DIM      (embedding dimensions; default 384 for all-MiniLM)
//   run with LD_LIBRARY_PATH=$SQLITE_EXT_DIR so lembed0.so finds libllama/libggml
program rag.spike;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.db.raw.sqlite3,
  mormot.db.raw.sqlite3.static;

var
  db: TSqlDataBase;
  extdir, model, modelName: RawUtf8;
  dim: integer;

procedure EnableLoadExtension;
var
  enabled: integer;
begin
  enabled := 0;
  if not Assigned(sqlite3.db_config) then
    raise ESqlite3Exception.CreateUtf8('sqlite3.db_config unavailable', []);
  if sqlite3.db_config(db.DB, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, @enabled) <> SQLITE_OK then
    raise ESqlite3Exception.CreateUtf8('enable load_extension failed: %',
      [Utf8ToString(sqlite3.errmsg(db.DB))]);
end;

procedure LoadExt(const aFile: RawUtf8);
var
  msg: PUtf8Char;
  fn: RawUtf8;
begin
  fn := extdir + '/' + aFile;
  msg := nil;
  if sqlite3.load_extension(db.DB, pointer(fn), nil, msg) <> SQLITE_OK then
    raise ESqlite3Exception.CreateUtf8('load_extension % failed: %',
      [fn, Utf8ToString(msg)]);
end;

procedure RegisterModel;
var
  r: TSqlRequest;
begin
  r.Prepare(db.DB,
    'INSERT INTO temp.lembed_models(name, model) VALUES (?, lembed_model_from_file(?));');
  try
    r.Bind(1, modelName);
    r.Bind(2, model);
    r.Step;
  finally
    r.Close;
  end;
end;

procedure AddDoc(aId: Int64; const aText: RawUtf8);
var
  r: TSqlRequest;
begin
  r.Prepare(db.DB, 'INSERT INTO documents(id, content) VALUES (?, ?);');
  try
    r.Bind(1, aId);
    r.Bind(2, aText);
    r.Step;
  finally
    r.Close;
  end;
  r.Prepare(db.DB,
    'INSERT INTO vec_documents(rowid, embedding) SELECT ?, lembed(?, ?);');
  try
    r.Bind(1, aId);
    r.Bind(2, modelName);
    r.Bind(3, aText);
    r.Step;
  finally
    r.Close;
  end;
end;

procedure Search(const aQuery: RawUtf8; aLimit: integer);
var
  r: TSqlRequest;
  content: RawUtf8;
  dist: double;
begin
  ConsoleWrite(FormatUtf8('query: "%"', [aQuery]), ccLightCyan);
  r.Prepare(db.DB,
    'WITH matches AS (' +
    '  SELECT rowid, distance FROM vec_documents' +
    '  WHERE embedding MATCH lembed(?, ?) AND k = ? ORDER BY distance) ' +
    'SELECT d.content, m.distance FROM matches m ' +
    'JOIN documents d ON d.id = m.rowid ORDER BY m.distance;');
  try
    r.Bind(1, modelName);
    r.Bind(2, aQuery);
    r.Bind(3, aLimit);
    while r.Step = SQLITE_ROW do
    begin
      r.FieldUtf8(0, content);
      dist := r.FieldDouble(1);
      ConsoleWrite(FormatUtf8('  dist=% | %', [dist, content]), ccLightGreen);
    end;
  finally
    r.Close;
  end;
end;

begin
  extdir := StringToUtf8(GetEnvironmentVariable('SQLITE_EXT_DIR'));
  model := StringToUtf8(GetEnvironmentVariable('LEMBED_MODEL'));
  dim := StrToIntDef(GetEnvironmentVariable('LEMBED_DIM'), 384);
  modelName := 'embedder';
  if (extdir = '') or (model = '') then
  begin
    ConsoleWrite('set SQLITE_EXT_DIR and LEMBED_MODEL', ccLightRed);
    exit;
  end;

  db := TSqlDataBase.Create(':memory:', '');
  try
    EnableLoadExtension;
    LoadExt('lembed0.so');
    LoadExt('vec0.so');
    ConsoleWrite(FormatUtf8('loaded: vec=%  lembed=%',
      [db.ExecuteJson('SELECT vec_version()'),
       db.ExecuteJson('SELECT lembed_version()')]), ccLightGray);

    RegisterModel;
    db.Execute('CREATE TABLE documents(id INTEGER PRIMARY KEY, content TEXT);');
    db.Execute(FormatUtf8(
      'CREATE VIRTUAL TABLE vec_documents USING vec0(embedding float[%]);', [dim]));

    AddDoc(1, 'Die Rechnung fuer die Dachsanierung betraegt 4500 Euro.');
    AddDoc(2, 'Kunde Mueller hat einen Termin zur Heizungswartung vereinbart.');
    AddDoc(3, 'Im Lager sind noch 20 Saecke Zement vorhanden.');

    Search('Was kostet die Reparatur am Dach?', 2);
    Search('Wann kommt der Techniker fuer die Heizung?', 2);
  except
    on E: Exception do
      ConsoleWrite(FormatUtf8('ERROR %: %', [E.ClassName, E.Message]), ccLightRed);
  end;
  db.Free;
end.
