/// standalone mORMot extension: TExternalProcess + RunRedirect with stdin
// - interactive child process with bidirectional stdin/stdout pipes
// - cross-platform: Windows (CreateProcessW + pipes) and POSIX FPC (fork+pipe)
// - background reader thread prevents pipe-buffer deadlocks
// - overloaded RunRedirect(cmd, stdinput, ...) for one-shot stdin pipe
// - adopted for landrix (LandrixAI MCP demos) from flydev-fr/mormot2-extensions
//   (mormot-os module); licensed under MPL/GPL/LGPL (follows mORMot)
unit mormot.ext.os;

interface
        
{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os;

type
  /// options for TExternalProcess and the RunRedirect stdinput overload
  // - epoEnvAddExisting merges provided env pairs with the current environment
  // - epoJobCloseChildren wraps the child in a Windows Job Object so its
  // children are auto-killed when the parent dies
  // - epoNewConsole gives the child its own console window
  // - epoNewProcessGroup creates the child in its own Windows process group
  // (enables targeted CTRL_BREAK_EVENT via GenerateConsoleCtrlEvent)
  TExtProcessOption = (
    epoEnvAddExisting,
    epoJobCloseChildren,
    epoNewConsole,
    epoNewProcessGroup);
  TExtProcessOptions = set of TExtProcessOption;

  /// interactive child process with bidirectional stdin/stdout pipes
  // - unlike RunRedirect() which is a one-shot blocking call, TExternalProcess
  // allows ongoing communication with a long-lived process (REPL, FFmpeg, etc.)
  // - a background reader thread continuously drains the child's stdout to
  // prevent pipe-buffer deadlocks when both sides are writing
  // - writing to stdin is done explicitly via Write() or WriteAndCloseStdin()
  // - typical use: Start('node -i'), then Write('2+3\n'), ReadAvailable()
  // - for FFmpeg graceful shutdown: Write('q'#10) then WaitFor()
  TExternalProcess = class(TSynPersistent)
  protected
    fCommand: TRunArg;
    fWorkDir: TFileName;
    fOptions: TExtProcessOptions;
    fOnOutput: TOnRedirect;
    fPid: cardinal;
    {$ifdef OSWINDOWS}
    fProcess: THandle;
    fProcessThread: THandle;
    fJob: THandle;
    fReaderHandle: THandle;
    {$endif OSWINDOWS}
    fStdinWrite: THandle;
    fStdoutRead: THandle;
    {$ifdef FPC}
    fReaderTid: PtrUInt;
    {$else}
    fReaderTid: TThreadID;
    {$endif}
    fReaderFinished: boolean;
    fOutput: RawByteString;
    fOutputSafe: TLightLock;
    fExitCode: integer;
    fStarted: boolean;
    fTerminated: boolean;
    function GetRunning: boolean;
    function GetPid: cardinal;
  public
    /// release all handles and terminate the child process if still running
    destructor Destroy; override;
    /// start a new external process with bidirectional stdin/stdout pipes
    // - returns true on success, false if the process could not be started
    // - cmd is the full command line (executable + arguments)
    function Start(const cmd: TRunArg; const env: TRunArg = '';
      const wrkdir: TFileName = '';
      options: TExtProcessOptions = []): boolean;
    /// write raw data to the process stdin pipe
    // - returns true if all bytes were written, false on error
    function Write(const data: RawByteString): boolean; overload;
    /// write raw data to the process stdin pipe
    function Write(p: pointer; len: PtrInt): boolean; overload;
    /// write data then close the stdin pipe (signal EOF to the child)
    function WriteAndCloseStdin(const data: RawByteString): boolean;
    /// close the stdin pipe, signaling EOF to the child process
    // - idempotent: safe to call multiple times
    procedure CloseStdin;
    /// read and consume all currently buffered output from the child
    // - returns '' if no output is available
    function ReadAvailable: RawByteString;
    /// check whether any output data is buffered and ready to read
    function HasDataAvailable: boolean;
    /// attempt graceful termination, then hard-kill after waitms
    // - on Windows: sends WM_QUIT, then TerminateProcess
    // - on POSIX: sends SIGTERM, then SIGKILL
    // - returns true if the process exited within the timeout
    function Terminate(waitms: cardinal = 5000): boolean;
    /// unconditionally hard-kill the process
    procedure Kill;
    /// block until the process exits or timeout expires
    // - returns the exit code, or -1 on timeout
    function WaitFor(waitms: cardinal = INFINITE): integer;
    /// whether the process was started and is still running
    property Running: boolean read GetRunning;
    /// exit code of the process (valid only after the process has exited)
    // - also captured automatically when Running transitions to false
    property ExitCode: integer read fExitCode;
    /// the OS process ID
    property Pid: cardinal read GetPid;
    /// optional callback for real-time output notification
    // - called from the background reader thread with new output chunks
    // - the return value is ignored (use Terminate to stop the process)
    // - set before calling Start() if needed
    property OnOutput: TOnRedirect read fOnOutput write fOnOutput;
  end;

/// one-shot command execution with stdin data - overloads mormot.core.os RunRedirect
// - the second parameter (stdinput) distinguishes this from the vanilla
// RunRedirect(cmd, exitcode, ...) where the second parameter is PInteger
// - writes stdinput to the child process, closes stdin (EOF), then reads
// all stdout until the child exits
// - uses TExternalProcess internally, so a background reader thread prevents
// deadlocks even when both input and output are large
// - returns '' on error; set exitcode^ to retrieve the child's exit code
// - optional onoutput callback is invoked from the reader thread
function RunRedirect(const cmd: TRunArg;
  const stdinput: RawByteString;
  exitcode: PLongint = nil;
  const onoutput: TOnRedirect = nil;
  waitfordelayms: cardinal = INFINITE;
  setresult: boolean = true;
  const wrkdir: TFileName = '';
  options: TExtProcessOptions = []): RawByteString; overload;


implementation

uses
  {$ifdef OSWINDOWS}
  Windows
  {$endif OSWINDOWS}
  {$ifdef OSPOSIX}
  {$ifdef FPC}
  BaseUnix,
  Unix
  {$endif FPC}
  {$endif OSPOSIX}
  ;

// platform-specific implementations are in separate .inc files

{$ifdef OSWINDOWS}
  {$include mormot.ext.os.windows.inc}
{$endif OSWINDOWS}

{$ifdef OSPOSIX}
  {$include mormot.ext.os.posix.inc}
{$endif OSPOSIX}


{ RunRedirect overload with stdinput }

function RunRedirect(const cmd: TRunArg;
  const stdinput: RawByteString;
  exitcode: PLongint;
  const onoutput: TOnRedirect;
  waitfordelayms: cardinal;
  setresult: boolean;
  const wrkdir: TFileName;
  options: TExtProcessOptions): RawByteString;
var
  proc: TExternalProcess;
begin
  result := '';
  proc := TExternalProcess.Create;
  try
    proc.OnOutput := onoutput;
    if not proc.Start(cmd, '', wrkdir, options) then
    begin
      if exitcode <> nil then
        exitcode^ := -1;
      exit;
    end;
    if stdinput <> '' then
      proc.WriteAndCloseStdin(stdinput)
    else
      proc.CloseStdin;
    proc.WaitFor(waitfordelayms);
    if setresult then
      result := proc.ReadAvailable;
    if exitcode <> nil then
      exitcode^ := proc.ExitCode;
  finally
    proc.Free;
  end;
end;

end.
