/// native inference engine - memory-mapped weight files
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - weights are mapped read-only, never copied: the OS pages them in on demand
//   and shares them between processes
// - a file is mapped as a whole; one that cannot fit the address space of a
//   32-bit process is refused with a clear message instead of failing somewhere
//   inside the OS call. The Q8_0 BGE-M3 (634 MB) fits and runs on Win32; windowed
//   mapping is only needed for larger models (the decoder phase)
unit mormot.ai.infer.memory;

{
  *****************************************************************************

    - TInferMappedFile: a whole file mapped read-only, or a copy of a buffer

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

{$ifndef FPC}{$ifdef CPUX64}
  {$EXCESSPRECISION OFF} // single expressions in single, as FPC computes them
{$endif}{$endif}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.ai.infer.tensor;

const
  /// the largest file a 32-bit process maps as a whole (1.5 GB of the 2-4 GB
  // address space, so that the heap and the scratch buffers keep room)
  INFER_MAX_MAP32 = Int64(1536) shl 20;

type
  /// a read-only file content, memory-mapped or held in a string
  TInferMappedFile = class
  protected
    fMap: TMemoryMap;
    fMapped: boolean;
    fContent: RawByteString;
    fBuffer: PAnsiChar;
    fSize: Int64;
    fFileName: TFileName;
  public
    /// map aFileName read-only - raises EInferError if missing or too large
    constructor Create(const aFileName: TFileName);
    /// hold a copy of aContent - for tests and small in-memory files
    constructor CreateFromContent(const aContent: RawByteString);
    /// unmap the file
    destructor Destroy; override;
    /// first byte of the content
    property Buffer: PAnsiChar
      read fBuffer;
    /// size of the content in bytes
    property Size: Int64
      read fSize;
    /// the mapped file, '' for an in-memory content
    property FileName: TFileName
      read fFileName;
  end;


implementation

{ TInferMappedFile }

constructor TInferMappedFile.Create(const aFileName: TFileName);
var
  size: Int64;
begin
  inherited Create;
  fFileName := aFileName;
  size := FileSize(aFileName);
  if size <= 0 then
    EInferError.RaiseUtf8('%.Create: missing or empty file %', [self, aFileName]);
  {$ifdef CPU32}
  if size > INFER_MAX_MAP32 then
    EInferError.RaiseUtf8('%.Create: % has % MB - too large to map whole in a ' +
      '32-bit process (use a quantised file)', [self, aFileName, size shr 20]);
  {$endif CPU32}
  if not fMap.Map(aFileName, {forcemap=}true) then
    EInferError.RaiseUtf8('%.Create: cannot map %', [self, aFileName]);
  fMapped := true;
  fBuffer := fMap.Buffer;
  fSize := fMap.Size;
  if fSize <> size then
    EInferError.RaiseUtf8('%.Create: mapped % of % bytes of %',
      [self, fSize, size, aFileName]);
end;

constructor TInferMappedFile.CreateFromContent(const aContent: RawByteString);
begin
  inherited Create;
  fContent := aContent;
  fBuffer := pointer(fContent);
  fSize := length(fContent);
end;

destructor TInferMappedFile.Destroy;
begin
  if fMapped then
    fMap.UnMap;
  inherited Destroy;
end;

end.
