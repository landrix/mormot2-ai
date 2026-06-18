/// MCP example tool parameter records
// - this unit is part of the mormot-mcp-server project
// - licensed under MPL/GPL/LGPL three license
// - adopted into the mormot.ai.* namespace for landrix (LandrixAI) from
//   flydev-fr/mormot2-extensions
unit mormot.ai.mcp.tools;

interface

{$I mormot.defines.inc}

uses
  mormot.core.base;

type
  TRunExecutableParams = record
    Path: RawUtf8;
    Args: RawUtf8;
    Wait: boolean;
  end;


implementation


end.
