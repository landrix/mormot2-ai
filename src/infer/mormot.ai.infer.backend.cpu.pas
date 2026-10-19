/// native inference engine - the scalar CPU reference backend
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - plain Pascal, no SIMD, no threads: it runs on every compiler and target down
//   to Delphi 7 / Win32 and is the yardstick for every optimised path
// - products and sums are computed in double (each factor widened before the
//   multiplication), results are stored as single: x87 keeps intermediates in
//   extended precision, so the last bit of a double may differ from SSE and NEON,
//   far below any stored single; the reference trades speed for the smallest
//   rounding error
// - Q8_0 weights are multiplied as ggml does (the arithmetic of the lembed oracle,
//   measured to matter: dequantized weights with F32 activations miss the cosine
//   gate on one corpus text): each activation row is quantized to Q8_0 blocks
//   (d = max|x| / 127 stored as FP16, q = x * (1 / d) rounded to nearest even
//   like the NEON and AVX paths of ggml-cpu - its plain C reference rounds ties
//   away from zero, the CPU kernels lembed runs do not; the AVX path also forms
//   the inverse as 127 / amax instead of 1 / d, which can move a near-tie by
//   one step: byte-equal to the aarch64 oracle, not to every ggml build), the
//   block products are exact integer sums, scaled by both FP16 scales and summed
//   in double
// - erf is our own (no Math unit): polynomial fits made for this unit by
//   tools/infer-ref/fit_erf.py - erf(x) = x P(x^2) up to 1, erfc(x) = exp(-x^2)
//   Q_k(x) on half units up to 6 (relative error below 1e-14), the continued
//   fraction of erfc beyond; negative GELU inputs use erfc, so the tail never
//   cancels. Some 40 ns a value: the GELU of a forward pass calls it millions
//   of times
unit mormot.ai.infer.backend.cpu;

{
  *****************************************************************************

    - InferErf / InferErfc: double-precision error function
    - InferQuantizeRowQ8_0: activations to Q8_0 blocks, as ggml does
    - TInferCpuBackend: scalar reference implementation of TInferBackend
    - TInferCpuFastBackend: the same results, on all cores, with SIMD kernels

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.ai.infer.tensor,
  mormot.ai.infer.kernels,
  mormot.ai.infer.threads,
  mormot.ai.infer.backend;

/// the error function: relative error below 1e-14 over the whole range
function InferErf(x: double): double;

/// the complementary error function 1 - erf(x)
// - from 1 on directly (relative error below 1e-14 - near 6 the rounding of
// x * x inside exp() alone is ~4e-15 - no cancellation); below 1
// as 1 - erf(x) (absolute error ~2e-16)
function InferErfc(x: double): double;

/// quantize aCount (a multiple of 32) singles into aCount div 32 Q8_0 blocks
// - ggml-cpu's quantize_row_q8_0: d = max|x| / 127 (kept as FP16), q = the
// single product x * (1 / d) rounded to nearest, ties to even
procedure InferQuantizeRowQ8_0(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array);

/// the plain Pascal quantizer behind InferQuantizeRowQ8_0 - the yardstick of
// the SIMD kernels, same contract
procedure InferQuantizeRowQ8_0Pascal(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array);

/// quantize aCount (a multiple of 32) weights into aCount div 32 Q8_0 blocks
// - the bytes of ggml's quantize_row_q8_0_ref, what a GGUF writer stores: as
// InferQuantizeRowQ8_0, but ties round away from zero (roundf), not to even -
// weights widened from BF16 have few mantissa bits and do hit ties
procedure InferQuantizeRowQ8_0Ref(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array);

/// quantize aCount (a multiple of 32) singles into aCount div 32 Q4_0 blocks
// - the bytes of ggml's quantize_row_q4_0_ref (what a GGUF writer stores): d =
// m / -8 with m the first value of the largest magnitude (kept as FP16), q =
// min(15, trunc(x * (1 / d) + 8.5)) with x * (1 / d) + 8.5 rounded to single
// once, as a fused multiply-add: the compilers contract it so on aarch64 and
// with -march=native on x86 (a build rounding twice differs in rare bytes)
// - NaN and infinity are refused (EInferError); a block whose 1 / d would
// overflow a single stores q = 8 (value 0) - ggml leaves that case undefined
procedure InferQuantizeRowQ4_0(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ4_0Array);

/// quantize aCount (a multiple of 256) activations into Q8_K blocks, as
// ggml-cpu's quantize_row_q8_K before a K-quant dot product: d = m / -127 with
// m the first value of the largest magnitude (d a single), q = the single
// product x * (-127 / m) rounded to nearest even, at most 127; bsums per 16
// - NaN and infinity are refused (EInferError)
procedure InferQuantizeRowQ8_K(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_KArray);

/// the dot product of aBlocks Q4_K super-blocks with Q8_K activations: per
// sub-block the exact integer sums, scaled in double (the reference)
function InferDotQ4K(w: PInferBlockQ4_K; x: PInferBlockQ8_K; aBlocks: PtrInt): double;

/// the dot product of aBlocks Q6_K super-blocks with Q8_K activations
function InferDotQ6K(w: PInferBlockQ6_K; x: PInferBlockQ8_K; aBlocks: PtrInt): double;

/// aBlocks Q4_0 blocks as Q8_0 blocks: the same scale, q - 8 as the signed
// value - exact, so every Q8_0 kernel computes the Q4_0 x Q8_0 dot product
procedure InferQ4ToQ8(aSrc: PInferBlockQ4_0; aBlocks: PtrInt; aDst: PInferBlockQ8_0);

type
  /// scalar reference backend: every operation; Linear takes F32, Q8_0, Q4_0,
  // Q4_K or Q6_K
  TInferCpuBackend = class(TInferBackend)
  protected
    /// query rows aFirst..aLast of one head: scores, masked softmax, weighted
    // values; aScratch is the [aSeq, aSeq] score matrix of this head
    procedure AttentionHead(q, k, v: PSingleArray; aSeq, aHeadDim, aWidth,
      aOffset, aFirst, aLast: PtrInt; aKeep: PIntegerArray; aScratch, y: PSingleArray);
  public
    function Supports(aOp: TInferOp; aWeights: TInferDType): boolean; override;
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); override;
    procedure LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
      gamma, beta: PSingleArray; aEps: double; y: PSingleArray); override;
    procedure Gelu(x: PSingleArray; aCount: PtrInt); override;
    procedure Add(x, y: PSingleArray; aCount: PtrInt); override;
    procedure SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
      aKeep: PIntegerArray); override;
    procedure Attention(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray); override;
    procedure L2Normalize(x: PSingleArray; aCount: PtrInt); override;
    procedure CausalAttention(q: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      keys, values: PSingleArray; aPast, aKvHeads: PtrInt; y: PSingleArray;
      aWindow: PtrInt = 0); override;
    procedure RmsNorm(x: PSingleArray; aRows, aCols: PtrInt; w: PSingleArray;
      aEps: double; y: PSingleArray); override;
    procedure SwiGlu(gate, up: PSingleArray; aCount: PtrInt); override;
    procedure Rope(x: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      aCos, aSin: PSingleArray); override;
  end;

  /// one CausalAttention: its operands and the arithmetic to use
  TInferCausalJob = record
    Q, Keys, Values, Y: PSingleArray;
    Rows, Heads, HeadDim, Past, KvHeads: PtrInt;
    /// the keys a row sees at most, itself included (0: all before it)
    Window: PtrInt;
    RowBlocks: PtrInt;
    /// true: float lanes (the fast path), false: the double reference
    Fast: boolean;
  end;

  /// one Linear of TInferCpuFastBackend: operands and quantized activations
  TInferLinearJob = record
    X, Y, Bias: PSingleArray;
    W: TInferWeight;
    Rows, InCount, OutCount, Blocks: PtrInt;
    /// the activation rows as Q8_0 blocks and their scales (double and single)
    Xq: PInferBlockQ8_0Array;
    Dx: PDoubleArray;
    Dxs: PSingleArray;
    /// the activation rows as Q8_K blocks, for Q4_K / Q6_K weights
    Xk: PInferBlockQ8_KArray;
    /// true: the FP32 arithmetic of the SIMD kernel (Q8_0 or F32 weights),
    // false: the exact one
    Fp32: boolean;
  end;

  /// the reference backend on all cores, with the SIMD kernels of this CPU
  // - Exact = true: bit-identical to TInferCpuBackend - work is split by output
  // feature (Linear), head (Attention) or element range (GELU), never inside one
  // sum, and the Q8_0 kernels return exact integer block sums
  // - Exact = false (default): float arithmetic where this CPU has a kernel (see
  // the Exact property), checked against the reference within a tolerance
  // - one parallel operation at a time: a second thread calling meanwhile
  // computes on its own thread instead of waiting for the pool
  TInferCpuFastBackend = class(TInferCpuBackend)
  protected
    fPool: TInferThreadPool;
    fThreads: integer;
    fBusy: TLightLock;
    fExact: boolean;
    // the job of the running parallel operation (fBusy held)
    fJob: TInferLinearJob;
    fX, fY, fQ, fK, fV: PSingleArray;
    fRows, fSeq, fHeadDim, fWidth: PtrInt;
    fKeep: PIntegerArray;
    fXq: array of TInferBlockQ8_0;
    fXk: array of TInferBlockQ8_K;
    fDx: TDoubleDynArray;
    fDxs: TSingleDynArray;
    fScratch: array of TSingleDynArray;
    fRowBlocks: PtrInt;
    fGamma, fBeta: PSingleArray;
    fEps: double;
    fCols: PtrInt;
    fCausal: TInferCausalJob;
    procedure CausalItem(aItem: integer);
    procedure LinearTile(aItem: integer);
    procedure QuantizeRow(aItem: integer);
    procedure AttentionItem(aItem: integer);
    procedure AttentionRows(h, aFirst, aLast: PtrInt; aScratch: PSingleArray);
    procedure AttentionSerial(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray);
    procedure GeluRange(aItem: integer);
    procedure LayerNormRows(aItem: integer);
    procedure RmsNormRows(aItem: integer);
    procedure SwiGluRange(aItem: integer);
    procedure RopeRows(aItem: integer);
    procedure RunParallel(const aJob: TOnInferItem; aCount: integer);
  public
    /// aThreads: the threads to compute on, the calling one included; 0 means
    // two thirds of the CPU cores (rounded), 1 no pool at all
    constructor Create(aThreads: integer = 0); reintroduce;
    /// stop the thread pool
    destructor Destroy; override;
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); override;
    procedure Gelu(x: PSingleArray; aCount: PtrInt); override;
    procedure LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
      gamma, beta: PSingleArray; aEps: double; y: PSingleArray); override;
    procedure Attention(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray); override;
    procedure CausalAttention(q: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      keys, values: PSingleArray; aPast, aKvHeads: PtrInt; y: PSingleArray;
      aWindow: PtrInt = 0); override;
    procedure RmsNorm(x: PSingleArray; aRows, aCols: PtrInt; w: PSingleArray;
      aEps: double; y: PSingleArray); override;
    procedure SwiGlu(gate, up: PSingleArray; aCount: PtrInt); override;
    procedure Rope(x: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      aCos, aSin: PSingleArray); override;
    /// the threads computing, the calling one included
    property Threads: integer
      read fThreads;
    /// true: the arithmetic of the reference for every operation (Q8_0 Linear
    // in double and block order, exact GELU, double attention), bit-identical to
    // TInferCpuBackend, just on all cores; false (default): float arithmetic
    // where this CPU has a kernel (Q8_0 Linear in float lanes like ggml, float
    // attention, a GELU table) - checked against the reference within a tolerance
    property Exact: boolean
      read fExact write fExact;
  end;


implementation

const
  INV_SQRTPI = 0.56418958354775628695;    // 1 / sqrt(pi)
  INV_SQRT2 = 0.70710678118654752440;     // 1 / sqrt(2)

  // our own polynomial fits: tools/infer-ref/fit_erf.py (mpmath, 60 digits);
  // doubles as their bits, every compiler reads them exactly
  // erf(x) = x * P(x^2 - 0.5) on [0, 1], worst relative error 2.61e-16
  ERF_P: array[0..29] of cardinal = (
    $B23CB7EA, $3FEEE51E {0.9654687386698673},
    $292151FD, $BFD1FD16 {-0.2810721780454342},
    $8017B042, $3FB45436 {0.07940998675593483},
    $B7B2F224, $BF92B902 {-0.018283884489152538},
    $249DF428, $3F6C82AA {0.003480274496665565},
    $1B81D64A, $BF426399 {-0.0005611894221277472},
    $8BCA9FA1, $3F148663 {7.829649525581277e-05},
    $38BADFD7, $BEE429E9 {-9.614808525668025e-06},
    $DC8EA554, $3EB1AD5E {1.0536449386123444e-06},
    $AE42AB27, $BE7BF8E5 {-1.042047688716128e-07},
    $4066AFD3, $3E4428F3 {9.387713728634682e-09},
    $13A7F7B0, $BE0AAD6D {-7.764158546227278e-10},
    $6C13C042, $3DD04FF1 {5.934372005891178e-11},
    $BF60C1B4, $BD929CD9 {-4.232037455094912e-12},
    $D3A74542, $3D53C01F {2.8067128195611264e-13});
  // erfc(x) = exp(-x^2) * Q_k(x - mid_k) on [1 + k/2, 1.5 + k/2], k = 0..9
  ERFC_Q: array[0..9, 0..35] of cardinal = (
    // [1.0, 1.5], worst relative error 3.98e-16
    ($2138767A, $3FD78A69 {0.3678229164523611},
     $DBFA8B07, $BFCABAAC {-0.20882187596460985},
     $5EEF7E58, $3FBB56F4 {0.1067955714965988},
     $AC624AD5, $BFA9B635 {-0.05021827439590757},
     $A6641F26, $3F968A25 {0.022011364250857167},
     $6D6C5895, $BF829963 {-0.009081627632934445},
     $5AABBEE6, $3F6D1B69 {0.003553109903229645},
     $94C61CE7, $BF55B8BC {-0.0013257829296849323},
     $FB6130FA, $3F3F0FE6 {0.0004739703102856343},
     $D22BA6CE, $BF255C07 {-0.00016296000929765523},
     $57B30D6E, $3F0C5703 {5.405405951244389e-05},
     $07F216E9, $BEF22FC4 {-1.7344078966383568e-05},
     $6893D136, $3ED6A18D {5.395666115893662e-06},
     $83F183FD, $BEBB5BC7 {-1.6306944557108957e-06},
     $2A476F91, $3EA0170A {4.795193331813261e-07},
     $6D79D105, $BE827347 {-1.374655116296733e-07},
     $AB57214D, $3E651514 {3.926893070053693e-08},
     $408CA486, $BE470992 {-1.0727619696709474e-08}),
    // [1.5, 2.0], worst relative error 4.38e-16
    ($2F1DC7E0, $3FD23CFC {0.2849722347374364},
     $38446447, $BFC0C3D5 {-0.1309763455144852},
     $EF0F810D, $3FAC8D0C {0.05576363008708727},
     $FE48945F, $BF96CB52 {-0.022259995241388327},
     $A11FFE73, $3F813648 {0.008404319207328847},
     $6A8EABED, $BF68BF71 {-0.0030209746514251374},
     $5C043331, $3F5106BD {0.001039204522444947},
     $84AB6B74, $BF368388 {-0.000343533353470419},
     $87E54DCA, $3F1CB4C6 {0.00010950528846836237},
     $2CD4591F, $BF01B291 {-3.375535525600354e-05},
     $3390EBDC, $3EE5273F {1.0086683334012006e-05},
     $F9628539, $BEC88FB2 {-2.9279380654034904e-06},
     $99376656, $3EABC103 {8.271324979894339e-07},
     $824EF291, $BE8E91DE {-2.2776276967046484e-07},
     $3166CC6C, $3E706E7D {6.121247184721773e-08},
     $648EA1BF, $BE514469 {-1.6081364049133937e-08},
     $265A6A32, $3E321204 {4.2073372335136154e-09},
     $7F8D9EF9, $BE122FF2 {-1.0586398409048978e-09}),
    // [2.0, 2.5], worst relative error 6.62e-16
    ($6D627932, $3FCD9444 {0.23108725873039188},
     $2BB37411, $BFB6A70D {-0.08848650280874916},
     $70E25A7B, $3FA06156 {0.03199262741070626},
     $9919A17A, $BF86883F {-0.011002060756440047},
     $561F7D33, $3F6DA595 {0.0036189953543580788},
     $1BB2F029, $BF52BD25 {-0.0011437284836537476},
     $3D3B2803, $3F36D774 {0.000348535422045715},
     $BC99E2D9, $BF1AED7E {-0.00010272108115739631},
     $CC92981C, $3EFEC773 {2.9353247360440577e-05},
     $6646667C, $BEE117A6 {-8.15028324367976e-06},
     $288385E7, $3EC27AF4 {2.2030220102476643e-06},
     $5AD13D44, $BEA37B9A {-5.806334023669654e-07},
     $6964BA02, $3E840E79 {1.4943286847633497e-07},
     $B94000B4, $BE642FE8 {-3.760148739595634e-08},
     $735E9F04, $3E43E2F8 {9.260421412183136e-09},
     $68FFA2DC, $BE2332A0 {-2.234913427924387e-09},
     $C217D54E, $3E027376 {5.369956737838093e-10},
     $22C1424F, $BDE1237A {-1.2469959906462586e-10}),
    // [2.5, 3.0], worst relative error 1.10e-15
    ($68FF27D7, $3FC8C9EB {0.1936620962790687},
     $81330099, $BFB03057 {-0.06323763756063484},
     $BAC83823, $3F943B98 {0.019758592987322864},
     $B30E6AB3, $BF784E9A {-0.005934337896997976},
     $FD72763E, $3F5C2C72 {0.0017195818852892143},
     $1ECB0904, $BF3F99E4 {-0.0004821950849810549},
     $16125571, $3F2131BB {0.00013118180053043772},
     $259675BF, $BF02312B {-3.469860957781454e-05},
     $B0EB9FE9, $3EE2BFB5 {8.94015604786798e-06},
     $9C48F41E, $BEC2DA32 {-2.2473734324888634e-06},
     $AAF6003B, $3EA2856F {5.519758214250832e-07},
     $B6190E10, $BE81CCF9 {-1.32625440451171e-07},
     $3E723E5D, $3E60C160 {3.120931762426605e-08},
     $415AB0BA, $BE3EEC75 {-7.19997663370746e-09},
     $1FAFF168, $3E1BFFE6 {1.629791522875503e-09},
     $9EB01EB1, $BDF8E64F {-3.6233763098245136e-10},
     $16078596, $3DD60B4A {8.019596575993681e-11},
     $FA329B15, $BDB2F0FD {-1.7227080558383554e-11}),
    // [3.0, 3.5], worst relative error 1.07e-15
    ($08D4BB45, $3FC54A7A {0.16633534842682188},
     $22B868A1, $BFA82A85 {-0.047199402321170376},
     $C9EE6425, $3F8A7EDD {0.012937290883018157},
     $9C47A2C4, $BF6C24B4 {-0.003435471300907574},
     $57A17F33, $3F4D0858 {0.0008860045775342717},
     $BA1C4911, $BF2D25EB {-0.0002223825695684763},
     $0238146D, $3F0C882F {5.442040881224121e-05},
     $25FA26B3, $BEEB45D0 {-1.3004640265340669e-05},
     $8D735C31, $3EC97DD7 {3.038831987471905e-06},
     $B5819D85, $BEA753CA {-6.952080680134817e-07},
     $1FD514EF, $3E84EC09 {1.5588115324420867e-07},
     $C47C5D42, $BE6268C3 {-3.428987632567693e-08},
     $52068D66, $3E3FCF8B {7.406510322072111e-09},
     $DC8C91DE, $BE1B0237 {-1.5721112070109953e-09},
     $B4680CCD, $3DF68CCF {3.281463333754513e-10},
     $5BE7C3D6, $BDD28757 {-6.740693331315234e-11},
     $CCD937D8, $3DAE50E0 {1.3786088938410364e-11},
     $8906C810, $BD88288B {-2.746489728803563e-12}),
    // [3.5, 4.0], worst relative error 1.16e-15
    ($19C14930, $3FC2A2AF {0.14558972127503855},
     $03ACDA11, $BFA2AA65 {-0.03645625753272353},
     $64F3CBF9, $3F822F06 {0.008878755527325298},
     $E05873AB, $BF61434A {-0.00210728287016911},
     $2A0DF889, $3F3FFF03 {0.00048822238209556744},
     $A1B1F551, $BF1CFCDE {-0.00011057957492429284},
     $0D260D9C, $3EF9B50D {2.451632537648977e-05},
     $AAD394D4, $BED65778 {-5.326672789273199e-06},
     $B3FECB1D, $3EB30C2F {1.135325604178969e-06},
     $2B3E0B4F, $BE8FE3E3 {-2.3760039413394554e-07},
     $31613954, $3E6A3BEE {4.8864825228343124e-08},
     $0E2DEF15, $BE453951 {-9.883145362638827e-09},
     $50EC9A6F, $3E20E5DB {1.9671718762564424e-09},
     $4EA2D09B, $BDFA7F25 {-3.855771939890968e-10},
     $3C2D8AF3, $3DD477D2 {7.446233824370416e-11},
     $875C67C6, $BDAF2CFC {-1.4177079848643914e-11},
     $BFAA1AFD, $3D87A4AB {2.6879258344959674e-12},
     $F85D7E04, $BD618559 {-4.979740448887448e-13}),
    // [4.0, 4.5], worst relative error 2.03e-15
    ($CE8C89AD, $3FC08E62 {0.12934527478598792},
     $33524970, $BF9DA395 {-0.028944331414615332},
     $691A5520, $3F79EF71 {0.006331866273872749},
     $26EDF541, $BF563732 {-0.0013559331671040983},
     $FDEC0456, $3F32A660 {0.00028457515684016577},
     $0E8F3B82, $BF0EB88E {-5.8595500213357555e-05},
     $975487B1, $3EE8D8E5 {1.1848093644465379e-05},
     $63867CF7, $BEC3C077 {-2.354600635537056e-06},
     $ECAD1974, $3E9EE335 {4.602602358582516e-07},
     $D3D921EB, $BE77C568 {-8.855436291991787e-08},
     $8B74BDCA, $3E5204AE {1.6780838688391694e-08},
     $2C7B278B, $BE2AEB42 {-3.133781543159465e-09},
     $D3DCF02D, $3E03D3BC {5.77044557732442e-10},
     $85B6D37A, $BDDCD02F {-1.0482190239807535e-10},
     $185F20A4, $3DB4A99E {1.8792496658836123e-11},
     $4C8BE957, $BD8D432E {-3.326752586966172e-12},
     $FBAE4E64, $3D64A396 {5.865963024038206e-13},
     $0B86BA65, $BD3C8661 {-1.013408673939828e-13}),
    // [4.5, 5.0], worst relative error 1.95e-15
    ($A3E77E9B, $3FBDC603 {0.11630270721024731},
     $BC4A104B, $BF981149 {-0.02350344859816315},
     $44F8B419, $3F7317C1 {0.00466132636897234},
     $883A33C8, $BF4DC1AF {-0.0009080988970296905},
     $C16255A3, $3F26CC10 {0.000173928304040655},
     $43BC5A27, $BF012F17 {-3.277578113463171e-05},
     $0A1C70E3, $3ED9818C {6.08111455038479e-06},
     $A21FAEDE, $BEB2A625 {-1.1115677200868462e-06},
     $CCB68ABA, $3E8AE1FA {2.0029196999307355e-07},
     $041779AD, $BE631C3E {-3.559574724884e-08},
     $FF0BE090, $3E3ACFA0 {6.24243411193597e-09},
     $A6A77C44, $BE12913C {-1.0807609482849086e-09},
     $ED03C209, $3DE9662F {1.8480327555343476e-10},
     $96BB2C05, $BDC12A28 {-3.122237437462642e-11},
     $1F5DDE8A, $3D96EE25 {5.213736118274698e-12},
     $53BB7030, $BD6E49E4 {-8.60854932077846e-13},
     $753F897E, $3D43EE2D {1.416138753572556e-13},
     $9D392457, $BD19C20B {-2.2877690602308322e-14}),
    // [5.0, 5.5], worst relative error 1.96e-15
    ($ACE146FE, $3FBB096F {0.1056127354688918},
     $B3B13590, $BF93E981 {-0.01944544467214865},
     $21161624, $3F6CDEAE {0.003524150940111397},
     $2A39EB5F, $BF449D49 {-0.0006291014910425433},
     $9AA11379, $3F1D03E1 {0.00011068405606902238},
     $3CCF878F, $BEF4230E {-1.9204078672070307e-05},
     $735CBB41, $3ECB93F4 {3.287547680217758e-06},
     $2EAABD00, $BEA2A435 {-5.555581002648793e-07},
     $30E519F7, $3E78E375 {9.271691345678677e-08},
     $D974901F, $BE506A3A {-1.5287623248168242e-08},
     $497E4F99, $3E256699 {2.4913782807153982e-09},
     $3CB86078, $BDFB961A {-4.0143404982115153e-10},
     $BB9F003C, $3DD195D5 {6.397492163217674e-11},
     $D4877C2F, $BDA62E7E {-1.008703419536307e-11},
     $CC2C8AC8, $3D7BB088 {1.5739708126733833e-12},
     $1292EC53, $BD511BEC {-2.431345213169403e-13},
     $1CF435F4, $3D25120E {3.7428776254054476e-14},
     $49864128, $BCF988B7 {-5.669697320697347e-15}),
    // [5.5, 6.0], worst relative error 3.66e-15
    ($49CD551E, $3FB8C140 {0.09669877816971392},
     $CDC18FE6, $BF90BC46 {-0.016343218143802483},
     $40E2C85A, $3F665350 {0.002725273842849635},
     $DA6D50F5, $BF3D662F {-0.00044859569827805586},
     $BE43629F, $3F131DDD {7.292428887540691e-05},
     $0BD28F2E, $BEE8900E {-1.1712414897786438e-05},
     $25ABA48F, $3EBF31A3 {1.8593010710449665e-06},
     $06D40841, $BE9395BE {-2.9183821122225146e-07},
     $DED6C5C4, $3E68530F {4.530783912925553e-08},
     $42509228, $BE3DE425 {-6.959585828674316e-09},
     $C7844BB6, $3E122D53 {1.058044122859079e-09},
     $42EB1466, $BDE5E2D6 {-1.5924220402302805e-10},
     $576F66CC, $3DBA1868 {2.3733575404189005e-11},
     $44B5EE41, $BD8ED1AA {-3.5037151450996483e-12},
     $B268F4E8, $3D6207C1 {5.124519285104998e-13},
     $E1CEA31D, $BD34E821 {-7.427575709753711e-14},
     $CB504843, $3D082A4D {1.0731526576354289e-14},
     $B9E7454C, $BCDB882E {-1.5283309609538977e-15}));
  // worst relative error of all erfc pieces: 3.66e-15

/// erfc by its continued fraction - used for x >= 6 only
function ErfcFraction(x: double): double;
var
  f: double;
  k: integer;
begin
  // erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
  // evaluated backwards; 40 terms are far beyond double precision at x >= 6
  f := x;
  for k := 40 downto 1 do
    f := x + (k * 0.5) / f;
  result := Exp(-x * x) * INV_SQRTPI / f;
end;

/// erf(x) for |x| <= 1: x * P(x^2 - 1/2)
function ErfSmall(x: double): double;
var
  c: PDoubleArray;
  t, r: double;
  i: PtrInt;
begin
  c := @ERF_P;
  t := x * x - 0.5;
  r := c^[high(ERF_P) shr 1];
  for i := (high(ERF_P) shr 1) - 1 downto 0 do
    r := r * t + c^[i];
  result := x * r;
end;

/// erfc(x) for 1 <= x < 6: exp(-x^2) * Q_k(x - mid_k), half-unit pieces
function ErfcPiece(x: double): double;
var
  c: PDoubleArray;
  u, r: double;
  k, i: PtrInt;
begin
  k := Trunc((x - 1) * 2);
  if k > 9 then
    k := 9; // x = 6 exactly
  c := @ERFC_Q[k];
  u := x - (1.25 + k * 0.5);
  r := c^[high(ERFC_Q[0]) shr 1];
  for i := (high(ERFC_Q[0]) shr 1) - 1 downto 0 do
    r := r * u + c^[i];
  result := Exp(-x * x) * r;
end;

function InferErf(x: double): double;
var
  a: double;
begin
  a := Abs(x);
  if a <= 1 then
    result := ErfSmall(x)
  else
  begin
    if a < 6 then
      result := 1 - ErfcPiece(a)
    else
      result := 1 - ErfcFraction(a);
    if x < 0 then
      result := -result;
  end;
end;

function InferErfc(x: double): double;
begin
  if x >= 6 then
    result := ErfcFraction(x)
  else if x >= 1 then
    result := ErfcPiece(x)
  else if x > -1 then
    result := 1 - ErfSmall(x)
  else if x > -6 then
    result := 2 - ErfcPiece(-x)
  else
    result := 2 - ErfcFraction(-x);
end;


{ TInferCpuBackend }

procedure InferQuantizeRowQ8_0(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array);
var
  bad: PtrInt;
begin
  if Assigned(InferQ8QuantizeRow) then
  begin
    // the SIMD kernel of this CPU: the very same bytes (tests/test.infer.gguf)
    bad := InferQ8QuantizeRow(pointer(x), aCount shr 5, pointer(aBlocks));
    if bad <> 0 then
      EInferError.RaiseUtf8('InferQuantizeRowQ8_0: non-finite activation in ' +
        'block %', [bad - 1]);
  end
  else
    InferQuantizeRowQ8_0Pascal(x, aCount, aBlocks);
end;

procedure QuantizeQ8_0(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array; aTiesAway: boolean);
var
  b, j: PtrInt;
  amax, v, d, id, t: single;
  f: integer;
  xb: PSingleArray;
begin
  for b := 0 to (aCount shr 5) - 1 do
  begin
    xb := @x^[b * 32];
    amax := 0;
    for j := 0 to 31 do
    begin
      // a NaN or an infinity has no Q8_0 form: refused by its exponent bits
      // (a comparison would skip a NaN), never truncated
      if PCardinal(@xb^[j])^ and $7f800000 = $7f800000 then
        EInferError.RaiseUtf8('InferQuantizeRowQ8_0: non-finite activation at %',
          [b * 32 + j]);
      v := Abs(xb^[j]);
      if v > amax then
        amax := v;
    end;
    // single results of one division each: correctly rounded on every FPU
    // (an x87 or double intermediate rounds twice, harmlessly for / of singles)
    d := amax / 127;
    aBlocks^[b].d := InferSingleToHalf(d);
    // d = 0, or so small that 1 / d overflows a single: its FP16 scale is 0,
    // the block adds nothing to any dot product - all q = 0 (ggml: 0 for
    // d = 0, undefined for the overflow)
    if d < 2.9387359e-39 then // 1 / 3.4028235e38: beyond, 1 / d is no single
    begin
      FillCharFast(aBlocks^[b].qs, SizeOf(aBlocks^[b].qs), 0);
      continue;
    end;
    id := 1 / d;
    for j := 0 to 31 do
    begin
      // one correctly rounded single product, as the SIMD kernels form it
      t := xb^[j] * id;
      // nearest, ties to even (or away from zero) - spelled out: Round
      // depends on the FPU mode
      f := Trunc(t);
      v := t - f; // exact: f and t are close singles
      if v > 0.5 then
        inc(f)
      else if v < -0.5 then
        dec(f)
      else if (v = 0.5) and (aTiesAway or odd(f)) then
        inc(f)
      else if (v = -0.5) and (aTiesAway or odd(f)) then
        dec(f);
      aBlocks^[b].qs[j] := f;
    end;
  end;
end;

procedure InferQuantizeRowQ8_0Pascal(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array);
begin
  QuantizeQ8_0(x, aCount, aBlocks, {tiesaway=}false);
end;

procedure InferQuantizeRowQ8_0Ref(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array);
begin
  QuantizeQ8_0(x, aCount, aBlocks, {tiesaway=}true);
end;

procedure InferQuantizeRowQ4_0(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ4_0Array);
var
  b, j, q0, q1: PtrInt;
  amax, m, v, d, id, t: single;
  p: double;
  xb: PSingleArray;
begin
  for b := 0 to (aCount shr 5) - 1 do
  begin
    xb := @x^[b * 32];
    amax := 0;
    m := 0;
    for j := 0 to 31 do
    begin
      if PCardinal(@xb^[j])^ and $7f800000 = $7f800000 then
        EInferError.RaiseUtf8('InferQuantizeRowQ4_0: non-finite value at %',
          [b * 32 + j]);
      v := Abs(xb^[j]);
      if amax < v then // the first of equal magnitudes, as ggml
      begin
        amax := v;
        m := xb^[j];
      end;
    end;
    d := m / -8; // one correctly rounded single division
    aBlocks^[b].d := InferSingleToHalf(d);
    // d = 0 (ggml: q = 8 as well), or so small that 1 / d is no single
    if Abs(d) < 2.9387359e-39 then // 1 / 3.4028235e38
    begin
      FillCharFast(aBlocks^[b].qs, SizeOf(aBlocks^[b].qs), $88);
      continue;
    end;
    id := 1 / d;
    for j := 0 to 15 do
    begin
      // the product of two singles is exact in a double: one rounding to
      // single after the sum, like the fused multiply-add of the C code
      p := xb^[j];
      p := p * id;
      t := p + 8.5;
      q0 := Trunc(t);
      if q0 > 15 then
        q0 := 15;
      p := xb^[j + 16];
      p := p * id;
      t := p + 8.5;
      q1 := Trunc(t);
      if q1 > 15 then
        q1 := 15;
      aBlocks^[b].qs[j] := q0 or (q1 shl 4);
    end;
  end;
end;

procedure InferQuantizeRowQ8_K(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_KArray);
var
  b, j, g, s: PtrInt;
  amax, m, v, iscale, t: single;
  f: integer;
  xb: PSingleArray;
  blk: PInferBlockQ8_K;
begin
  for b := 0 to (aCount shr 8) - 1 do
  begin
    xb := @x^[b * 256];
    blk := @aBlocks^[b];
    amax := 0;
    m := 0;
    for j := 0 to 255 do
    begin
      if PCardinal(@xb^[j])^ and $7f800000 = $7f800000 then
        EInferError.RaiseUtf8('InferQuantizeRowQ8_K: non-finite activation at %',
          [b * 256 + j]);
      v := Abs(xb^[j]);
      if v > amax then // the first of equal magnitudes, as ggml
      begin
        amax := v;
        m := xb^[j];
      end;
    end;
    // m so small that -127 / m is no single: a zero block (ggml: only m = 0)
    if amax < 3.7320437e-37 then // 127 / 3.4028235e38
    begin
      FillCharFast(blk^, SizeOf(blk^), 0);
      continue;
    end;
    iscale := -127 / m; // one correctly rounded single division
    for j := 0 to 255 do
    begin
      t := iscale * xb^[j];
      // nearest, ties to even - spelled out: Round depends on the FPU mode
      f := Trunc(t);
      v := t - f;
      if v > 0.5 then
        inc(f)
      else if v < -0.5 then
        dec(f)
      else if (v = 0.5) and odd(f) then
        inc(f)
      else if (v = -0.5) and odd(f) then
        dec(f);
      if f > 127 then
        f := 127;
      blk^.qs[j] := f;
    end;
    for g := 0 to 15 do
    begin
      s := 0;
      for j := 16 * g to 16 * g + 15 do
        inc(s, blk^.qs[j]);
      blk^.bsums[g] := s;
    end;
    blk^.d := 1 / iscale;
  end;
end;

/// a K-quant dot product of aBlocks super-blocks: the integer parts from
// aKernel (TInferKSums), then the half-float scales in double
function DotK(aType: TInferDType; w, x: pointer; aBlocks: PtrInt;
  aKernel: TInferKSums): double;
var
  sums: array[0..1023] of integer; // 512 blocks at a time: no heap
  n, b, size: PtrInt;
  wq: PInferBlockQ4_K;
  w6: PInferBlockQ6_K;
  xb: PInferBlockQ8_K;
  dx: double;
begin
  result := 0;
  size := INFER_DTYPE_SIZE[aType];
  xb := x;
  while aBlocks > 0 do
  begin
    n := aBlocks;
    if n > 512 then
      n := 512;
    aKernel(w, xb, n, @sums);
    if aType = idtQ4_K then
    begin
      wq := w;
      for b := 0 to n - 1 do
      begin
        dx := xb^.d;
        result := result + (INFER_HALF[wq^.d] * dx) * sums[b * 2] -
          (INFER_HALF[wq^.dmin] * dx) * sums[b * 2 + 1];
        inc(wq);
        inc(xb);
      end;
    end
    else
    begin
      w6 := w;
      for b := 0 to n - 1 do
      begin
        dx := xb^.d;
        result := result + (INFER_HALF[w6^.d] * dx) * sums[b * 2];
        inc(w6);
        inc(xb);
      end;
    end;
    w := PAnsiChar(w) + n * size;
    dec(aBlocks, n);
  end;
end;

/// DotK of one K-quant weight row against the two activation rows x and
// x + aXStride (bytes), the ints of both from aKernel (TInferKSums2): each
// result in the arithmetic of DotK, so bit-identical to two DotK calls
procedure DotK2(aType: TInferDType; w, x: pointer; aXStride, aBlocks: PtrInt;
  aKernel: TInferKSums2; out aA, aB: double);
var
  sums: array[0..2047] of integer; // 512 blocks at a time: no heap
  n, b, size: PtrInt;
  wq: PInferBlockQ4_K;
  w6: PInferBlockQ6_K;
  xa, xb: PInferBlockQ8_K;
  dx: double;
begin
  aA := 0;
  aB := 0;
  size := INFER_DTYPE_SIZE[aType];
  xa := x;
  while aBlocks > 0 do
  begin
    n := aBlocks;
    if n > 512 then
      n := 512;
    aKernel(w, xa, n, @sums, aXStride);
    xb := pointer(PAnsiChar(xa) + aXStride);
    if aType = idtQ4_K then
    begin
      wq := w;
      for b := 0 to n - 1 do
      begin
        dx := xa^.d;
        aA := aA + (INFER_HALF[wq^.d] * dx) * sums[b * 4] -
          (INFER_HALF[wq^.dmin] * dx) * sums[b * 4 + 1];
        dx := xb^.d;
        aB := aB + (INFER_HALF[wq^.d] * dx) * sums[b * 4 + 2] -
          (INFER_HALF[wq^.dmin] * dx) * sums[b * 4 + 3];
        inc(wq);
        inc(xa);
        inc(xb);
      end;
    end
    else
    begin
      w6 := w;
      for b := 0 to n - 1 do
      begin
        dx := xa^.d;
        aA := aA + (INFER_HALF[w6^.d] * dx) * sums[b * 4];
        dx := xb^.d;
        aB := aB + (INFER_HALF[w6^.d] * dx) * sums[b * 4 + 2];
        inc(w6);
        inc(xa);
        inc(xb);
      end;
    end;
    w := PAnsiChar(w) + n * size;
    dec(aBlocks, n);
  end;
end;

function InferDotQ4K(w: PInferBlockQ4_K; x: PInferBlockQ8_K; aBlocks: PtrInt): double;
begin
  result := DotK(idtQ4_K, w, x, aBlocks, InferQ4KSumsPascal);
end;

function InferDotQ6K(w: PInferBlockQ6_K; x: PInferBlockQ8_K; aBlocks: PtrInt): double;
begin
  result := DotK(idtQ6_K, w, x, aBlocks, InferQ6KSumsPascal);
end;

/// the K-quant Linear of the reference: activations as Q8_K, exact block sums
procedure LinearK(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
  aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  r, o, nb: PtrInt;
  acc: double;
  xk: array of TInferBlockQ8_K;
begin
  nb := aIn shr 8;
  SetLength(xk, nb);
  for r := 0 to aRows - 1 do
  begin
    InferQuantizeRowQ8_K(@x^[r * aIn], aIn, pointer(xk));
    for o := 0 to aOut - 1 do
    begin
      if bias <> nil then
        acc := bias^[o]
      else
        acc := 0;
      if w.DType = idtQ4_K then
        acc := acc + InferDotQ4K(@PByteArray(w.Data)^[o * nb * SizeOf(TInferBlockQ4_K)],
          pointer(xk), nb)
      else
        acc := acc + InferDotQ6K(@PByteArray(w.Data)^[o * nb * SizeOf(TInferBlockQ6_K)],
          pointer(xk), nb);
      y^[r * aOut + o] := acc;
    end;
  end;
end;

// unaligned Int64 loads are fine on these CPUs (Q4_0 qs sits at offset 2)
{$ifdef CPUINTEL}
  {$define INFERQ4WORDS}
{$endif CPUINTEL}
{$ifdef CPUAARCH64}
  {$define INFERQ4WORDS}
{$endif CPUAARCH64}

{$ifdef INFERQ4WORDS}
// eight nibbles x (0..15) of an Int64 as the eight int8 x - 8, bitwise only (no
// carry between the bytes): y = x xor 8 is x - 8 where x >= 8, else x + 8,
// which then only needs its high nibble set (0xF0) where bit 3 of y is set
function Q4Minus8(x: Int64): Int64;
  {$ifdef HASINLINE} inline; {$endif}
var
  t: Int64;
begin
  result := x xor Int64($0808080808080808);
  t := result and Int64($0808080808080808);
  result := result or (t shl 1) or (t shl 2) or (t shl 3) or (t shl 4);
end;
{$endif INFERQ4WORDS}

procedure InferQ4ToQ8(aSrc: PInferBlockQ4_0; aBlocks: PtrInt; aDst: PInferBlockQ8_0);
var
  b: PtrInt;
  {$ifdef INFERQ4WORDS}
  s, d: PInt64Array;
  {$else}
  j, v: PtrInt;
  {$endif INFERQ4WORDS}
begin
  for b := 0 to aBlocks - 1 do
  begin
    aDst^.d := aSrc^.d;
    // qs[j] low nibble -> qs[j], high nibble -> qs[j + 16]
    {$ifdef INFERQ4WORDS}
    // 8 bytes at a time
    s := @aSrc^.qs;
    d := @aDst^.qs;
    d^[0] := Q4Minus8(s^[0] and Int64($0F0F0F0F0F0F0F0F));
    d^[1] := Q4Minus8(s^[1] and Int64($0F0F0F0F0F0F0F0F));
    d^[2] := Q4Minus8((s^[0] shr 4) and Int64($0F0F0F0F0F0F0F0F));
    d^[3] := Q4Minus8((s^[1] shr 4) and Int64($0F0F0F0F0F0F0F0F));
    {$else}
    // other CPUs (32-bit ARM) may fault on an unaligned Int64: by bytes
    for j := 0 to 15 do
    begin
      v := aSrc^.qs[j];
      aDst^.qs[j] := (v and 15) - 8;
      aDst^.qs[j + 16] := (v shr 4) - 8;
    end;
    {$endif INFERQ4WORDS}
    inc(aSrc);
    inc(aDst);
  end;
end;

const
  /// CausalRows keeps its scores on the stack up to this many keys
  CAUSAL_STACK = 4096;

/// rows aFirst..aLast of head h of a causal attention job
// - the reference: scores and sums in double, the softmax divided at the end;
// with aJob.Fast and a float kernel for this CPU and head size, float lanes
procedure CausalRows(const aJob: TInferCausalJob; h, aFirst, aLast: PtrInt);
var
  r, g, j, t, d, total, first, cnt, kvw, qw: PtrInt;
  scale, m, sum, acc, a: double;
  q, k, outp: PSingleArray;
  sc, sums: PDoubleArray;
  s: PSingleArray;
  // no heap below CAUSAL_STACK keys: the workers call this for every head and
  // token at once, and a shared memory manager (Delphi's) serializes them
  sStack: array[0..CAUSAL_STACK - 1] of single;
  scStack: array[0..CAUSAL_STACK - 1] of double;
  sumsStack: array[0..255] of double;
  sHeap: TSingleDynArray;
  scHeap, sumsHeap: TDoubleDynArray;
begin
  d := aJob.HeadDim;
  kvw := aJob.KvHeads * d;
  qw := aJob.Heads * d;
  g := h div (aJob.Heads div aJob.KvHeads); // grouped-query: the shared KV head
  scale := 1 / Sqrt(d);
  if aJob.Fast and Assigned(InferDotsF32) and (d and 15 = 0) then
  begin
    s := @sStack;
    if aJob.Past + aLast + 1 > CAUSAL_STACK then
    begin
      SetLength(sHeap, aJob.Past + aLast + 1);
      s := pointer(sHeap);
    end;
    for r := aFirst to aLast do
    begin
      total := aJob.Past + r + 1; // causal: up to and including this token
      first := 0; // a sliding window: the last Window keys only
      if (aJob.Window > 0) and (total > aJob.Window) then
        first := total - aJob.Window;
      cnt := total - first;
      InferDotsF32(@aJob.Q^[r * qw + h * d], @aJob.Keys^[first * kvw + g * d],
        kvw * SizeOf(single), cnt, d, s);
      m := s^[0] * scale;
      for j := 0 to cnt - 1 do
      begin
        s^[j] := s^[j] * scale;
        if s^[j] > m then
          m := s^[j];
      end;
      sum := 0;
      for j := 0 to cnt - 1 do
      begin
        a := Exp(s^[j] - m);
        s^[j] := a;
        sum := sum + a;
      end;
      sum := 1 / sum;
      for j := 0 to cnt - 1 do
        s^[j] := s^[j] * sum;
      outp := @aJob.Y^[r * qw + h * d];
      FillCharFast(outp^, d * SizeOf(single), 0);
      InferAxpyRowsF32(pointer(s), @aJob.Values^[first * kvw + g * d],
        kvw * SizeOf(single), cnt, d, outp);
    end;
    exit;
  end;
  sc := @scStack;
  if aJob.Past + aLast + 1 > CAUSAL_STACK then
  begin
    SetLength(scHeap, aJob.Past + aLast + 1);
    sc := pointer(scHeap);
  end;
  sums := @sumsStack;
  if d > length(sumsStack) then
  begin
    SetLength(sumsHeap, d);
    sums := pointer(sumsHeap);
  end;
  for r := aFirst to aLast do
  begin
    total := aJob.Past + r + 1;
    first := 0;
    if (aJob.Window > 0) and (total > aJob.Window) then
      first := total - aJob.Window;
    q := @aJob.Q^[r * qw + h * d];
    m := 0;
    for j := first to total - 1 do
    begin
      k := @aJob.Keys^[j * kvw + g * d];
      acc := 0;
      for t := 0 to d - 1 do
      begin
        a := q^[t]; // widen before multiplying: same result on every FPU
        acc := acc + a * k^[t];
      end;
      acc := acc * scale;
      sc^[j] := acc;
      if (j = first) or (acc > m) then
        m := acc;
    end;
    sum := 0;
    for j := first to total - 1 do
    begin
      sc^[j] := Exp(sc^[j] - m);
      sum := sum + sc^[j];
    end;
    for t := 0 to d - 1 do
      sums^[t] := 0;
    for j := first to total - 1 do
    begin
      k := @aJob.Values^[j * kvw + g * d];
      acc := sc^[j];
      for t := 0 to d - 1 do
        sums^[t] := sums^[t] + acc * k^[t];
    end;
    outp := @aJob.Y^[r * qw + h * d];
    sum := 1 / sum;
    for t := 0 to d - 1 do
      outp^[t] := sums^[t] * sum;
  end;
end;

function CausalJob(q: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
  keys, values: PSingleArray; aPast, aKvHeads: PtrInt; y: PSingleArray;
  aWindow: PtrInt; aFast: boolean): TInferCausalJob;
begin
  if (aRows <= 0) or (aHeads <= 0) or (aHeadDim <= 0) or (aPast < 0) or
     (aKvHeads <= 0) or (aHeads mod aKvHeads <> 0) or (aWindow < 0) then
    EInferError.RaiseUtf8('CausalAttention: % rows, % heads of %, % KV heads, ' +
      'past %, window %', [aRows, aHeads, aHeadDim, aKvHeads, aPast, aWindow]);
  result.Q := q;
  result.Keys := keys;
  result.Values := values;
  result.Y := y;
  result.Rows := aRows;
  result.Heads := aHeads;
  result.HeadDim := aHeadDim;
  result.Past := aPast;
  result.KvHeads := aKvHeads;
  result.Window := aWindow;
  result.RowBlocks := 1;
  result.Fast := aFast;
end;

procedure TInferCpuBackend.CausalAttention(q: PSingleArray; aRows, aHeads,
  aHeadDim: PtrInt; keys, values: PSingleArray; aPast, aKvHeads: PtrInt;
  y: PSingleArray; aWindow: PtrInt);
var
  job: TInferCausalJob;
  h: PtrInt;
begin
  job := CausalJob(q, aRows, aHeads, aHeadDim, keys, values, aPast, aKvHeads, y,
    aWindow, false);
  for h := 0 to aHeads - 1 do
    CausalRows(job, h, 0, aRows - 1);
end;

function TInferCpuBackend.Supports(aOp: TInferOp; aWeights: TInferDType): boolean;
begin
  if aOp = ioLinear then
    result := aWeights in [idtF32, idtQ8_0, idtQ4_0, idtQ4_K, idtQ6_K]
  else
    result := aWeights = idtF32;
end;

procedure LinearQ8_0(x: PSingleArray; aRows, aIn: PtrInt; w: PInferBlockQ8_0Array;
  aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  r, o, b, j, nb: PtrInt;
  sum: integer;
  acc, dx, dw: double;
  xq: array of TInferBlockQ8_0;
  wb, xb: PInferBlockQ8_0;
  dxs: array of double;
begin
  nb := aIn shr 5;
  SetLength(xq, nb);
  SetLength(dxs, nb);
  for r := 0 to aRows - 1 do
  begin
    InferQuantizeRowQ8_0(@x^[r * aIn], aIn, pointer(xq));
    for b := 0 to nb - 1 do
      dxs[b] := InferHalfToSingle(xq[b].d);
    for o := 0 to aOut - 1 do
    begin
      wb := @w^[o * nb];
      if bias <> nil then
        acc := bias^[o]
      else
        acc := 0;
      for b := 0 to nb - 1 do
      begin
        xb := @xq[b];
        sum := 0; // at most 32 * 127 * 128: no overflow
        for j := 0 to 31 do
          inc(sum, integer(wb^.qs[j]) * xb^.qs[j]);
        dw := InferHalfToSingle(wb^.d);
        dx := dxs[b];
        acc := acc + sum * (dw * dx);
        inc(wb);
      end;
      y^[r * aOut + o] := acc;
    end;
  end;
end;

procedure TInferCpuBackend.Linear(x: PSingleArray; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  r, o, i: PtrInt;
  acc, a: double;
  xr, wo: PSingleArray;
  w8: array of TInferBlockQ8_0;
begin
  case w.DType of
    idtF32:
      ;
    idtQ8_0:
      begin
        if aIn and 31 <> 0 then
          EInferError.RaiseUtf8('%.Linear: % inputs are no whole Q8_0 blocks',
            [self, aIn]);
        LinearQ8_0(x, aRows, aIn, w.Data, aOut, bias, y);
        exit;
      end;
    idtQ4_K, idtQ6_K:
      begin
        if aIn and 255 <> 0 then
          EInferError.RaiseUtf8('%.Linear: % inputs are no whole % blocks',
            [self, aIn, INFER_DTYPE_NAME[w.DType]]);
        LinearK(x, aRows, aIn, w, aOut, bias, y);
        exit;
      end;
    idtQ4_0:
      begin
        if aIn and 31 <> 0 then
          EInferError.RaiseUtf8('%.Linear: % inputs are no whole Q4_0 blocks',
            [self, aIn]);
        // the reference: the whole matrix as Q8_0 blocks, then the Q8_0 path
        SetLength(w8, aOut * (aIn shr 5));
        InferQ4ToQ8(w.Data, length(w8), pointer(w8));
        LinearQ8_0(x, aRows, aIn, pointer(w8), aOut, bias, y);
        exit;
      end;
  else
    EInferError.RaiseUtf8('%.Linear: % weights not supported',
      [self, INFER_DTYPE_NAME[w.DType]]);
  end;
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aIn];
    for o := 0 to aOut - 1 do
    begin
      wo := @PSingleArray(w.Data)^[o * aIn];
      if bias <> nil then
        acc := bias^[o]
      else
        acc := 0;
      for i := 0 to aIn - 1 do
      begin
        a := xr^[i]; // widen before multiplying: same result on every FPU
        acc := acc + a * wo^[i];
      end;
      y^[r * aOut + o] := acc;
    end;
  end;
end;

procedure TInferCpuBackend.LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
  gamma, beta: PSingleArray; aEps: double; y: PSingleArray);
var
  r, c: PtrInt;
  mean, variance, d, inv: double;
  xr, yr: PSingleArray;
begin
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aCols];
    yr := @y^[r * aCols];
    // two passes: the one-pass formula loses digits on nearly constant rows
    mean := 0;
    for c := 0 to aCols - 1 do
      mean := mean + xr^[c];
    mean := mean / aCols;
    variance := 0;
    for c := 0 to aCols - 1 do
    begin
      d := xr^[c] - mean;
      variance := variance + d * d;
    end;
    variance := variance / aCols;
    inv := 1 / Sqrt(variance + aEps);
    for c := 0 to aCols - 1 do
      yr^[c] := (xr^[c] - mean) * inv * gamma^[c] + beta^[c];
  end;
end;

procedure TInferCpuBackend.Gelu(x: PSingleArray; aCount: PtrInt);
var
  i: PtrInt;
  v, z: double;
begin
  for i := 0 to aCount - 1 do
  begin
    v := x^[i];
    z := v * INV_SQRT2;
    if z >= 0 then
      x^[i] := 0.5 * v * (1 + InferErf(z))
    else
      x^[i] := 0.5 * v * InferErfc(-z); // 1 + erf(z) = erfc(-z), no cancellation
  end;
end;

procedure TInferCpuBackend.Add(x, y: PSingleArray; aCount: PtrInt);
var
  i: PtrInt;
begin
  for i := 0 to aCount - 1 do
    x^[i] := x^[i] + y^[i];
end;

procedure TInferCpuBackend.SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
  aKeep: PIntegerArray);
var
  r, c: PtrInt;
  m, sum, e: double;
  xr: PSingleArray;
  any: boolean;
begin
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aCols];
    any := false;
    m := 0;
    for c := 0 to aCols - 1 do
      if (aKeep = nil) or (aKeep^[c] <> 0) then
        if not any or (xr^[c] > m) then
        begin
          m := xr^[c];
          any := true;
        end;
    if not any then
    begin
      FillCharFast(xr^, aCols * SizeOf(single), 0);
      continue;
    end;
    // subtract the maximum: exp() of a large logit would overflow
    sum := 0;
    for c := 0 to aCols - 1 do
      if (aKeep = nil) or (aKeep^[c] <> 0) then
      begin
        e := Exp(xr^[c] - m);
        xr^[c] := e;
        sum := sum + e;
      end
      else
        xr^[c] := 0;
    sum := 1 / sum;
    for c := 0 to aCols - 1 do
      xr^[c] := xr^[c] * sum;
  end;
end;

procedure TInferCpuBackend.Attention(q, k, v: PSingleArray;
  aSeq, aHeads, aHeadDim: PtrInt; aKeep: PIntegerArray; aScratch, y: PSingleArray);
var
  h: PtrInt;
begin
  for h := 0 to aHeads - 1 do
    AttentionHead(q, k, v, aSeq, aHeadDim, aHeads * aHeadDim, h * aHeadDim,
      0, aSeq - 1, aKeep, aScratch, y);
end;

procedure TInferCpuBackend.AttentionHead(q, k, v: PSingleArray; aSeq, aHeadDim,
  aWidth, aOffset, aFirst, aLast: PtrInt; aKeep: PIntegerArray;
  aScratch, y: PSingleArray);
var
  i, j, d, width, off: PtrInt;
  scale, acc, acc1, acc2, acc3, a: double;
  qi, kj, k1, k2, k3, p, vj: PSingleArray;
  sums: TDoubleDynArray;
begin
  width := aWidth;
  off := aOffset;
  scale := 1 / Sqrt(aHeadDim);
  SetLength(sums, aHeadDim);
  begin
    // scores[i, j] = q_i . k_j / sqrt(d) - four keys at a time: four
    // independent sums, each over d in order (the same bits as one by one)
    for i := aFirst to aLast do
    begin
      qi := @q^[i * width + off];
      j := 0;
      while j + 3 < aSeq do
      begin
        kj := @k^[j * width + off];
        k1 := @k^[(j + 1) * width + off];
        k2 := @k^[(j + 2) * width + off];
        k3 := @k^[(j + 3) * width + off];
        acc := 0;
        acc1 := 0;
        acc2 := 0;
        acc3 := 0;
        for d := 0 to aHeadDim - 1 do
        begin
          a := qi^[d];
          acc := acc + a * kj^[d];
          acc1 := acc1 + a * k1^[d];
          acc2 := acc2 + a * k2^[d];
          acc3 := acc3 + a * k3^[d];
        end;
        aScratch^[i * aSeq + j] := acc * scale;
        aScratch^[i * aSeq + j + 1] := acc1 * scale;
        aScratch^[i * aSeq + j + 2] := acc2 * scale;
        aScratch^[i * aSeq + j + 3] := acc3 * scale;
        inc(j, 4);
      end;
      while j < aSeq do
      begin
        kj := @k^[j * width + off];
        acc := 0;
        for d := 0 to aHeadDim - 1 do
        begin
          a := qi^[d];
          acc := acc + a * kj^[d];
        end;
        aScratch^[i * aSeq + j] := acc * scale;
        inc(j);
      end;
    end;
    SoftmaxMasked(@aScratch^[aFirst * aSeq], aLast - aFirst + 1, aSeq, aKeep);
    // y_i = sum_j p[i, j] v_j - j ascending for every d, as one sum per d,
    // but reading v row by row (contiguous) instead of down a column
    for i := aFirst to aLast do
    begin
      p := @aScratch^[i * aSeq];
      for d := 0 to aHeadDim - 1 do
        sums[d] := 0;
      for j := 0 to aSeq - 1 do
      begin
        a := p^[j];
        vj := @v^[j * width + off];
        for d := 0 to aHeadDim - 1 do
          sums[d] := sums[d] + a * vj^[d];
      end;
      for d := 0 to aHeadDim - 1 do
        y^[i * width + off + d] := sums[d];
    end;
  end;
end;

procedure TInferCpuBackend.L2Normalize(x: PSingleArray; aCount: PtrInt);
var
  i: PtrInt;
  sum, v: double;
begin
  sum := 0;
  for i := 0 to aCount - 1 do
  begin
    v := x^[i];
    sum := sum + v * v;
  end;
  if sum = 0 then
    exit;
  sum := 1 / Sqrt(sum);
  for i := 0 to aCount - 1 do
    x^[i] := x^[i] * sum;
end;

procedure TInferCpuBackend.RmsNorm(x: PSingleArray; aRows, aCols: PtrInt;
  w: PSingleArray; aEps: double; y: PSingleArray);
var
  r, c: PtrInt;
  sum, a, inv: double;
  xr, yr: PSingleArray;
  n: single;
begin
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aCols];
    yr := @y^[r * aCols];
    sum := 0;
    for c := 0 to aCols - 1 do
    begin
      a := xr^[c];
      sum := sum + a * a;
    end;
    inv := 1 / Sqrt(sum / aCols + aEps);
    for c := 0 to aCols - 1 do
    begin
      // the reference rounds the normalized value to single before the weight
      n := xr^[c] * inv;
      yr^[c] := w^[c] * n;
    end;
  end;
end;

procedure TInferCpuBackend.SwiGlu(gate, up: PSingleArray; aCount: PtrInt);
var
  i: PtrInt;
  a: double;
  b: single;
begin
  for i := 0 to aCount - 1 do
  begin
    // silu(g) * u, each step rounded to single like the reference
    a := gate^[i];
    b := a / (1 + Exp(-a));
    gate^[i] := b * up^[i];
  end;
end;

procedure TInferCpuBackend.Rope(x: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
  aCos, aSin: PSingleArray);
var
  r, h, i, half: PtrInt;
  c, s, a, b: double;
  v: PSingleArray;
begin
  half := aHeadDim shr 1;
  for r := 0 to aRows - 1 do
    for i := 0 to half - 1 do
    begin
      c := aCos^[r * half + i];
      s := aSin^[r * half + i];
      for h := 0 to aHeads - 1 do
      begin
        v := @x^[(r * aHeads + h) * aHeadDim];
        a := v^[i];
        b := v^[i + half];
        v^[i] := a * c - b * s;
        v^[i + half] := b * c + a * s;
      end;
    end;
end;

{ TInferCpuFastBackend }

const
  /// output features per Linear work item: 16 Q8_0 rows of 1024 inputs are
  // 17 KB, kept in L1 while every token row passes over them
  TILE = 16;
  /// below this many multiply-adds a Linear stays on the calling thread
  PARALLEL_MIN = 1 shl 16;

constructor TInferCpuFastBackend.Create(aThreads: integer);
begin
  inherited Create;
  if aThreads <= 0 then
    // two thirds of the cores: the workers spin between operations, and all of
    // them spinning starves the OS and the other processes (measured: 12 of 12
    // cores are 20 % slower than 8 in a WSL VM)
    aThreads := (SystemInfo.dwNumberOfProcessors * 2 + 2) div 3;
  if aThreads < 1 then
    aThreads := 1;
  fThreads := aThreads;
  if fThreads > 1 then
    // the calling thread computes too: the pool holds one thread less
    fPool := TInferThreadPool.Create(fThreads - 1);
end;

destructor TInferCpuFastBackend.Destroy;
begin
  fPool.Free;
  inherited Destroy;
end;

procedure TInferCpuFastBackend.RunParallel(const aJob: TOnInferItem;
  aCount: integer);
var
  i: integer;
begin
  if fPool = nil then
    for i := 0 to aCount - 1 do
      aJob(i)
  else
    fPool.Run(aJob, aCount);
end;

procedure LinearQuantizeRow(const aJob: TInferLinearJob; aRow: PtrInt);
var
  b, first: PtrInt;
begin
  first := aRow * aJob.Blocks;
  if aJob.W.DType in [idtQ4_K, idtQ6_K] then
  begin
    InferQuantizeRowQ8_K(@aJob.X^[aRow * aJob.InCount], aJob.InCount,
      @aJob.Xk^[first]);
    exit;
  end;
  InferQuantizeRowQ8_0(@aJob.X^[aRow * aJob.InCount], aJob.InCount,
    @aJob.Xq^[first]);
  for b := first to first + aJob.Blocks - 1 do
  begin
    // a block beyond 127 * 65504 has the FP16 scale +Inf (ggml writes it too):
    // every product with it is Inf or NaN, and the SIMD kernels must not trap
    // inside (see kernels) - refused here, before any of them runs
    if aJob.Xq^[b].d and $7fff = $7c00 then
      EInferError.RaiseUtf8('Linear: activation beyond the Q8_0 range in block %',
        [b - first]);
    aJob.Dx^[b] := INFER_HALF[aJob.Xq^[b].d];
    aJob.Dxs^[b] := INFER_HALF[aJob.Xq^[b].d];
  end;
end;

/// output features aTile * TILE .. of every token row
procedure LinearTileWork(const aJob: TInferLinearJob; aTile: PtrInt);
var
  o, oEnd, r, i, nb, k, stride: PtrInt;
  acc, acc2, a, dw: double;
  k1: TInferKSums;
  k2: TInferKSums2;
  accs: single;
  acc8: array[0..7] of double;
  out8: array[0..7] of single;
  local: array[0..2047] of integer; // no heap traffic up to 16384 inputs
  heap: TIntegerDynArray;
  sums: PIntegerArray;
  xr, wo: PSingleArray;
  wq, wb: PInferBlockQ8_0;
  pair: boolean;
  // the weight rows of the tile as Q8_0: the matrix itself, or its Q4_0 rows
  // unpacked once per tile (on the stack up to 128 blocks per row)
  wbase: PInferBlockQ8_0Array;
  wfirst: PtrInt;
  unpacked: array[0..TILE * 128 - 1] of TInferBlockQ8_0;
  unpackedHeap: array of TInferBlockQ8_0;

  function Bias(aOut: PtrInt): double;
  begin
    if aJob.Bias <> nil then
      result := aJob.Bias^[aOut]
    else
      result := 0;
  end;

  // four weight rows against token row aRow without a fused kernel: the int
  // sums of the 4-row kernel, scaled in the reference's order
  procedure Rows4(aRow, aFirst: PtrInt);
  var
    j, c: PtrInt;
    d: PDoubleArray;
  begin
    d := @aJob.Dx^[aRow * nb];
    InferQ8BlockDots4(wq, @aJob.Xq^[aRow * nb], nb, sums, stride);
    for c := 0 to nb - 1 do
      for j := 0 to 3 do
      begin
        dw := INFER_HALF[wbase^[(o - wfirst + j) * nb + c].d];
        acc8[aFirst + j] := acc8[aFirst + j] + sums^[c * 4 + j] * (dw * d^[c]);
      end;
  end;

  // one weight row against token row aRow: the reference loop
  procedure Row1(aRow: PtrInt);
  var
    c: PtrInt;
    d: PDoubleArray;
  begin
    d := @aJob.Dx^[aRow * nb];
    InferQ8BlockDots(wq, @aJob.Xq^[aRow * nb], nb, sums);
    acc := Bias(o);
    wb := wq;
    for c := 0 to nb - 1 do
    begin
      dw := INFER_HALF[wb^.d];
      acc := acc + sums^[c] * (dw * d^[c]);
      inc(wb);
    end;
    aJob.Y^[aRow * aJob.OutCount + o] := acc;
  end;

  // one weight row against token row aRow in single precision, block by block
  // (the tail of the FP32 path: same tolerance contract, simpler order)
  procedure Row1F32(aRow: PtrInt);
  var
    c: PtrInt;
    d: PSingleArray;
  begin
    d := @aJob.Dxs^[aRow * nb];
    InferQ8BlockDots(wq, @aJob.Xq^[aRow * nb], nb, sums);
    accs := 0;
    wb := wq;
    for c := 0 to nb - 1 do
    begin
      accs := accs + sums^[c] * (INFER_HALF[wb^.d] * d^[c]);
      inc(wb);
    end;
    aJob.Y^[aRow * aJob.OutCount + o] := Bias(o) + accs;
  end;

begin
  oEnd := MinPtrInt((aTile + 1) * TILE, aJob.OutCount) - 1;
  if aJob.W.DType in [idtQ4_K, idtQ6_K] then
  begin
    // K-quants: per output and row - two rows at a time where the CPU has a
    // kernel for it (the weights unpacked once for both)
    nb := aJob.Blocks;
    if aJob.W.DType = idtQ4_K then
    begin
      k1 := InferQ4KSums;
      k2 := InferQ4KSums2;
    end
    else
    begin
      k1 := InferQ6KSums;
      k2 := InferQ6KSums2;
    end;
    stride := nb * INFER_DTYPE_SIZE[aJob.W.DType];
    for o := aTile * TILE to oEnd do
    begin
      r := 0;
      while r < aJob.Rows do
      begin
        // the integer sums of this CPU's kernel, then the arithmetic of the
        // reference: Exact and the float path are bit-identical to it
        if Assigned(k2) and (r + 1 < aJob.Rows) then
        begin
          DotK2(aJob.W.DType, @PByteArray(aJob.W.Data)^[o * stride], @aJob.Xk^[r * nb],
            nb * SizeOf(TInferBlockQ8_K), nb, k2, acc, acc2);
          aJob.Y^[r * aJob.OutCount + o] := Bias(o) + acc;
          aJob.Y^[(r + 1) * aJob.OutCount + o] := Bias(o) + acc2;
          inc(r, 2);
        end
        else
        begin
          acc := DotK(aJob.W.DType, @PByteArray(aJob.W.Data)^[o * stride],
            @aJob.Xk^[r * nb], nb, k1);
          aJob.Y^[r * aJob.OutCount + o] := Bias(o) + acc;
          inc(r);
        end;
      end;
    end;
  end
  else if aJob.W.DType <> idtF32 then
  begin
    nb := aJob.Blocks;
    wbase := aJob.W.Data;
    wfirst := 0;
    if aJob.W.DType = idtQ4_0 then
    begin
      wfirst := aTile * TILE;
      if nb <= 128 then
        wbase := @unpacked
      else
      begin
        SetLength(unpackedHeap, (oEnd - wfirst + 1) * nb);
        wbase := pointer(unpackedHeap);
      end;
      if Assigned(InferQ4Unpack) then
        InferQ4Unpack(@PInferBlockQ4_0Array(aJob.W.Data)^[wfirst * nb],
          (oEnd - wfirst + 1) * nb, @wbase^[0])
      else
        InferQ4ToQ8(@PInferBlockQ4_0Array(aJob.W.Data)^[wfirst * nb],
          (oEnd - wfirst + 1) * nb, @wbase^[0]);
    end;
    sums := @local;
    if nb * 4 > length(local) then
    begin
      SetLength(heap, nb * 4);
      sums := pointer(heap);
    end;
    stride := nb * SizeOf(TInferBlockQ8_0);
    // the tile stays in cache while every token row passes over it; token rows
    // go by two (each weight block loaded once for both), weight rows by four
    // (each activation block loaded once for four)
    r := 0;
    while r < aJob.Rows do
    begin
      if aJob.Fp32 then
      begin
        // the FP32 arithmetic: float lanes, reduced at the end (see kernels)
        pair := r + 1 < aJob.Rows;
        o := aTile * TILE;
        while o + 3 <= oEnd do
        begin
          wq := @wbase^[(o - wfirst) * nb];
          if pair then
            InferQ8DotF32x4x2(wq, @aJob.Xq^[r * nb], @aJob.Xq^[(r + 1) * nb], nb,
              @aJob.Dxs^[r * nb], @aJob.Dxs^[(r + 1) * nb], stride, @out8)
          else
            InferQ8DotF32x4x2(wq, @aJob.Xq^[r * nb], @aJob.Xq^[r * nb], nb,
              @aJob.Dxs^[r * nb], @aJob.Dxs^[r * nb], stride, @out8);
          for k := 0 to 3 do
            aJob.Y^[r * aJob.OutCount + o + k] := Bias(o + k) + out8[k];
          if pair then
            for k := 0 to 3 do
              aJob.Y^[(r + 1) * aJob.OutCount + o + k] := Bias(o + k) + out8[4 + k];
          inc(o, 4);
        end;
        while o <= oEnd do
        begin
          wq := @wbase^[(o - wfirst) * nb];
          Row1F32(r);
          if pair then
            Row1F32(r + 1);
          inc(o);
        end;
      end
      else
      begin
        // the exact arithmetic of the reference, every output in block order
        pair := Assigned(InferQ8Linear4x2) and (r + 1 < aJob.Rows);
        o := aTile * TILE;
        while o + 3 <= oEnd do
        begin
          wq := @wbase^[(o - wfirst) * nb];
          for k := 0 to 3 do
            acc8[k] := Bias(o + k);
          if pair then
          begin
            for k := 0 to 3 do
              acc8[4 + k] := Bias(o + k);
            InferQ8Linear4x2(wq, @aJob.Xq^[r * nb], nb, @aJob.Dx^[r * nb], stride,
              @acc8);
          end
          else if Assigned(InferQ8Linear4) then
            InferQ8Linear4(wq, @aJob.Xq^[r * nb], nb, @aJob.Dx^[r * nb], stride,
              @acc8)
          else
            Rows4(r, 0);
          for k := 0 to 3 do
            aJob.Y^[r * aJob.OutCount + o + k] := acc8[k];
          if pair then
            for k := 0 to 3 do
              aJob.Y^[(r + 1) * aJob.OutCount + o + k] := acc8[4 + k];
          inc(o, 4);
        end;
        while o <= oEnd do
        begin
          wq := @wbase^[(o - wfirst) * nb];
          Row1(r);
          if pair then
            Row1(r + 1);
          inc(o);
        end;
      end;
      inc(r, 1 + ord(pair));
    end;
  end
  else if aJob.Fp32 then
    // F32 weights in float lanes: the tile of weight rows stays in cache while
    // every token row passes over it
    for r := 0 to aJob.Rows - 1 do
    begin
      o := aTile * TILE;
      InferDotsF32(@aJob.X^[r * aJob.InCount],
        @PSingleArray(aJob.W.Data)^[o * aJob.InCount], aJob.InCount * SizeOf(single),
        oEnd - o + 1, aJob.InCount, @aJob.Y^[r * aJob.OutCount + o]);
      if aJob.Bias <> nil then
        for o := o to oEnd do
          aJob.Y^[r * aJob.OutCount + o] := aJob.Y^[r * aJob.OutCount + o] +
            aJob.Bias^[o];
    end
  else
    // F32 weights: the reference loop, split by output feature
    for r := 0 to aJob.Rows - 1 do
    begin
      xr := @aJob.X^[r * aJob.InCount];
      for o := aTile * TILE to oEnd do
      begin
        wo := @PSingleArray(aJob.W.Data)^[o * aJob.InCount];
        acc := Bias(o);
        for i := 0 to aJob.InCount - 1 do
        begin
          a := xr^[i];
          acc := acc + a * wo^[i];
        end;
        aJob.Y^[r * aJob.OutCount + o] := acc;
      end;
    end;
end;

procedure TInferCpuFastBackend.LinearTile(aItem: integer);
begin
  LinearTileWork(fJob, aItem);
end;

procedure TInferCpuFastBackend.QuantizeRow(aItem: integer);
begin
  LinearQuantizeRow(fJob, aItem);
end;

procedure TInferCpuFastBackend.Linear(x: PSingleArray; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  job: TInferLinearJob;
  xq: array of TInferBlockQ8_0;
  xk: array of TInferBlockQ8_K;
  dx: TDoubleDynArray;
  dxs: TSingleDynArray;
  r, t, n: PtrInt;
  kq: boolean;
begin
  if not (w.DType in [idtF32, idtQ8_0, idtQ4_0, idtQ4_K, idtQ6_K]) or
     ((w.DType <> idtF32) and (aIn mod INFER_DTYPE_BLOCK[w.DType] <> 0)) then
  begin
    inherited Linear(x, aRows, aIn, w, aOut, bias, y); // raises
    exit;
  end;
  job.X := x;
  job.Y := y;
  job.Bias := bias;
  job.W := w;
  job.Rows := aRows;
  job.InCount := aIn;
  job.OutCount := aOut;
  kq := w.DType in [idtQ4_K, idtQ6_K];
  if kq then
    job.Blocks := aIn shr 8 // Q8_K blocks of 256
  else
    job.Blocks := aIn shr 5;
  if kq then
    job.Fp32 := false // the exact block sums for now
  else if w.DType <> idtF32 then // Q4_0 tiles become Q8_0: the same kernels
    job.Fp32 := not fExact and Assigned(InferQ8DotF32x4x2)
  else
    // the float row kernel needs whole lanes of 16
    job.Fp32 := not fExact and Assigned(InferDotsF32) and (aIn > 0) and
      (aIn and 15 = 0);
  n := (aOut + TILE - 1) div TILE;
  if (Int64(aRows) * aIn * aOut >= PARALLEL_MIN) and fBusy.TryLock then
  try
    // on the pool: the buffers of the backend, items shared by every thread
    if kq then
    begin
      if length(fXk) < aRows * job.Blocks then
        SetLength(fXk, aRows * job.Blocks);
      job.Xk := pointer(fXk);
    end
    else if w.DType <> idtF32 then
    begin
      if length(fXq) < aRows * job.Blocks then
      begin
        SetLength(fXq, aRows * job.Blocks);
        SetLength(fDx, aRows * job.Blocks);
        SetLength(fDxs, aRows * job.Blocks);
      end;
      job.Xq := pointer(fXq);
      job.Dx := pointer(fDx);
      job.Dxs := pointer(fDxs);
    end;
    fJob := job;
    if w.DType <> idtF32 then
      RunParallel(QuantizeRow, aRows);
    RunParallel(LinearTile, n);
    exit;
  finally
    fBusy.UnLock;
  end;
  // too small, or the pool is busy with another caller: the same kernels and
  // arithmetic on this thread, with buffers of its own
  if kq then
  begin
    SetLength(xk, aRows * job.Blocks);
    job.Xk := pointer(xk);
    for r := 0 to aRows - 1 do
      LinearQuantizeRow(job, r);
  end
  else if w.DType <> idtF32 then
  begin
    SetLength(xq, aRows * job.Blocks);
    SetLength(dx, aRows * job.Blocks);
    SetLength(dxs, aRows * job.Blocks);
    job.Xq := pointer(xq);
    job.Dx := pointer(dx);
    job.Dxs := pointer(dxs);
    for r := 0 to aRows - 1 do
      LinearQuantizeRow(job, r);
  end;
  for t := 0 to n - 1 do
    LinearTileWork(job, t);
end;

const
  /// query rows per Attention work item
  ATTENTION_ROWS = 16;

procedure TInferCpuFastBackend.AttentionRows(h, aFirst, aLast: PtrInt;
  aScratch: PSingleArray);
var
  i, j, off: PtrInt;
  scale: single;
  s, yr: PSingleArray;
begin
  off := h * fHeadDim;
  if fExact or not Assigned(InferDotsF32) or (fHeadDim <= 0) or
     (fHeadDim and 15 <> 0) then
  begin
    // exact mode, or no float kernel for this CPU or head size: the reference
    // arithmetic
    AttentionHead(fQ, fK, fV, fSeq, fHeadDim, fWidth, off, aFirst, aLast, fKeep,
      aScratch, fY);
    exit;
  end;
  // the float arithmetic of the fast path: scores and value sums in float
  // lanes (FMA), the masked softmax as in the reference
  scale := 1 / Sqrt(fHeadDim);
  for i := aFirst to aLast do
  begin
    s := @aScratch^[i * fSeq];
    InferDotsF32(@fQ^[i * fWidth + off], @fK^[off], fWidth * SizeOf(single), fSeq,
      fHeadDim, s);
    for j := 0 to fSeq - 1 do
      s^[j] := s^[j] * scale;
    SoftmaxMasked(s, 1, fSeq, fKeep);
    yr := @fY^[i * fWidth + off];
    FillCharFast(yr^, fHeadDim * SizeOf(single), 0);
    InferAxpyRowsF32(pointer(s), @fV^[off], fWidth * SizeOf(single), fSeq,
      fHeadDim, yr);
  end;
end;

procedure TInferCpuFastBackend.AttentionItem(aItem: integer);
var
  h, r: PtrInt;
begin
  // item = head * row blocks + row block: rows of one head are independent
  h := aItem div fRowBlocks;
  r := (aItem mod fRowBlocks) * ATTENTION_ROWS;
  AttentionRows(h, r, MinPtrInt(r + ATTENTION_ROWS, fSeq) - 1,
    pointer(fScratch[h]));
end;

procedure TInferCpuFastBackend.Attention(q, k, v: PSingleArray;
  aSeq, aHeads, aHeadDim: PtrInt; aKeep: PIntegerArray; aScratch, y: PSingleArray);
var
  h: PtrInt;
  busy: boolean;
begin
  busy := not fBusy.TryLock;
  if busy then
  begin
    // the pool serves another caller: the same arithmetic on this thread, with
    // the caller's scratch and a local copy of the operands
    AttentionSerial(q, k, v, aSeq, aHeads, aHeadDim, aKeep, aScratch, y);
    exit;
  end;
  try
    fQ := q;
    fK := k;
    fV := v;
    fY := y;
    fSeq := aSeq;
    fHeadDim := aHeadDim;
    fWidth := aHeads * aHeadDim;
    fKeep := aKeep;
    // one score matrix per head: heads run side by side
    if length(fScratch) < aHeads then
      SetLength(fScratch, aHeads);
    for h := 0 to aHeads - 1 do
      if length(fScratch[h]) < aSeq * aSeq then
        SetLength(fScratch[h], aSeq * aSeq);
    fRowBlocks := (aSeq + ATTENTION_ROWS - 1) div ATTENTION_ROWS;
    RunParallel(AttentionItem, aHeads * fRowBlocks);
  finally
    fBusy.UnLock;
  end;
end;

const
  /// query rows per item of a causal attention
  CAUSAL_ROWS = 8;

procedure TInferCpuFastBackend.CausalItem(aItem: integer);
var
  h, r: PtrInt;
begin
  // item = head * row blocks + row block: the rows of one head are independent
  h := aItem div fCausal.RowBlocks;
  r := (aItem mod fCausal.RowBlocks) * CAUSAL_ROWS;
  CausalRows(fCausal, h, r, MinPtrInt(r + CAUSAL_ROWS, fCausal.Rows) - 1);
end;

procedure TInferCpuFastBackend.CausalAttention(q: PSingleArray; aRows, aHeads,
  aHeadDim: PtrInt; keys, values: PSingleArray; aPast, aKvHeads: PtrInt;
  y: PSingleArray; aWindow: PtrInt);
var
  job: TInferCausalJob;
  h: PtrInt;
begin
  // split by head and row block, never inside one sum: Exact stays
  // bit-identical to the reference
  job := CausalJob(q, aRows, aHeads, aHeadDim, keys, values, aPast, aKvHeads, y,
    aWindow, not fExact);
  job.RowBlocks := (aRows + CAUSAL_ROWS - 1) div CAUSAL_ROWS;
  if fBusy.TryLock then
  try
    fCausal := job;
    RunParallel(CausalItem, aHeads * job.RowBlocks);
    exit;
  finally
    fBusy.UnLock;
  end;
  // the pool serves another caller: the same arithmetic on this thread
  for h := 0 to aHeads - 1 do
    CausalRows(job, h, 0, aRows - 1);
end;

procedure TInferCpuFastBackend.AttentionSerial(q, k, v: PSingleArray;
  aSeq, aHeads, aHeadDim: PtrInt; aKeep: PIntegerArray; aScratch, y: PSingleArray);
var
  other: TInferCpuFastBackend;
  h: PtrInt;
begin
  // AttentionRows reads its operands from fields: a thread of its own needs an
  // object of its own - a pool-less instance computes on this thread
  other := TInferCpuFastBackend.Create(1);
  try
    other.fQ := q;
    other.fK := k;
    other.fV := v;
    other.fY := y;
    other.fSeq := aSeq;
    other.fHeadDim := aHeadDim;
    other.fWidth := aHeads * aHeadDim;
    other.fKeep := aKeep;
    other.fExact := fExact; // the arithmetic of this backend, not the default
    for h := 0 to aHeads - 1 do
      other.AttentionRows(h, 0, aSeq - 1, aScratch);
  finally
    other.Free;
  end;
end;

const
  /// the GELU table of the fast backend: cubic Hermite on [GELU_LO, GELU_HI]
  // with GELU_STEPS nodes per unit, value and slope of the exact GELU at each
  GELU_LO = -5;
  GELU_HI = 8;
  GELU_STEPS = 64;
  GELU_NODES = (GELU_HI - GELU_LO) * GELU_STEPS;

var
  GeluValue, GeluSlope: array[0..GELU_NODES] of double;

procedure InitGeluTable;
var
  i: PtrInt;
  x: double;
begin
  for i := 0 to GELU_NODES do
  begin
    x := GELU_LO + i / GELU_STEPS;
    // gelu(x) = x Phi(x), gelu'(x) = Phi(x) + x phi(x)
    GeluValue[i] := 0.5 * x * InferErfc(-x * INV_SQRT2);
    GeluSlope[i] := 0.5 * InferErfc(-x * INV_SQRT2) +
      x * Exp(-0.5 * x * x) * 0.39894228040143267794; // 1 / sqrt(2 pi)
  end;
end;

/// GELU from the table: absolute error below 1e-9 on [-5, 8] (cubic Hermite,
// step 1/64), the exact formula outside - x from 8 on (gelu = x in single),
// below -5 by erfc; some 3 ns a value instead of ~20
procedure GeluTable(x: PSingleArray; aCount: PtrInt);
var
  i, k: PtrInt;
  v, t, t2, t3, f: double;
const
  H = 1 / GELU_STEPS;
begin
  for i := 0 to aCount - 1 do
  begin
    // a NaN or an infinity: the reference formula, never a table index
    if PCardinal(@x^[i])^ and $7f800000 = $7f800000 then
    begin
      x^[i] := 0.5 * x^[i] * InferErfc(-x^[i] * INV_SQRT2);
      continue;
    end;
    v := x^[i];
    if v >= GELU_HI then
      continue; // x Phi(x) = x within single precision
    if v < GELU_LO then
    begin
      x^[i] := 0.5 * v * InferErfc(-v * INV_SQRT2);
      continue;
    end;
    t := (v - GELU_LO) * GELU_STEPS;
    k := Trunc(t);
    if k >= GELU_NODES then
      k := GELU_NODES - 1;
    t := t - k;
    t2 := t * t;
    t3 := t2 * t;
    f := (2 * t3 - 3 * t2 + 1) * GeluValue[k] + (t3 - 2 * t2 + t) * H * GeluSlope[k] +
      (3 * t2 - 2 * t3) * GeluValue[k + 1] + (t3 - t2) * H * GeluSlope[k + 1];
    x^[i] := f;
  end;
end;

procedure TInferCpuFastBackend.GeluRange(aItem: integer);
var
  first, last: PtrInt;
begin
  // work item i is the slice [i * 4096, (i + 1) * 4096) of the buffer
  first := PtrInt(aItem) shl 12;
  last := MinPtrInt(PtrInt(aItem + 1) shl 12, fRows);
  if fExact then
    inherited Gelu(@fX^[first], last - first)
  else
    GeluTable(@fX^[first], last - first);
end;

procedure TInferCpuFastBackend.Gelu(x: PSingleArray; aCount: PtrInt);
begin
  if (aCount < 8192) or not fBusy.TryLock then
  begin
    // the same arithmetic on this thread
    if fExact then
      inherited Gelu(x, aCount)
    else
      GeluTable(x, aCount);
    exit;
  end;
  try
    fX := x;
    fRows := aCount;
    RunParallel(GeluRange, (aCount + 4095) shr 12);
  finally
    fBusy.UnLock;
  end;
end;

const
  /// rows per LayerNorm work item
  LAYERNORM_ROWS = 2;

procedure TInferCpuFastBackend.LayerNormRows(aItem: integer);
var
  first: PtrInt;
begin
  // the reference arithmetic, each row on one thread: bit-identical
  first := PtrInt(aItem) * LAYERNORM_ROWS;
  inherited LayerNorm(@fX^[first * fCols], MinPtrInt(LAYERNORM_ROWS, fRows - first),
    fCols, fGamma, fBeta, fEps, @fY^[first * fCols]);
end;

procedure TInferCpuFastBackend.LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
  gamma, beta: PSingleArray; aEps: double; y: PSingleArray);
begin
  if (aRows < 2 * LAYERNORM_ROWS) or (aRows * aCols < 8192) or
     not fBusy.TryLock then
  begin
    inherited LayerNorm(x, aRows, aCols, gamma, beta, aEps, y);
    exit;
  end;
  try
    fX := x;
    fY := y;
    fRows := aRows;
    fCols := aCols;
    fGamma := gamma;
    fBeta := beta;
    fEps := aEps;
    RunParallel(LayerNormRows, (aRows + LAYERNORM_ROWS - 1) div LAYERNORM_ROWS);
  finally
    fBusy.UnLock;
  end;
end;

const
  /// rows per RmsNorm work item
  RMSNORM_ROWS = 4;
  /// elements per SwiGlu work item: a decoded token has a few of them
  SWIGLU_ITEM = 1024;

procedure TInferCpuFastBackend.RmsNormRows(aItem: integer);
var
  first: PtrInt;
begin
  // the reference arithmetic, each row on one thread: bit-identical
  first := PtrInt(aItem) * RMSNORM_ROWS;
  inherited RmsNorm(@fX^[first * fCols], MinPtrInt(RMSNORM_ROWS, fRows - first),
    fCols, fGamma, fEps, @fY^[first * fCols]);
end;

procedure TInferCpuFastBackend.RmsNorm(x: PSingleArray; aRows, aCols: PtrInt;
  w: PSingleArray; aEps: double; y: PSingleArray);
begin
  if (aRows < 2 * RMSNORM_ROWS) or (aRows * aCols < 8192) or
     not fBusy.TryLock then
  begin
    inherited RmsNorm(x, aRows, aCols, w, aEps, y);
    exit;
  end;
  try
    fX := x;
    fY := y;
    fRows := aRows;
    fCols := aCols;
    fGamma := w;
    fEps := aEps;
    RunParallel(RmsNormRows, (aRows + RMSNORM_ROWS - 1) div RMSNORM_ROWS);
  finally
    fBusy.UnLock;
  end;
end;

procedure TInferCpuFastBackend.SwiGluRange(aItem: integer);
var
  first: PtrInt;
begin
  // the reference arithmetic on a slice: bit-identical
  first := PtrInt(aItem) * SWIGLU_ITEM;
  inherited SwiGlu(@fX^[first], @fY^[first], MinPtrInt(SWIGLU_ITEM, fRows - first));
end;

procedure TInferCpuFastBackend.SwiGlu(gate, up: PSingleArray; aCount: PtrInt);
begin
  if (aCount < 2 * SWIGLU_ITEM) or not fBusy.TryLock then
  begin
    inherited SwiGlu(gate, up, aCount);
    exit;
  end;
  try
    fX := gate;
    fY := up;
    fRows := aCount;
    RunParallel(SwiGluRange, (aCount + SWIGLU_ITEM - 1) div SWIGLU_ITEM);
  finally
    fBusy.UnLock;
  end;
end;

const
  /// rows per Rope work item
  ROPE_ROWS = 4;

procedure TInferCpuFastBackend.RopeRows(aItem: integer);
var
  first, half: PtrInt;
begin
  // the reference arithmetic, each row on one thread: bit-identical
  first := PtrInt(aItem) * ROPE_ROWS;
  half := fHeadDim shr 1;
  inherited Rope(@fX^[first * fWidth * fHeadDim], MinPtrInt(ROPE_ROWS, fRows - first),
    fWidth, fHeadDim, @fGamma^[first * half], @fBeta^[first * half]);
end;

procedure TInferCpuFastBackend.Rope(x: PSingleArray; aRows, aHeads,
  aHeadDim: PtrInt; aCos, aSin: PSingleArray);
begin
  if (aRows < 2 * ROPE_ROWS) or not fBusy.TryLock then
  begin
    inherited Rope(x, aRows, aHeads, aHeadDim, aCos, aSin);
    exit;
  end;
  try
    fX := x;
    fRows := aRows;
    fWidth := aHeads;
    fHeadDim := aHeadDim;
    fGamma := aCos;
    fBeta := aSin;
    RunParallel(RopeRows, (aRows + ROPE_ROWS - 1) div ROPE_ROWS);
  finally
    fBusy.UnLock;
  end;
end;

initialization
  InitGeluTable;

end.
