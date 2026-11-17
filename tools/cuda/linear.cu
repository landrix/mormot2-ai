// LandrixAI - CUDA: y[r, o] = bias[o] + sum_i w[o, i] * x[r, i] (backlog B3,
// slice C2) - the CUDA form of tools/vk/linear.comp
// - w in its storage type: F32, Q8_0, Q4_K, Q6_K, Q5_0, MXFP4, Q4_0 or Q5_1 (ggml block layouts, see
//   mormot.ai.infer.tensor), dequantized per element in float, x in F32 - the
//   arithmetic of InferRowToF32 followed by an F32 dot product, in another
//   summation order (a tolerance, not bits)
// - one block of 128 threads per output feature (grid x) and token row
//   (grid y): correct first, the speed comes with slice C5 (linear8, gemm8
//   below) and backlog B20 (matvec8, one token row)
// - the half scales sit at even byte offsets in every block layout: a 16-bit
//   load is aligned; converted by cvt.f32.f16 (no cuda_fp16.h: it needs headers
//   the nvcc wheel does not carry)

// dtype: 1 = F32, 2 = Q8_0, 3 = Q4_K, 4 = Q6_K, 5 = Q5_0, 6 = MXFP4, 7 = Q4_0,
// 8 = Q5_1
// (TInferCudaBackend)

// MXFP4 (slice D5d): 17 bytes per 32 values - e (E8M0), qs[16]; value j is
// kvalues[q] * 2^(e - 128) with q the low nibble of qs[j] below 16, the high
// one from 16: ggml's block_mxfp4 (kvalues_mxfp4, the E2M1 values doubled,
// and E8M0_TO_FP32_HALF) - every product exact in float
__constant__ float KV_MXFP4[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f,
  12.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};

__device__ __forceinline__ float e8m0Half(unsigned int e)
{
  // 2^(e - 128): a subnormal for e 0 and 1
  return __int_as_float(e < 2u ? 0x00200000u << e : (e - 1u) << 23);
}

__device__ __forceinline__ float halfAt(const unsigned char* p)
{
  unsigned short h = *reinterpret_cast<const unsigned short*>(p);
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

// the weight of input i in the row at byte row
__device__ float weight(const unsigned char* row, unsigned int dtype, unsigned int i)
{
  if (dtype == 1u)
    return reinterpret_cast<const float*>(row)[i];
  if (dtype == 2u)
  {
    // Q8_0: 34 bytes per 32 values, value = d * qs[j]
    const unsigned char* b = row + (i >> 5) * 34u;
    return halfAt(b) * float(static_cast<signed char>(b[2u + (i & 31u)]));
  }
  if (dtype == 3u)
  {
    // Q4_K: 144 bytes per 256 values (InferScaleMinK4, DequantQ4K)
    const unsigned char* b = row + (i >> 8) * 144u;
    unsigned int k = i & 255u;
    unsigned int j = k >> 5;
    unsigned int sc, m;
    if (j < 4u)
    {
      sc = b[4u + j] & 63u;
      m = b[8u + j] & 63u;
    }
    else
    {
      sc = (b[8u + j] & 15u) | ((b[j] >> 6) << 4);
      m = (b[8u + j] >> 4) | ((b[4u + j] >> 6) << 4);
    }
    unsigned int q = b[16u + 32u * (j >> 1) + (k & 31u)];
    q = (j & 1u) ? q >> 4 : q & 15u;
    float d1 = halfAt(b) * float(sc);
    float m1 = halfAt(b + 2u) * float(m);
    return d1 * float(q) - m1;
  }
  if (dtype == 6u)
  {
    // MXFP4: 17 bytes per 32 values
    const unsigned char* b = row + (i >> 5) * 17u;
    unsigned int j = i & 31u;
    unsigned int q = b[1u + (j & 15u)];
    q = j >= 16u ? q >> 4 : q & 15u;
    return e8m0Half(b[0]) * KV_MXFP4[q];
  }
  if (dtype == 5u)
  {
    // Q5_0: 22 bytes per 32 values - d, qh[4], qs[16]; value = d * (q - 16)
    // with the low 4 bits of value j in qs[j & 15] (low nibble below 16,
    // high from 16) and its fifth bit bit j of qh
    const unsigned char* b = row + (i >> 5) * 22u;
    unsigned int j = i & 31u;
    unsigned int q = b[6u + (j & 15u)];
    q = j >= 16u ? q >> 4 : q & 15u;
    q |= ((b[2u + (j >> 3)] >> (j & 7u)) & 1u) << 4;
    return halfAt(b) * float(int(q) - 16);
  }
  if (dtype == 7u)
  {
    // Q4_0: 18 bytes per 32 values - d, qs[16]; value = d * (q - 8) with q
    // the low nibble of qs[j & 15] below 16, the high one from 16
    const unsigned char* b = row + (i >> 5) * 18u;
    unsigned int j = i & 31u;
    unsigned int q = b[2u + (j & 15u)];
    q = j >= 16u ? q >> 4 : q & 15u;
    return halfAt(b) * float(int(q) - 8);
  }
  if (dtype == 8u)
  {
    // Q5_1: 24 bytes per 32 values - d, m, qh[4], qs[16]; value = d * q + m
    // with q (0..31) packed as in Q5_0
    const unsigned char* b = row + (i >> 5) * 24u;
    unsigned int j = i & 31u;
    unsigned int q = b[8u + (j & 15u)];
    q = j >= 16u ? q >> 4 : q & 15u;
    q |= ((b[4u + (j >> 3)] >> (j & 7u)) & 1u) << 4;
    return halfAt(b) * float(q) + halfAt(b + 2u);
  }
  // Q6_K: 210 bytes per 256 values - ql[128], qh[64], scales[16], d (Q6KValue)
  const unsigned char* b = row + (i >> 8) * 210u;
  unsigned int k = i & 255u;
  unsigned int n = k >> 7;
  unsigned int ii = k & 127u;
  unsigned int l = ii & 31u;
  unsigned int quarter = ii >> 5;
  unsigned int h = (b[128u + 32u * n + l] >> (2u * quarter)) & 3u;
  unsigned int ql = b[64u * n + l + ((quarter & 1u) ? 32u : 0u)];
  ql = quarter >= 2u ? ql >> 4 : ql & 15u;
  int q = int(ql | (h << 4)) - 32;
  float t = halfAt(b + 208u) * float(static_cast<signed char>(b[192u + (k >> 4)]));
  return t * float(q);
}

extern "C" __global__ void linear(const unsigned char* w, const float* x,
  const float* bias, float* y, unsigned int rows, unsigned int nin,
  unsigned int nout, unsigned int dtype, unsigned int hasBias)
{
  __shared__ float partial[4];
  unsigned int o = blockIdx.x;
  unsigned int r = blockIdx.y;
  unsigned int t = threadIdx.x;
  // bytes of one weight row
  unsigned long long rowBytes;
  if (dtype == 1u)
    rowBytes = nin * 4ull;
  else if (dtype == 2u)
    rowBytes = (nin >> 5) * 34ull;
  else if (dtype == 3u)
    rowBytes = (nin >> 8) * 144ull;
  else if (dtype == 5u)
    rowBytes = (nin >> 5) * 22ull;
  else if (dtype == 6u)
    rowBytes = (nin >> 5) * 17ull;
  else if (dtype == 7u)
    rowBytes = (nin >> 5) * 18ull;
  else if (dtype == 8u)
    rowBytes = (nin >> 5) * 24ull;
  else
    rowBytes = (nin >> 8) * 210ull;
  const unsigned char* row = w + o * rowBytes;
  const float* xr = x + (unsigned long long)r * nin;
  float s = 0.0f;
  for (unsigned int i = t; i < nin; i += 128u)
    s += weight(row, dtype, i) * xr[i];
  // within the warp, then the four warps
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    s += __shfl_down_sync(0xffffffffu, s, off);
  if ((t & 31u) == 0u)
    partial[t >> 5] = s;
  __syncthreads();
  if (t == 0u)
  {
    float sum = partial[0] + partial[1] + partial[2] + partial[3];
    if (hasBias)
      sum += bias[o];
    y[(unsigned long long)r * nout + o] = sum;
  }
}

// ---- slice C5: the fast forms, CUDA ports of tools/vk/linear8.comp and
// gemm8.comp - same parameters and arithmetic contract as linear (the
// weights dequantized as InferRowToF32), another summation order
// - the weights read as 32-bit words: a weight buffer starts 4-aligned

// 8 bytes from byte address a (any alignment) into two words, little-endian
__device__ __forceinline__ uint2 bytes8(const unsigned int* w, unsigned long long a)
{
  unsigned int s = (unsigned int)(a & 3u) * 8u;
  unsigned long long i = a >> 2;
  unsigned int w0 = w[i];
  unsigned int w1 = w[i + 1u];
  if (s == 0u)
    return make_uint2(w0, w1);
  unsigned int w2 = w[i + 2u];
  return make_uint2((w0 >> s) | (w1 << (32u - s)), (w1 >> s) | (w2 << (32u - s)));
}

__device__ __forceinline__ unsigned int byteAt(const unsigned int* w, unsigned long long a)
{
  return (w[a >> 2] >> ((unsigned int)(a & 3u) * 8u)) & 255u;
}

__device__ __forceinline__ float halfBits(unsigned int h)
{
  float f;
  unsigned short hs = (unsigned short)h;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(hs));
  return f;
}

__device__ __forceinline__ float halfWord(const unsigned int* w, unsigned long long a)
{
  return halfBits(byteAt(w, a) | (byteAt(w, a + 1u) << 8));
}

// signed byte k (0..3) of v
__device__ __forceinline__ float s8(unsigned int v, unsigned int k)
{
  return float((int)(v << (24u - 8u * k)) >> 24);
}

// unsigned byte k (0..3) of v
__device__ __forceinline__ unsigned int u8(unsigned int v, unsigned int k)
{
  return (v >> (k * 8u)) & 255u;
}

// the bytes of one weight row
__device__ __forceinline__ unsigned long long rowBytesOf(unsigned int dtype,
  unsigned int nin)
{
  if (dtype == 1u)
    return nin * 4ull;
  if (dtype == 2u)
    return (nin >> 5) * 34ull;
  if (dtype == 3u)
    return (nin >> 8) * 144ull;
  if (dtype == 5u)
    return (nin >> 5) * 22ull;
  if (dtype == 6u)
    return (nin >> 5) * 17ull;
  if (dtype == 7u)
    return (nin >> 5) * 18ull;
  if (dtype == 8u)
    return (nin >> 5) * 24ull;
  return (nin >> 8) * 210ull;
}

// the 8 weights e .. e + 7 of the row that starts at byte rowBase (weights8
// of linear8.comp)
__device__ __forceinline__ void weights8(const unsigned int* w, unsigned int dtype,
  unsigned long long rowBase, unsigned int e, float* wv)
{
  if (dtype == 1u)
  {
    const float* f = reinterpret_cast<const float*>(w) + (rowBase >> 2) + e;
#pragma unroll
    for (unsigned int k = 0u; k < 8u; k++)
      wv[k] = f[k];
  }
  else if (dtype == 2u)
  {
    // Q8_0: 34 bytes per 32 values, value = d * qs[j]; 8 values in one block
    unsigned long long base = rowBase + (e >> 5) * 34ull;
    float d = halfWord(w, base);
    uint2 q = bytes8(w, base + 2u + (e & 31u));
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      wv[k] = d * s8(q.x, k);
      wv[k + 4u] = d * s8(q.y, k);
    }
  }
  else if (dtype == 3u)
  {
    // Q4_K: 144 bytes per 256 values (InferScaleMinK4, DequantQ4K); the 8
    // values lie in sub-block j
    // - a row is a whole number of 144-byte blocks and starts 4-aligned (as
    //   an expert at ids * stride, stride a multiple of 4), so the block is
    //   read as words: d and dmin in word 0, the 12 scale bytes in words 1..3,
    //   the nibbles from word 4 - one 64-bit address per block (backlog B19)
    const unsigned int* blk = w + ((rowBase + (e >> 8) * 144ull) >> 2);
    unsigned int k8 = e & 255u;
    unsigned int j = k8 >> 5;
    unsigned int h0 = blk[0];
    // byte j & 3 of the scale words: bytes 4 + (j & 3), 8 + (j & 3) and
    // 12 + (j & 3) of the block
    unsigned int s1 = u8(blk[1], j & 3u);
    unsigned int s2 = u8(blk[2], j & 3u);
    unsigned int sc, m;
    if (j < 4u)
    {
      sc = s1 & 63u;
      m = s2 & 63u;
    }
    else
    {
      unsigned int s3 = u8(blk[3], j & 3u);
      sc = (s3 & 15u) | ((s1 >> 6) << 4);
      m = (s3 >> 4) | ((s2 >> 6) << 4);
    }
    float d1 = halfBits(h0 & 65535u) * float(sc);
    float m1 = halfBits(h0 >> 16) * float(m);
    // (16 + 32 * (j / 2) + (k8 & 31)) / 4: k8 & 31 a multiple of 8
    const unsigned int* qw = blk + 4u + 8u * (j >> 1) + ((k8 & 31u) >> 2);
    unsigned int q0 = qw[0];
    unsigned int q1 = qw[1];
    unsigned int sh = (j & 1u) * 4u;
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      wv[k] = d1 * float((u8(q0, k) >> sh) & 15u) - m1;
      wv[k + 4u] = d1 * float((u8(q1, k) >> sh) & 15u) - m1;
    }
  }
  else if (dtype == 6u)
  {
    // MXFP4: 17 bytes per 32 values, 8 values in one block: 8 bytes of qs
    unsigned long long base = rowBase + (e >> 5) * 17ull;
    unsigned int j = e & 31u;
    float d = e8m0Half(byteAt(w, base));
    uint2 q = bytes8(w, base + 1u + (j & 15u));
    unsigned int sh = j >= 16u ? 4u : 0u;
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      wv[k] = d * KV_MXFP4[(u8(q.x, k) >> sh) & 15u];
      wv[k + 4u] = d * KV_MXFP4[(u8(q.y, k) >> sh) & 15u];
    }
  }
  else if (dtype == 5u)
  {
    // Q5_0: 22 bytes per 32 values - d, qh[4], qs[16]; value = d * (q - 16)
    // with the low 4 bits of value j in qs[j & 15] (low nibble below 16,
    // high from 16) and its fifth bit bit j of qh; 8 values in one block -
    // the 8 fifth bits are one byte of qh (as linear8.comp, where the
    // Adreno X1-85 driver got the whole qh word shifted by j wrong)
    unsigned long long base = rowBase + (e >> 5) * 22ull;
    unsigned int j = e & 31u;
    float d = halfWord(w, base);
    unsigned int hb = byteAt(w, base + 2u + (j >> 3));
    uint2 q = bytes8(w, base + 6u + (j & 15u));
    unsigned int sh = j >= 16u ? 4u : 0u;
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      wv[k] = d * float(int(((u8(q.x, k) >> sh) & 15u) | (((hb >> k) & 1u) << 4)) - 16);
      wv[k + 4u] = d * float(int(((u8(q.y, k) >> sh) & 15u) |
        (((hb >> (k + 4u)) & 1u) << 4)) - 16);
    }
  }
  else if (dtype == 7u)
  {
    // Q4_0: 18 bytes per 32 values - d, qs[16]; value = d * (q - 8), the low
    // nibble of qs[j & 15] below 16, the high one from 16; a block starts at
    // an even byte (bytes8 takes any alignment)
    unsigned long long base = rowBase + (e >> 5) * 18ull;
    unsigned int j = e & 31u;
    float d = halfWord(w, base);
    uint2 q = bytes8(w, base + 2u + (j & 15u));
    unsigned int sh = j >= 16u ? 4u : 0u;
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      wv[k] = d * float(int((u8(q.x, k) >> sh) & 15u) - 8);
      wv[k + 4u] = d * float(int((u8(q.y, k) >> sh) & 15u) - 8);
    }
  }
  else if (dtype == 8u)
  {
    // Q5_1: 24 bytes per 32 values - d, m, qh[4], qs[16]; value = d * q + m,
    // q packed as in Q5_0 (the 8 fifth bits one byte of qh)
    unsigned long long base = rowBase + (e >> 5) * 24ull;
    unsigned int j = e & 31u;
    float d = halfWord(w, base);
    float m = halfWord(w, base + 2u);
    unsigned int hb = byteAt(w, base + 4u + (j >> 3));
    uint2 q = bytes8(w, base + 8u + (j & 15u));
    unsigned int sh = j >= 16u ? 4u : 0u;
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      wv[k] = d * float(((u8(q.x, k) >> sh) & 15u) | (((hb >> k) & 1u) << 4)) + m;
      wv[k + 4u] = d * float(((u8(q.y, k) >> sh) & 15u) |
        (((hb >> (k + 4u)) & 1u) << 4)) + m;
    }
  }
  else
  {
    // Q6_K: 210 bytes per 256 values - ql[128], qh[64], scales[16], d
    // (Q6KValue); the 8 values share one scale (a group of 16)
    unsigned long long base = rowBase + (e >> 8) * 210ull;
    unsigned int k8 = e & 255u;
    unsigned int n = k8 >> 7;
    unsigned int ii = k8 & 127u;
    unsigned int l = ii & 31u;
    unsigned int quarter = ii >> 5;
    uint2 ql = bytes8(w, base + 64u * n + l + ((quarter & 1u) != 0u ? 32u : 0u));
    uint2 qh = bytes8(w, base + 128u + 32u * n + l);
    unsigned int lsh = quarter >= 2u ? 4u : 0u;
    unsigned int hsh = 2u * quarter;
    float t = halfWord(w, base + 208u) *
      float((int)(byteAt(w, base + 192u + (k8 >> 4)) << 24) >> 24);
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      int a = int(((u8(ql.x, k) >> lsh) & 15u) | (((u8(qh.x, k) >> hsh) & 3u) << 4)) - 32;
      int b = int(((u8(ql.y, k) >> lsh) & 15u) | (((u8(qh.y, k) >> hsh) & 3u) << 4)) - 32;
      wv[k] = t * float(a);
      wv[k + 4u] = t * float(b);
    }
  }
}

#define ROWS8 8

// linear8: a block of 128 = 4 output features (one per warp, grid x) x 32
// lanes, up to ROWS8 token rows (grid y); each lane dequantizes 8 consecutive
// weights per step of 256 inputs and multiplies them with every row, the 32
// partial sums of a feature added by warp shuffles
// - nin a multiple of 8 (F32) / 32 (Q8_0) / 256 (K-quants); x 16-byte aligned
extern "C" __global__ void linear8(const unsigned int* w, const float* x,
  const float* bias, float* y, unsigned int rows, unsigned int nin,
  unsigned int nout, unsigned int dtype, unsigned int hasBias)
{
  unsigned int t = threadIdx.x;
  unsigned int lane = t & 31u;
  unsigned int o = blockIdx.x * 4u + (t >> 5);
  unsigned int r0 = blockIdx.y * ROWS8;
  unsigned int nr = min((unsigned int)ROWS8, rows - r0);
  if (o >= nout)
    return; // a whole warp: the shuffles below stay within live warps
  unsigned long long rowBase = o * rowBytesOf(dtype, nin);
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc[ROWS8];
#pragma unroll
  for (unsigned int r = 0u; r < ROWS8; r++)
    acc[r] = 0.0f;
  for (unsigned int e = lane * 8u; e < nin; e += 256u)
  {
    float wv[8];
    weights8(w, dtype, rowBase, e, wv);
#pragma unroll
    for (unsigned int r = 0u; r < ROWS8; r++)
      if (r < nr)
      {
        unsigned long long xi = ((unsigned long long)(r0 + r) * nin + e) >> 2;
        float4 xa = x4[xi];
        float4 xb = x4[xi + 1u];
        acc[r] += wv[0] * xa.x + wv[1] * xa.y + wv[2] * xa.z + wv[3] * xa.w +
                  wv[4] * xb.x + wv[5] * xb.y + wv[6] * xb.z + wv[7] * xb.w;
      }
  }
  float b = 0.0f;
  if (hasBias)
    b = bias[o];
#pragma unroll
  for (unsigned int r = 0u; r < ROWS8; r++)
    if (r < nr)
    {
      float v = acc[r];
      for (unsigned int off = 16u; off > 0u; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
      if (lane == 0u)
        y[(unsigned long long)(r0 + r) * nout + o] = v + b;
    }
}

// linear8ids (slice D3c): linear8 for one token row over the k experts a
// route kernel chose - slot s (grid y) reads its expert id from ids[s], the
// expert's matrix at byte ids[s] * stride of w (the experts of a layer in one
// buffer); x is row s of x when xPerSlot (the down projection), else row 0
// (gate and up); y is row s of y. The arithmetic of linear8 for one row.
// - bias: [experts, nout] floats, the expert's row added (gpt-oss, slice
//   D5d); 0 for none
extern "C" __global__ void linear8ids(const unsigned int* w, const float* x,
  const int* ids, float* y, unsigned long long stride, unsigned int nin,
  unsigned int nout, unsigned int dtype, unsigned int xPerSlot,
  const float* bias)
{
  unsigned int t = threadIdx.x;
  unsigned int lane = t & 31u;
  unsigned int o = blockIdx.x * 4u + (t >> 5);
  unsigned int slot = blockIdx.y;
  if (o >= nout)
    return; // a whole warp, as in linear8
  unsigned long long rowBase = (unsigned long long)ids[slot] * stride +
    o * rowBytesOf(dtype, nin);
  const float4* x4 = reinterpret_cast<const float4*>(x +
    (xPerSlot ? (unsigned long long)slot * nin : 0ull));
  float acc = 0.0f;
  for (unsigned int e = lane * 8u; e < nin; e += 256u)
  {
    float wv[8];
    weights8(w, dtype, rowBase, e, wv);
    float4 xa = x4[e >> 2];
    float4 xb = x4[(e >> 2) + 1u];
    acc += wv[0] * xa.x + wv[1] * xa.y + wv[2] * xa.z + wv[3] * xa.w +
           wv[4] * xb.x + wv[5] * xb.y + wv[6] * xb.z + wv[7] * xb.w;
  }
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0u)
  {
    float b = 0.0f;
    if (bias != 0)
      b = bias[(unsigned long long)ids[slot] * nout + o];
    y[(unsigned long long)slot * nout + o] = acc + b;
  }
}

#define TM 64
#define TN 64
#define TK 32

// gemm8: many token rows (prefill) - a block of 256 computes a tile of TM
// rows (grid y) x TN features (grid x): per step of TK inputs it stages the
// activations and the dequantized weights in shared memory (16 KB, as float4
// of 4 rows / 4 features), then each thread adds 4 rows x 4 features
// - nin a multiple of 32 (F32, Q8_0) / 256 (K-quants); x 16-byte aligned
extern "C" __global__ void gemm8(const unsigned int* w, const float* x,
  const float* bias, float* y, unsigned int rows, unsigned int nin,
  unsigned int nout, unsigned int dtype, unsigned int hasBias)
{
  // [k][4 rows or 4 features]: xs[k * 16 + j] holds rows 4j .. 4j + 3
  __shared__ float4 xs[TK * (TM / 4)];
  __shared__ float4 ws[TK * (TN / 4)];
  unsigned int t = threadIdx.x;
  unsigned int r0 = blockIdx.y * TM;
  unsigned int o0 = blockIdx.x * TN;
  unsigned long long rowBytes = rowBytesOf(dtype, nin);
  const float4* x4 = reinterpret_cast<const float4*>(x);
  // staging: thread t loads 8 consecutive inputs of row / feature t / 4
  unsigned int ls = t >> 2;
  unsigned int lk = (t & 3u) * 8u;
  // compute: features 4 tx .. 4 tx + 3 of rows 4 ty .. 4 ty + 3
  unsigned int tx = t & 15u;
  unsigned int ty = t >> 4;
  float4 acc[4];
#pragma unroll
  for (unsigned int i = 0u; i < 4u; i++)
    acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float* xsf = reinterpret_cast<float*>(xs);
  float* wsf = reinterpret_cast<float*>(ws);
  for (unsigned int k0 = 0u; k0 < nin; k0 += TK)
  {
    // the activations: zero beyond the last row
    unsigned int r = r0 + ls;
    float4 xa = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 xb = xa;
    if (r < rows)
    {
      unsigned long long xi = ((unsigned long long)r * nin + k0 + lk) >> 2;
      xa = x4[xi];
      xb = x4[xi + 1u];
    }
    // the weights, dequantized: zero beyond the last feature
    float wv[8];
    unsigned int o = o0 + ls;
    if (o < nout)
      weights8(w, dtype, o * rowBytes, k0 + lk, wv);
    else
#pragma unroll
      for (unsigned int k = 0u; k < 8u; k++)
        wv[k] = 0.0f;
    // lane ls & 3 of the float4 at [k][ls / 4]
    unsigned int j = ls >> 2;
    unsigned int c = ls & 3u;
    float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    __syncthreads(); // the previous step has read xs / ws
#pragma unroll
    for (unsigned int k = 0u; k < 8u; k++)
    {
      xsf[((lk + k) * (TM / 4) + j) * 4u + c] = xv[k];
      wsf[((lk + k) * (TN / 4) + j) * 4u + c] = wv[k];
    }
    __syncthreads();
#pragma unroll 8
    for (unsigned int k = 0u; k < TK; k++)
    {
      float4 a = xs[k * (TM / 4) + ty];
      float4 b = ws[k * (TN / 4) + tx];
      acc[0].x += a.x * b.x; acc[0].y += a.x * b.y; acc[0].z += a.x * b.z; acc[0].w += a.x * b.w;
      acc[1].x += a.y * b.x; acc[1].y += a.y * b.y; acc[1].z += a.y * b.z; acc[1].w += a.y * b.w;
      acc[2].x += a.z * b.x; acc[2].y += a.z * b.y; acc[2].z += a.z * b.z; acc[2].w += a.z * b.w;
      acc[3].x += a.w * b.x; acc[3].y += a.w * b.y; acc[3].z += a.w * b.z; acc[3].w += a.w * b.w;
    }
  }
#pragma unroll
  for (unsigned int i = 0u; i < 4u; i++)
  {
    unsigned int r = r0 + ty * 4u + i;
    if (r < rows)
    {
      float v[4] = {acc[i].x, acc[i].y, acc[i].z, acc[i].w};
#pragma unroll
      for (unsigned int c = 0u; c < 4u; c++)
      {
        unsigned int o = o0 + tx * 4u + c;
        if (o < nout)
        {
          float b = 0.0f;
          if (hasBias)
            b = bias[o];
          y[(unsigned long long)r * nout + o] = v[c] + b;
        }
      }
    }
  }
}

// ---- backlog B20: matvec8, the CUDA port of tools/vk/matvec8.comp - one
// token row (decoding) of Q8_0, Q4_K or Q6_K weights, same parameters as
// linear8 (rows is 1), the dequantization of InferRowToF32, another summation
// order
// - a block of 128 = 4 warps, FEAT consecutive output features per warp; a
//   lane takes units of 32 weights - a Q8_0 block, a quarter of a Q6_K half,
//   both nibbles of 16 Q4_K quant bytes (16 values of two sub-blocks, every
//   byte read once) - reading their scales once and their quants as whole
//   words, the activations of a unit loaded once for the FEAT
//   features
// - a row starts 2-aligned (Q8_0, Q6_K) or 16-aligned (Q4_K, read as 16-byte
//   words: blocks of 144 bytes, and LinearShape checks the weight address):
//   rows of whole blocks, an expert at a multiple of its matrix

// the 4 bytes from bit sh (0 or 16) of word i on (wordAt of matvec8.comp)
__device__ __forceinline__ unsigned int wordAtMv(const unsigned int* w,
  unsigned long long i, unsigned int sh)
{
  return (w[i] >> sh) | ((w[i + (sh >> 4)] << 16) << (16u - sh));
}

// the 4 bytes at the even byte address a as a word
__device__ __forceinline__ unsigned int word2(const unsigned int* w,
  unsigned long long a)
{
  return wordAtMv(w, a >> 2, (unsigned int)(a & 2u) * 8u);
}

// the 2 bytes at the even byte address a (never past their own word)
__device__ __forceinline__ unsigned int half16(const unsigned int* w,
  unsigned long long a)
{
  return (w[a >> 2] >> ((unsigned int)(a & 2u) * 8u)) & 0xffffu;
}

// byte k of the little-endian words s0 s1 s2
__device__ __forceinline__ unsigned int byte12(unsigned int s0, unsigned int s1,
  unsigned int s2, unsigned int k)
{
  unsigned int v = k < 4u ? s0 : (k < 8u ? s1 : s2);
  return (v >> ((k & 3u) * 8u)) & 255u;
}

// the 4 signed bytes of v times a
__device__ __forceinline__ float dot4s(unsigned int v, float4 a)
{
  return s8(v, 0u) * a.x + s8(v, 1u) * a.y + s8(v, 2u) * a.z + s8(v, 3u) * a.w;
}

// the 4 nibbles at bit sh of the bytes of v times a
__device__ __forceinline__ float dot4n(unsigned int v, unsigned int sh, float4 a)
{
  return float((v >> sh) & 15u) * a.x + float((v >> (sh + 8u)) & 15u) * a.y +
         float((v >> (sh + 16u)) & 15u) * a.z + float((v >> (sh + 24u)) & 15u) * a.w;
}

// the 6-bit scale and min of sub-block j of a Q4_K block whose 12 scale
// bytes are the words s0 s1 s2 (InferScaleMinK4)
__device__ __forceinline__ void scaleMinK4(unsigned int s0, unsigned int s1,
  unsigned int s2, unsigned int j, unsigned int& sc, unsigned int& m)
{
  if (j < 4u)
  {
    sc = byte12(s0, s1, s2, j) & 63u;
    m = byte12(s0, s1, s2, j + 4u) & 63u;
  }
  else
  {
    sc = (byte12(s0, s1, s2, j + 4u) & 15u) |
         ((byte12(s0, s1, s2, j - 4u) >> 6) << 4);
    m = (byte12(s0, s1, s2, j + 4u) >> 4) | ((byte12(s0, s1, s2, j) >> 6) << 4);
  }
}

// the float4 of x for activations k (0..7) of unit u: 8 in a row, but for
// Q4_K the values 16 (uu % 2) .. + 15 of sub-blocks 2 (uu / 2) and 2 (uu / 2)
// + 1 (uu = u % 8)
template <unsigned int DT>
__device__ __forceinline__ unsigned int xAt(unsigned int u, unsigned int k)
{
  if (DT == 3u)
    return (u >> 3) * 64u + ((u >> 1) & 3u) * 16u + (u & 1u) * 4u + (k & 3u) +
           (k >> 2) * 8u;
  return u * 8u + k;
}

// unit u (32 weights) of the row at byte rowBase times its activations xv
template <unsigned int DT>
__device__ __forceinline__ float unitDot(const unsigned int* w,
  unsigned long long rowBase, unsigned int u, const float4* xv)
{
  float s = 0.0f;
  if (DT == 2u)
  {
    // Q8_0: 34 bytes per 32 values, value = d * qs[j]; the quants from byte
    // a + 2 on: word qi, bit qs
    unsigned long long a = rowBase + u * 34ull;
    float d = halfBits(half16(w, a));
    unsigned long long qi = (a >> 2) + ((a >> 1) & 1u);
    unsigned int qs = 16u - (unsigned int)(a & 2u) * 8u;
#pragma unroll
    for (unsigned int k = 0u; k < 8u; k++)
      s += dot4s(wordAtMv(w, qi + k, qs), xv[k]);
    return d * s;
  }
  else if (DT == 3u)
  {
    // Q4_K: 144 bytes per 256 values (InferScaleMinK4, DequantQ4K); value =
    // d * sc * q - dmin * m. Unit uu = u % 8 of block u / 8: the 16 bytes at
    // 16 + 32 * (uu / 2) + 16 * (uu % 2), both nibbles - values 16 (uu % 2) ..
    // + 15 of sub-blocks j = 2 (uu / 2) (low, xv[0..3]) and j + 1 (high,
    // xv[4..7]); every quant byte read once (xAt maps the activations)
    unsigned long long i = (rowBase + (u >> 3) * 144ull) >> 2;
    unsigned int uu = u & 7u;
    unsigned int j = uu & 6u;
    // the block header and the unit's quants as one 16-byte load each (a
    // block of 144 bytes and its rows start at a multiple of 16; LinearShape
    // takes matvec8 for Q4_K only at such a weight address) - backlog B20,
    // 5 to 15 % faster than four word loads each on an RTX 5090
    uint4 hd = __ldg(reinterpret_cast<const uint4*>(w + i));
    uint4 qv = __ldg(reinterpret_cast<const uint4*>(w + i + 4u + 4u * uu));
    unsigned int h = hd.x;
    unsigned int s0 = hd.y;
    unsigned int s1 = hd.z;
    unsigned int s2 = hd.w;
    unsigned int sc0, m0, sc1, m1;
    scaleMinK4(s0, s1, s2, j, sc0, m0);
    scaleMinK4(s0, s1, s2, j + 1u, sc1, m1);
    unsigned int qa[4] = {qv.x, qv.y, qv.z, qv.w};
    float s1v = 0.0f;
    float sx0 = 0.0f;
    float sx1 = 0.0f;
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      unsigned int v = qa[k];
      s += dot4n(v, 0u, xv[k]);
      s1v += dot4n(v, 4u, xv[k + 4u]);
      sx0 += xv[k].x + xv[k].y + xv[k].z + xv[k].w;
      sx1 += xv[k + 4u].x + xv[k + 4u].y + xv[k + 4u].z + xv[k + 4u].w;
    }
    return halfBits(h & 65535u) * (float(sc0) * s + float(sc1) * s1v) -
           halfBits(h >> 16) * (float(m0) * sx0 + float(m1) * sx1);
  }
  else
  {
    // Q6_K: 210 bytes per 256 values - ql[128], qh[64], scales[16], d
    // (Q6KValue); unit uu = u % 8 of block u / 8: the values 32 uu .. 32 uu +
    // 31 of half uu / 4, quarter uu % 4, under two scales of 16 values
    unsigned long long base = rowBase + (u >> 3) * 210ull;
    unsigned int uu = u & 7u;
    unsigned int n = uu >> 2;
    unsigned int quarter = uu & 3u;
    unsigned long long la = base + 64u * n + ((quarter & 1u) != 0u ? 32u : 0u);
    unsigned long long ha = base + 128u + 32u * n;
    unsigned int lsh = quarter >= 2u ? 4u : 0u;
    unsigned int hsh = 2u * quarter;
    unsigned int scw = half16(w, base + 192u + 2u * uu); // the two int8 scales
    float d = halfBits(half16(w, base + 208u));
    float s1 = 0.0f;
#pragma unroll
    for (unsigned int k = 0u; k < 8u; k++)
    {
      // per byte: q = (ql nibble) | (2 bits of qh) << 4, value q - 32
      unsigned int q = ((word2(w, la + 4u * k) >> lsh) & 0x0f0f0f0fu) |
                       (((word2(w, ha + 4u * k) >> hsh) & 0x03030303u) << 4);
      float4 a = xv[k];
      float v = (float(q & 255u) - 32.0f) * a.x +
                (float((q >> 8) & 255u) - 32.0f) * a.y +
                (float((q >> 16) & 255u) - 32.0f) * a.z +
                (float(q >> 24) - 32.0f) * a.w;
      if (k < 4u)
        s += v;
      else
        s1 += v;
    }
    return d * (s8(scw, 0u) * s + s8(scw, 1u) * s1);
  }
}

template <unsigned int DT, unsigned int FEAT>
__device__ __forceinline__ void matvec8Body(const unsigned int* w,
  const float* x, const float* bias, float* y, unsigned int nin,
  unsigned int nout, unsigned int hasBias)
{
  unsigned int lane = threadIdx.x & 31u;
  unsigned int o0 = (blockIdx.x * 4u + (threadIdx.x >> 5)) * FEAT;
  if (o0 >= nout)
    return; // a whole warp: the shuffles below stay within live warps
  unsigned long long rowBytes = rowBytesOf(DT, nin);
  unsigned int units = nin >> 5;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc[FEAT];
#pragma unroll
  for (unsigned int f = 0u; f < FEAT; f++)
    acc[f] = 0.0f;
  for (unsigned int u = lane; u < units; u += 32u)
  {
    float4 xv[8];
#pragma unroll
    for (unsigned int k = 0u; k < 8u; k++)
      xv[k] = x4[xAt<DT>(u, k)];
#pragma unroll
    for (unsigned int f = 0u; f < FEAT; f++)
      if (o0 + f < nout)
        acc[f] += unitDot<DT>(w, (o0 + f) * rowBytes, u, xv);
  }
#pragma unroll
  for (unsigned int f = 0u; f < FEAT; f++)
    if (o0 + f < nout) // the same for the whole warp
    {
      float v = acc[f];
      for (unsigned int off = 16u; off > 0u; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
      if (lane == 0u)
        y[o0 + f] = v + (hasBias ? bias[o0 + f] : 0.0f);
    }
}

// one kernel per weight type and FEAT (as matvec8.comp: the code of the
// other types costs registers); rows and dtype only for the parameters of
// linear8
#define MATVEC8(name, dt, feat) \
extern "C" __global__ void name(const unsigned int* w, const float* x, \
  const float* bias, float* y, unsigned int rows, unsigned int nin, \
  unsigned int nout, unsigned int dtype, unsigned int hasBias) \
{ \
  matvec8Body<dt, feat>(w, x, bias, y, nin, nout, hasBias); \
}

MATVEC8(matvec8q8, 2u, 1u)
MATVEC8(matvec8q4k, 3u, 1u)
MATVEC8(matvec8q6k, 4u, 1u)
MATVEC8(matvec8q8f2, 2u, 2u)
MATVEC8(matvec8q4kf2, 3u, 2u)
MATVEC8(matvec8q6kf2, 4u, 2u)
