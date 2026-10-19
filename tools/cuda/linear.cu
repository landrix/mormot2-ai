// LandrixAI - CUDA: y[r, o] = bias[o] + sum_i w[o, i] * x[r, i] (backlog B3,
// slice C2) - the CUDA form of tools/vk/linear.comp
// - w in its storage type: F32, Q8_0, Q4_K or Q6_K (ggml block layouts, see
//   mormot.ai.infer.tensor), dequantized per element in float, x in F32 - the
//   arithmetic of InferRowToF32 followed by an F32 dot product, in another
//   summation order (a tolerance, not bits)
// - one block of 128 threads per output feature (grid x) and token row
//   (grid y): correct first, the speed comes with slice C5
// - the half scales sit at even byte offsets in every block layout: a 16-bit
//   load is aligned; converted by cvt.f32.f16 (no cuda_fp16.h: it needs headers
//   the nvcc wheel does not carry)

// dtype: 1 = F32, 2 = Q8_0, 3 = Q4_K, 4 = Q6_K (TInferCudaBackend)

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
    unsigned long long base = rowBase + (e >> 8) * 144ull;
    unsigned int k8 = e & 255u;
    unsigned int j = k8 >> 5;
    unsigned int sc, m;
    if (j < 4u)
    {
      sc = byteAt(w, base + 4u + j) & 63u;
      m = byteAt(w, base + 8u + j) & 63u;
    }
    else
    {
      sc = (byteAt(w, base + 8u + j) & 15u) | ((byteAt(w, base + j) >> 6) << 4);
      m = (byteAt(w, base + 8u + j) >> 4) | ((byteAt(w, base + 4u + j) >> 6) << 4);
    }
    float d1 = halfWord(w, base) * float(sc);
    float m1 = halfWord(w, base + 2u) * float(m);
    // 16 + 32 * (j / 2) + (k8 & 31): a multiple of 8, the block 4-aligned
    unsigned long long qi = (base + 16u + 32u * (j >> 1) + (k8 & 31u)) >> 2;
    unsigned int q0 = w[qi];
    unsigned int q1 = w[qi + 1u];
    unsigned int sh = (j & 1u) * 4u;
#pragma unroll
    for (unsigned int k = 0u; k < 4u; k++)
    {
      wv[k] = d1 * float((u8(q0, k) >> sh) & 15u) - m1;
      wv[k + 4u] = d1 * float((u8(q1, k) >> sh) & 15u) - m1;
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
extern "C" __global__ void linear8ids(const unsigned int* w, const float* x,
  const int* ids, float* y, unsigned long long stride, unsigned int nin,
  unsigned int nout, unsigned int dtype, unsigned int xPerSlot)
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
    y[(unsigned long long)slot * nout + o] = acc + 0.0f;
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
