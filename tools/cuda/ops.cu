// LandrixAI - CUDA: the operations of a Qwen3 layer besides Linear (backlog B3,
// slice C3) - the CUDA forms of tools/vk/rmsnorm.comp, addnorm.comp,
// qkrope.comp, swiglu.comp, attention.comp and attnmerge.comp (slice C5)
// - float arithmetic as the Vulkan shaders: the logits match the CPU within a
//   tolerance, not in bits
// - every block reducing over a row has 128 threads (four warps)
// - fill0, gatherrows and scatteradd: a mixture of experts (phase D, D3)
// - qkrope16 and attention16: the same with the KV cache in half floats
//   (TQwen3Cuda.KvF16), converted by cvt (no cuda_fp16.h needed)

#define BLOCK 128u

// a half float (its 16 bits) from a float, rounded to nearest even, and back
__device__ __forceinline__ unsigned short toHalf(float f)
{
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}

__device__ __forceinline__ float fromHalf(unsigned short h)
{
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

// the cache entries of either type: float or half float (unsigned short)
__device__ __forceinline__ void store(float* c, unsigned long long i, float v)
{
  c[i] = v;
}

__device__ __forceinline__ void store(unsigned short* c, unsigned long long i,
  float v)
{
  c[i] = toHalf(v);
}

__device__ __forceinline__ float load(const float* c, unsigned long long i)
{
  return c[i];
}

__device__ __forceinline__ float load(const unsigned short* c,
  unsigned long long i)
{
  return fromHalf(c[i]);
}

// the 4 keys from index 4 i on, i.e. key slot i (16 bytes, or 8 for half
// floats)
__device__ __forceinline__ float4 load4(const float* c, unsigned long long i)
{
  return reinterpret_cast<const float4*>(c)[i];
}

__device__ __forceinline__ float4 load4(const unsigned short* c,
  unsigned long long i)
{
  uint2 u = reinterpret_cast<const uint2*>(c)[i];
  return make_float4(fromHalf((unsigned short)(u.x & 0xffffu)),
    fromHalf((unsigned short)(u.x >> 16)), fromHalf((unsigned short)(u.y & 0xffffu)),
    fromHalf((unsigned short)(u.y >> 16)));
}

// the sum of v over the block (all 128 threads call it; the result in all)
__device__ float blockSum(float v, float* shared4)
{
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  __syncthreads(); // shared4 may still be read from a previous call
  if ((threadIdx.x & 31u) == 0u)
    shared4[threadIdx.x >> 5] = v;
  __syncthreads();
  return shared4[0] + shared4[1] + shared4[2] + shared4[3];
}

// the maximum of v over the block
__device__ float blockMax(float v, float* shared4)
{
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  __syncthreads();
  if ((threadIdx.x & 31u) == 0u)
    shared4[threadIdx.x >> 5] = v;
  __syncthreads();
  return fmaxf(fmaxf(shared4[0], shared4[1]), fmaxf(shared4[2], shared4[3]));
}

// y[r] = w * x[r] / sqrt(mean(x[r]^2) + eps), one block per row; y may be x
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* y,
  unsigned int cols, float eps)
{
  __shared__ float part[4];
  unsigned long long base = (unsigned long long)blockIdx.x * cols;
  float s = 0.0f;
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
  {
    float a = x[base + c];
    s += a * a;
  }
  float inv = rsqrtf(blockSum(s, part) / float(cols) + eps);
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
    y[base + c] = w[c] * (x[base + c] * inv);
}

// x[r] += t[r], then y[r] = w * x[r] / rms(x[r]): the residual and the next
// norm in one launch, one block per row
extern "C" __global__ void addnorm(float* x, const float* t, const float* w,
  float* y, unsigned int cols, float eps)
{
  __shared__ float part[4];
  unsigned long long base = (unsigned long long)blockIdx.x * cols;
  float s = 0.0f;
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
  {
    float a = x[base + c] + t[base + c];
    x[base + c] = a;
    s += a * a;
  }
  float inv = rsqrtf(blockSum(s, part) / float(cols) + eps);
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
    y[base + c] = w[c] * (x[base + c] * inv);
}

// g = silu(g) * u over n values; with limit > 0 the clamped SwiGLU of gpt-oss
// (slice D5d): (clamp(u, -limit, limit) + 1) * g * sigmoid(alpha * g), g
// clamped from above - the steps of TInferCpuBackend.SwiGlu
extern "C" __global__ void swiglu(float* g, const float* u, unsigned int n,
  float alpha, float limit)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float a = g[i];
    if (limit > 0.0f)
    {
      float b = u[i];
      // by comparison, as the CPU: a NaN stays a NaN (fminf would drop it)
      if (a > limit)
        a = limit;
      if (b > limit)
        b = limit;
      else if (b < -limit)
        b = -limit;
      float sg = 1.0f / (1.0f + expf(-(a * alpha)));
      g[i] = (b + 1.0f) * (a * sg);
    }
    else
      g[i] = (a / (1.0f + expf(-a))) * u[i];
  }
}

// per token row (grid y) and head (grid x: the heads, then the KV heads):
// RMSNorm of q (qn) or k (kn) when norm is set (Qwen3; Qwen2 has none: qn
// and kn are not read), rotary positions unless rope is 0 (a NoPE layer,
// backlog D4b) (cs: the cos of every row, then the
// sin), q in place, k and v stored into the cache at past + row
// - rot: the first rot values of a head rotate (rotate_half within them, cs
//   holds rot / 2 per row), the others pass (Qwen3.5, slice D7e1); 0: all
// - headDim <= 256, even; T: the type of the cache (float or half float)
template <typename T>
__device__ __forceinline__ void qkropeT(float* q, const float* k,
  const float* v, const float* qn, const float* kn, const float* cs, T* kc,
  T* vc, unsigned int rows, unsigned int heads, unsigned int kvHeads,
  unsigned int headDim, unsigned int past, float eps, unsigned int norm,
  unsigned int rope, unsigned int rot)
{
  __shared__ float part[4];
  __shared__ float normed[256];
  unsigned int t = threadIdx.x;
  unsigned int r = blockIdx.y;
  unsigned int hh = blockIdx.x;
  unsigned int d = headDim;
  unsigned int half = (rot == 0u ? d : rot) >> 1;
  bool isQ = hh < heads;
  unsigned int h = isQ ? hh : hh - heads;
  unsigned long long src = isQ ? ((unsigned long long)r * heads + h) * d
                               : ((unsigned long long)r * kvHeads + h) * d;
  float s = 0.0f;
  for (unsigned int c = t; c < d; c += BLOCK)
  {
    float a = isQ ? q[src + c] : k[src + c];
    s += a * a;
  }
  float inv = rsqrtf(blockSum(s, part) / float(d) + eps);
  for (unsigned int c = t; c < d; c += BLOCK)
  {
    float a = isQ ? q[src + c] : k[src + c];
    normed[c] = norm ? (isQ ? qn[c] : kn[c]) * (a * inv) : a;
  }
  __syncthreads();
  unsigned long long cbase = (unsigned long long)r * half;
  unsigned long long sbase = (unsigned long long)rows * half + cbase;
  unsigned long long kvw = (unsigned long long)kvHeads * d;
  unsigned long long dst = (past + (unsigned long long)r) * kvw + (unsigned long long)h * d;
  for (unsigned int i = t; i < half; i += BLOCK)
  {
    float a = normed[i];
    float b = normed[i + half];
    float x0 = a;
    float x1 = b;
    if (rope)
    {
      float c = cs[cbase + i];
      float sn = cs[sbase + i];
      x0 = a * c - b * sn;
      x1 = b * c + a * sn;
    }
    if (isQ)
    {
      q[src + i] = x0;
      q[src + i + half] = x1;
    }
    else
    {
      store(kc, dst + i, x0);
      store(kc, dst + i + half, x1);
    }
  }
  // beyond the rotated share: as normed
  for (unsigned int c = 2u * half + t; c < d; c += BLOCK)
    if (isQ)
      q[src + c] = normed[c];
    else
      store(kc, dst + c, normed[c]);
  if (!isQ)
    for (unsigned int c = t; c < d; c += BLOCK)
      store(vc, dst + c, v[src + c]);
}

extern "C" __global__ void qkrope(float* q, const float* k, const float* v,
  const float* qn, const float* kn, const float* cs, float* kc, float* vc,
  unsigned int rows, unsigned int heads, unsigned int kvHeads,
  unsigned int headDim, unsigned int past, float eps, unsigned int norm,
  unsigned int rope, unsigned int rot)
{
  qkropeT(q, k, v, qn, kn, cs, kc, vc, rows, heads, kvHeads, headDim, past,
    eps, norm, rope, rot);
}

extern "C" __global__ void qkrope16(float* q, const float* k, const float* v,
  const float* qn, const float* kn, const float* cs, unsigned short* kc,
  unsigned short* vc, unsigned int rows, unsigned int heads,
  unsigned int kvHeads, unsigned int headDim, unsigned int past, float eps,
  unsigned int norm, unsigned int rope, unsigned int rot)
{
  qkropeT(q, k, v, qn, kn, cs, kc, vc, rows, heads, kvHeads, headDim, past,
    eps, norm, rope, rot);
}

// causal attention with grouped-query heads - the CUDA form of
// tools/vk/attention.comp (slice C5)
// - y[r, h] = softmax(q[r, h] . keys[0 .. past + r] / sqrt(d)) . values, the
//   keys and values of the KV head h / (heads / kvHeads), rows at kvWidth
// - one block of 128 per (head block: grid x, row block: grid y, split: grid
//   z); a block holds up to MAXQ queries - headBlock heads of one KV head times
//   rowBlock rows - so every key and value read serves all of them; the keys
//   go by in tiles of 128, their scores stay in shared memory and the softmax
//   is kept online (running maximum, sums rescaled)
// - the keys of a row are cut into splits ranges (flash-decoding), so a single
//   decoded token still fills the device; splits = 1: y is written normalized;
//   splits > 1: each range writes its unnormalized sums, its maximum and its
//   sum of exponentials to part[] ([heads, rows, splits, headDim + 2]), and
//   attnmerge combines them into y
// - headDim a multiple of 4 and at most 256 (the queries kept as float4, each
//   thread owns the output dimensions t and t + 128); keys 16-byte aligned
//   (8 for half floats); T: the type of the cache
// - window > 0 (backlog D4b): row r sees only the keys past + r - window + 1 ..
//   past + r; the cache holds every row, the ranges start at the window
// - a block sums the values of all keys lo .. hi of its queries, those
//   outside a query's range with a weight of 0 - and 0 times a NaN or infinite
//   value is NaN: a layer with a window therefore runs one row per block
//   (TQwen3Cuda), where every query shares one range; at the causal edge of
//   a prompt chunk the earlier rows of a group still sum the later rows'
//   values with a weight of 0, as since slice C5
// - sinks (gpt-oss, slice D5d): [heads] logits that join the denominator of
//   the softmax without a value - added once, here when splits is 1, else
//   by attnmerge, and part of the maximum as on the CPU; 0 for none - a
//   sink more than about 103 above every score leaves the keys a share below
//   the smallest float: 0, as the weights of the fast CPU backend (single)
#define MAXQ 8
#define NEG -3.402823466e38f

template <typename T>
__device__ __forceinline__ void attentionT(const float* q, const T* keys,
  const T* vals, float* y, float* part, unsigned int rows,
  unsigned int heads, unsigned int headDim, unsigned int kvHeads,
  unsigned int past, unsigned int splits, unsigned int headBlock,
  unsigned int rowBlock, unsigned int window, const float* sinks)
{
  __shared__ float4 qs[MAXQ * 64];   // the queries
  __shared__ float ps[MAXQ * 128];   // the scores, then the weights of a tile
  __shared__ float red[MAXQ * 128];  // the maximum of a tile
  unsigned int t = threadIdx.x;
  unsigned int hb = headBlock;
  unsigned int h0 = blockIdx.x * hb;
  unsigned int r0 = blockIdx.y * rowBlock;
  unsigned int s = blockIdx.z;
  unsigned int d = headDim;
  unsigned int d4 = d >> 2;
  unsigned long long kvw = (unsigned long long)kvHeads * d;
  unsigned long long qw = (unsigned long long)heads * d;
  unsigned int g = h0 / (heads / kvHeads);
  // query i: head h0 + i % hb, row r0 + i / hb
  unsigned int nq = hb * min(rowBlock, rows - r0);
  float scale = rsqrtf(float(d));
  for (unsigned int i = t; i < nq * d4; i += BLOCK)
  {
    unsigned int qi = i / d4;
    unsigned int k = i - qi * d4;
    unsigned long long qb = (r0 + qi / hb) * qw + (h0 + qi % hb) * d + 4u * k;
    qs[qi * 64u + k] = make_float4(q[qb], q[qb + 1u], q[qb + 2u], q[qb + 3u]);
  }
  // the keys j0 .. j1 - 1 of each query in this split (empty for a late
  // split of an early row); both grow with the row, the block reads lo .. hi
  unsigned int j0[MAXQ], j1[MAXQ];
  unsigned int lo = 0xffffffffu;
  unsigned int hi = 0u;
#pragma unroll
  for (unsigned int i = 0u; i < MAXQ; i++)
  {
    j0[i] = 0u;
    j1[i] = 0u;
    if (i < nq)
    {
      unsigned int total = past + r0 + i / hb + 1u;
      // a sliding window: the keys before the last window ones are not read
      unsigned int first = (window != 0u) && (total > window) ? total - window : 0u;
      unsigned int chunk = (total - first + splits - 1u) / splits;
      j0[i] = min(total, first + s * chunk);
      j1[i] = min(total, j0[i] + chunk);
      lo = min(lo, j0[i]);
      hi = max(hi, j1[i]);
    }
  }
  __syncthreads();
  float m[MAXQ], l[MAXQ], acc0[MAXQ], acc1[MAXQ];
#pragma unroll
  for (unsigned int i = 0u; i < MAXQ; i++)
  {
    m[i] = NEG;
    l[i] = 0.0f;
    acc0[i] = 0.0f;
    acc1[i] = 0.0f;
  }
  unsigned long long vk = (unsigned long long)g * d + t;
  for (unsigned int base = lo; base < hi; base += BLOCK)
  {
    // the scores of key base + t
    unsigned int j = base + t;
    float sv[MAXQ];
#pragma unroll
    for (unsigned int i = 0u; i < MAXQ; i++)
      sv[i] = 0.0f;
    if (j < hi)
    {
      unsigned long long kp = (j * kvw + (unsigned long long)g * d) >> 2;
#pragma unroll 8
      for (unsigned int k = 0u; k < d4; k++)
      {
        float4 kv = load4(keys, kp + k);
#pragma unroll
        for (unsigned int i = 0u; i < MAXQ; i++)
          if (i < nq)
          {
            float4 qv = qs[i * 64u + k];
            sv[i] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
          }
      }
    }
#pragma unroll
    for (unsigned int i = 0u; i < MAXQ; i++)
    {
      float v = NEG;
      if ((i < nq) && (j >= j0[i]) && (j < j1[i]))
        v = sv[i] * scale;
      ps[i * 128u + t] = v;
      red[i * 128u + t] = v;
    }
    __syncthreads();
    for (unsigned int w = 64u; w > 0u; w >>= 1)
    {
      if (t < w)
#pragma unroll
        for (unsigned int i = 0u; i < MAXQ; i++)
          red[i * 128u + t] = fmaxf(red[i * 128u + t], red[i * 128u + t + w]);
      __syncthreads();
    }
    // the new maxima: the sums so far rescaled, the weights of this tile
#pragma unroll
    for (unsigned int i = 0u; i < MAXQ; i++)
    {
      float mn = fmaxf(m[i], red[i * 128u]);
      float f = expf(m[i] - mn);
      acc0[i] *= f;
      acc1[i] *= f;
      l[i] *= f;
      m[i] = mn;
      float e = 0.0f;
      if ((i < nq) && (j >= j0[i]) && (j < j1[i]))
        e = expf(ps[i * 128u + t] - mn);
      ps[i * 128u + t] = e;
    }
    __syncthreads();
    // the weighted values of this tile
    unsigned int n = min(BLOCK, hi - base);
#pragma unroll 8
    for (unsigned int jj = 0u; jj < n; jj++)
    {
      unsigned long long vb = (base + jj) * kvw + vk;
      float v0 = 0.0f;
      float v1 = 0.0f;
      if (t < d)
        v0 = load(vals, vb);
      if (t + 128u < d)
        v1 = load(vals, vb + 128u);
#pragma unroll
      for (unsigned int i = 0u; i < MAXQ; i++)
      {
        float w = ps[i * 128u + jj];
        l[i] += w;
        acc0[i] += w * v0;
        acc1[i] += w * v1;
      }
    }
    __syncthreads(); // ps and red are refilled by the next tile
  }
#pragma unroll
  for (unsigned int i = 0u; i < MAXQ; i++)
    if (i < nq)
    {
      unsigned int h = h0 + i % hb;
      unsigned int r = r0 + i / hb;
      if (splits == 1u)
      {
        unsigned long long yb = r * qw + (unsigned long long)h * d + t;
        // the sink joins the maximum, as on the CPU: a sink far above every
        // score would overflow exp(sink - m)
        float f = 1.0f;
        float den = l[i];
        if (sinks != 0)
        {
          float mm = fmaxf(m[i], sinks[h]);
          f = expf(m[i] - mm);
          den = l[i] * f + expf(sinks[h] - mm); // a share without a value
        }
        if (t < d)
          y[yb] = acc0[i] * f / den;
        if (t + 128u < d)
          y[yb + 128u] = acc1[i] * f / den;
      }
      else
      {
        unsigned long long pb = (((unsigned long long)h * rows + r) * splits + s) * (d + 2u);
        if (t < d)
          part[pb + t] = acc0[i];
        if (t + 128u < d)
          part[pb + t + 128u] = acc1[i];
        if (t == 0u)
        {
          part[pb + d] = m[i];
          part[pb + d + 1u] = l[i];
        }
      }
    }
}

extern "C" __global__ void attention(const float* q, const float* keys,
  const float* vals, float* y, float* part, unsigned int rows,
  unsigned int heads, unsigned int headDim, unsigned int kvHeads,
  unsigned int past, unsigned int splits, unsigned int headBlock,
  unsigned int rowBlock, unsigned int window, const float* sinks)
{
  attentionT(q, keys, vals, y, part, rows, heads, headDim, kvHeads, past,
    splits, headBlock, rowBlock, window, sinks);
}

extern "C" __global__ void attention16(const float* q,
  const unsigned short* keys, const unsigned short* vals, float* y,
  float* part, unsigned int rows, unsigned int heads, unsigned int headDim,
  unsigned int kvHeads, unsigned int past, unsigned int splits,
  unsigned int headBlock, unsigned int rowBlock, unsigned int window,
  const float* sinks)
{
  attentionT(q, keys, vals, y, part, rows, heads, headDim, kvHeads, past,
    splits, headBlock, rowBlock, window, sinks);
}

// combines the key ranges of attention (flash-decoding), the CUDA form of
// tools/vk/attnmerge.comp: per (head: grid x, row: grid y) M = max of the
// range maxima, y = sum_s exp(m_s - M) * acc_s / sum_s exp(m_s - M) * l_s
// - an empty range has m = -FLT_MAX and l = 0: its weight is 0
// - sinks: [heads] logits added to the denominator (gpt-oss), 0 for none
extern "C" __global__ void attnmerge(const float* part, float* y,
  unsigned int rows, unsigned int heads, unsigned int headDim,
  unsigned int splits, const float* sinks)
{
  unsigned int t = threadIdx.x;
  unsigned int h = blockIdx.x;
  unsigned int r = blockIdx.y;
  unsigned int d = headDim;
  unsigned int stride = d + 2u;
  const float* b = part + ((unsigned long long)h * rows + r) * splits * stride;
  // the sink joins the maximum: exp(sink - mx) cannot overflow
  float mx = NEG;
  if (sinks != 0)
    mx = sinks[h];
  for (unsigned int s = 0u; s < splits; s++)
    mx = fmaxf(mx, b[s * stride + d]);
  float den = 0.0f;
  for (unsigned int s = 0u; s < splits; s++)
    den += expf(b[s * stride + d] - mx) * b[s * stride + d + 1u];
  if (sinks != 0)
    den += expf(sinks[h] - mx); // a share without a value
  for (unsigned int k = t; k < d; k += BLOCK)
  {
    float acc = 0.0f;
    for (unsigned int s = 0u; s < splits; s++)
      acc += expf(b[s * stride + d] - mx) * b[s * stride + k];
    y[((unsigned long long)r * heads + h) * d + k] = acc / den;
  }
}

// a mixture of experts (phase D, slice D3): the rows of the tokens of one
// expert gathered into a buffer of their own, its SwiGLU block computed by
// the Linear kernels over them, its output added back into the rows of its
// tokens with their router weights - one expert after the other, in id order,
// as the CPU (TDecoderModel.MoeBlock)

// y[i] = 0 for i < n
extern "C" __global__ void fill0(float* y, unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = 0.0f;
}

// dst[r] = src[idx[r]], rows of cols floats: one block per gathered row
extern "C" __global__ void gatherrows(const float* src, const int* idx, float* dst,
  unsigned int cols)
{
  const float* s = src + (unsigned long long)idx[blockIdx.x] * cols;
  float* d = dst + (unsigned long long)blockIdx.x * cols;
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
    d[c] = s[c];
}

// dst[idx[r]] += w[r] * src[r]: one block per row - the tokens of one expert
// are distinct, so no two blocks of a launch write the same row
extern "C" __global__ void scatteradd(const float* src, const int* idx,
  const float* w, float* dst, unsigned int cols)
{
  const float* s = src + (unsigned long long)blockIdx.x * cols;
  float* d = dst + (unsigned long long)idx[blockIdx.x] * cols;
  float f = w[blockIdx.x];
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
    d[c] = d[c] + f * s[c];
}

// ---- slice D3c: the choice of the experts on the device, for a decode step
// (one token row): no wait for the host, the step stays one CUDA graph

// the router of TDecoderModel.Route on one row of logits: softmax in double,
// the k highest logits (the lower id on a tie), each weight rounded to float,
// renormed to their sum when norm - then sorted by expert id, so the experts
// are summed in id order as the CPU does; one thread (experts <= 4096,
// k <= 64, checked by the host)
extern "C" __global__ void route(const float* logits, int* ids, float* w,
  unsigned int experts, unsigned int k, unsigned int norm)
{
  if (threadIdx.x != 0u || blockIdx.x != 0u)
    return;
  unsigned int taken[128];
  int chosen[64];
  float wt[64];
  for (unsigned int i = 0u; i < 128u; i++)
    taken[i] = 0u;
  double m = logits[0];
  for (unsigned int i = 1u; i < experts; i++)
    if (logits[i] > m)
      m = logits[i];
  double sum = 0.0;
  for (unsigned int i = 0u; i < experts; i++)
    sum += exp((double)logits[i] - m);
  double picked = 0.0;
  for (unsigned int j = 0u; j < k; j++)
  {
    int best = -1;
    for (unsigned int i = 0u; i < experts; i++)
      if (((taken[i >> 5] >> (i & 31u)) & 1u) == 0u &&
          (best < 0 || logits[i] > logits[best]))
        best = (int)i;
    taken[best >> 5] |= 1u << (best & 31);
    chosen[j] = best;
    wt[j] = (float)(exp((double)logits[best] - m) / sum);
    picked += (double)wt[j];
  }
  if (norm)
    for (unsigned int j = 0u; j < k; j++)
      wt[j] = (float)((double)wt[j] / picked);
  // by id: an insertion sort of k entries
  for (unsigned int j = 1u; j < k; j++)
  {
    int id = chosen[j];
    float v = wt[j];
    unsigned int t = j;
    while (t > 0u && chosen[t - 1u] > id)
    {
      chosen[t] = chosen[t - 1u];
      wt[t] = wt[t - 1u];
      t--;
    }
    chosen[t] = id;
    wt[t] = v;
  }
  for (unsigned int j = 0u; j < k; j++)
  {
    ids[j] = chosen[j];
    w[j] = wt[j];
  }
}

// y[i] = sum over the k slots, in slot (= id) order, of w[s] * d[s, i] - the
// arithmetic of fill0 followed by one scatteradd per expert
extern "C" __global__ void moecombine(const float* d, const float* w, float* y,
  unsigned int k, unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float acc = 0.0f;
  for (unsigned int s = 0u; s < k; s++)
    acc = acc + w[s] * d[(unsigned long long)s * n + i];
  y[i] = acc;
}
