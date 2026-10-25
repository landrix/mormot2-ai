// LandrixAI - CUDA: the Gated DeltaNet of Qwen3.5 and the output gate of its
// full attention (phase D, slice D7e1) - the device forms of
// TDecoderModel.DeltaPrep, TInferBackend.DeltaRule, the gated RMSNorm and
// GateBySigmoid in mormot.ai.infer.decoder
// - float arithmetic: the logits match the CPU within a tolerance, not in bits
// - the states stay on the device: conv [channels, kw - 1] (oldest first) and
//   S [value heads, value dim, key dim] per layer, as in TDecoderContext

#define BLOCK 128u
// the largest key dimension and convolution kernel handled
#define MAXK 256u
#define MAXKW 8u

// silu and sigmoid as the CPU: 0 below -100 (exp would overflow there)
__device__ __forceinline__ float silu(float x)
{
  return x < -100.0f ? 0.0f : x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid(float x)
{
  return x < -100.0f ? 0.0f : 1.0f / (1.0f + expf(-x));
}

// the sum of v over the block of 128 (all threads call it, all get it)
__device__ float blockSum(float v, float* shared4)
{
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  __syncthreads();
  if ((threadIdx.x & 31u) == 0u)
    shared4[threadIdx.x >> 5] = v;
  __syncthreads();
  return shared4[0] + shared4[1] + shared4[2] + shared4[3];
}

// the sum of v over the warp (every lane gets it)
__device__ __forceinline__ float warpSum(float v)
{
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// the causal depthwise convolution of every channel (one thread each) over
// rows tokens, in place, then silu: weight kw - 1 takes the new input, the
// others the last kw - 1 inputs kept in conv (oldest first), which move on
extern "C" __global__ void dnconv(float* qkv, const float* w, float* conv,
  unsigned int rows, unsigned int channels, unsigned int kw)
{
  unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  unsigned int k1 = kw - 1u;
  float st[MAXKW];
  for (unsigned int j = 0u; j < k1; j++)
    st[j] = conv[(unsigned long long)c * k1 + j];
  for (unsigned int t = 0u; t < rows; t++)
  {
    unsigned long long i = (unsigned long long)t * channels + c;
    float x = qkv[i];
    float acc = w[c * kw + k1] * x;
    for (unsigned int j = 0u; j < k1; j++)
      acc += w[c * kw + j] * st[j];
    for (unsigned int j = 0u; j + 1u < k1; j++)
      st[j] = st[j + 1u];
    if (k1 > 0u)
      st[k1 - 1u] = x;
    qkv[i] = silu(acc);
  }
  for (unsigned int j = 0u; j < k1; j++)
    conv[(unsigned long long)c * k1 + j] = st[j];
}

// per token row (grid y) and item (grid x): a Q or K head L2-normed
// (x / sqrt(sum x^2 + 1e-6)), a query also times dk^-0.5; then per value head
// beta = sigmoid(b), the decay exp(-exp(A_log) * softplus(a + dt_bias)) (negA
// holds -exp(A_log); beyond 20 softplus is its input) and silu of its z
extern "C" __global__ void dnprep(float* qkv, float* beta, float* decay,
  float* z, const float* dtBias, const float* negA, unsigned int keyHeads,
  unsigned int valueHeads, unsigned int keyDim, unsigned int valueDim,
  unsigned int channels)
{
  __shared__ float part[4];
  unsigned int item = blockIdx.x;
  unsigned long long t = blockIdx.y;
  if (item < 2u * keyHeads)
  {
    float* s = qkv + t * channels + (unsigned long long)item * keyDim;
    float acc = 0.0f;
    for (unsigned int j = threadIdx.x; j < keyDim; j += BLOCK)
      acc += s[j] * s[j];
    float a = 1.0f / sqrtf(blockSum(acc, part) + 1e-6f);
    float scale = 1.0f / sqrtf(float(keyDim));
    for (unsigned int j = threadIdx.x; j < keyDim; j += BLOCK)
    {
      float v = s[j] * a;
      if (item < keyHeads)
        v = v * scale;
      s[j] = v;
    }
    return;
  }
  unsigned int i = item - 2u * keyHeads;
  if (threadIdx.x == 0u)
  {
    unsigned long long f = t * valueHeads + i;
    beta[f] = sigmoid(beta[f]);
    float x = decay[f] + dtBias[i];
    float sp = x > 20.0f ? x : log1pf(expf(x));
    decay[f] = expf(negA[i] * sp);
  }
  float* zz = z + (t * valueHeads + i) * valueDim;
  for (unsigned int j = threadIdx.x; j < valueDim; j += BLOCK)
    zz[j] = silu(zz[j]);
}

// the delta rule over rows tokens: per value head (grid x) a column j of S
// per warp (grid y: four columns per block) - S[j] decayed, what it predicts
// for the key, the correction written in, the column read by the query; S[j]
// stays in the registers of its warp over all the tokens
// - qkv rows at stride: [q all | k all | v all]; a key head serves
//   valueHeads / keyHeads neighbouring value heads; y [rows, valueHeads *
//   valueDim]
extern "C" __global__ void deltarule(const float* qkv, const float* beta,
  const float* decay, float* state, float* y, unsigned int rows,
  unsigned int stride, unsigned int keyHeads, unsigned int valueHeads,
  unsigned int keyDim, unsigned int valueDim)
{
  unsigned int head = blockIdx.x;
  unsigned int j = blockIdx.y * 4u + (threadIdx.x >> 5);
  unsigned int lane = threadIdx.x & 31u;
  if (j >= valueDim)
    return; // the whole warp: no block-wide sync follows
  unsigned int kd = keyHeads * keyDim;
  unsigned int vd = valueHeads * valueDim;
  unsigned int off = (head / (valueHeads / keyHeads)) * keyDim;
  float* row = state + ((unsigned long long)head * valueDim + j) * keyDim;
  float s[MAXK / 32u];
  for (unsigned int m = 0u; m < MAXK / 32u; m++)
  {
    unsigned int i = lane + 32u * m;
    s[m] = i < keyDim ? row[i] : 0.0f;
  }
  for (unsigned int t = 0u; t < rows; t++)
  {
    const float* r = qkv + (unsigned long long)t * stride;
    const float* q = r + off;
    const float* k = r + kd + off;
    float v = r[2u * kd + head * valueDim + j];
    float g = decay[(unsigned long long)t * valueHeads + head];
    float b = beta[(unsigned long long)t * valueHeads + head];
    float a = 0.0f;
    for (unsigned int m = 0u; m < MAXK / 32u; m++)
    {
      unsigned int i = lane + 32u * m;
      if (i < keyDim)
      {
        s[m] = s[m] * g;
        a += s[m] * k[i];
      }
    }
    a = warpSum(a);
    float delta = (v - a) * b;
    float o = 0.0f;
    for (unsigned int m = 0u; m < MAXK / 32u; m++)
    {
      unsigned int i = lane + 32u * m;
      if (i < keyDim)
      {
        s[m] = s[m] + k[i] * delta;
        o += s[m] * q[i];
      }
    }
    o = warpSum(o);
    if (lane == 0u)
      y[(unsigned long long)t * vd + head * valueDim + j] = o;
  }
  for (unsigned int m = 0u; m < MAXK / 32u; m++)
  {
    unsigned int i = lane + 32u * m;
    if (i < keyDim)
      row[i] = s[m];
  }
}

// y[r] = w * x[r] / sqrt(mean(x[r]^2) + eps) * z[r], one block per row: the
// gated RMSNorm of a value head (z holds silu(z) already); y may be x
extern "C" __global__ void gatednorm(const float* x, const float* w,
  const float* z, float* y, unsigned int cols, float eps)
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
    y[base + c] = (w[c] * (x[base + c] * inv)) * z[base + c];
}

// the query projection with an output gate holds per head its query, then
// its gate: q and g get them apart, n heads of d values each
extern "C" __global__ void splitgate(const float* qg, float* q, float* g,
  unsigned int n, unsigned int d)
{
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (unsigned long long)n * d)
    return;
  unsigned long long r = i / d;
  unsigned long long c = i - r * d;
  q[i] = qg[r * 2u * d + c];
  g[i] = qg[r * 2u * d + d + c];
}

// a[i] *= sigmoid(g[i]) over n values
extern "C" __global__ void gatemul(float* a, const float* g, unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] = a[i] * sigmoid(g[i]);
}
