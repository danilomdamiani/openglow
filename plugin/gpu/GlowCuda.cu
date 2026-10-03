// OpenGlow CUDA kernels (Premiere Pro GPU filter). Same math, line by line,
// as OpenGlow.hlsl and core/src/glow.cpp; keep all three in sync.
//
// Each block of 16x16 threads computes 16x16 outputs. Neighbouring outputs
// read overlapping inputs, so the block first loads its input footprint into
// shared memory once, then filters from there. The arithmetic is unchanged.
#include "GlowCuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace openglow_gpu {
namespace {

constexpr int kGroupSize = 16;
// 4x4 tent at stride 2: 16 outputs read 2 * 16 + 2 = 34 inputs per axis,
// starting one pixel before twice the block's first output.
constexpr int kDownTile = 2 * kGroupSize + 2;
// 2x bilinear: 16 outputs read source indices 8b - 1 .. 8b + 8.
constexpr int kUpTile = kGroupSize / 2 + 2;

__constant__ float kTent[4] = {1.0f / 8, 3.0f / 8, 3.0f / 8, 1.0f / 8};

__device__ inline float to_linear(float v) { return v * fabsf(v); }
__device__ inline float to_gamma(float v) { return v < 0.0f ? -sqrtf(-v) : sqrtf(v); }

// Soft-knee bright pass on linear RGB (bright_pass in glow.cpp).
__device__ inline void bright_pass(float& r, float& g, float& b, float t, float k) {
  const float m = fmaxf(r, fmaxf(g, b));
  float soft = fminf(fmaxf(m - t + k, 0.0f), 2.0f * k);
  soft = soft * soft / (4.0f * k + 1e-5f);
  const float contrib = fmaxf(soft, m - t) / fmaxf(m, 1e-5f);
  r *= contrib;
  g *= contrib;
  b *= contrib;
}

// Frames: BGRA, 32f (float4) or 16f (four halves, read as one 8-byte word).
__device__ inline float4 load_frame(const char* __restrict__ base, int pitch, int half_float,
                                    int x, int y) {
  const char* row = base + static_cast<size_t>(pitch) * y;
  if (half_float) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(row) + x);
    const __half2 bg = *reinterpret_cast<const __half2*>(&v.x);
    const __half2 ra = *reinterpret_cast<const __half2*>(&v.y);
    const float2 lo = __half22float2(bg), hi = __half22float2(ra);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
  }
  return __ldg(reinterpret_cast<const float4*>(row) + x);
}

__device__ inline void store_frame(char* base, int pitch, int half_float, int x, int y, float4 v) {
  char* row = base + static_cast<size_t>(pitch) * y;
  if (half_float) {
    const __half2 bg = __halves2half2(__float2half_rn(v.x), __float2half_rn(v.y));
    const __half2 ra = __halves2half2(__float2half_rn(v.z), __float2half_rn(v.w));
    uint2 packed;
    packed.x = *reinterpret_cast<const unsigned int*>(&bg);
    packed.y = *reinterpret_cast<const unsigned int*>(&ra);
    reinterpret_cast<uint2*>(row)[x] = packed;
  } else {
    reinterpret_cast<float4*>(row)[x] = v;
  }
}

__device__ inline void tap(int d, int src_size, int& i0, int& i1, float& f) {
  const float s = fmaxf(0.0f, (d + 0.5f) * 0.5f - 0.5f);
  const int si = static_cast<int>(s);
  i0 = min(si, src_size - 1);
  i1 = min(i0 + 1, src_size - 1);
  f = s - si;
}

__device__ inline float4 lerp4(float4 a, float4 b, float t) {
  return make_float4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t,
                     a.w + (b.w - a.w) * t);
}

// Loads the footprint of this block's 2x upsample of `plane` into `tile`.
// Returns the source index of tile[0] (per axis).
__device__ inline int2 load_up_tile(const float4* __restrict__ plane, int sw, int sh,
                                    float4* tile) {
  const int2 base = make_int2(blockIdx.x * (kGroupSize / 2) - 1,
                              blockIdx.y * (kGroupSize / 2) - 1);
  const int index = threadIdx.y * kGroupSize + threadIdx.x;
  if (index < kUpTile * kUpTile) {
    const int sx = min(max(base.x + index % kUpTile, 0), sw - 1);
    const int sy = min(max(base.y + index / kUpTile, 0), sh - 1);
    tile[index] = __ldg(plane + sy * sw + sx);
  }
  __syncthreads();
  return base;
}

// Plain 2x bilinear from global memory (for a few samples only).
__device__ inline float4 bilinear(const float4* __restrict__ plane, int sw, int sh, int x,
                                  int y) {
  int x0, x1, y0, y1;
  float fx, fy;
  tap(x, sw, x0, x1, fx);
  tap(y, sh, y0, y1, fy);
  const float4 top = lerp4(__ldg(plane + y0 * sw + x0), __ldg(plane + y0 * sw + x1), fx);
  const float4 bot = lerp4(__ldg(plane + y1 * sw + x0), __ldg(plane + y1 * sw + x1), fx);
  return lerp4(top, bot, fy);
}

// Level 0 for the composite: like load_up_tile, but each sample first gets
// its upsample_add from the next level (same arithmetic as upsample_add).
__device__ inline int2 load_collapsed_tile(const float4* __restrict__ plane, int sw, int sh,
                                           const float4* __restrict__ next, int nw, int nh,
                                           float weight, float4* tile) {
  if (!next) return load_up_tile(plane, sw, sh, tile);
  const int2 base = make_int2(blockIdx.x * (kGroupSize / 2) - 1,
                              blockIdx.y * (kGroupSize / 2) - 1);
  const int index = threadIdx.y * kGroupSize + threadIdx.x;
  if (index < kUpTile * kUpTile) {
    const int sx = min(max(base.x + index % kUpTile, 0), sw - 1);
    const int sy = min(max(base.y + index / kUpTile, 0), sh - 1);
    const float4 up = bilinear(next, nw, nh, sx, sy);
    float4 d = __ldg(plane + sy * sw + sx);
    d.x += up.x * weight;
    d.y += up.y * weight;
    d.z += up.z * weight;
    d.w += up.w * weight;
    tile[index] = d;
  }
  __syncthreads();
  return base;
}

__device__ inline float4 bilinear_up(const float4* tile, int2 base, int sw, int sh, int x,
                                     int y) {
  int x0, x1, y0, y1;
  float fx, fy;
  tap(x, sw, x0, x1, fx);
  tap(y, sh, y0, y1, fy);
  x0 -= base.x;
  x1 -= base.x;
  y0 -= base.y;
  y1 -= base.y;
  const float4 top = lerp4(tile[y0 * kUpTile + x0], tile[y0 * kUpTile + x1], fx);
  const float4 bot = lerp4(tile[y1 * kUpTile + x0], tile[y1 * kUpTile + x1], fx);
  return lerp4(top, bot, fy);
}

__global__ void downsample_first(const char* __restrict__ src, int src_pitch, int src_half,
                                 int sw, int sh, float4* __restrict__ dst, int dw, int dh,
                                 float gain, float threshold, float knee) {
  __shared__ float3 tile[kDownTile * kDownTile];
  const int bx = blockIdx.x * 2 * kGroupSize - 1, by = blockIdx.y * 2 * kGroupSize - 1;
  for (int k = threadIdx.y * kGroupSize + threadIdx.x; k < kDownTile * kDownTile;
       k += kGroupSize * kGroupSize) {
    const int sx = min(max(bx + k % kDownTile, 0), sw - 1);
    const int sy = min(max(by + k / kDownTile, 0), sh - 1);
    const float4 p = load_frame(src, src_pitch, src_half, sx, sy);
    float cb = to_linear(p.x), cg = to_linear(p.y), cr = to_linear(p.z);
    if (threshold > 0.0f) bright_pass(cb, cg, cr, threshold, knee);
    tile[k] = make_float3(cb * p.w * gain, cg * p.w * gain, cr * p.w * gain);
  }
  __syncthreads();

  const int x = blockIdx.x * kGroupSize + threadIdx.x;
  const int y = blockIdx.y * kGroupSize + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float3 acc = make_float3(0, 0, 0);
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      const float w = kTent[j] * kTent[i];
      const float3 v = tile[(2 * threadIdx.y + j) * kDownTile + 2 * threadIdx.x + i];
      acc.x += w * v.x;
      acc.y += w * v.y;
      acc.z += w * v.z;
    }
  }
  dst[y * dw + x] = make_float4(acc.x, acc.y, acc.z, 0.0f);
}

__global__ void downsample(const float4* __restrict__ src, int sw, int sh,
                           float4* __restrict__ dst, int dw, int dh) {
  __shared__ float4 tile[kDownTile * kDownTile];
  const int bx = blockIdx.x * 2 * kGroupSize - 1, by = blockIdx.y * 2 * kGroupSize - 1;
  for (int k = threadIdx.y * kGroupSize + threadIdx.x; k < kDownTile * kDownTile;
       k += kGroupSize * kGroupSize) {
    const int sx = min(max(bx + k % kDownTile, 0), sw - 1);
    const int sy = min(max(by + k / kDownTile, 0), sh - 1);
    tile[k] = __ldg(src + sy * sw + sx);
  }
  __syncthreads();

  const int x = blockIdx.x * kGroupSize + threadIdx.x;
  const int y = blockIdx.y * kGroupSize + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float4 acc = make_float4(0, 0, 0, 0);
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      const float w = kTent[j] * kTent[i];
      const float4 p = tile[(2 * threadIdx.y + j) * kDownTile + 2 * threadIdx.x + i];
      acc.x += w * p.x;
      acc.y += w * p.y;
      acc.z += w * p.z;
      acc.w += w * p.w;
    }
  }
  dst[y * dw + x] = acc;
}

__global__ void upsample_add(const float4* __restrict__ src, int sw, int sh,
                             float4* __restrict__ dst, int dw, int dh, float weight) {
  __shared__ float4 tile[kUpTile * kUpTile];
  const int2 base = load_up_tile(src, sw, sh, tile);
  const int x = blockIdx.x * kGroupSize + threadIdx.x;
  const int y = blockIdx.y * kGroupSize + threadIdx.y;
  if (x >= dw || y >= dh) return;
  const float4 up = bilinear_up(tile, base, sw, sh, x, y);
  float4 d = dst[y * dw + x];
  d.x += up.x * weight;
  d.y += up.y * weight;
  d.z += up.z * weight;
  d.w += up.w * weight;
  dst[y * dw + x] = d;
}

// src and dst may be the same frame (in place), so no __restrict__ on them.
__global__ void composite(const char* src, int src_pitch, int src_half,
                          const float4* __restrict__ glow, int gw, int gh,
                          const float4* __restrict__ next, int nw, int nh, float next_weight,
                          char* dst, int dst_pitch, int dst_half, int w, int h, float tint_b,
                          float tint_g, float tint_r) {
  __shared__ float4 tile[kUpTile * kUpTile];
  const int2 base = load_collapsed_tile(glow, gw, gh, next, nw, nh, next_weight, tile);
  const int x = blockIdx.x * kGroupSize + threadIdx.x;
  const int y = blockIdx.y * kGroupSize + threadIdx.y;
  if (x >= w || y >= h) return;
  const float4 g = bilinear_up(tile, base, gw, gh, x, y);
  const float4 p = load_frame(src, src_pitch, src_half, x, y);
  // Premultiplied source + glow; alpha grows by the glow's coverage (with
  // alpha 1 this is exactly source + glow).
  const float gb = g.x * tint_b, gg = g.y * tint_g, gr = g.z * tint_r;
  const float pb = to_linear(p.x) * p.w + gb;
  const float pg = to_linear(p.y) * p.w + gg;
  const float pr = to_linear(p.z) * p.w + gr;
  const float coverage = fminf(fmaxf(fmaxf(gb, fmaxf(gg, gr)), 0.0f), 1.0f);
  const float out_a = p.w + coverage * (1.0f - p.w);
  float4 q = make_float4(0, 0, 0, out_a);
  if (out_a > 0.0f) {
    q.x = to_gamma(pb / out_a);
    q.y = to_gamma(pg / out_a);
    q.z = to_gamma(pr / out_a);
  }
  store_frame(dst, dst_pitch, dst_half, x, y, q);
}

dim3 Grid(int w, int h) {
  return dim3((w + kGroupSize - 1) / kGroupSize, (h + kGroupSize - 1) / kGroupSize, 1);
}

const dim3 kBlock(kGroupSize, kGroupSize, 1);

cudaStream_t Stream(void* stream) { return static_cast<cudaStream_t>(stream); }

bool Launched() { return cudaPeekAtLastError() == cudaSuccess; }

// Frames are read and written a whole pixel (8 or 16 bytes) at a time.
bool PixelAligned(const Frame& f) { return f.pitch % (f.half ? 8 : 16) == 0; }

}  // namespace

bool CudaDownsampleFirst(const Frame& src, const Plane& dst, float gain, float threshold,
                         float knee, void* stream) {
  if (!PixelAligned(src)) return false;
  downsample_first<<<Grid(dst.width, dst.height), kBlock, 0, Stream(stream)>>>(
      static_cast<const char*>(src.data), src.pitch, src.half ? 1 : 0, src.width, src.height,
      static_cast<float4*>(dst.data), dst.width, dst.height, gain, threshold, knee);
  return Launched();
}

bool CudaDownsample(const Plane& src, const Plane& dst, void* stream) {
  downsample<<<Grid(dst.width, dst.height), kBlock, 0, Stream(stream)>>>(
      static_cast<const float4*>(src.data), src.width, src.height,
      static_cast<float4*>(dst.data), dst.width, dst.height);
  return Launched();
}

bool CudaUpsampleAdd(const Plane& src, const Plane& dst, float src_weight, void* stream) {
  upsample_add<<<Grid(dst.width, dst.height), kBlock, 0, Stream(stream)>>>(
      static_cast<const float4*>(src.data), src.width, src.height,
      static_cast<float4*>(dst.data), dst.width, dst.height, src_weight);
  return Launched();
}

bool CudaComposite(const Frame& src, const Plane& glow, const Plane& next, float next_weight,
                   const Frame& dst, const float tint_bgr[3], void* stream) {
  if (!PixelAligned(src) || !PixelAligned(dst)) return false;
  composite<<<Grid(dst.width, dst.height), kBlock, 0, Stream(stream)>>>(
      static_cast<const char*>(src.data), src.pitch, src.half ? 1 : 0,
      static_cast<const float4*>(glow.data), glow.width, glow.height,
      static_cast<const float4*>(next.data), next.width, next.height, next_weight,
      static_cast<char*>(dst.data), dst.pitch, dst.half ? 1 : 0, dst.width, dst.height,
      tint_bgr[0], tint_bgr[1], tint_bgr[2]);
  return Launched();
}

bool CudaFinish(void* stream) { return cudaStreamSynchronize(Stream(stream)) == cudaSuccess; }

}  // namespace openglow_gpu
