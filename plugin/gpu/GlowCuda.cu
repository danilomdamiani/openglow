// OpenGlow CUDA kernels (Premiere Pro GPU filter). Same math, line by line,
// as OpenGlow.hlsl and core/src/glow.cpp; keep all three in sync.
#include "GlowCuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace openglow_gpu {
namespace {

constexpr int kGroupSize = 16;

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

__device__ inline float4 load_frame(const char* base, int pitch, int half_float, int x, int y) {
  const char* row = base + static_cast<size_t>(pitch) * y;
  if (half_float) {
    const __half* p = reinterpret_cast<const __half*>(row) + 4 * x;
    return make_float4(__half2float(p[0]), __half2float(p[1]), __half2float(p[2]),
                       __half2float(p[3]));
  }
  return reinterpret_cast<const float4*>(row)[x];
}

__device__ inline void store_frame(char* base, int pitch, int half_float, int x, int y, float4 v) {
  char* row = base + static_cast<size_t>(pitch) * y;
  if (half_float) {
    __half* p = reinterpret_cast<__half*>(row) + 4 * x;
    p[0] = __float2half_rn(v.x);
    p[1] = __float2half_rn(v.y);
    p[2] = __float2half_rn(v.z);
    p[3] = __float2half_rn(v.w);
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

__device__ inline float4 bilinear(const float4* plane, int sw, int sh, int x, int y) {
  int x0, x1, y0, y1;
  float fx, fy;
  tap(x, sw, x0, x1, fx);
  tap(y, sh, y0, y1, fy);
  const float4 top = lerp4(plane[y0 * sw + x0], plane[y0 * sw + x1], fx);
  const float4 bot = lerp4(plane[y1 * sw + x0], plane[y1 * sw + x1], fx);
  return lerp4(top, bot, fy);
}

__global__ void downsample_first(const char* src, int src_pitch, int src_half, int sw, int sh,
                                 float4* dst, int dw, int dh, float gain, float threshold,
                                 float knee) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float4 acc = make_float4(0, 0, 0, 0);
  for (int j = 0; j < 4; ++j) {
    const int sy = min(max(2 * y - 1 + j, 0), sh - 1);
    for (int i = 0; i < 4; ++i) {
      const int sx = min(max(2 * x - 1 + i, 0), sw - 1);
      const float w = kTent[j] * kTent[i];
      const float4 p = load_frame(src, src_pitch, src_half, sx, sy);
      float cb = to_linear(p.x), cg = to_linear(p.y), cr = to_linear(p.z);
      if (threshold > 0.0f) bright_pass(cb, cg, cr, threshold, knee);
      acc.x += w * (cb * p.w * gain);
      acc.y += w * (cg * p.w * gain);
      acc.z += w * (cr * p.w * gain);
    }
  }
  acc.w = 0.0f;
  dst[y * dw + x] = acc;
}

__global__ void downsample(const float4* src, int sw, int sh, float4* dst, int dw, int dh) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float4 acc = make_float4(0, 0, 0, 0);
  for (int j = 0; j < 4; ++j) {
    const int sy = min(max(2 * y - 1 + j, 0), sh - 1);
    for (int i = 0; i < 4; ++i) {
      const int sx = min(max(2 * x - 1 + i, 0), sw - 1);
      const float w = kTent[j] * kTent[i];
      const float4 p = src[sy * sw + sx];
      acc.x += w * p.x;
      acc.y += w * p.y;
      acc.z += w * p.z;
      acc.w += w * p.w;
    }
  }
  dst[y * dw + x] = acc;
}

__global__ void upsample_add(const float4* src, int sw, int sh, float4* dst, int dw, int dh,
                             float weight) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  const float4 up = bilinear(src, sw, sh, x, y);
  float4 d = dst[y * dw + x];
  d.x += up.x * weight;
  d.y += up.y * weight;
  d.z += up.z * weight;
  d.w += up.w * weight;
  dst[y * dw + x] = d;
}

__global__ void composite(const char* src, int src_pitch, int src_half, const float4* glow,
                          int gw, int gh, char* dst, int dst_pitch, int dst_half, int w, int h,
                          float tint_b, float tint_g, float tint_r) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const float4 g = bilinear(glow, gw, gh, x, y);
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

}  // namespace

bool CudaDownsampleFirst(const Frame& src, const Plane& dst, float gain, float threshold,
                         float knee, void* stream) {
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

bool CudaComposite(const Frame& src, const Plane& glow, const Frame& dst,
                   const float tint_bgr[3], void* stream) {
  composite<<<Grid(dst.width, dst.height), kBlock, 0, Stream(stream)>>>(
      static_cast<const char*>(src.data), src.pitch, src.half ? 1 : 0,
      static_cast<const float4*>(glow.data), glow.width, glow.height,
      static_cast<char*>(dst.data), dst.pitch, dst.half ? 1 : 0, dst.width, dst.height,
      tint_bgr[0], tint_bgr[1], tint_bgr[2]);
  return Launched();
}

bool CudaFinish(void* stream) { return cudaStreamSynchronize(Stream(stream)) == cudaSuccess; }

}  // namespace openglow_gpu
