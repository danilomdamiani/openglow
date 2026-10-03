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
                                 float4* dst, int dw, int dh, float gain) {
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
      acc.x += w * (to_linear(p.x) * gain);
      acc.y += w * (to_linear(p.y) * gain);
      acc.z += w * (to_linear(p.z) * gain);
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
  float4 q;
  q.x = to_gamma(to_linear(p.x) + g.x * tint_b);
  q.y = to_gamma(to_linear(p.y) + g.y * tint_g);
  q.z = to_gamma(to_linear(p.z) + g.z * tint_r);
  q.w = p.w;
  store_frame(dst, dst_pitch, dst_half, x, y, q);
}

dim3 Grid(int w, int h) {
  return dim3((w + kGroupSize - 1) / kGroupSize, (h + kGroupSize - 1) / kGroupSize, 1);
}

const dim3 kBlock(kGroupSize, kGroupSize, 1);

cudaStream_t Stream(void* stream) { return static_cast<cudaStream_t>(stream); }

bool Launched() { return cudaPeekAtLastError() == cudaSuccess; }

}  // namespace

bool CudaDownsampleFirst(const Frame& src, const Plane& dst, float gain, void* stream) {
  downsample_first<<<Grid(dst.width, dst.height), kBlock, 0, Stream(stream)>>>(
      static_cast<const char*>(src.data), src.pitch, src.half ? 1 : 0, src.width, src.height,
      static_cast<float4*>(dst.data), dst.width, dst.height, gain);
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
