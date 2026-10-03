// OpenGlow DirectX 12 kernels (Premiere Pro GPU filter). One entry point per
// pass, each compiled to its own .cso. The math mirrors core/src/glow.cpp line
// by line (see GlowGpu.h for the order of the passes); keep them in sync.
//
// Frames are Premiere's BGRA 32f or 16f buffers (straight alpha) with a pitch
// in bytes.
// Pyramid levels are tightly packed float4 BGRA planes.
//
// 16f is read with f16tof32 rather than native half types, which older GPUs
// (e.g. Pascal) don't support in D3D12.

cbuffer Params : register(b0) {
  int src_pitch;  // bytes
  int src_half;
  int src_width;
  int src_height;
  int dst_pitch;  // bytes
  int dst_half;
  int dst_width;
  int dst_height;
  int glow_width;
  int glow_height;
  float scale;  // gain (first downsample) or source weight (upsample-add)
  float tint_b;
  float tint_g;
  float tint_r;
  float threshold;  // bright pass in linear light, 0 = off
  float knee;
};

RWByteAddressBuffer buf0 : register(u0);
RWByteAddressBuffer buf1 : register(u1);
RWByteAddressBuffer buf2 : register(u2);

// Parameters are root constants and buffers root UAVs: nothing to allocate
// per dispatch. Root parameter order: constants, u0, u1[, u2].
#define ROOT_2 "RootConstants(num32BitConstants = 16, b0), UAV(u0), UAV(u1)"
#define ROOT_3 "RootConstants(num32BitConstants = 16, b0), UAV(u0), UAV(u1), UAV(u2)"

static const float kTent[4] = {1.0f / 8, 3.0f / 8, 3.0f / 8, 1.0f / 8};

float to_linear(float v) { return v * abs(v); }
float to_gamma(float v) { return v < 0.0f ? -sqrt(-v) : sqrt(v); }

// Soft-knee bright pass on linear RGB (bright_pass in glow.cpp).
float3 bright_pass(float3 c, float t, float k) {
  const float b = max(c.x, max(c.y, c.z));
  float soft = min(max(b - t + k, 0.0f), 2.0f * k);
  soft = soft * soft / (4.0f * k + 1e-5f);
  const float contrib = max(soft, b - t) / max(b, 1e-5f);
  return c * contrib;
}

float4 load_frame(RWByteAddressBuffer b, int pitch, int half_float, int x, int y) {
  if (half_float) {
    const uint2 v = b.Load2(pitch * y + 8 * x);
    return float4(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y), f16tof32(v.y >> 16));
  }
  return asfloat(b.Load4(pitch * y + 16 * x));
}

void store_frame(RWByteAddressBuffer b, int pitch, int half_float, int x, int y, float4 p) {
  if (half_float) {
    b.Store2(pitch * y + 8 * x, uint2(f32tof16(p.x) | (f32tof16(p.y) << 16),
                                      f32tof16(p.z) | (f32tof16(p.w) << 16)));
  } else {
    b.Store4(pitch * y + 16 * x, asuint(p));
  }
}

float4 load_plane(RWByteAddressBuffer b, int width, int x, int y) {
  return asfloat(b.Load4(16 * (y * width + x)));
}

void store_plane(RWByteAddressBuffer b, int width, int x, int y, float4 p) {
  b.Store4(16 * (y * width + x), asuint(p));
}

// Bilinear tap for an exact 2x upsample (make_taps in glow.cpp).
void tap(int d, int src_size, out int i0, out int i1, out float f) {
  const float s = max(0.0f, (d + 0.5f) * 0.5f - 0.5f);
  const int si = (int)s;
  i0 = min(si, src_size - 1);
  i1 = min(i0 + 1, src_size - 1);
  f = s - si;
}

float4 bilinear(RWByteAddressBuffer b, int sw, int sh, int x, int y) {
  int x0, x1, y0, y1;
  float fx, fy;
  tap(x, sw, x0, x1, fx);
  tap(y, sh, y0, y1, fy);
  const float4 a = load_plane(b, sw, x0, y0), bb = load_plane(b, sw, x1, y0);
  const float4 c = load_plane(b, sw, x0, y1), d = load_plane(b, sw, x1, y1);
  const float4 top = a + (bb - a) * fx;
  const float4 bot = c + (d - c) * fx;
  return top + (bot - top) * fy;
}

// Frame (u0) -> pyramid level 0 (u1): linearize, bright pass, weight by
// (straight) alpha, apply exposure, 4x4 tent.
[RootSignature(ROOT_2)]
[numthreads(16, 16, 1)]
void downsample_first(uint3 id : SV_DispatchThreadID) {
  const int x = id.x, y = id.y;
  if (x >= dst_width || y >= dst_height) return;
  float4 acc = 0;
  for (int j = 0; j < 4; ++j) {
    const int sy = clamp(2 * y - 1 + j, 0, src_height - 1);
    for (int i = 0; i < 4; ++i) {
      const int sx = clamp(2 * x - 1 + i, 0, src_width - 1);
      const float w = kTent[j] * kTent[i];
      const float4 p = load_frame(buf0, src_pitch, src_half, sx, sy);
      float3 c = float3(to_linear(p.x), to_linear(p.y), to_linear(p.z));
      if (threshold > 0.0f) c = bright_pass(c, threshold, knee);
      acc.x += w * (c.x * p.w * scale);
      acc.y += w * (c.y * p.w * scale);
      acc.z += w * (c.z * p.w * scale);
    }
  }
  acc.w = 0.0f;
  store_plane(buf1, dst_width, x, y, acc);
}

// Pyramid level (u0) -> next level (u1), 4x4 tent.
[RootSignature(ROOT_2)]
[numthreads(16, 16, 1)]
void downsample(uint3 id : SV_DispatchThreadID) {
  const int x = id.x, y = id.y;
  if (x >= dst_width || y >= dst_height) return;
  float4 acc = 0;
  for (int j = 0; j < 4; ++j) {
    const int sy = clamp(2 * y - 1 + j, 0, src_height - 1);
    for (int i = 0; i < 4; ++i) {
      const int sx = clamp(2 * x - 1 + i, 0, src_width - 1);
      acc += (kTent[j] * kTent[i]) * load_plane(buf0, src_width, sx, sy);
    }
  }
  store_plane(buf1, dst_width, x, y, acc);
}

// dst (u1) += upsample(src (u0)) * scale.
[RootSignature(ROOT_2)]
[numthreads(16, 16, 1)]
void upsample_add(uint3 id : SV_DispatchThreadID) {
  const int x = id.x, y = id.y;
  if (x >= dst_width || y >= dst_height) return;
  const float4 up = bilinear(buf0, src_width, src_height, x, y) * scale;
  store_plane(buf1, dst_width, x, y, load_plane(buf1, dst_width, x, y) + up);
}

// Source frame (u0) + upsampled glow (u1) -> output frame (u2).
[RootSignature(ROOT_3)]
[numthreads(16, 16, 1)]
void composite(uint3 id : SV_DispatchThreadID) {
  const int x = id.x, y = id.y;
  if (x >= dst_width || y >= dst_height) return;
  const float4 g = bilinear(buf1, glow_width, glow_height, x, y);
  const float4 p = load_frame(buf0, src_pitch, src_half, x, y);
  // Premultiplied source + glow; alpha grows by the glow's coverage (with
  // alpha 1 this is exactly source + glow).
  const float3 glow = float3(g.x * tint_b, g.y * tint_g, g.z * tint_r);
  const float3 premult = float3(to_linear(p.x), to_linear(p.y), to_linear(p.z)) * p.w + glow;
  const float coverage = min(max(max(glow.x, max(glow.y, glow.z)), 0.0f), 1.0f);
  const float out_a = p.w + coverage * (1.0f - p.w);
  float4 q = 0;
  if (out_a > 0.0f) {
    q.x = to_gamma(premult.x / out_a);
    q.y = to_gamma(premult.y / out_a);
    q.z = to_gamma(premult.z / out_a);
  }
  q.w = out_a;
  store_frame(buf2, dst_pitch, dst_half, x, y, q);
}
