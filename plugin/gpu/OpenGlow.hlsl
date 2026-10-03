// OpenGlow DirectX 12 kernels (Premiere Pro GPU filter). One entry point per
// pass, each compiled to its own .cso. The math mirrors core/src/glow.cpp line
// by line (see GlowGpu.h for the order of the passes); keep them in sync.
//
// Frames are Premiere's BGRA 32f or 16f buffers (straight alpha). The passes
// that touch frames are compiled twice, with HALF_FRAMES 0 and 1 ("_16f").
// Pyramid levels are tightly packed float4 BGRA planes.
//
// Buffers are structured (one element per pixel) rather than byte-address:
// a raw Load4 only promises 4-byte alignment, which some GPUs (e.g. Pascal)
// split into four loads. 16f is read with f16tof32 rather than native half
// types, which older GPUs don't support in D3D12.

#ifndef HALF_FRAMES
#define HALF_FRAMES 0
#endif

cbuffer Params : register(b0) {
  int src_pitch;  // pixels
  int src_half;   // informational: the shader variant decides
  int src_width;
  int src_height;
  int dst_pitch;  // pixels
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
  int next_width;  // composite: level 1, added to level 0 on the fly; 0 = none
  int next_height;
  float next_weight;
};

#if HALF_FRAMES
typedef uint2 FramePixel;  // four packed halves
#else
typedef float4 FramePixel;
#endif

// Frames live in space1, pyramid planes in space0.
RWStructuredBuffer<FramePixel> src_frame : register(u0, space1);
RWStructuredBuffer<FramePixel> dst_frame : register(u1, space1);
RWStructuredBuffer<float4> plane_a : register(u0);  // source level / glow
RWStructuredBuffer<float4> plane_b : register(u1);  // destination level
RWStructuredBuffer<float4> plane_c : register(u2);  // composite: next level

// Parameters are root constants and buffers root UAVs: nothing to allocate
// per dispatch. The root parameter order is the order the host binds buffers.
#define CONSTANTS "RootConstants(num32BitConstants = 19, b0)"
#define ROOT_FIRST CONSTANTS ", UAV(u0, space = 1), UAV(u1)"
#define ROOT_PLANES CONSTANTS ", UAV(u0), UAV(u1)"
#define ROOT_COMPOSITE CONSTANTS ", UAV(u0, space = 1), UAV(u0), UAV(u1, space = 1), UAV(u2)"

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

float4 unpack(FramePixel v) {
#if HALF_FRAMES
  return float4(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y), f16tof32(v.y >> 16));
#else
  return v;
#endif
}

FramePixel pack(float4 p) {
#if HALF_FRAMES
  return uint2(f32tof16(p.x) | (f32tof16(p.y) << 16), f32tof16(p.z) | (f32tof16(p.w) << 16));
#else
  return p;
#endif
}

float4 load_src(int x, int y) { return unpack(src_frame[y * src_pitch + x]); }
void store_dst(int x, int y, float4 p) { dst_frame[y * dst_pitch + x] = pack(p); }

// Each group of 16x16 threads computes 16x16 outputs. Neighbouring outputs
// read overlapping inputs, so the group first loads its input footprint into
// groupshared memory once (UAV reads may bypass the L1 cache), then filters
// from there. The arithmetic is unchanged.
#define GROUP 16

// 4x4 tent at stride 2: 16 outputs read 2 * 16 + 2 = 34 inputs per axis,
// starting one pixel before twice the group's first output.
#define DOWN_TILE 34
groupshared float3 g_first[DOWN_TILE * DOWN_TILE];
groupshared float4 g_down[DOWN_TILE * DOWN_TILE];

// 2x bilinear: 16 outputs read source indices 8g - 1 .. 8g + 8.
#define UP_TILE 10
groupshared float4 g_up[UP_TILE * UP_TILE];

// Bilinear tap for an exact 2x upsample (make_taps in glow.cpp).
void tap(int d, int src_size, out int i0, out int i1, out float f) {
  const float s = max(0.0f, (d + 0.5f) * 0.5f - 0.5f);
  const int si = (int)s;
  i0 = min(si, src_size - 1);
  i1 = min(i0 + 1, src_size - 1);
  f = s - si;
}

// Loads the plane_a footprint of this group's 2x upsample into g_up.
// Returns the source index of g_up[0] (per axis).
int2 load_up_tile(uint3 group, uint index, int sw, int sh) {
  const int2 base = int2(group.xy) * (GROUP / 2) - 1;
  if (index < UP_TILE * UP_TILE) {
    const int sx = clamp(base.x + (int)(index % UP_TILE), 0, sw - 1);
    const int sy = clamp(base.y + (int)(index / UP_TILE), 0, sh - 1);
    g_up[index] = plane_a[sy * sw + sx];
  }
  GroupMemoryBarrierWithGroupSync();
  return base;
}

// Plain 2x bilinear of plane_c from memory (for a few samples only).
float4 bilinear_c(int sw, int sh, int x, int y) {
  int x0, x1, y0, y1;
  float fx, fy;
  tap(x, sw, x0, x1, fx);
  tap(y, sh, y0, y1, fy);
  const float4 a = plane_c[y0 * sw + x0], b = plane_c[y0 * sw + x1];
  const float4 c = plane_c[y1 * sw + x0], d = plane_c[y1 * sw + x1];
  const float4 top = a + (b - a) * fx;
  const float4 bot = c + (d - c) * fx;
  return top + (bot - top) * fy;
}

// Level 0 for the composite: like load_up_tile, but each sample first gets
// its upsample_add from the next level (same arithmetic as upsample_add).
int2 load_collapsed_tile(uint3 group, uint index, int sw, int sh) {
  if (next_width == 0) return load_up_tile(group, index, sw, sh);
  const int2 base = int2(group.xy) * (GROUP / 2) - 1;
  if (index < UP_TILE * UP_TILE) {
    const int sx = clamp(base.x + (int)(index % UP_TILE), 0, sw - 1);
    const int sy = clamp(base.y + (int)(index / UP_TILE), 0, sh - 1);
    const float4 up = bilinear_c(next_width, next_height, sx, sy) * next_weight;
    g_up[index] = plane_a[sy * sw + sx] + up;
  }
  GroupMemoryBarrierWithGroupSync();
  return base;
}

float4 bilinear_up(int2 base, int sw, int sh, int x, int y) {
  int x0, x1, y0, y1;
  float fx, fy;
  tap(x, sw, x0, x1, fx);
  tap(y, sh, y0, y1, fy);
  x0 -= base.x;
  x1 -= base.x;
  y0 -= base.y;
  y1 -= base.y;
  const float4 a = g_up[y0 * UP_TILE + x0], b = g_up[y0 * UP_TILE + x1];
  const float4 c = g_up[y1 * UP_TILE + x0], d = g_up[y1 * UP_TILE + x1];
  const float4 top = a + (b - a) * fx;
  const float4 bot = c + (d - c) * fx;
  return top + (bot - top) * fy;
}

// Frame -> pyramid level 0 (plane_b): linearize, bright pass, weight by
// (straight) alpha, apply exposure, 4x4 tent.
[RootSignature(ROOT_FIRST)]
[numthreads(GROUP, GROUP, 1)]
void downsample_first(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID,
                      uint index : SV_GroupIndex) {
  const int2 base = int2(group.xy) * (2 * GROUP) - 1;
  for (uint k = index; k < DOWN_TILE * DOWN_TILE; k += GROUP * GROUP) {
    const int sx = clamp(base.x + (int)(k % DOWN_TILE), 0, src_width - 1);
    const int sy = clamp(base.y + (int)(k / DOWN_TILE), 0, src_height - 1);
    const float4 p = load_src(sx, sy);
    float3 c = float3(to_linear(p.x), to_linear(p.y), to_linear(p.z));
    if (threshold > 0.0f) c = bright_pass(c, threshold, knee);
    g_first[k] = float3(c.x * p.w * scale, c.y * p.w * scale, c.z * p.w * scale);
  }
  GroupMemoryBarrierWithGroupSync();

  const int x = group.x * GROUP + local.x, y = group.y * GROUP + local.y;
  if (x >= dst_width || y >= dst_height) return;
  float3 acc = 0;
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      const float w = kTent[j] * kTent[i];
      acc += w * g_first[(2 * local.y + j) * DOWN_TILE + 2 * local.x + i];
    }
  }
  plane_b[y * dst_width + x] = float4(acc, 0.0f);
}

// Pyramid level (plane_a) -> next level (plane_b), 4x4 tent.
[RootSignature(ROOT_PLANES)]
[numthreads(GROUP, GROUP, 1)]
void downsample(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID,
                uint index : SV_GroupIndex) {
  const int2 base = int2(group.xy) * (2 * GROUP) - 1;
  for (uint k = index; k < DOWN_TILE * DOWN_TILE; k += GROUP * GROUP) {
    const int sx = clamp(base.x + (int)(k % DOWN_TILE), 0, src_width - 1);
    const int sy = clamp(base.y + (int)(k / DOWN_TILE), 0, src_height - 1);
    g_down[k] = plane_a[sy * src_width + sx];
  }
  GroupMemoryBarrierWithGroupSync();

  const int x = group.x * GROUP + local.x, y = group.y * GROUP + local.y;
  if (x >= dst_width || y >= dst_height) return;
  float4 acc = 0;
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      acc += (kTent[j] * kTent[i]) * g_down[(2 * local.y + j) * DOWN_TILE + 2 * local.x + i];
    }
  }
  plane_b[y * dst_width + x] = acc;
}

// plane_b += upsample(plane_a) * scale.
[RootSignature(ROOT_PLANES)]
[numthreads(GROUP, GROUP, 1)]
void upsample_add(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID,
                  uint index : SV_GroupIndex) {
  const int2 base = load_up_tile(group, index, src_width, src_height);
  const int x = group.x * GROUP + local.x, y = group.y * GROUP + local.y;
  if (x >= dst_width || y >= dst_height) return;
  const float4 up = bilinear_up(base, src_width, src_height, x, y) * scale;
  plane_b[y * dst_width + x] = plane_b[y * dst_width + x] + up;
}

// Source frame + upsampled glow (plane_a) -> output frame.
[RootSignature(ROOT_COMPOSITE)]
[numthreads(GROUP, GROUP, 1)]
void composite(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID,
               uint index : SV_GroupIndex) {
  const int2 base = load_collapsed_tile(group, index, glow_width, glow_height);
  const int x = group.x * GROUP + local.x, y = group.y * GROUP + local.y;
  if (x >= dst_width || y >= dst_height) return;
  const float4 g = bilinear_up(base, glow_width, glow_height, x, y);
  const float4 p = load_src(x, y);
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
  store_dst(x, y, q);
}