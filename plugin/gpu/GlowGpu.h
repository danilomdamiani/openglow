// GPU glow: the sequence of passes, independent of the GPU API. Each backend
// (DirectX 12, CUDA) implements the four kernels; RunGlow drives them through
// the same steps as openglow::render_glow on the CPU:
//
//   downsample_first  frame -> level 0 (linearize, exposure, 4x4 tent)
//   downsample        level k-1 -> level k
//   upsample_add      level k += 2x bilinear(level k+1) [* last_weight once]
//   composite         frame + tinted 2x bilinear(level 0) -> output frame
#pragma once

#include <cstddef>

#include "openglow/glow.h"

namespace openglow_gpu {

// A frame owned by the host: BGRA, 32f or 16f, rows `pitch` bytes apart.
struct Frame {
  void* data = nullptr;  // backend handle (ID3D12Resource*, CUDA pointer)
  int width = 0;
  int height = 0;
  int pitch = 0;  // bytes
  bool half = false;

  std::size_t bytes() const { return static_cast<std::size_t>(pitch) * height; }
};

// A float4 BGRA pyramid level, tightly packed.
struct Plane {
  void* data = nullptr;
  int width = 0;
  int height = 0;

  std::size_t bytes() const { return static_cast<std::size_t>(width) * height * 16; }
};

class Backend {
 public:
  virtual ~Backend() = default;

  // Device memory for pyramid levels. Returns nullptr on failure.
  virtual void* Allocate(std::size_t bytes) = 0;
  virtual void Free(void* memory) = 0;

  virtual bool DownsampleFirst(const Frame& src, const Plane& dst, float gain) = 0;
  virtual bool Downsample(const Plane& src, const Plane& dst) = 0;
  virtual bool UpsampleAdd(const Plane& src, const Plane& dst, float src_weight) = 0;
  // tint_bgr already includes the 1/total normalization.
  virtual bool Composite(const Frame& src, const Plane& glow, const Frame& dst,
                         const float tint_bgr[3]) = 0;
  // Passes may be queued; this submits them and waits until the GPU is done,
  // so the pyramid can be freed. Called once per render, even after a failure.
  virtual bool Finish() = 0;
};

// Renders src with glow into dst (same size; may be the same buffer).
inline bool RunGlow(Backend& backend, const Frame& src, const Frame& dst,
                    const openglow::GlowParams& params) {
  const int width = src.width < dst.width ? src.width : dst.width;
  const int height = src.height < dst.height ? src.height : dst.height;
  const openglow::GlowPlan plan = openglow::plan_glow(width, height, params);
  if (plan.levels == 0) return true;

  Frame in = src, out = dst;
  in.width = out.width = width;
  in.height = out.height = height;

  Plane levels[openglow::GlowPlan::kMaxLevels];
  bool ok = true;
  for (int k = 0; k < plan.levels && ok; ++k) {
    levels[k].width = plan.width[k];
    levels[k].height = plan.height[k];
    levels[k].data = backend.Allocate(levels[k].bytes());
    ok = levels[k].data != nullptr;
  }

  if (ok) ok = backend.DownsampleFirst(in, levels[0], plan.gain);
  for (int k = 1; k < plan.levels && ok; ++k) ok = backend.Downsample(levels[k - 1], levels[k]);
  // The deepest level fades in by last_weight (the CPU scales it in place).
  for (int k = plan.levels - 2; k >= 0 && ok; --k) {
    const float weight = k == plan.levels - 2 ? plan.last_weight : 1.0f;
    ok = backend.UpsampleAdd(levels[k + 1], levels[k], weight);
  }
  if (ok) {
    // With a single level there is no upsample to carry last_weight (it is
    // always 1 then, but keep the math identical to the CPU).
    const float w = plan.levels == 1 ? plan.last_weight : 1.0f;
    const float tint_bgr[3] = {plan.tint[2] * w, plan.tint[1] * w, plan.tint[0] * w};
    ok = backend.Composite(in, levels[0], out, tint_bgr);
  }
  ok = backend.Finish() && ok;

  for (int k = 0; k < plan.levels; ++k) {
    if (levels[k].data) backend.Free(levels[k].data);
  }
  return ok;
}

}  // namespace openglow_gpu
