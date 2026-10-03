// OpenGlow core: the glow algorithm, independent of any Adobe SDK.
//
// Images are 4 channels of 32-bit float, interleaved. The channel order is
// given by ChannelOrder so host buffers (BGRA in Premiere, ARGB in After
// Effects) can be processed in place without reshuffling.
#pragma once

#include <cstddef>

namespace openglow {

struct GlowParams {
  float exposure = 0.0f;  // in stops; the glow is scaled by 2^exposure
  float radius = 50.0f;   // in pixels, relative to a 1080-pixel-tall frame
  bool tint = false;
  float tint_color[3] = {1.0f, 1.0f, 1.0f};  // RGB, 0..1
  float threshold = 0.0f;  // 0..1 (gamma); only brighter parts glow, with a soft knee
};

// Index of each channel inside a pixel.
struct ChannelOrder {
  int r, g, b, a;
};
constexpr ChannelOrder kRGBA{0, 1, 2, 3};
constexpr ChannelOrder kBGRA{2, 1, 0, 3};
constexpr ChannelOrder kARGB{1, 2, 3, 0};

struct ImageView {
  float* pixels = nullptr;
  int width = 0;
  int height = 0;
  std::size_t stride = 0;  // floats per row (>= width * 4)

  float* row(int y) const { return pixels + static_cast<std::size_t>(y) * stride; }
};

struct ConstImageView {
  const float* pixels = nullptr;
  int width = 0;
  int height = 0;
  std::size_t stride = 0;

  const float* row(int y) const { return pixels + static_cast<std::size_t>(y) * stride; }
};

// The shape of one glow render: pyramid sizes and the constants applied along
// the way. The CPU path and the GPU kernels both follow it, so they match.
struct GlowPlan {
  static constexpr int kMaxLevels = 32;

  int levels = 0;               // pyramid levels, >= 1 for a non-empty frame
  int width[kMaxLevels] = {};   // level k is (about) half of level k-1;
  int height[kMaxLevels] = {};  // level 0 is half of the frame
  float gain = 1.0f;            // 2^exposure, applied when linearizing
  float last_weight = 1.0f;     // fade of the deepest level, in (0, 1]
  float tint[3] = {1, 1, 1};    // RGB factor on the collapsed glow,
                                // including the 1/total normalization
  float threshold = 0.0f;       // bright pass in linear light; 0 = off
  float knee = 0.0f;            // half-width of the soft knee
};

// Plans a render of a width x height frame. levels is 0 if the frame is empty.
GlowPlan plan_glow(int width, int height, const GlowParams& params);

// Renders src with glow into dst. Both must have the same size; they may
// point at the same memory. Color values are display-referred (gamma), 1.0 is
// white; values above 1.0 (32-bit float) are kept. Alpha is straight (not
// premultiplied). Only visible pixels emit glow, and the glow also covers
// transparent areas: output alpha is src alpha plus the glow's coverage, so
// opaque frames keep alpha 1 and transparent ones get an aura around shapes.
void render_glow(const ConstImageView& src, const ImageView& dst, const GlowParams& params,
                 ChannelOrder order = kRGBA);

}  // namespace openglow
