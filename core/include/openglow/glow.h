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

// Renders src with glow into dst. Both must have the same size; they may
// point at the same memory. Color values are display-referred (gamma), 1.0 is
// white; values above 1.0 (32-bit float) are kept. Alpha is copied from src.
void render_glow(const ConstImageView& src, const ImageView& dst, const GlowParams& params,
                 ChannelOrder order = kRGBA);

}  // namespace openglow
