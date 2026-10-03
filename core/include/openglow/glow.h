// OpenGlow core: the glow algorithm, independent of any Adobe SDK.
//
// Images are RGBA, 32-bit float per channel, interleaved. The plugin layer
// converts whatever pixel format the host hands us into this layout.
#pragma once

#include <cstddef>

namespace openglow {

struct GlowParams {
  float exposure = 0.0f;  // in stops; the glow is scaled by 2^exposure
  float radius = 50.0f;   // in pixels, relative to a 1080p frame
  bool tint = false;
  float tint_color[3] = {1.0f, 1.0f, 1.0f};
};

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

// Renders src with glow into dst. Both must have the same size.
// Stage 0: copies src to dst unchanged; the glow arrives in stage 1.
void render_glow(const ConstImageView& src, const ImageView& dst, const GlowParams& params);

}  // namespace openglow
