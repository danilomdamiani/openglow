#include "openglow/glow.h"

#include <algorithm>
#include <cstring>

namespace openglow {

void render_glow(const ConstImageView& src, const ImageView& dst, const GlowParams& /*params*/) {
  const int width = std::min(src.width, dst.width);
  const int height = std::min(src.height, dst.height);
  const std::size_t row_bytes = static_cast<std::size_t>(width) * 4 * sizeof(float);
  for (int y = 0; y < height; ++y) {
    std::memcpy(dst.row(y), src.row(y), row_bytes);
  }
}

}  // namespace openglow
