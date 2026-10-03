// Minimal test runner: no framework, so CI stays fast on every platform.
#include "openglow/glow.h"

#include <cstdio>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,     \
                   __LINE__, #cond);                                  \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

static void test_passthrough_copies_pixels_with_padded_rows() {
  const int w = 7, h = 5;
  const std::size_t src_stride = w * 4 + 3;  // padded rows, like host buffers
  const std::size_t dst_stride = w * 4 + 9;
  std::vector<float> src(src_stride * h, -1.0f);
  std::vector<float> dst(dst_stride * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w * 4; ++x) src[y * src_stride + x] = static_cast<float>(y * 100 + x);

  openglow::ConstImageView in{src.data(), w, h, src_stride};
  openglow::ImageView out{dst.data(), w, h, dst_stride};
  openglow::render_glow(in, out, openglow::GlowParams{});

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w * 4; ++x) CHECK(dst[y * dst_stride + x] == src[y * src_stride + x]);
    // Padding past the row must stay untouched.
    for (std::size_t x = w * 4; x < dst_stride; ++x) CHECK(dst[y * dst_stride + x] == 0.0f);
  }
}

int main() {
  test_passthrough_copies_pixels_with_padded_rows();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all core tests passed\n");
  return 0;
}
