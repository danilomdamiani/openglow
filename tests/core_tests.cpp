// Minimal test runner: no framework, so CI stays fast on every platform.
#include "openglow/glow.h"

#include <cmath>
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

namespace {

struct Image {
  int w, h;
  std::size_t stride;
  std::vector<float> px;

  Image(int w_, int h_, std::size_t pad = 0)
      : w(w_), h(h_), stride(w_ * 4 + pad), px(stride * h_, 0.0f) {}
  float* at(int x, int y) { return &px[y * stride + x * 4]; }
  const float* at(int x, int y) const { return &px[y * stride + x * 4]; }
  openglow::ImageView view() { return {px.data(), w, h, stride}; }
  openglow::ConstImageView cview() const { return {px.data(), w, h, stride}; }
};

Image opaque(int w, int h) {
  Image img(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) img.at(x, y)[3] = 1.0f;
  return img;
}

Image render(const Image& src, const openglow::GlowParams& p,
             openglow::ChannelOrder order = openglow::kRGBA) {
  Image dst(src.w, src.h);
  openglow::render_glow(src.cview(), dst.view(), p, order);
  return dst;
}

// Sum of squared (linear) red over the image, i.e. the light added.
double linear_red_energy(const Image& img) {
  double e = 0;
  for (int y = 0; y < img.h; ++y)
    for (int x = 0; x < img.w; ++x) e += img.at(x, y)[0] * img.at(x, y)[0];
  return e;
}

void test_black_stays_black() {
  Image src = opaque(64, 36);
  Image dst = render(src, {});
  for (int y = 0; y < dst.h; ++y)
    for (int x = 0; x < dst.w; ++x) {
      CHECK(dst.at(x, y)[0] == 0.0f);
      CHECK(dst.at(x, y)[3] == 1.0f);
    }
}

void test_bright_point_spreads_symmetrically() {
  Image src = opaque(129, 129);
  src.at(64, 64)[0] = src.at(64, 64)[1] = src.at(64, 64)[2] = 50.0f;
  openglow::GlowParams p;
  p.radius = 200;
  Image dst = render(src, p);
  // Glow reaches well away from the point...
  CHECK(dst.at(64 + 20, 64)[0] > 0.01f);
  // ...falls off with distance...
  CHECK(dst.at(64 + 5, 64)[0] > dst.at(64 + 20, 64)[0]);
  CHECK(dst.at(64 + 20, 64)[0] > dst.at(64 + 50, 64)[0]);
  // ...and is roughly symmetric (the pyramid grid can shift it by a pixel).
  const float l = dst.at(64 - 20, 64)[0], r = dst.at(64 + 20, 64)[0];
  const float u = dst.at(64, 64 - 20)[0], d = dst.at(64, 64 + 20)[0];
  CHECK(std::fabs(l - r) < 0.25f * r);
  CHECK(std::fabs(u - d) < 0.25f * d);
  CHECK(std::fabs(l - u) < 0.25f * u);
}

void test_larger_radius_spreads_wider() {
  Image src = opaque(256, 256);
  src.at(128, 128)[0] = 50.0f;
  openglow::GlowParams small, large;
  small.radius = 30;
  large.radius = 600;
  const Image a = render(src, small);
  const Image b = render(src, large);
  CHECK(b.at(128 + 60, 128)[0] > a.at(128 + 60, 128)[0]);
}

void test_radius_change_is_smooth() {
  // Crossing a power of two must not make the glow jump.
  Image src = opaque(256, 256);
  src.at(128, 128)[0] = 50.0f;
  openglow::GlowParams p;
  float prev = -1;
  for (float r = 100; r <= 160; r += 2) {
    p.radius = r;
    const float v = render(src, p).at(128 + 30, 128)[0];
    if (prev >= 0) CHECK(std::fabs(v - prev) < 0.1f * prev + 1e-4f);
    prev = v;
  }
}

void test_exposure_scales_glow() {
  Image src = opaque(96, 96);
  src.at(48, 48)[0] = 10.0f;
  openglow::GlowParams p0, p1;
  p1.exposure = 1.0f;
  const Image a = render(src, p0);
  const Image b = render(src, p1);
  // Away from the source the result is pure glow, linear in 2^exposure.
  const float la = a.at(60, 48)[0] * a.at(60, 48)[0];
  const float lb = b.at(60, 48)[0] * b.at(60, 48)[0];
  CHECK(la > 0);
  CHECK(std::fabs(lb / la - 2.0f) < 0.01f);
}

void test_energy_does_not_depend_on_radius() {
  Image src = opaque(256, 256);
  src.at(128, 128)[0] = 20.0f;
  const double base = 20.0 * 20.0;
  openglow::GlowParams p;
  p.radius = 40;
  const double e_small = linear_red_energy(render(src, p)) - base;
  p.radius = 400;
  const double e_large = linear_red_energy(render(src, p)) - base;
  CHECK(e_small > 0);
  CHECK(std::fabs(e_large / e_small - 1.0) < 0.15);
}

void test_tint_colors_glow_and_keeps_luminance() {
  Image src = opaque(96, 96);
  for (int c = 0; c < 3; ++c) src.at(48, 48)[c] = 10.0f;
  openglow::GlowParams plain, red;
  red.tint = true;
  red.tint_color[0] = 1.0f;
  red.tint_color[1] = 0.0f;
  red.tint_color[2] = 0.0f;
  const Image a = render(src, plain);
  const Image b = render(src, red);
  const float* pa = a.at(60, 48);
  const float* pb = b.at(60, 48);
  CHECK(pb[0] > pa[0]);
  CHECK(pb[1] == 0.0f);
  CHECK(pb[2] == 0.0f);
  auto luma = [](const float* p) {
    return 0.2126f * p[0] * p[0] + 0.7152f * p[1] * p[1] + 0.0722f * p[2] * p[2];
  };
  CHECK(std::fabs(luma(pb) / luma(pa) - 1.0f) < 0.01f);
}

void test_tint_off_ignores_color() {
  Image src = opaque(32, 32);
  src.at(16, 16)[0] = 5.0f;
  openglow::GlowParams a, b;
  b.tint_color[1] = 0.0f;  // color set but tint unchecked
  CHECK(render(src, a).px == render(src, b).px);
}

void test_opaque_alpha_stays_opaque() {
  Image src = opaque(40, 30);
  for (int y = 0; y < 30; ++y)
    for (int x = 0; x < 40; ++x) src.at(x, y)[0] = (x + y) / 70.0f;
  src.at(20, 15)[1] = 6.0f;
  const Image dst = render(src, {});
  for (int y = 0; y < 30; ++y)
    for (int x = 0; x < 40; ++x) {
      CHECK(dst.at(x, y)[3] == 1.0f);
      CHECK(dst.at(x, y)[0] >= src.at(x, y)[0] - 1e-6f);  // glow only adds light
    }
}

// A bright opaque square on a fully transparent background, like a title.
Image square_on_transparent(float garbage) {
  Image src(96, 96);
  for (int y = 0; y < 96; ++y)
    for (int x = 0; x < 96; ++x) {
      float* p = src.at(x, y);
      p[0] = p[1] = p[2] = garbage;  // RGB under alpha 0 must not matter
    }
  for (int y = 44; y < 52; ++y)
    for (int x = 44; x < 52; ++x) {
      float* p = src.at(x, y);
      p[0] = 1.0f;
      p[1] = 0.6f;
      p[2] = 0.2f;
      p[3] = 1.0f;
    }
  return src;
}

void test_glow_spills_onto_transparent_background() {
  openglow::GlowParams p;
  p.radius = 60;
  p.exposure = 2.0f;
  const Image dst = render(square_on_transparent(0.0f), p);
  // The square stays opaque.
  CHECK(dst.at(47, 47)[3] == 1.0f);
  // Outside it there is now an aura: visible, colored like the source, and
  // fading with distance.
  const float* near = dst.at(51 + 3, 47);
  const float* mid = dst.at(51 + 10, 47);
  const float* far = dst.at(51 + 30, 47);
  CHECK(near[3] > 0.0f);
  CHECK(mid[3] > 0.0f);
  CHECK(near[3] > mid[3]);
  CHECK(mid[3] > far[3]);
  CHECK(near[0] > near[1] && near[1] > near[2]);  // orange like the square
  CHECK(mid[0] > 0.0f);
}

void test_transparent_rgb_is_ignored() {
  openglow::GlowParams p;
  p.radius = 60;
  CHECK(render(square_on_transparent(0.0f), p).px == render(square_on_transparent(5.0f), p).px);
}

void test_threshold_zero_is_unchanged() {
  Image src = opaque(64, 48);
  src.at(20, 20)[0] = 3.0f;
  src.at(40, 30)[1] = 0.4f;
  openglow::GlowParams off, zero;
  zero.threshold = 0.0f;
  zero.radius = off.radius = 80;
  CHECK(render(src, off).px == render(src, zero).px);
}

void test_threshold_keeps_only_bright_parts() {
  openglow::GlowParams p;
  p.radius = 600;  // relative to 1080 lines: about 35 px on this 64 px frame
  p.threshold = 0.6f;

  // A dim area (0.3, below the threshold) adds no glow at all...
  Image dim = opaque(64, 64);
  for (int y = 16; y < 48; ++y)
    for (int x = 16; x < 48; ++x) dim.at(x, y)[0] = dim.at(x, y)[1] = dim.at(x, y)[2] = 0.3f;
  const Image dim_out = render(dim, p);
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x) CHECK(std::fabs(dim_out.at(x, y)[0] - dim.at(x, y)[0]) < 1e-6f);
  // ...while without the threshold it does glow.
  openglow::GlowParams no_threshold = p;
  no_threshold.threshold = 0.0f;
  CHECK(render(dim, no_threshold).at(10, 32)[0] > 0.01f);

  // A bright spot (1.0, above the threshold) still glows.
  Image spot = opaque(64, 64);
  for (int y = 30; y < 34; ++y)
    for (int x = 30; x < 34; ++x) spot.at(x, y)[0] = spot.at(x, y)[1] = spot.at(x, y)[2] = 1.0f;
  CHECK(render(spot, p).at(33 + 6, 32)[0] > 0.01f);
}

void test_channel_order_matches_rgba() {
  // A red point in BGRA must glow red, same as in RGBA.
  Image rgba = opaque(48, 48), bgra = opaque(48, 48), argb(48, 48);
  rgba.at(24, 24)[0] = 5.0f;
  bgra.at(24, 24)[2] = 5.0f;
  for (int y = 0; y < 48; ++y)
    for (int x = 0; x < 48; ++x) argb.at(x, y)[0] = 1.0f;
  argb.at(24, 24)[1] = 5.0f;
  openglow::GlowParams p;
  p.tint = true;
  p.tint_color[0] = 1.0f;
  p.tint_color[1] = 0.5f;
  p.tint_color[2] = 0.25f;
  const Image a = render(rgba, p, openglow::kRGBA);
  const Image b = render(bgra, p, openglow::kBGRA);
  const Image c = render(argb, p, openglow::kARGB);
  for (int y = 0; y < 48; ++y)
    for (int x = 0; x < 48; ++x) {
      const float* pa = a.at(x, y);
      const float* pb = b.at(x, y);
      const float* pc = c.at(x, y);
      CHECK(pa[0] == pb[2] && pa[1] == pb[1] && pa[2] == pb[0] && pa[3] == pb[3]);
      CHECK(pa[0] == pc[1] && pa[1] == pc[2] && pa[2] == pc[3] && pa[3] == pc[0]);
    }
}

void test_in_place_matches_separate_buffers() {
  Image src = opaque(50, 40);
  src.at(10, 10)[1] = 3.0f;
  src.at(40, 30)[2] = 2.0f;
  const Image expected = render(src, {});
  openglow::render_glow(src.cview(), src.view(), {});
  CHECK(src.px == expected.px);
}

void test_tiny_and_odd_sizes() {
  const int sizes[][2] = {{1, 1}, {1, 7}, {7, 1}, {2, 3}, {3, 2}, {33, 17}};
  for (const auto& s : sizes) {
    Image src = opaque(s[0], s[1]);
    src.at(0, 0)[0] = 1.0f;
    openglow::GlowParams p;
    p.radius = 2000;
    const Image dst = render(src, p);
    for (float v : dst.px) CHECK(std::isfinite(v));
    CHECK(dst.at(0, 0)[0] >= 1.0f);
  }
}

void test_padded_rows_are_left_alone() {
  Image src(20, 10, 3);
  Image dst(20, 10, 9);
  for (float& v : src.px) v = 0.5f;
  openglow::render_glow(src.cview(), dst.view(), {});
  for (int y = 0; y < 10; ++y)
    for (std::size_t x = 20 * 4; x < dst.stride; ++x) CHECK(dst.px[y * dst.stride + x] == 0.0f);
}

}  // namespace

int main() {
  test_black_stays_black();
  test_bright_point_spreads_symmetrically();
  test_larger_radius_spreads_wider();
  test_radius_change_is_smooth();
  test_exposure_scales_glow();
  test_energy_does_not_depend_on_radius();
  test_tint_colors_glow_and_keeps_luminance();
  test_tint_off_ignores_color();
  test_opaque_alpha_stays_opaque();
  test_glow_spills_onto_transparent_background();
  test_transparent_rgb_is_ignored();
  test_threshold_zero_is_unchanged();
  test_threshold_keeps_only_bright_parts();
  test_channel_order_matches_rgba();
  test_in_place_matches_separate_buffers();
  test_tiny_and_odd_sizes();
  test_padded_rows_are_left_alone();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all core tests passed\n");
  return 0;
}
