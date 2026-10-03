// Glow via a mip pyramid (the "bloom" technique from real-time rendering):
//
//   1. Linearize the image and scale it by 2^exposure.
//   2. Downsample repeatedly by 2 with a 4x4 tent filter. Level k is the image
//      blurred by roughly 2^k pixels.
//   3. Walk back up, upsampling each level bilinearly and adding it to the
//      one above. Summing all levels gives a wide, soft falloff, close to the
//      inverse-square look of physically based glows.
//   4. Tint, then add the glow onto the original and return to gamma.
//
// Each level has a quarter of the pixels of the previous one, so the cost is
// about the same for a radius of 20 or 500 pixels.
#include "openglow/glow.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

namespace openglow {
namespace {

// Gamma 2.0 instead of the exact Rec.709 curve: square and sqrt are much
// cheaper than pow and look the same for a glow. Sign is kept so negative
// float values survive the round trip.
inline float to_linear(float v) { return v * std::fabs(v); }
inline float to_gamma(float v) { return v < 0.0f ? -std::sqrt(-v) : std::sqrt(v); }

// RGBA float buffer owned by the pyramid.
struct Plane {
  int width = 0;
  int height = 0;
  std::unique_ptr<float[]> data;

  Plane() = default;
  Plane(int w, int h)
      : width(w), height(h), data(new float[static_cast<std::size_t>(w) * h * 4]) {}
  float* row(int y) { return data.get() + static_cast<std::size_t>(y) * width * 4; }
  const float* row(int y) const {
    return data.get() + static_cast<std::size_t>(y) * width * 4;
  }
};

// Runs fn(y_begin, y_end) over [0, rows) on the available cores.
template <typename Fn>
void parallel_rows(int rows, Fn fn) {
  static const int hw = std::max(1u, std::thread::hardware_concurrency());
  // Small planes aren't worth a thread each.
  const int threads = std::min(hw, std::max(1, rows / 32));
  if (threads <= 1) {
    fn(0, rows);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  const int chunk = (rows + threads - 1) / threads;
  for (int t = 1; t < threads; ++t) {
    const int begin = t * chunk;
    const int end = std::min(rows, begin + chunk);
    if (begin < end) pool.emplace_back(fn, begin, end);
  }
  fn(0, std::min(rows, chunk));
  for (auto& th : pool) th.join();
}

// Half-size downsample with a separable [1 3 3 1]/8 tent over a 4x4
// footprint. Smooth enough that small bright points don't flicker.
// load(p, out) reads one source pixel into RGBA.
template <typename RowFn, typename LoadFn>
void downsample(int sw, int sh, RowFn src_row, LoadFn load, Plane& dst) {
  parallel_rows(dst.height, [&](int y0, int y1) {
    static const float k[4] = {1.0f / 8, 3.0f / 8, 3.0f / 8, 1.0f / 8};
    for (int y = y0; y < y1; ++y) {
      const float* rows[4];
      for (int j = 0; j < 4; ++j) rows[j] = src_row(std::clamp(2 * y - 1 + j, 0, sh - 1));
      float* out = dst.row(y);
      for (int x = 0; x < dst.width; ++x) {
        int cols[4];
        for (int i = 0; i < 4; ++i) cols[i] = std::clamp(2 * x - 1 + i, 0, sw - 1) * 4;
        float acc[4] = {0, 0, 0, 0};
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const float w = k[j] * k[i];
            float p[4];
            load(rows[j] + cols[i], p);
            acc[0] += w * p[0];
            acc[1] += w * p[1];
            acc[2] += w * p[2];
            acc[3] += w * p[3];
          }
        }
        out[x * 4 + 0] = acc[0];
        out[x * 4 + 1] = acc[1];
        out[x * 4 + 2] = acc[2];
        out[x * 4 + 3] = acc[3];
      }
    }
  });
}

void downsample(const Plane& src, Plane& dst) {
  downsample(
      src.width, src.height, [&](int y) { return src.row(y); },
      [](const float* p, float* out) {
        out[0] = p[0];
        out[1] = p[1];
        out[2] = p[2];
        out[3] = p[3];
      },
      dst);
}

// Bilinear taps for a 2x upsample. Each level is exactly half of the one
// above (odd sizes round up and clamp at the edge), so pixel centers line up
// with a fixed factor of 2, not with the ratio of the sizes.
struct BilinearTap {
  int i0, i1;
  float f;
};

std::vector<BilinearTap> make_taps(int src_size, int dst_size) {
  std::vector<BilinearTap> taps(dst_size);
  for (int d = 0; d < dst_size; ++d) {
    const float s = std::max(0.0f, (d + 0.5f) * 0.5f - 0.5f);
    const int i0 = std::min(static_cast<int>(s), src_size - 1);
    taps[d] = {i0, std::min(i0 + 1, src_size - 1), s - static_cast<int>(s)};
  }
  return taps;
}

// dst = dst * dst_weight + upsample(src)
void upsample_add(const Plane& src, Plane& dst, float dst_weight) {
  const auto tx = make_taps(src.width, dst.width);
  const auto ty = make_taps(src.height, dst.height);
  parallel_rows(dst.height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const float* r0 = src.row(ty[y].i0);
      const float* r1 = src.row(ty[y].i1);
      const float fy = ty[y].f;
      float* out = dst.row(y);
      for (int x = 0; x < dst.width; ++x) {
        const int a = tx[x].i0 * 4, b = tx[x].i1 * 4;
        const float fx = tx[x].f;
        for (int c = 0; c < 4; ++c) {
          const float top = r0[a + c] + (r0[b + c] - r0[a + c]) * fx;
          const float bot = r1[a + c] + (r1[b + c] - r1[a + c]) * fx;
          out[x * 4 + c] = out[x * 4 + c] * dst_weight + top + (bot - top) * fy;
        }
      }
    }
  });
}

}  // namespace

GlowPlan plan_glow(int width, int height, const GlowParams& params) {
  GlowPlan plan;
  if (width <= 0 || height <= 0) return plan;

  // Radius is relative to a 1080p frame so the look doesn't change between
  // proxies, 4K, or Premiere's reduced playback resolution.
  const float radius_px = std::max(1.0f, params.radius * height / 1080.0f);
  // Level k blurs by about 2^k pixels. The fractional part fades the last
  // level in, so dragging the slider is smooth instead of stepping.
  const float levels_f = std::max(1.0f, std::log2(radius_px));
  int levels = std::min(static_cast<int>(std::ceil(levels_f)), GlowPlan::kMaxLevels);
  float last_weight = levels_f - std::floor(levels_f);
  if (last_weight <= 0.0f) last_weight = 1.0f;

  // Size the pyramid, stopping when a level would be smaller than 1 pixel.
  int count = 0;
  int w = width, h = height;
  while (count < levels) {
    w = (w + 1) / 2;
    h = (h + 1) / 2;
    plan.width[count] = w;
    plan.height[count] = h;
    ++count;
    if (w == 1 && h == 1) break;
  }
  if (count < levels) {
    levels = count;
    last_weight = 1.0f;
  }
  plan.levels = levels;
  plan.last_weight = last_weight;
  plan.gain = std::exp2(params.exposure);

  // Every level contributes equally (the last one partially), normalized so
  // the total glow energy doesn't depend on the radius. Tint keeps the glow's
  // luminance so changing the color doesn't change how bright it looks.
  const float total = (levels - 1) + last_weight;
  float tint[3] = {1.0f, 1.0f, 1.0f};
  if (params.tint) {
    const float* c = params.tint_color;
    const float luma = 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
    for (int i = 0; i < 3; ++i) tint[i] = luma > 1e-6f ? c[i] / luma : 0.0f;
  }
  for (int i = 0; i < 3; ++i) plan.tint[i] = tint[i] / total;
  return plan;
}

void render_glow(const ConstImageView& src, const ImageView& dst, const GlowParams& params,
                 ChannelOrder order) {
  const int width = std::min(src.width, dst.width);
  const int height = std::min(src.height, dst.height);
  const GlowPlan plan = plan_glow(width, height, params);
  const int levels = plan.levels;
  if (levels == 0) return;

  std::vector<Plane> pyramid;
  pyramid.reserve(levels);
  for (int k = 0; k < levels; ++k) pyramid.emplace_back(plan.width[k], plan.height[k]);

  // The first level reads the source directly, linearizing and applying the
  // exposure on the fly, so no full-resolution copy is made.
  const float gain = plan.gain;
  downsample(
      width, height, [&](int y) { return src.row(y); },
      [&](const float* p, float* out) {
        out[0] = to_linear(p[order.r]) * gain;
        out[1] = to_linear(p[order.g]) * gain;
        out[2] = to_linear(p[order.b]) * gain;
        out[3] = 0.0f;
      },
      pyramid[0]);
  for (int k = 1; k < levels; ++k) downsample(pyramid[k - 1], pyramid[k]);

  // Collapse: pyramid[k] = pyramid[k] + upsample(pyramid[k+1]). Scaling the
  // deepest level by last_weight before the walk applies its fade.
  {
    Plane& deepest = pyramid[levels - 1];
    const std::size_t n = static_cast<std::size_t>(deepest.width) * deepest.height * 4;
    for (std::size_t i = 0; i < n; ++i) deepest.data[i] *= plan.last_weight;
  }
  for (int k = levels - 2; k >= 0; --k) upsample_add(pyramid[k + 1], pyramid[k], 1.0f);

  const float tr = plan.tint[0], tg = plan.tint[1], tb = plan.tint[2];

  // Upsample the collapsed glow to full resolution and composite.
  const Plane& glow = pyramid[0];
  const auto tx = make_taps(glow.width, width);
  const auto ty = make_taps(glow.height, height);
  parallel_rows(height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const float* r0 = glow.row(ty[y].i0);
      const float* r1 = glow.row(ty[y].i1);
      const float fy = ty[y].f;
      const float* in = src.row(y);
      float* out = dst.row(y);
      for (int x = 0; x < width; ++x) {
        const int a = tx[x].i0 * 4, b = tx[x].i1 * 4;
        const float fx = tx[x].f;
        float g[3];
        for (int c = 0; c < 3; ++c) {
          const float top = r0[a + c] + (r0[b + c] - r0[a + c]) * fx;
          const float bot = r1[a + c] + (r1[b + c] - r1[a + c]) * fx;
          g[c] = top + (bot - top) * fy;
        }
        const float* p = in + x * 4;
        float* q = out + x * 4;
        const float alpha = p[order.a];
        q[order.r] = to_gamma(to_linear(p[order.r]) + g[0] * tr);
        q[order.g] = to_gamma(to_linear(p[order.g]) + g[1] * tg);
        q[order.b] = to_gamma(to_linear(p[order.b]) + g[2] * tb);
        q[order.a] = alpha;
      }
    }
  });
}

}  // namespace openglow
