// Glow via a mip pyramid (the "bloom" technique from real-time rendering):
//
//   1. Linearize the image, keep what passes the threshold, weight it by
//      alpha and scale it by 2^exposure.
//   2. Downsample repeatedly by 2 with a 4x4 tent filter. Level k is the image
//      blurred by roughly 2^k pixels.
//   3. Walk back up, upsampling each level bilinearly and adding it to the
//      one above. Summing all levels gives a wide, soft falloff, close to the
//      inverse-square look of physically based glows.
//   4. Tint, add the glow onto the original (premultiplied, so it spills
//      onto transparent areas) and return to gamma.
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

// Soft-knee bright pass on linear RGB: keeps what is above the threshold t,
// easing in over [t - knee, t + knee] so edges don't pop. Scales all three
// channels by the same factor, so hue is kept.
inline void bright_pass(float c[3], float t, float knee) {
  const float b = std::max(c[0], std::max(c[1], c[2]));
  float soft = std::min(std::max(b - t + knee, 0.0f), 2.0f * knee);
  soft = soft * soft / (4.0f * knee + 1e-5f);
  const float contrib = std::max(soft, b - t) / std::max(b, 1e-5f);
  c[0] *= contrib;
  c[1] *= contrib;
  c[2] *= contrib;
}

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

  // The threshold slider is in gamma like the image; compare in linear.
  const float t = std::max(0.0f, params.threshold);
  plan.threshold = to_linear(t);
  plan.knee = 0.5f * plan.threshold;
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

  // The first level reads the source directly, so no full-resolution copy is
  // made: linearize, keep what passes the threshold, weight by alpha (only
  // visible pixels emit light; RGB under alpha 0 is ignored) and expose.
  const float gain = plan.gain, threshold = plan.threshold, knee = plan.knee;
  downsample(
      width, height, [&](int y) { return src.row(y); },
      [&](const float* p, float* out) {
        float c[3] = {to_linear(p[order.r]), to_linear(p[order.g]), to_linear(p[order.b])};
        if (threshold > 0.0f) bright_pass(c, threshold, knee);
        const float a = p[order.a];
        out[0] = c[0] * a * gain;
        out[1] = c[1] * a * gain;
        out[2] = c[2] * a * gain;
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
        // Composite premultiplied: source (linear * alpha) plus glow. The glow
        // covers what it lights up, so alpha grows by the glow's coverage. With
        // alpha 1 this is exactly source + glow, alpha 1.
        const float* p = in + x * 4;
        float* q = out + x * 4;
        const float alpha = p[order.a];
        const float gr = g[0] * tr, gg = g[1] * tg, gb = g[2] * tb;
        const float pr = to_linear(p[order.r]) * alpha + gr;
        const float pg = to_linear(p[order.g]) * alpha + gg;
        const float pb = to_linear(p[order.b]) * alpha + gb;
        const float coverage = std::min(std::max(std::max(gr, std::max(gg, gb)), 0.0f), 1.0f);
        const float out_a = alpha + coverage * (1.0f - alpha);
        if (out_a > 0.0f) {
          q[order.r] = to_gamma(pr / out_a);
          q[order.g] = to_gamma(pg / out_a);
          q[order.b] = to_gamma(pb / out_a);
        } else {
          q[order.r] = q[order.g] = q[order.b] = 0.0f;
        }
        q[order.a] = out_a;
      }
    }
  });
}

}  // namespace openglow
