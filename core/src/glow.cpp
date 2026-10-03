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
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
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

// Persistent worker threads: a render runs about ten parallel passes, and
// starting threads for each one cost more than some of the passes.
//
// Several renders may run at once (hosts render frames in parallel), so jobs
// queue up; the caller of Run also works on its own job, so it always
// finishes even when every worker is busy elsewhere.
class ThreadPool {
 public:
  static ThreadPool& get() {
    // Never destroyed: the workers stay parked until the process exits
    // (joining threads while a plugin DLL unloads can deadlock).
    static ThreadPool* pool = new ThreadPool(
        std::max(1, static_cast<int>(std::thread::hardware_concurrency())) - 1);
    return *pool;
  }

  int threads() const { return workers_ + 1; }

  // Calls fn(i) for every i in [0, count), and returns when all are done.
  void run(int count, const std::function<void(int)>& fn) {
    Job job{&fn, count};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jobs_.push_back(&job);
    }
    work_.notify_all();
    work_on(job);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return job.done.load() == count && job.users == 0; });
    const auto it = std::find(jobs_.begin(), jobs_.end(), &job);
    if (it != jobs_.end()) jobs_.erase(it);
  }

 private:
  struct Job {
    const std::function<void(int)>* fn;
    int count;
    std::atomic<int> next{0};
    std::atomic<int> done{0};
    int users = 0;  // workers holding a pointer to it, guarded by mutex_
  };

  explicit ThreadPool(int workers) : workers_(workers) {
    for (int i = 0; i < workers; ++i) std::thread([this] { worker(); }).detach();
  }

  // Runs chunks of the job until none are left. Returns whether it ran any.
  bool work_on(Job& job) {
    bool any = false;
    for (int i; (i = job.next.fetch_add(1)) < job.count;) {
      (*job.fn)(i);
      any = true;
      if (job.done.fetch_add(1) + 1 == job.count) {
        std::lock_guard<std::mutex> lock(mutex_);
        done_.notify_all();
      }
    }
    return any;
  }

  void worker() {
    for (;;) {
      Job* job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_.wait(lock, [&] { return !jobs_.empty(); });
        job = jobs_.front();
        ++job->users;
      }
      const bool ran = work_on(*job);
      std::lock_guard<std::mutex> lock(mutex_);
      // A job with nothing left to take leaves the queue for the next one.
      if (!ran && !jobs_.empty() && jobs_.front() == job) jobs_.pop_front();
      if (--job->users == 0) done_.notify_all();
    }
  }

  const int workers_;
  std::mutex mutex_;
  std::condition_variable work_, done_;
  std::deque<Job*> jobs_;
};

// Runs fn(y_begin, y_end) over [0, rows) on the available cores.
template <typename Fn>
void parallel_rows(int rows, Fn fn) {
  ThreadPool& pool = ThreadPool::get();
  // Small planes aren't worth splitting.
  const int chunks = std::min(pool.threads(), std::max(1, rows / 32));
  if (chunks <= 1) {
    fn(0, rows);
    return;
  }
  const int chunk = (rows + chunks - 1) / chunks;
  pool.run(chunks, [&](int i) {
    const int begin = i * chunk;
    const int end = std::min(rows, begin + chunk);
    if (begin < end) fn(begin, end);
  });
}

// Half-size downsample with a separable [1 3 3 1]/8 tent over a 4x4
// footprint. Smooth enough that small bright points don't flicker.
// make_rows() is called once per thread and returns a function that gives
// source row y as RGBA floats.
template <typename MakeRows>
void downsample(int sw, int sh, MakeRows make_rows, Plane& dst) {
  parallel_rows(dst.height, [&](int y0, int y1) {
    static const float k[4] = {1.0f / 8, 3.0f / 8, 3.0f / 8, 1.0f / 8};
    auto src_row = make_rows();
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
            const float* p = rows[j] + cols[i];
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
      src.width, src.height, [&] { return [&](int y) { return src.row(y); }; }, dst);
}

// Source rows for the first level, converted once each: linearize, keep what
// passes the threshold, weight by alpha (only visible pixels emit light; RGB
// under alpha 0 is ignored) and expose. Consecutive output rows share two of
// their four source rows, so a ring of four converted rows is enough.
class LinearRows {
 public:
  LinearRows(const ConstImageView& src, int width, ChannelOrder order, const GlowPlan& plan)
      : src_(src), width_(width), order_(order), plan_(plan),
        buffer_(static_cast<std::size_t>(width) * 4 * 4) {}

  const float* operator()(int y) {
    float* row = buffer_.data() + static_cast<std::size_t>(y & 3) * width_ * 4;
    if (cached_[y & 3] != y) {
      const float* in = src_.row(y);
      for (int x = 0; x < width_; ++x) {
        const float* p = in + x * 4;
        float c[3] = {to_linear(p[order_.r]), to_linear(p[order_.g]), to_linear(p[order_.b])};
        if (plan_.threshold > 0.0f) bright_pass(c, plan_.threshold, plan_.knee);
        const float a = p[order_.a];
        float* out = row + x * 4;
        out[0] = c[0] * a * plan_.gain;
        out[1] = c[1] * a * plan_.gain;
        out[2] = c[2] * a * plan_.gain;
        out[3] = 0.0f;
      }
      cached_[y & 3] = y;
    }
    return row;
  }

 private:
  const ConstImageView& src_;
  const int width_;
  const ChannelOrder order_;
  const GlowPlan& plan_;
  std::vector<float> buffer_;
  int cached_[4] = {-1, -1, -1, -1};
};
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

  // The first level reads the source directly (a few converted rows at a
  // time), so no full-resolution copy is made.
  downsample(
      width, height, [&] { return LinearRows(src, width, order, plan); }, pyramid[0]);
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
