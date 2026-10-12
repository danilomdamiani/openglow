// Shared GPU-vs-CPU checks for the GPU backends (gpu_dx_test, gpu_cuda_test).
// A test supplies a GpuApi (buffers, copies, a Backend); RunAll renders the
// same frames on the GPU and with the CPU core and compares them, then times
// a 1080p and a 4K frame.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "GlowGpu.h"
#include "openglow/glow.h"

namespace gpu_test {

// IEEE half <-> float on the host (round to nearest even).
inline float HalfToFloat(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {  // subnormal: normalize
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        --exp;
      }
      bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline uint16_t FloatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000);
  const int exp = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;
  if (exp >= 31) return sign | 0x7C00;  // overflow (and inf/nan) -> inf
  if (exp <= 0) {                       // subnormal or zero
    if (exp < -10) return sign;
    mant |= 0x800000;
    const int shift = 14 - exp;
    uint32_t h = mant >> shift;
    const uint32_t rest = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rest > half || (rest == half && (h & 1))) ++h;
    return sign | static_cast<uint16_t>(h);
  }
  uint32_t h = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t rest = mant & 0x1FFF;
  if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) ++h;  // may carry into exp: fine
  return sign | static_cast<uint16_t>(h);
}

// A BGRA host image with a padded pitch, as Premiere hands them out.
struct HostFrame {
  int width, height, pitch;  // pitch in bytes
  bool half;
  std::vector<uint8_t> bytes;

  HostFrame(int w, int h, bool half_float, int pad)
      : width(w), height(h), pitch(w * (half_float ? 8 : 16) + pad), half(half_float),
        bytes(static_cast<std::size_t>(pitch) * h, 0) {}

  float Get(int x, int y, int c) const {
    const uint8_t* row = bytes.data() + static_cast<std::size_t>(pitch) * y;
    if (half) {
      uint16_t v;
      std::memcpy(&v, row + (x * 4 + c) * 2, 2);
      return HalfToFloat(v);
    }
    float v;
    std::memcpy(&v, row + (x * 4 + c) * 4, 4);
    return v;
  }
  void Set(int x, int y, int c, float v) {
    uint8_t* row = bytes.data() + static_cast<std::size_t>(pitch) * y;
    if (half) {
      const uint16_t h = FloatToHalf(v);
      std::memcpy(row + (x * 4 + c) * 2, &h, 2);
    } else {
      std::memcpy(row + (x * 4 + c) * 4, &v, 4);
    }
  }
};

// Dim gradient, a few hot spots (above 1.0, like 32-bit footage) and partial
// alpha, so every term of the glow is exercised.
inline HostFrame MakeFrame(int w, int h, bool half, int pad) {
  HostFrame f(w, h, half, pad);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float u = static_cast<float>(x) / w, v = static_cast<float>(y) / h;
      float b = 0.1f * u, g = 0.2f * v, r = 0.05f + 0.1f * u * v;
      const int cx = x % 97, cy = y % 61;
      if (cx < 3 && cy < 3) {
        r = 4.0f;
        g = 2.5f;
        b = 1.0f;
      }
      if ((x * 7 + y * 13) % 211 == 0) b = 3.0f;
      f.Set(x, y, 0, b);
      f.Set(x, y, 1, g);
      f.Set(x, y, 2, r);
      f.Set(x, y, 3, 0.5f + 0.5f * u);
    }
  }
  return f;
}

// A title-like frame: transparent background (with garbage RGB, which must be
// ignored), an opaque bright bar and a half-transparent disc.
inline HostFrame MakeShapeFrame(int w, int h, bool half, int pad) {
  HostFrame f(w, h, half, pad);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float b = 7.0f, g = 0.3f, r = 2.0f, a = 0.0f;
      if (x > w / 4 && x < w / 2 && y > h / 3 && y < h / 2) {
        b = 0.2f;
        g = 0.7f;
        r = 1.0f;
        a = 1.0f;
      }
      const float dx = x - 0.7f * w, dy = y - 0.6f * h;
      if (dx * dx + dy * dy < 0.01f * w * w) {
        b = 1.5f;
        g = 0.4f;
        r = 0.1f;
        a = 0.5f;
      }
      f.Set(x, y, 0, b);
      f.Set(x, y, 1, g);
      f.Set(x, y, 2, r);
      f.Set(x, y, 3, a);
    }
  }
  return f;
}

// CPU reference on exactly the values the GPU reads.
inline std::vector<float> Reference(const HostFrame& in, const openglow::GlowParams& params) {
  const std::size_t stride = static_cast<std::size_t>(in.width) * 4;
  std::vector<float> src(stride * in.height), dst(stride * in.height);
  for (int y = 0; y < in.height; ++y)
    for (int x = 0; x < in.width; ++x)
      for (int c = 0; c < 4; ++c) src[y * stride + x * 4 + c] = in.Get(x, y, c);
  openglow::render_glow({src.data(), in.width, in.height, stride},
                        {dst.data(), in.width, in.height, stride}, params, openglow::kBGRA);
  return dst;
}

// What a test provides: device buffers, copies, and a backend that uses the
// given allocator.
struct GpuApi {
  virtual ~GpuApi() = default;
  virtual const char* Name() const = 0;
  virtual void* CreateBuffer(std::size_t bytes) = 0;
  virtual void DestroyBuffer(void* buffer) = 0;
  virtual void Upload(void* dst, const void* data, std::size_t bytes) = 0;
  virtual void Download(void* src, void* data, std::size_t bytes) = 0;
  virtual std::unique_ptr<openglow_gpu::Backend> MakeBackend(
      std::function<void*(std::size_t)> allocate, std::function<void(void*)> free) = 0;
};

// Pools freed pyramid buffers by size, like Premiere's AllocateDeviceMemory
// does, so the timings measure the glow rather than buffer creation.
class Runner {
 public:
  explicit Runner(GpuApi& api) : api_(api) {}
  ~Runner() {
    for (auto& entry : pool_) api_.DestroyBuffer(entry.second);
  }

  std::unique_ptr<openglow_gpu::Backend> Backend() {
    return api_.MakeBackend(
        [this](std::size_t bytes) -> void* {
          ++live_;
          auto it = pool_.find(bytes);
          if (it != pool_.end()) {
            void* b = it->second;
            pool_.erase(it);
            return b;
          }
          void* b = api_.CreateBuffer(bytes);
          sizes_[b] = bytes;
          return b;
        },
        [this](void* memory) {
          --live_;
          pool_.emplace(sizes_[memory], memory);
        });
  }

  void Check(const char* name, int w, int h, bool half, int pad, bool in_place,
             const openglow::GlowParams& params, bool shapes = false) {
    HostFrame in = shapes ? MakeShapeFrame(w, h, half, pad) : MakeFrame(w, h, half, pad);
    void* src_buf = api_.CreateBuffer(in.bytes.size());
    api_.Upload(src_buf, in.bytes.data(), in.bytes.size());
    void* dst_buf = in_place ? src_buf : api_.CreateBuffer(in.bytes.size());

    openglow_gpu::Frame src{src_buf, w, h, in.pitch, half};
    openglow_gpu::Frame dst = src;
    dst.data = dst_buf;
    auto backend = Backend();
    const bool ok = openglow_gpu::RunGlow(*backend, src, dst, params);

    HostFrame out(w, h, half, pad);
    api_.Download(dst_buf, out.bytes.data(), out.bytes.size());
    if (!in_place) api_.DestroyBuffer(dst_buf);
    api_.DestroyBuffer(src_buf);
    const std::vector<float> ref = Reference(in, params);

    // 32f: float rounding only. 16f: the output is rounded to half too.
    // Colors are compared premultiplied by alpha: where the glow barely covers
    // a transparent pixel, its color is a ratio of tiny numbers and invisible.
    const double tol = half ? 2e-3 : 2e-5;
    double worst = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const float* r = &ref[(static_cast<std::size_t>(y) * w + x) * 4];
        const double ra = r[3], oa = out.Get(x, y, 3);
        for (int c = 0; c < 4; ++c) {
          const double rv = c == 3 ? ra : r[c] * ra;
          const double ov = c == 3 ? oa : out.Get(x, y, c) * oa;
          worst = std::max(worst, std::fabs(ov - rv) / std::max(1.0, std::fabs(rv)));
        }
      }

    const bool pass = ok && worst <= tol && live_ == 0;
    if (!pass) ++failures_;
    std::printf("%-4s %-36s %4dx%-4d max rel err %.2e (tol %.0e)%s\n", pass ? "ok" : "FAIL",
                name, w, h, worst, tol, ok ? "" : "  [render failed]");
  }

  void Time(int w, int h, bool half) {
    HostFrame in = MakeFrame(w, h, half, 0);
    void* buf = api_.CreateBuffer(in.bytes.size());
    api_.Upload(buf, in.bytes.data(), in.bytes.size());
    openglow_gpu::Frame frame{buf, w, h, in.pitch, half};
    for (float radius : {20.0f, 500.0f}) {
      openglow::GlowParams params;
      params.radius = radius;
      auto backend = Backend();
      openglow_gpu::RunGlow(*backend, frame, frame, params);  // warm-up
      const int runs = 30;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < runs; ++i) openglow_gpu::RunGlow(*backend, frame, frame, params);
      const auto t1 = std::chrono::steady_clock::now();
      std::printf("time %dx%d %s radius %4.0f: %6.2f ms/frame (%s)\n", w, h, half ? "16f" : "32f",
                  radius, std::chrono::duration<double, std::milli>(t1 - t0).count() / runs,
                  api_.Name());
    }
    api_.DestroyBuffer(buf);
  }

  int RunAll() {
    openglow::GlowParams base;
    base.radius = 50;

    openglow::GlowParams bright = base;
    bright.exposure = 1.5f;
    bright.radius = 137.3f;  // fractional last level
    bright.tint = true;
    bright.tint_color[0] = 1.0f;
    bright.tint_color[1] = 0.45f;
    bright.tint_color[2] = 0.1f;

    openglow::GlowParams tiny = base;
    tiny.radius = 1;  // a single level

    openglow::GlowParams huge = base;
    huge.radius = 2000;  // pyramid runs out of pixels

    openglow::GlowParams aura = base;
    aura.radius = 400;
    aura.exposure = 1.0f;
    openglow::GlowParams thresholded = bright;
    thresholded.threshold = 0.6f;
    openglow::GlowParams thresholded_aura = aura;
    thresholded_aura.threshold = 0.8f;

    Check("32f default", 301, 173, false, 0, false, base);
    Check("32f exposure+tint+fraction, padded", 301, 173, false, 48, false, bright);
    Check("32f radius 1", 64, 40, false, 0, false, tiny);
    Check("32f radius 2000", 257, 129, false, 16, false, huge);
    Check("32f in place", 200, 120, false, 32, true, bright);
    Check("16f default", 301, 173, true, 0, false, base);
    Check("16f exposure+tint, padded, in place", 299, 171, true, 24, true, bright);
    Check("32f 1080p", 1920, 1080, false, 0, false, bright);
    Check("32f shapes on transparent", 320, 180, false, 0, false, aura, true);
    Check("16f shapes on transparent, in place", 320, 180, true, 16, true, aura, true);
    Check("32f threshold 0.6", 301, 173, false, 0, false, thresholded);
    Check("16f threshold 0.6, padded", 301, 173, true, 40, false, thresholded);
    Check("32f shapes + threshold 0.8", 320, 180, false, 0, false, thresholded_aura, true);

    Time(1920, 1080, false);
    Time(3840, 2160, false);
    Time(3840, 2160, true);

    std::printf(failures_ ? "%d check(s) failed\n" : "all checks passed\n", failures_);
    return failures_ ? 1 : 0;
  }

 private:
  GpuApi& api_;
  int live_ = 0;
  int failures_ = 0;
  std::multimap<std::size_t, void*> pool_;
  std::map<void*, std::size_t> sizes_;
};

}  // namespace gpu_test
