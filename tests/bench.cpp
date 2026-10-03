// Times render_glow on full HD and 4K frames: openglow_bench [iterations]
#include "openglow/glow.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 10;
  const int sizes[][2] = {{1920, 1080}, {3840, 2160}};
  const float radii[] = {20.0f, 500.0f};
  for (const auto& s : sizes) {
    const int w = s[0], h = s[1];
    std::vector<float> src(static_cast<std::size_t>(w) * h * 4), dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i % 97) / 97.0f;
    const openglow::ConstImageView in{src.data(), w, h, static_cast<std::size_t>(w) * 4};
    const openglow::ImageView out{dst.data(), w, h, static_cast<std::size_t>(w) * 4};
    for (float radius : radii) {
      openglow::GlowParams p;
      p.radius = radius;
      openglow::render_glow(in, out, p);  // warm up
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iterations; ++i) openglow::render_glow(in, out, p);
      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
      std::printf("%dx%d radius %5.0f: %7.2f ms/frame\n", w, h, radius, ms);
    }
  }
  return 0;
}
