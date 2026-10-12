// Runs the CUDA glow on the local GPU and compares it with the CPU core (see
// gpu_test_common.h). Exits 77 (skipped) when there is no CUDA device.
#include <cuda_runtime.h>

#include <string>

#include "GlowCuda.h"
#include "gpu_test_common.h"

namespace {

constexpr int kSkip = 77;

class CudaApi : public gpu_test::GpuApi {
 public:
  bool Init() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    // Premiere hands the filter its own stream; use a non-default one too.
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess;
  }
  ~CudaApi() override {
    if (stream_) cudaStreamDestroy(stream_);
  }

  const char* Name() const override { return "CUDA"; }

  void* CreateBuffer(std::size_t bytes) override {
    void* p = nullptr;
    return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
  }
  void DestroyBuffer(void* buffer) override { cudaFree(buffer); }
  void Upload(void* dst, const void* data, std::size_t bytes) override {
    // From pageable memory, cudaMemcpy may return before the copy lands, and
    // the render stream doesn't wait for the default stream: wait here.
    cudaMemcpy(dst, data, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
  }
  void Download(void* src, void* data, std::size_t bytes) override {
    cudaMemcpy(data, src, bytes, cudaMemcpyDeviceToHost);
  }

  std::unique_ptr<openglow_gpu::Backend> MakeBackend(std::function<void*(std::size_t)> allocate,
                                                     std::function<void(void*)> free) override {
    return std::make_unique<openglow_gpu::CudaBackend>(stream_, std::move(allocate),
                                                        std::move(free));
  }

 private:
  cudaStream_t stream_ = nullptr;
};

// --profile: times each pass of a 4K 32f render with CUDA events.
class ProfilingBackend : public openglow_gpu::Backend {
 public:
  explicit ProfilingBackend(openglow_gpu::Backend& inner) : inner_(inner) {}
  void* Allocate(std::size_t bytes) override { return inner_.Allocate(bytes); }
  void Free(void* memory) override { inner_.Free(memory); }
  bool DownsampleFirst(const openglow_gpu::Frame& s, const openglow_gpu::Plane& d, float g,
                       float t, float k) override {
    return Timed("downsample_first", [&] { return inner_.DownsampleFirst(s, d, g, t, k); });
  }
  bool Downsample(const openglow_gpu::Plane& s, const openglow_gpu::Plane& d) override {
    return Timed("downsample", [&] { return inner_.Downsample(s, d); });
  }
  bool UpsampleAdd(const openglow_gpu::Plane& s, const openglow_gpu::Plane& d, float w) override {
    return Timed("upsample_add", [&] { return inner_.UpsampleAdd(s, d, w); });
  }
  bool Composite(const openglow_gpu::Frame& s, const openglow_gpu::Plane& g,
                 const openglow_gpu::Plane& n, float nw, const openglow_gpu::Frame& d,
                 const float t[3]) override {
    return Timed("composite", [&] { return inner_.Composite(s, g, n, nw, d, t); });
  }
  bool Finish() override { return inner_.Finish(); }

  void Print(int runs) {
    for (auto& e : totals_) std::printf("  %-18s %6.3f ms\n", e.first.c_str(), e.second / runs);
  }

 private:
  template <typename Fn>
  bool Timed(const char* name, Fn fn) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    const bool ok = fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    totals_[name] += ms;
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ok;
  }

  openglow_gpu::Backend& inner_;
  std::map<std::string, double> totals_;
};

void Profile(CudaApi& api, int w, int h, bool half) {
  gpu_test::HostFrame in = gpu_test::MakeFrame(w, h, half, 0);
  void* buf = api.CreateBuffer(in.bytes.size());
  api.Upload(buf, in.bytes.data(), in.bytes.size());
  gpu_test::Runner runner(api);
  auto inner = runner.Backend();
  // Same stream as the events: the default (legacy) stream.
  auto direct = std::make_unique<openglow_gpu::CudaBackend>(
      nullptr, [&](std::size_t b) { return inner->Allocate(b); },
      [&](void* m) { inner->Free(m); });
  ProfilingBackend profiled(*direct);
  openglow::GlowParams params;
  params.radius = 500;
  openglow_gpu::Frame frame{buf, w, h, in.pitch, half};
  openglow_gpu::RunGlow(*direct, frame, frame, params);
  const int runs = 20;
  for (int i = 0; i < runs; ++i) openglow_gpu::RunGlow(profiled, frame, frame, params);
  std::printf("profile %dx%d %s:\n", w, h, half ? "16f" : "32f");
  profiled.Print(runs);
  api.DestroyBuffer(buf);
}

}  // namespace

int main(int argc, char** argv) {
  CudaApi api;
  if (!api.Init()) {
    std::printf("no CUDA device, skipping\n");
    return kSkip;
  }
  if (argc > 1 && std::string(argv[1]) == "--profile") {
    Profile(api, 3840, 2160, false);
    Profile(api, 3840, 2160, true);
    return 0;
  }
  gpu_test::Runner runner(api);
  return runner.RunAll();
}
