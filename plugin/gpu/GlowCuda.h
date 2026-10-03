// CUDA backend for the GPU glow (kernels in GlowCuda.cu). Pointers are device
// memory, pitches are in bytes, `stream` is the host's CUstream /
// cudaStream_t. Each launcher returns true if the launch succeeded.
#pragma once

#include <functional>
#include <utility>

#include "GlowGpu.h"

namespace openglow_gpu {

bool CudaDownsampleFirst(const Frame& src, const Plane& dst, float gain, float threshold,
                         float knee, void* stream);
bool CudaDownsample(const Plane& src, const Plane& dst, void* stream);
bool CudaUpsampleAdd(const Plane& src, const Plane& dst, float src_weight, void* stream);
bool CudaComposite(const Frame& src, const Plane& glow, const Plane& next, float next_weight,
                   const Frame& dst, const float tint_bgr[3], void* stream);
// Waits for every launch on the stream.
bool CudaFinish(void* stream);

// Launches on one stream; memory comes from the host (Premiere's GPU device
// suite, or cudaMalloc in the tests).
class CudaBackend : public Backend {
 public:
  using AllocateFn = std::function<void*(std::size_t)>;
  using FreeFn = std::function<void(void*)>;

  CudaBackend(void* stream, AllocateFn allocate, FreeFn free)
      : stream_(stream), allocate_(std::move(allocate)), free_(std::move(free)) {}

  void* Allocate(std::size_t bytes) override { return allocate_(bytes); }
  void Free(void* memory) override { free_(memory); }

  bool DownsampleFirst(const Frame& src, const Plane& dst, float gain, float threshold,
                       float knee) override {
    return CudaDownsampleFirst(src, dst, gain, threshold, knee, stream_);
  }
  bool Downsample(const Plane& src, const Plane& dst) override {
    return CudaDownsample(src, dst, stream_);
  }
  bool UpsampleAdd(const Plane& src, const Plane& dst, float src_weight) override {
    return CudaUpsampleAdd(src, dst, src_weight, stream_);
  }
  bool Composite(const Frame& src, const Plane& glow, const Plane& next, float next_weight,
                 const Frame& dst, const float tint_bgr[3]) override {
    return CudaComposite(src, glow, next, next_weight, dst, tint_bgr, stream_);
  }
  bool Finish() override { return CudaFinish(stream_); }

 private:
  void* stream_;
  AllocateFn allocate_;
  FreeFn free_;
};

}  // namespace openglow_gpu
