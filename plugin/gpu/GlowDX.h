// DirectX 12 backend for the GPU glow, built on DirectXUtils from the
// Premiere Pro SDK (compiled from the SDK, never copied into this repo).
#pragma once

#include <functional>
#include <mutex>
#include <string>

#include "DirectXUtils.h"
#include "GlowGpu.h"

namespace openglow_gpu {

// Compiled kernels for one device. Files are "<dir>OpenGlow_<pass>.cso/.rs",
// built from OpenGlow.hlsl.
struct DXShaders {
  ShaderObjectPtr downsample_first;
  ShaderObjectPtr downsample;
  ShaderObjectPtr upsample_add;
  ShaderObjectPtr composite;

  bool Load(DXContext& context, const std::wstring& dir);
};

// Everything one device needs. Passes are recorded into DXContext's single
// command list, so renders on the same device must hold the mutex.
struct DXDevice {
  DXContextPtr context;
  DXShaders shaders;
  std::mutex mutex;
};

class DXBackend : public Backend {
 public:
  using AllocateFn = std::function<void*(std::size_t)>;
  using FreeFn = std::function<void(void*)>;

  DXBackend(DXDevice& device, AllocateFn allocate, FreeFn free)
      : device_(device), allocate_(std::move(allocate)), free_(std::move(free)) {}

  void* Allocate(std::size_t bytes) override { return allocate_(bytes); }
  void Free(void* memory) override { free_(memory); }

  bool DownsampleFirst(const Frame& src, const Plane& dst, float gain) override;
  bool Downsample(const Plane& src, const Plane& dst) override;
  bool UpsampleAdd(const Plane& src, const Plane& dst, float src_weight) override;
  bool Composite(const Frame& src, const Plane& glow, const Frame& dst,
                 const float tint_bgr[3]) override;
  bool Finish() override;

 private:
  DXDevice& device_;
  AllocateFn allocate_;
  FreeFn free_;
};

}  // namespace openglow_gpu
