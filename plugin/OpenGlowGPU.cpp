// Premiere Pro GPU filter for OpenGlow. Premiere looks up xGPUFilterEntry in
// the same .aex as the CPU effect (matched through the PiPL) and calls it
// whenever the project renders on the GPU. If an instance can't be created,
// Premiere renders that clip with the CPU effect instead.
//
// Premiere on Windows hands out CUDA or DirectX 12 devices (OpenCL was dropped
// in 2021). DirectX is always built; CUDA when nvcc was found at configure
// time (OPENGLOW_HAS_CUDA).
#include "AEConfig.h"
#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "PrSDKAESupport.h"
#include "PrGPUFilterModule.h"

#include <memory>
#include <mutex>
#include <vector>

#include "OpenGlowParams.h"
#include "gpu/GlowDX.h"
#include "gpu/GlowGpu.h"
#if OPENGLOW_HAS_CUDA
#include "gpu/GlowCuda.h"
#endif

namespace {

using openglow_gpu::Frame;
using openglow_gpu::Plane;

// One DirectX context and shader set per Premiere device index.
std::mutex g_dx_devices_mutex;
std::vector<std::unique_ptr<openglow_gpu::DXDevice>> g_dx_devices;

openglow_gpu::DXDevice* GetDXDevice(csSDK_uint32 index, const PrGPUDeviceInfo& info) {
  std::lock_guard<std::mutex> lock(g_dx_devices_mutex);
  if (index >= g_dx_devices.size()) g_dx_devices.resize(index + 1);
  if (!g_dx_devices[index]) {
    auto device = std::make_unique<openglow_gpu::DXDevice>();
    device->context = std::make_shared<DXContext>();
    if (!device->context->Initialize(static_cast<ID3D12Device*>(info.outDeviceHandle),
                                     static_cast<ID3D12CommandQueue*>(info.outCommandQueueHandle))) {
      return nullptr;
    }
    // Shaders live in "DirectX_Assets" next to the .aex.
    std::wstring cso, rs;
    if (!GetShaderPath(L"", cso, rs)) return nullptr;
    const std::wstring dir = cso.substr(0, cso.size() - 4);  // strip ".cso"
    if (!device->shaders.Load(*device->context, dir)) return nullptr;
    g_dx_devices[index] = std::move(device);
  }
  return g_dx_devices[index].get();
}

#if OPENGLOW_HAS_CUDA
class CudaBackend : public openglow_gpu::Backend {
 public:
  CudaBackend(PrSDKGPUDeviceSuite* suite, csSDK_uint32 device, void* stream)
      : suite_(suite), device_(device), stream_(stream) {}

  void* Allocate(std::size_t bytes) override {
    void* memory = nullptr;
    return suite_->AllocateDeviceMemory(device_, bytes, &memory) == suiteError_NoError ? memory
                                                                                      : nullptr;
  }
  void Free(void* memory) override { suite_->FreeDeviceMemory(device_, memory); }

  bool DownsampleFirst(const Frame& src, const Plane& dst, float gain) override {
    return openglow_gpu::CudaDownsampleFirst(src, dst, gain, stream_);
  }
  bool Downsample(const Plane& src, const Plane& dst) override {
    return openglow_gpu::CudaDownsample(src, dst, stream_);
  }
  bool UpsampleAdd(const Plane& src, const Plane& dst, float src_weight) override {
    return openglow_gpu::CudaUpsampleAdd(src, dst, src_weight, stream_);
  }
  bool Composite(const Frame& src, const Plane& glow, const Frame& dst,
                 const float tint_bgr[3]) override {
    return openglow_gpu::CudaComposite(src, glow, dst, tint_bgr, stream_);
  }
  bool Finish() override { return openglow_gpu::CudaFinish(stream_); }

 private:
  PrSDKGPUDeviceSuite* suite_;
  csSDK_uint32 device_;
  void* stream_;
};
#endif

float ParamFloat(const PrParam& p) {
  switch (p.mType) {
    case kPrParamType_Float32:
      return p.mFloat32;
    case kPrParamType_Float64:
      return static_cast<float>(p.mFloat64);
    case kPrParamType_Int32:
      return static_cast<float>(p.mInt32);
    default:
      return 0.0f;
  }
}

bool ParamBool(const PrParam& p) {
  switch (p.mType) {
    case kPrParamType_Bool:
      return p.mBool != 0;
    case kPrParamType_Int32:
      return p.mInt32 != 0;
    default:
      return false;
  }
}

// Color params arrive packed as ARGB in an integer, but the SDK doesn't
// document the depth. Alpha is always opaque, which tells them apart: 16 bits
// per channel puts it in the upper 32 bits, 8 bits per channel in bits 24-31.
void ParamColor(const PrParam& p, float rgb[3]) {
  const unsigned long long v = p.mType == kPrParamType_Int32
                                   ? static_cast<unsigned int>(p.mInt32)
                                   : static_cast<unsigned long long>(p.mInt64);
  if (v >> 32) {
    rgb[0] = ((v >> 32) & 0xFFFF) / 65535.0f;
    rgb[1] = ((v >> 16) & 0xFFFF) / 65535.0f;
    rgb[2] = (v & 0xFFFF) / 65535.0f;
  } else {
    rgb[0] = ((v >> 16) & 0xFF) / 255.0f;
    rgb[1] = ((v >> 8) & 0xFF) / 255.0f;
    rgb[2] = (v & 0xFF) / 255.0f;
  }
}

class OpenGlowGPU : public PrGPUFilterBase {
 public:
  // Called once when Premiere unloads the GPU filter (the index is the
  // filter's, not a device's), so drop every device.
  static prSuiteError Shutdown(piSuitesPtr /*suites*/, csSDK_int32 /*index*/) {
    std::lock_guard<std::mutex> lock(g_dx_devices_mutex);
    g_dx_devices.clear();
    return suiteError_NoError;
  }

  prSuiteError Initialize(PrGPUFilterInstance* instance) override {
    PrGPUFilterBase::Initialize(instance);
    if (!mDeviceInfo.outMeetsMinimumRequirementsForAcceleration) return suiteError_Fail;
    switch (mDeviceInfo.outDeviceFramework) {
      case PrGPUDeviceFramework_DirectX:
        dx_device_ = GetDXDevice(mDeviceIndex, mDeviceInfo);
        return dx_device_ ? suiteError_NoError : suiteError_Fail;
#if OPENGLOW_HAS_CUDA
      case PrGPUDeviceFramework_CUDA:
        return suiteError_NoError;
#endif
      default:
        return suiteError_Fail;  // Premiere falls back to the CPU effect.
    }
  }

  prSuiteError Render(const PrGPUFilterRenderParams* render, const PPixHand* in_frames,
                      csSDK_size_t in_frame_count, PPixHand* out_frame) override {
    if (in_frame_count < 1 || !out_frame) return suiteError_Fail;

    const PrTime time = render->inClipTime;
    openglow::GlowParams params;
    params.exposure = ParamFloat(GetParam(OPENGLOW_EXPOSURE, time));
    params.radius = ParamFloat(GetParam(OPENGLOW_RADIUS, time));
    params.tint = ParamBool(GetParam(OPENGLOW_TINT, time));
    ParamColor(GetParam(OPENGLOW_TINT_COLOR, time), params.tint_color);

    Frame src, dst;
    if (!DescribeFrame(in_frames[0], src) || !DescribeFrame(*out_frame, dst)) {
      return suiteError_Fail;
    }

    bool ok = false;
    if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_DirectX) {
      openglow_gpu::DXBackend backend(
          *dx_device_, [this](std::size_t bytes) { return AllocateDevice(bytes); },
          [this](void* memory) { mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, memory); });
      std::lock_guard<std::mutex> lock(dx_device_->mutex);
      ok = openglow_gpu::RunGlow(backend, src, dst, params);
    }
#if OPENGLOW_HAS_CUDA
    else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA) {
      CudaBackend backend(mGPUDeviceSuite, mDeviceIndex, mDeviceInfo.outCommandQueueHandle);
      ok = openglow_gpu::RunGlow(backend, src, dst, params);
    }
#endif
    return ok ? suiteError_NoError : suiteError_Fail;
  }

 private:
  void* AllocateDevice(std::size_t bytes) {
    void* memory = nullptr;
    if (mGPUDeviceSuite->AllocateDeviceMemory(mDeviceIndex, bytes, &memory) != suiteError_NoError) {
      return nullptr;
    }
    return memory;
  }

  bool DescribeFrame(PPixHand pix, Frame& frame) {
    PrPixelFormat format = PrPixelFormat_Invalid;
    mPPixSuite->GetPixelFormat(pix, &format);
    if (format != PrPixelFormat_GPU_BGRA_4444_32f && format != PrPixelFormat_GPU_BGRA_4444_16f) {
      return false;
    }
    prRect bounds = {};
    mPPixSuite->GetBounds(pix, &bounds);
    csSDK_int32 row_bytes = 0;
    mPPixSuite->GetRowBytes(pix, &row_bytes);
    void* data = nullptr;
    mGPUDeviceSuite->GetGPUPPixData(pix, &data);

    frame.data = data;
    frame.width = bounds.right - bounds.left;
    frame.height = bounds.bottom - bounds.top;
    frame.pitch = row_bytes;
    frame.half = format == PrPixelFormat_GPU_BGRA_4444_16f;
    return data && frame.width > 0 && frame.height > 0 && row_bytes > 0;
  }

  openglow_gpu::DXDevice* dx_device_ = nullptr;
};

}  // namespace

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<OpenGlowGPU>)
