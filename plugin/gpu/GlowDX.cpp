#include "GlowDX.h"

#include <initializer_list>

namespace openglow_gpu {
namespace {

// Mirrors the cbuffer in OpenGlow.hlsl (bound as 16 root constants): 4-byte
// scalars only, so the C and HLSL packing rules agree.
struct Params {
  int src_pitch = 0;
  int src_half = 0;
  int src_width = 0;
  int src_height = 0;
  int dst_pitch = 0;
  int dst_half = 0;
  int dst_width = 0;
  int dst_height = 0;
  int glow_width = 0;
  int glow_height = 0;
  float scale = 1.0f;
  float tint_b = 1.0f;
  float tint_g = 1.0f;
  float tint_r = 1.0f;
  float threshold = 0.0f;
  float knee = 0.0f;
};
static_assert(sizeof(Params) == 64, "Params must match the HLSL cbuffer");

struct View {
  void* resource;
};

// Must match [numthreads] in OpenGlow.hlsl.
constexpr UINT kGroupSize = 16;

// Records one pass into the device's command list; nothing runs until
// Finish(). Parameters go in as root constants and buffers as root UAVs (see
// the root signatures in OpenGlow.hlsl), so a pass allocates nothing.
bool Dispatch(DXDevice& device, const ShaderObjectPtr& shader, const Params& params,
              std::initializer_list<View> views, int width, int height) {
  if (!shader || !shader->mPipelineState) return false;
  ID3D12GraphicsCommandList* list = device.context->mCommandList.Get();
  list->SetComputeRootSignature(shader->mRootSignature.Get());
  list->SetPipelineState(shader->mPipelineState.Get());
  list->SetComputeRoot32BitConstants(0, sizeof(Params) / 4, &params, 0);
  UINT slot = 1;
  for (const View& view : views) {
    auto* resource = static_cast<ID3D12Resource*>(view.resource);
    if (!resource) return false;
    list->SetComputeRootUnorderedAccessView(slot++, resource->GetGPUVirtualAddress());
  }
  list->Dispatch((width + kGroupSize - 1) / kGroupSize, (height + kGroupSize - 1) / kGroupSize, 1);

  // Each pass reads what the previous one wrote.
  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.UAV.pResource = nullptr;
  list->ResourceBarrier(1, &barrier);
  return true;
}

void SetSource(Params& p, const Frame& f) {
  p.src_pitch = f.pitch;
  p.src_half = f.half ? 1 : 0;
  p.src_width = f.width;
  p.src_height = f.height;
}

}  // namespace

bool DXShaders::Load(DXContext& context, const std::wstring& dir) {
  const struct {
    const wchar_t* name;
    ShaderObjectPtr* shader;
  } passes[] = {
      {L"downsample_first", &downsample_first},
      {L"downsample", &downsample},
      {L"upsample_add", &upsample_add},
      {L"composite", &composite},
  };
  for (const auto& pass : passes) {
    const std::wstring base = dir + L"OpenGlow_" + pass.name;
    *pass.shader = std::make_shared<ShaderObject>();
    if (!context.LoadShader((base + L".cso").c_str(), (base + L".rs").c_str(), *pass.shader)) {
      return false;
    }
  }
  return true;
}

bool DXBackend::DownsampleFirst(const Frame& src, const Plane& dst, float gain, float threshold,
                                float knee) {
  Params p;
  SetSource(p, src);
  p.dst_width = dst.width;
  p.dst_height = dst.height;
  p.scale = gain;
  p.threshold = threshold;
  p.knee = knee;
  return Dispatch(device_, device_.shaders.downsample_first, p,
                  {{src.data}, {dst.data}}, dst.width, dst.height);
}

bool DXBackend::Downsample(const Plane& src, const Plane& dst) {
  Params p;
  p.src_width = src.width;
  p.src_height = src.height;
  p.dst_width = dst.width;
  p.dst_height = dst.height;
  return Dispatch(device_, device_.shaders.downsample, p,
                  {{src.data}, {dst.data}}, dst.width, dst.height);
}

bool DXBackend::UpsampleAdd(const Plane& src, const Plane& dst, float src_weight) {
  Params p;
  p.src_width = src.width;
  p.src_height = src.height;
  p.dst_width = dst.width;
  p.dst_height = dst.height;
  p.scale = src_weight;
  return Dispatch(device_, device_.shaders.upsample_add, p,
                  {{src.data}, {dst.data}}, dst.width, dst.height);
}

bool DXBackend::Composite(const Frame& src, const Plane& glow, const Frame& dst,
                          const float tint_bgr[3]) {
  Params p;
  SetSource(p, src);
  p.dst_pitch = dst.pitch;
  p.dst_half = dst.half ? 1 : 0;
  p.dst_width = dst.width;
  p.dst_height = dst.height;
  p.glow_width = glow.width;
  p.glow_height = glow.height;
  p.tint_b = tint_bgr[0];
  p.tint_g = tint_bgr[1];
  p.tint_r = tint_bgr[2];
  return Dispatch(device_, device_.shaders.composite, p,
                  {{src.data}, {glow.data}, {dst.data}},
                  dst.width, dst.height);
}

bool DXBackend::Finish() {
  // Submits the recorded passes and waits for them.
  device_.context->CloseWaitAndReset();
  return device_.context->mDevice->GetDeviceRemovedReason() == S_OK;
}

}  // namespace openglow_gpu