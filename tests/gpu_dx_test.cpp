// Runs the DirectX 12 glow on the local GPU and compares it with the CPU core,
// for 32f and 16f frames, padded pitches, in-place renders and edge-case
// radii. Then times a 1080p and a 4K frame. Exits 77 (skipped) when there is
// no DirectX 12 hardware device.
//
// Usage: openglow_gpu_dx_test <DirectX_Assets dir with trailing slash>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <DirectXPackedVector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "GlowDX.h"
#include "GlowGpu.h"
#include "openglow/glow.h"

using Microsoft::WRL::ComPtr;

namespace {

constexpr int kSkip = 77;

struct Gpu {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12Fence> fence;
  UINT64 fence_value = 0;
  HANDLE event = nullptr;
  std::string adapter;

  bool Init() {
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return false;
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0;
         factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                             IID_PPV_ARGS(&a)) != DXGI_ERROR_NOT_FOUND;
         ++i) {
      DXGI_ADAPTER_DESC1 desc;
      a->GetDesc1(&desc);
      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
      if (SUCCEEDED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        char name[128];
        std::snprintf(name, sizeof(name), "%ls", desc.Description);
        adapter = name;
        break;
      }
    }
    if (!device) return false;
    // DXContext records compute command lists, like Premiere's queue.
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)))) return false;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                              IID_PPV_ARGS(&allocator)))) {
      return false;
    }
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(),
                                         nullptr, IID_PPV_ARGS(&list)))) {
      return false;
    }
    list->Close();
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    return true;
  }

  ID3D12Resource* CreateBuffer(std::size_t bytes, D3D12_HEAP_TYPE heap) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    if (heap == D3D12_HEAP_TYPE_DEFAULT) rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (heap == D3D12_HEAP_TYPE_UPLOAD) state = D3D12_RESOURCE_STATE_GENERIC_READ;
    if (heap == D3D12_HEAP_TYPE_READBACK) state = D3D12_RESOURCE_STATE_COPY_DEST;
    ID3D12Resource* r = nullptr;
    device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                    IID_PPV_ARGS(&r));
    return r;
  }

  void Submit() {
    list->Close();
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.Get(), ++fence_value);
    fence->SetEventOnCompletion(fence_value, event);
    WaitForSingleObject(event, INFINITE);
  }

  void Upload(ID3D12Resource* dst, const void* data, std::size_t bytes) {
    ComPtr<ID3D12Resource> staging;
    staging.Attach(CreateBuffer(bytes, D3D12_HEAP_TYPE_UPLOAD));
    void* p = nullptr;
    staging->Map(0, nullptr, &p);
    std::memcpy(p, data, bytes);
    staging->Unmap(0, nullptr);
    allocator->Reset();
    list->Reset(allocator.Get(), nullptr);
    list->CopyBufferRegion(dst, 0, staging.Get(), 0, bytes);
    Submit();
  }

  void Download(ID3D12Resource* src, void* data, std::size_t bytes) {
    ComPtr<ID3D12Resource> staging;
    staging.Attach(CreateBuffer(bytes, D3D12_HEAP_TYPE_READBACK));
    allocator->Reset();
    list->Reset(allocator.Get(), nullptr);
    list->CopyBufferRegion(staging.Get(), 0, src, 0, bytes);
    Submit();
    void* p = nullptr;
    staging->Map(0, nullptr, &p);
    std::memcpy(data, p, bytes);
    staging->Unmap(0, nullptr);
  }
};

float HalfToFloat(uint16_t h) { return DirectX::PackedVector::XMConvertHalfToFloat(h); }
uint16_t FloatToHalf(float f) { return DirectX::PackedVector::XMConvertFloatToHalf(f); }

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
    if (half) return HalfToFloat(reinterpret_cast<const uint16_t*>(row)[x * 4 + c]);
    return reinterpret_cast<const float*>(row)[x * 4 + c];
  }
  void Set(int x, int y, int c, float v) {
    uint8_t* row = bytes.data() + static_cast<std::size_t>(pitch) * y;
    if (half) {
      reinterpret_cast<uint16_t*>(row)[x * 4 + c] = FloatToHalf(v);
    } else {
      reinterpret_cast<float*>(row)[x * 4 + c] = v;
    }
  }
};

// Dim gradient, a few hot spots (above 1.0, like 32-bit footage) and partial
// alpha, so every term of the glow is exercised.
HostFrame MakeFrame(int w, int h, bool half, int pad) {
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

// CPU reference on exactly the values the GPU reads.
std::vector<float> Reference(const HostFrame& in, const openglow::GlowParams& params) {
  const std::size_t stride = static_cast<std::size_t>(in.width) * 4;
  std::vector<float> src(stride * in.height), dst(stride * in.height);
  for (int y = 0; y < in.height; ++y)
    for (int x = 0; x < in.width; ++x)
      for (int c = 0; c < 4; ++c) src[y * stride + x * 4 + c] = in.Get(x, y, c);
  openglow::render_glow({src.data(), in.width, in.height, stride},
                        {dst.data(), in.width, in.height, stride}, params, openglow::kBGRA);
  return dst;
}

// Pools freed buffers by size, like Premiere's AllocateDeviceMemory does, so
// the timings measure the glow rather than buffer creation.
struct Harness {
  Gpu gpu;
  openglow_gpu::DXDevice device;
  int live_buffers = 0;
  std::multimap<std::size_t, ComPtr<ID3D12Resource>> pool;
  std::map<void*, std::size_t> sizes;

  openglow_gpu::DXBackend Backend() {
    return openglow_gpu::DXBackend(
        device,
        [this](std::size_t bytes) -> void* {
          ++live_buffers;
          auto it = pool.find(bytes);
          ComPtr<ID3D12Resource> r;
          if (it != pool.end()) {
            r = it->second;
            pool.erase(it);
          } else {
            r.Attach(gpu.CreateBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT));
          }
          sizes[r.Get()] = bytes;
          return r.Detach();
        },
        [this](void* memory) {
          --live_buffers;
          ComPtr<ID3D12Resource> r;
          r.Attach(static_cast<ID3D12Resource*>(memory));
          pool.emplace(sizes[memory], r);
        });
  }
};

int failures = 0;

void Check(Harness& h, const char* name, int w, int hgt, bool half, int pad, bool in_place,
           const openglow::GlowParams& params) {
  HostFrame in = MakeFrame(w, hgt, half, pad);
  ComPtr<ID3D12Resource> src_buf, dst_buf;
  src_buf.Attach(h.gpu.CreateBuffer(in.bytes.size(), D3D12_HEAP_TYPE_DEFAULT));
  h.gpu.Upload(src_buf.Get(), in.bytes.data(), in.bytes.size());
  if (!in_place) {
    dst_buf.Attach(h.gpu.CreateBuffer(in.bytes.size(), D3D12_HEAP_TYPE_DEFAULT));
  }

  openglow_gpu::Frame src{src_buf.Get(), w, hgt, in.pitch, half};
  openglow_gpu::Frame dst = src;
  if (!in_place) dst.data = dst_buf.Get();

  auto backend = h.Backend();
  const bool ok = openglow_gpu::RunGlow(backend, src, dst, params);

  HostFrame out(w, hgt, half, pad);
  h.gpu.Download(in_place ? src_buf.Get() : dst_buf.Get(), out.bytes.data(), out.bytes.size());
  const std::vector<float> ref = Reference(in, params);

  // 32f: float rounding only. 16f: the output is rounded to half too.
  const double tol = half ? 2e-3 : 2e-5;
  double worst = 0;
  for (int y = 0; y < hgt; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 4; ++c) {
        const double r = ref[(static_cast<std::size_t>(y) * w + x) * 4 + c];
        const double err = std::fabs(out.Get(x, y, c) - r) / std::max(1.0, std::fabs(r));
        worst = std::max(worst, err);
      }

  const bool pass = ok && worst <= tol && h.live_buffers == 0;
  if (!pass) ++failures;
  std::printf("%-4s %-34s %4dx%-4d max rel err %.2e (tol %.0e)%s\n", pass ? "ok" : "FAIL",
              name, w, hgt, worst, tol, ok ? "" : "  [dispatch failed]");
}

void Time(Harness& h, int w, int hgt) {
  HostFrame in = MakeFrame(w, hgt, false, 0);
  ComPtr<ID3D12Resource> buf;
  buf.Attach(h.gpu.CreateBuffer(in.bytes.size(), D3D12_HEAP_TYPE_DEFAULT));
  h.gpu.Upload(buf.Get(), in.bytes.data(), in.bytes.size());
  openglow_gpu::Frame frame{buf.Get(), w, hgt, in.pitch, false};

  for (float radius : {20.0f, 500.0f}) {
    openglow::GlowParams params;
    params.radius = radius;
    auto backend = h.Backend();
    openglow_gpu::RunGlow(backend, frame, frame, params);  // warm-up
    const int runs = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) openglow_gpu::RunGlow(backend, frame, frame, params);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("time %dx%d radius %4.0f: %7.2f ms/frame (DirectX 12, 32f)\n", w, hgt, radius,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / runs);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <DirectX_Assets dir>\n", argv[0]);
    return 2;
  }
  Harness h;
  if (!h.gpu.Init()) {
    std::printf("no DirectX 12 hardware device, skipping\n");
    return kSkip;
  }
  std::printf("device: %s\n", h.gpu.adapter.c_str());

  h.device.context = std::make_shared<DXContext>();
  if (!h.device.context->Initialize(h.gpu.device.Get(), h.gpu.queue.Get())) {
    std::printf("FAIL: DXContext::Initialize\n");
    return 1;
  }
  std::string dir = argv[1];
  if (!h.device.shaders.Load(*h.device.context, std::wstring(dir.begin(), dir.end()))) {
    std::printf("FAIL: could not load shaders from %s\n", dir.c_str());
    return 1;
  }

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

  Check(h, "32f default", 301, 173, false, 0, false, base);
  Check(h, "32f exposure+tint+fraction, padded", 301, 173, false, 48, false, bright);
  Check(h, "32f radius 1", 64, 40, false, 0, false, tiny);
  Check(h, "32f radius 2000", 257, 129, false, 16, false, huge);
  Check(h, "32f in place", 200, 120, false, 32, true, bright);
  Check(h, "16f default", 301, 173, true, 0, false, base);
  Check(h, "16f exposure+tint, padded, in place", 299, 171, true, 24, true, bright);
  Check(h, "32f 1080p", 1920, 1080, false, 0, false, bright);

  Time(h, 1920, 1080);
  Time(h, 3840, 2160);

  std::printf(failures ? "%d check(s) failed\n" : "all checks passed\n", failures);
  return failures ? 1 : 0;
}
