// Runs the DirectX 12 glow on the local GPU and compares it with the CPU core
// (see gpu_test_common.h). Exits 77 (skipped) when there is no DirectX 12
// hardware device.
//
// Usage: openglow_gpu_dx_test <DirectX_Assets dir with trailing slash>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include <string>

#include "GlowDX.h"
#include "gpu_test_common.h"

using Microsoft::WRL::ComPtr;

namespace {

constexpr int kSkip = 77;

class DXApi : public gpu_test::GpuApi {
 public:
  bool Init(const std::string& shader_dir) {
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
      if (SUCCEEDED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)))) {
        std::printf("device: %ls\n", desc.Description);
        break;
      }
    }
    if (!device_) return false;
    // DXContext records compute command lists, like Premiere's queue.
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)))) return false;
    if (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                               IID_PPV_ARGS(&allocator_)))) {
      return false;
    }
    if (FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator_.Get(),
                                          nullptr, IID_PPV_ARGS(&list_)))) {
      return false;
    }
    list_->Close();
    if (FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) return false;
    event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    dx_.context = std::make_shared<DXContext>();
    if (!dx_.context->Initialize(device_.Get(), queue_.Get())) {
      std::printf("FAIL: DXContext::Initialize\n");
      return false;
    }
    if (!dx_.shaders.Load(*dx_.context, std::wstring(shader_dir.begin(), shader_dir.end()))) {
      std::printf("FAIL: could not load shaders from %s\n", shader_dir.c_str());
      return false;
    }
    return true;
  }

  bool HasDevice() const { return device_ != nullptr; }

  const char* Name() const override { return "DirectX 12"; }

  void* CreateBuffer(std::size_t bytes) override { return Create(bytes, D3D12_HEAP_TYPE_DEFAULT); }
  void DestroyBuffer(void* buffer) override { static_cast<ID3D12Resource*>(buffer)->Release(); }

  void Upload(void* dst, const void* data, std::size_t bytes) override {
    ComPtr<ID3D12Resource> staging;
    staging.Attach(Create(bytes, D3D12_HEAP_TYPE_UPLOAD));
    void* p = nullptr;
    staging->Map(0, nullptr, &p);
    std::memcpy(p, data, bytes);
    staging->Unmap(0, nullptr);
    allocator_->Reset();
    list_->Reset(allocator_.Get(), nullptr);
    list_->CopyBufferRegion(static_cast<ID3D12Resource*>(dst), 0, staging.Get(), 0, bytes);
    Submit();
  }

  void Download(void* src, void* data, std::size_t bytes) override {
    ComPtr<ID3D12Resource> staging;
    staging.Attach(Create(bytes, D3D12_HEAP_TYPE_READBACK));
    allocator_->Reset();
    list_->Reset(allocator_.Get(), nullptr);
    list_->CopyBufferRegion(staging.Get(), 0, static_cast<ID3D12Resource*>(src), 0, bytes);
    Submit();
    void* p = nullptr;
    staging->Map(0, nullptr, &p);
    std::memcpy(data, p, bytes);
    staging->Unmap(0, nullptr);
  }

  std::unique_ptr<openglow_gpu::Backend> MakeBackend(std::function<void*(std::size_t)> allocate,
                                                     std::function<void(void*)> free) override {
    return std::make_unique<openglow_gpu::DXBackend>(dx_, std::move(allocate), std::move(free));
  }

 private:
  ID3D12Resource* Create(std::size_t bytes, D3D12_HEAP_TYPE heap) {
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
    device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                     IID_PPV_ARGS(&r));
    return r;
  }

  void Submit() {
    list_->Close();
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    queue_->Signal(fence_.Get(), ++fence_value_);
    fence_->SetEventOnCompletion(fence_value_, event_);
    WaitForSingleObject(event_, INFINITE);
  }

 public:
  ID3D12Device* device() const { return device_.Get(); }
  ID3D12CommandQueue* queue() const { return queue_.Get(); }
  openglow_gpu::DXDevice& dx() { return dx_; }

 private:
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> list_;
  ComPtr<ID3D12Fence> fence_;
  UINT64 fence_value_ = 0;
  HANDLE event_ = nullptr;
  openglow_gpu::DXDevice dx_;
};

// --profile: times each pass of a 4K render with GPU timestamps recorded into
// the same command list as the passes.
class ProfilingBackend : public openglow_gpu::Backend {
 public:
  ProfilingBackend(DXApi& api, openglow_gpu::Backend& inner) : api_(api), inner_(inner) {
    D3D12_QUERY_HEAP_DESC qd = {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = kMaxQueries;
    api.device()->CreateQueryHeap(&qd, IID_PPV_ARGS(&heap_));
    readback_.Attach(static_cast<ID3D12Resource*>(nullptr));
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = kMaxQueries * 8;
    rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    api.device()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                          IID_PPV_ARGS(&readback_));
    api.queue()->GetTimestampFrequency(&frequency_);
  }

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
  bool Finish() override {
    List()->ResolveQueryData(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, next_, readback_.Get(),
                             0);
    const bool ok = inner_.Finish();
    UINT64* ticks = nullptr;
    readback_->Map(0, nullptr, reinterpret_cast<void**>(&ticks));
    for (UINT i = 0; i < next_; i += 2) {
      totals_[names_[i / 2]] += 1000.0 * (ticks[i + 1] - ticks[i]) / frequency_;
    }
    readback_->Unmap(0, nullptr);
    next_ = 0;
    names_.clear();
    return ok;
  }

  void Print(int runs) {
    for (auto& e : totals_) std::printf("  %-18s %6.3f ms\n", e.first.c_str(), e.second / runs);
  }

 private:
  static constexpr UINT kMaxQueries = 128;

  ID3D12GraphicsCommandList* List() { return api_.dx().context->mCommandList.Get(); }

  template <typename Fn>
  bool Timed(const char* name, Fn fn) {
    List()->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, next_++);
    const bool ok = fn();
    List()->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, next_++);
    names_.push_back(name);
    return ok;
  }

  DXApi& api_;
  openglow_gpu::Backend& inner_;
  ComPtr<ID3D12QueryHeap> heap_;
  ComPtr<ID3D12Resource> readback_;
  UINT64 frequency_ = 1;
  UINT next_ = 0;
  std::vector<std::string> names_;
  std::map<std::string, double> totals_;
};

void Profile(DXApi& api, int w, int h, bool half) {
  gpu_test::HostFrame in = gpu_test::MakeFrame(w, h, half, 0);
  void* buf = api.CreateBuffer(in.bytes.size());
  api.Upload(buf, in.bytes.data(), in.bytes.size());
  gpu_test::Runner runner(api);
  auto inner = runner.Backend();
  ProfilingBackend profiled(api, *inner);
  openglow::GlowParams params;
  params.radius = 500;
  openglow_gpu::Frame frame{buf, w, h, in.pitch, half};
  openglow_gpu::RunGlow(*inner, frame, frame, params);
  const int runs = 20;
  for (int i = 0; i < runs; ++i) openglow_gpu::RunGlow(profiled, frame, frame, params);
  std::printf("profile %dx%d %s:\n", w, h, half ? "16f" : "32f");
  profiled.Print(runs);
  api.DestroyBuffer(buf);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <DirectX_Assets dir>\n", argv[0]);
    return 2;
  }
  DXApi api;
  if (!api.Init(argv[1])) {
    if (!api.HasDevice()) {
      std::printf("no DirectX 12 hardware device, skipping\n");
      return kSkip;
    }
    return 1;
  }
  if (argc > 2 && std::string(argv[2]) == "--profile") {
    Profile(api, 3840, 2160, false);
    Profile(api, 3840, 2160, true);
    return 0;
  }
  gpu_test::Runner runner(api);
  return runner.RunAll();
}
