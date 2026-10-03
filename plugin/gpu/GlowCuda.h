// CUDA kernels for the GPU glow (GlowCuda.cu). Pointers are device memory,
// pitches are in bytes, `stream` is the host's CUstream / cudaStream_t.
// Each launcher returns true if the launch succeeded.
#pragma once

#include "GlowGpu.h"

namespace openglow_gpu {

bool CudaDownsampleFirst(const Frame& src, const Plane& dst, float gain, float threshold,
                         float knee, void* stream);
bool CudaDownsample(const Plane& src, const Plane& dst, void* stream);
bool CudaUpsampleAdd(const Plane& src, const Plane& dst, float src_weight, void* stream);
bool CudaComposite(const Frame& src, const Plane& glow, const Frame& dst,
                   const float tint_bgr[3], void* stream);
// Waits for every launch on the stream.
bool CudaFinish(void* stream);

}  // namespace openglow_gpu
