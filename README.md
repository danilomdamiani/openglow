# OpenGlow

An open-source glow effect for Adobe Premiere Pro (and After Effects), in the
spirit of Deep Glow, built for speed.

Status: **stage 2**. The glow has Exposure, Radius, Tint, Tint Color and
Threshold, and runs on the GPU in Premiere Pro on Windows (DirectX 12, and CUDA
when built with the CUDA toolkit), with the CPU as fallback.

- **Threshold** keeps only the bright parts of the image (with a soft knee), so
  on video the highlights bloom instead of the whole frame.
- On titles and shapes over transparency, the glow spills out around them: only
  visible pixels emit light, and the output alpha grows by the glow's coverage.
  Opaque footage stays opaque.

## How it works

OpenGlow is a native C++ plugin written against the After Effects effect API,
which Premiere Pro hosts natively (it is how most third-party effects run in
Premiere). The glow algorithm lives in `core/`, with no Adobe dependencies, so
it can be tested on any platform.

```
core/        glow algorithm (plain C++17) + public header
plugin/      thin adapter for the Adobe SDK (parameters, PiPL resource, render)
plugin/gpu/  GPU kernels (HLSL for DirectX 12, CUDA) and the pass sequence
tests/       core tests, run in CI on Windows, macOS and Linux
```

On the GPU, Premiere calls a separate entry point (`xGPUFilterEntry`, from the
Premiere Pro SDK) in the same `.aex`. The kernels follow the CPU code pass by
pass, and `openglow_gpu_dx_test` checks that both give the same image on the
local GPU.

The glow is a mip pyramid: the image is halved repeatedly with a soft filter,
then the levels are added back up. The cost barely depends on the radius.
`openglow_bench` times it:

```
build/tests/openglow_bench
```

## Building

You need CMake 3.21+ and:

- Windows: Visual Studio 2022 (Desktop development with C++).
- macOS: Xcode (builds a universal arm64 + x86_64 bundle).

### Core and tests only

```
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

### The plugin

Adobe does not allow redistributing its SDK, so download the **After Effects
SDK** from the [Adobe Developer Console](https://developer.adobe.com/after-effects/)
and unzip it anywhere (for example `sdk/AfterEffectsSDK`, which git ignores).
Point `AE_SDK_ROOT` at the folder that contains `Examples`:

```
cmake -S . -B build -DAE_SDK_ROOT=path/to/AfterEffectsSDK
cmake --build build --config Release
```

On macOS add `-DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"` for a universal build.

### GPU rendering in Premiere (Windows)

Also download the **Premiere Pro SDK** (same Developer Console) and point
`PREMIERE_SDK_ROOT` at the folder that contains `Examples`:

```
cmake -S . -B build -DAE_SDK_ROOT=path/to/AfterEffectsSDK -DPREMIERE_SDK_ROOT=path/to/PremiereProSDK
```

- DirectX 12 is always built. Its shader compiler (`dxc`) ships with the
  Windows SDK that Visual Studio installs.
- CUDA is built when CMake finds the CUDA toolkit. Premiere uses CUDA on NVIDIA
  cards. GTX 9xx/10xx cards need CUDA 12.x (CUDA 13 dropped them).

Premiere dropped OpenCL in 2021, so there is no OpenCL path.

### Installing

Copy the result into Adobe's shared plugin folder and restart Premiere:

- Windows: `build/plugin/Release/OpenGlow.aex` to
  `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\`, and the files in
  `build/plugin/Release/DirectX_Assets/` to `MediaCore\DirectX_Assets\` (the
  DirectX shaders)
- macOS: `build/plugin/OpenGlow.plugin` to
  `/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/`

The effect shows up under **Video Effects > OpenGlow > OpenGlow**.

## License

MIT, see [LICENSE](LICENSE).
