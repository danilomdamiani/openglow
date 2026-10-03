# OpenGlow

An open-source glow effect for Adobe Premiere Pro (and After Effects), in the
spirit of Deep Glow, built for speed.

Status: **stage 0**. The effect loads in Premiere with its controls (Exposure,
Radius, Tint, Tint Color) and passes the image through unchanged. The glow
itself arrives in stage 1.

## How it works

OpenGlow is a native C++ plugin written against the After Effects effect API,
which Premiere Pro hosts natively (it is how most third-party effects run in
Premiere). The glow algorithm lives in `core/`, with no Adobe dependencies, so
it can be tested on any platform.

```
core/     glow algorithm (plain C++17) + public header
plugin/   thin adapter for the Adobe SDK (parameters, PiPL resource, render)
tests/    core tests, run in CI on Windows, macOS and Linux
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

### Installing

Copy the result into Adobe's shared plugin folder and restart Premiere:

- Windows: `build/plugin/Release/OpenGlow.aex` to
  `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\`
- macOS: `build/plugin/OpenGlow.plugin` to
  `/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/`

The effect shows up under **Video Effects > OpenGlow > OpenGlow**.

## License

MIT, see [LICENSE](LICENSE).
