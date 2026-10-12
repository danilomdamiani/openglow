// OpenGlow plugin entry point (After Effects effect API, hosted natively by
// Premiere Pro). Reads the controls, adapts the host's pixel format and hands
// the frame to the core glow.
#include "OpenGlow.h"

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <vector>

#include "openglow/glow.h"

namespace {

// Acquires a PICA suite for the lifetime of the object.
template <typename Suite>
class ScopedSuite {
 public:
  ScopedSuite(PF_InData* in_data, const char* name, int32_t version)
      : in_data_(in_data), name_(name), version_(version) {
    const void* suite = nullptr;
    if (in_data_->pica_basicP->AcquireSuite(name_, version_, &suite) == kSPNoError) {
      suite_ = static_cast<const Suite*>(suite);
    }
  }
  ~ScopedSuite() {
    if (suite_) in_data_->pica_basicP->ReleaseSuite(name_, version_);
  }
  ScopedSuite(const ScopedSuite&) = delete;
  ScopedSuite& operator=(const ScopedSuite&) = delete;

  const Suite* get() const { return suite_; }
  const Suite* operator->() const { return suite_; }

 private:
  PF_InData* in_data_;
  const char* name_;
  int32_t version_;
  const Suite* suite_ = nullptr;
};

bool IsPremiere(const PF_InData* in_data) { return in_data->appl_id == 'PrMr'; }

PF_Err About(PF_InData* /*in_data*/, PF_OutData* out_data) {
  std::snprintf(out_data->return_msg, sizeof(out_data->return_msg),
                "%s v%d.%d\r%s", OPENGLOW_NAME, OPENGLOW_MAJOR_VERSION,
                OPENGLOW_MINOR_VERSION, OPENGLOW_SUPPORT_URL);
  return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data) {
  out_data->my_version = PF_VERSION(OPENGLOW_MAJOR_VERSION, OPENGLOW_MINOR_VERSION,
                                    OPENGLOW_BUG_VERSION, OPENGLOW_STAGE_VERSION,
                                    OPENGLOW_BUILD_VERSION);

  // Keep in sync with AE_Effect_Global_OutFlags(_2) in OpenGlowPiPL.r.
  out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;
  out_data->out_flags2 = PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

  if (IsPremiere(in_data)) {
    // Tell Premiere which pixel formats we handle so it skips conversions.
    ScopedSuite<PF_PixelFormatSuite1> pixel_formats(in_data, kPFPixelFormatSuite,
                                                    kPFPixelFormatSuiteVersion1);
    if (!pixel_formats.get()) return A_Err_MISSING_SUITE;
    pixel_formats->ClearSupportedPixelFormats(in_data->effect_ref);
    pixel_formats->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
    pixel_formats->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
  }
  return PF_Err_NONE;
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data) {
  PF_Err err = PF_Err_NONE;
  PF_ParamDef def;

  AEFX_CLR_STRUCT(def);
  PF_ADD_FLOAT_SLIDERX("Exposure", -10, 10, -4, 4, 1.2, PF_Precision_HUNDREDTHS,
                       PF_ValueDisplayFlag_NONE, 0, EXPOSURE_DISK_ID);

  AEFX_CLR_STRUCT(def);
  // The slider spans the whole range so the default sits on it.
  PF_ADD_FLOAT_SLIDERX("Radius", 1, 2000, 1, 2000, 1000, PF_Precision_TENTHS,
                       PF_ValueDisplayFlag_PIXEL, 0, RADIUS_DISK_ID);

  AEFX_CLR_STRUCT(def);
  PF_ADD_CHECKBOXX("Tint", FALSE, 0, TINT_DISK_ID);

  AEFX_CLR_STRUCT(def);
  PF_ADD_COLOR("Tint Color", 255, 255, 255, TINT_COLOR_DISK_ID);

  AEFX_CLR_STRUCT(def);
  PF_ADD_FLOAT_SLIDERX("Threshold", 0, 1, 0, 1, 0, PF_Precision_HUNDREDTHS,
                       PF_ValueDisplayFlag_NONE, 0, THRESHOLD_DISK_ID);

  out_data->num_params = OPENGLOW_NUM_PARAMS;
  return err;
}

enum class HostFormat { Unknown, BGRA8, BGRA32f, ARGB8, ARGB16 };

HostFormat GetHostFormat(PF_InData* in_data, PF_EffectWorld* world) {
  if (IsPremiere(in_data)) {
    ScopedSuite<PF_PixelFormatSuite1> pixel_formats(in_data, kPFPixelFormatSuite,
                                                    kPFPixelFormatSuiteVersion1);
    if (!pixel_formats.get()) return HostFormat::Unknown;
    PrPixelFormat format = PrPixelFormat_Invalid;
    pixel_formats->GetPixelFormat(world, &format);
    switch (format) {
      case PrPixelFormat_BGRA_4444_8u:
        return HostFormat::BGRA8;
      case PrPixelFormat_BGRA_4444_32f:
        return HostFormat::BGRA32f;
      default:
        return HostFormat::Unknown;
    }
  }
  // After Effects: 8 or 16 bits per channel ARGB (no float until smart render).
  return PF_WORLD_IS_DEEP(world) ? HostFormat::ARGB16 : HostFormat::ARGB8;
}

openglow::GlowParams ReadParams(PF_ParamDef* params[]) {
  openglow::GlowParams p;
  p.exposure = static_cast<float>(params[OPENGLOW_EXPOSURE]->u.fs_d.value);
  p.radius = static_cast<float>(params[OPENGLOW_RADIUS]->u.fs_d.value);
  p.tint = params[OPENGLOW_TINT]->u.bd.value != 0;
  const PF_Pixel& color = params[OPENGLOW_TINT_COLOR]->u.cd.value;
  p.tint_color[0] = color.red / 255.0f;
  p.tint_color[1] = color.green / 255.0f;
  p.tint_color[2] = color.blue / 255.0f;
  p.threshold = static_cast<float>(params[OPENGLOW_THRESHOLD]->u.fs_d.value);
  return p;
}

// Integer formats go through a float copy: unpack, glow in place, pack.
// The core works on straight alpha; After Effects' 8/16-bit worlds are
// premultiplied, so those are unpremultiplied going in and premultiplied
// coming out. Premiere's BGRA is straight already.
template <typename Channel>
void RenderInteger(PF_EffectWorld* input, PF_EffectWorld* output, int width, int height,
                   float max_value, const openglow::GlowParams& params,
                   openglow::ChannelOrder order, bool premultiplied) {
  const std::size_t stride = static_cast<std::size_t>(width) * 4;
  std::vector<float> buffer(stride * height);
  const float to_float = 1.0f / max_value;
  const int color[3] = {order.r, order.g, order.b};
  for (int y = 0; y < height; ++y) {
    const Channel* in = reinterpret_cast<const Channel*>(
        reinterpret_cast<const char*>(input->data) + static_cast<std::ptrdiff_t>(y) * input->rowbytes);
    float* row = buffer.data() + y * stride;
    for (std::size_t i = 0; i < stride; ++i) row[i] = in[i] * to_float;
    if (premultiplied) {
      for (int x = 0; x < width; ++x) {
        float* p = row + x * 4;
        const float a = p[order.a];
        for (int c : color) p[c] = a > 0.0f ? p[c] / a : 0.0f;
      }
    }
  }

  const openglow::ImageView view{buffer.data(), width, height, stride};
  openglow::render_glow({view.pixels, width, height, stride}, view, params, order);

  for (int y = 0; y < height; ++y) {
    Channel* out = reinterpret_cast<Channel*>(
        reinterpret_cast<char*>(output->data) + static_cast<std::ptrdiff_t>(y) * output->rowbytes);
    float* row = buffer.data() + y * stride;
    for (int x = 0; x < width; ++x) {
      float* p = row + x * 4;
      const float a = std::clamp(p[order.a], 0.0f, 1.0f);
      for (int c = 0; c < 4; ++c) {
        float v = std::clamp(p[c], 0.0f, 1.0f);
        if (premultiplied && c != order.a) v *= a;
        out[x * 4 + c] = static_cast<Channel>(v * max_value + 0.5f);
      }
    }
  }
}

PF_Err Render(PF_InData* in_data, PF_OutData* /*out_data*/, PF_ParamDef* params[],
              PF_LayerDef* output) {
  PF_EffectWorld* input = &params[OPENGLOW_INPUT]->u.ld;
  const openglow::GlowParams glow = ReadParams(params);
  const int width = std::min(input->width, output->width);
  const int height = std::min(input->height, output->height);
  if (width <= 0 || height <= 0) return PF_Err_NONE;

  switch (GetHostFormat(in_data, output)) {
    case HostFormat::BGRA32f:
      if (input->rowbytes > 0 && output->rowbytes > 0 && input->rowbytes % sizeof(float) == 0 &&
          output->rowbytes % sizeof(float) == 0) {
        // Float buffers are processed directly, no copies.
        const openglow::ConstImageView in{reinterpret_cast<const float*>(input->data), width,
                                          height, input->rowbytes / sizeof(float)};
        const openglow::ImageView out{reinterpret_cast<float*>(output->data), width, height,
                                      output->rowbytes / sizeof(float)};
        openglow::render_glow(in, out, glow, openglow::kBGRA);
      } else {
        return PF_Err_BAD_CALLBACK_PARAM;
      }
      break;
    case HostFormat::BGRA8:
      RenderInteger<A_u_char>(input, output, width, height, 255.0f, glow, openglow::kBGRA,
                              false);
      break;
    case HostFormat::ARGB8:
      RenderInteger<A_u_char>(input, output, width, height, PF_MAX_CHAN8, glow, openglow::kARGB,
                              true);
      break;
    case HostFormat::ARGB16:
      RenderInteger<A_u_short>(input, output, width, height, PF_MAX_CHAN16, glow,
                               openglow::kARGB, true);
      break;
    default:
      return PF_Err_BAD_CALLBACK_PARAM;
  }
  return PF_Err_NONE;
}

}  // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr,
                                                     PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* /*inSPBasicSuitePtr*/,
                                                     const char* /*inHostName*/,
                                                     const char* /*inHostVersion*/) {
  // The macro assigns to a local named `result`.
  PF_Err result = PF_Err_INVALID_CALLBACK;
  result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, OPENGLOW_NAME,
                                   OPENGLOW_MATCH_NAME, OPENGLOW_CATEGORY, AE_RESERVED_INFO,
                                   "EffectMain", OPENGLOW_SUPPORT_URL);
  return result;
}

PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                  PF_LayerDef* output, void* /*extra*/) {
  PF_Err err = PF_Err_NONE;
  try {
    switch (cmd) {
      case PF_Cmd_ABOUT:
        err = About(in_data, out_data);
        break;
      case PF_Cmd_GLOBAL_SETUP:
        err = GlobalSetup(in_data, out_data);
        break;
      case PF_Cmd_PARAMS_SETUP:
        err = ParamsSetup(in_data, out_data);
        break;
      case PF_Cmd_RENDER:
        err = Render(in_data, out_data, params, output);
        break;
      default:
        break;
    }
  } catch (PF_Err& thrown) {
    err = thrown;
  } catch (...) {
    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
  }
  return err;
}
