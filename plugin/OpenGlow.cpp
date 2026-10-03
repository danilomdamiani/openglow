// OpenGlow plugin entry point (After Effects effect API, hosted natively by
// Premiere Pro). Stage 0: registers the effect and its controls and passes the
// image through unchanged.
#include "OpenGlow.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

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
    if (!pixel_formats.get()) return PF_Err_MISSING_SUITE;
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
  PF_ADD_FLOAT_SLIDERX("Exposure", -10, 10, -4, 4, 0, PF_Precision_HUNDREDTHS,
                       PF_ValueDisplayFlag_NONE, 0, EXPOSURE_DISK_ID);

  AEFX_CLR_STRUCT(def);
  PF_ADD_FLOAT_SLIDERX("Radius", 1, 2000, 1, 500, 50, PF_Precision_TENTHS,
                       PF_ValueDisplayFlag_PIXEL, 0, RADIUS_DISK_ID);

  AEFX_CLR_STRUCT(def);
  PF_ADD_CHECKBOXX("Tint", FALSE, 0, TINT_DISK_ID);

  AEFX_CLR_STRUCT(def);
  PF_ADD_COLOR("Tint Color", 255, 255, 255, TINT_COLOR_DISK_ID);

  out_data->num_params = OPENGLOW_NUM_PARAMS;
  return err;
}

// Bytes per pixel of a host buffer, or 0 if we don't know the format.
int BytesPerPixel(PF_InData* in_data, PF_EffectWorld* world) {
  if (IsPremiere(in_data)) {
    ScopedSuite<PF_PixelFormatSuite1> pixel_formats(in_data, kPFPixelFormatSuite,
                                                    kPFPixelFormatSuiteVersion1);
    if (!pixel_formats.get()) return 0;
    PrPixelFormat format = PrPixelFormat_Invalid;
    pixel_formats->GetPixelFormat(world, &format);
    switch (format) {
      case PrPixelFormat_BGRA_4444_8u:
        return 4;
      case PrPixelFormat_BGRA_4444_32f:
        return 16;
      default:
        return 0;
    }
  }
  // After Effects: 8 or 16 bits per channel ARGB (no float until smart render).
  return PF_WORLD_IS_DEEP(world) ? 8 : 4;
}

PF_Err Render(PF_InData* in_data, PF_OutData* /*out_data*/, PF_ParamDef* params[],
              PF_LayerDef* output) {
  PF_EffectWorld* input = &params[OPENGLOW_INPUT]->u.ld;

  const int bpp = BytesPerPixel(in_data, output);
  if (bpp == 0) return PF_Err_BAD_CALLBACK_PARAM;

  const int width = std::min(input->width, output->width);
  const int height = std::min(input->height, output->height);
  const size_t row_bytes = static_cast<size_t>(width) * bpp;
  const char* src = reinterpret_cast<const char*>(input->data);
  char* dst = reinterpret_cast<char*>(output->data);
  for (int y = 0; y < height; ++y) {
    std::memcpy(dst + static_cast<size_t>(y) * output->rowbytes,
                src + static_cast<size_t>(y) * input->rowbytes, row_bytes);
  }
  return PF_Err_NONE;
}

}  // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr,
                                                     PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* /*inSPBasicSuitePtr*/,
                                                     const char* /*inHostName*/,
                                                     const char* /*inHostVersion*/) {
  return PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, OPENGLOW_NAME,
                                 OPENGLOW_MATCH_NAME, OPENGLOW_CATEGORY, AE_RESERVED_INFO,
                                 "EffectMain", OPENGLOW_SUPPORT_URL);
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
