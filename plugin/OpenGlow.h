#pragma once

#include "AEConfig.h"

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#include "entry.h"
#include "SPBasic.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "PrSDKAESupport.h"

#include "OpenGlowParams.h"

// Keep in sync with AE_Effect_Version in OpenGlowPiPL.r:
// PF_VERSION(0, 1, 0, PF_Stage_DEVELOP, 1) == 32769.
#define OPENGLOW_MAJOR_VERSION 0
#define OPENGLOW_MINOR_VERSION 1
#define OPENGLOW_BUG_VERSION 0
#define OPENGLOW_STAGE_VERSION PF_Stage_DEVELOP
#define OPENGLOW_BUILD_VERSION 1

extern "C" {
DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data,
                            PF_ParamDef* params[], PF_LayerDef* output, void* extra);
}
