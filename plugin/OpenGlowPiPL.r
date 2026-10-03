// PiPL resource: tells the host about the effect before loading the code.
// Flags and version must match GlobalSetup in OpenGlow.cpp.
#include "AEConfig.h"
#include "AE_EffectVers.h"

#ifndef AE_OS_WIN
#include <AE_General.r>
#endif

resource 'PiPL' (16000) {
  {
    Kind { AEEffect },
    Name { "OpenGlow" },
    Category { "OpenGlow" },
#ifdef AE_OS_WIN
#if defined(AE_PROC_INTELx64)
    CodeWin64X86 { "EffectMain" },
#elif defined(AE_PROC_ARM64)
    CodeWinARM64 { "EffectMain" },
#endif
#elif defined(AE_OS_MAC)
    CodeMacIntel64 { "EffectMain" },
    CodeMacARM64 { "EffectMain" },
#endif
    AE_PiPL_Version { 2, 0 },
    AE_Effect_Spec_Version { PF_PLUG_IN_VERSION, PF_PLUG_IN_SUBVERS },
    AE_Effect_Version { 32769 },  /* 0.1 */
    AE_Effect_Info_Flags { 0 },
    AE_Effect_Global_OutFlags { 0x02000000 },    /* DEEP_COLOR_AWARE */
    AE_Effect_Global_OutFlags_2 { 0x08000000 },  /* SUPPORTS_THREADED_RENDERING */
    AE_Effect_Match_Name { "OpenGlow Glow" },
    AE_Reserved_Info { 0 },
    AE_Effect_Support_URL { "https://github.com/danilomdamiani/openglow" }
  }
};
