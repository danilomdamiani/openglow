# Premiere Pro GPU filter for Windows, compiled into OpenGlow.aex.
# Included from plugin/CMakeLists.txt when PREMIERE_SDK_ROOT is set.

set(PR_SDK_EXAMPLES "${PREMIERE_SDK_ROOT}/Examples")
set(PR_GPU_UTILS "${PR_SDK_EXAMPLES}/Projects/GPUVideoFilter/Utils")
if(NOT EXISTS "${PR_GPU_UTILS}/PrGPUFilterModule.h")
  message(FATAL_ERROR "PREMIERE_SDK_ROOT does not look like the Premiere Pro SDK: "
                      "${PR_GPU_UTILS}/PrGPUFilterModule.h not found")
endif()

# DirectX Shader Compiler: shipped with the Windows SDK, or DXC_BASE_PATH.
find_program(DXC dxc
  HINTS "$ENV{DXC_BASE_PATH}/bin/x64"
        "$ENV{WindowsSdkVerBinPath}/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
  REQUIRED)

# Premiere headers come first: both SDKs ship a PrSDKAESupport.h and the GPU
# filter needs Premiere's.
set(PR_SDK_INCLUDES
  "${PR_SDK_EXAMPLES}/Headers"
  "${PR_SDK_EXAMPLES}/Headers/SP"
  "${PR_GPU_UTILS}"
  ${AE_SDK_INCLUDES}
)
# PrSDKTypes.h defines NOMINMAX itself.
set(PR_SDK_DEFINES MSWindows WIN32 _WINDOWS PRWIN_ENV)

# DirectX helpers (SDK's DirectXUtils + our backend). A separate library so
# the GPU test can drive the same code without Premiere.
add_library(openglow_dx STATIC
  "${CMAKE_CURRENT_LIST_DIR}/GlowDX.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/GlowDX.h"
  "${CMAKE_CURRENT_LIST_DIR}/GlowGpu.h"
  "${PR_GPU_UTILS}/DirectXUtils.cpp"
)
target_include_directories(openglow_dx PUBLIC "${CMAKE_CURRENT_LIST_DIR}" ${PR_SDK_INCLUDES})
target_compile_definitions(openglow_dx PUBLIC ${PR_SDK_DEFINES})
target_link_libraries(openglow_dx PUBLIC openglow_core d3d12 d3dcompiler)

# One .cso + root signature per pass, in DirectX_Assets/ next to the .aex.
set(DX_ASSETS "${CMAKE_CURRENT_BINARY_DIR}/DirectX_Assets")
set(HLSL_SOURCE "${CMAKE_CURRENT_LIST_DIR}/OpenGlow.hlsl")
set(DX_SHADER_OUTPUTS "")
foreach(pass downsample_first downsample upsample_add composite)
  set(out "${DX_ASSETS}/OpenGlow_${pass}")
  add_custom_command(
    OUTPUT "${out}.cso" "${out}.rs"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${DX_ASSETS}"
    COMMAND "${DXC}" -nologo -T cs_6_0 -E ${pass} -Fo "${out}.cso" -Frs "${out}.rs"
            "${HLSL_SOURCE}"
    DEPENDS "${HLSL_SOURCE}"
    COMMENT "Compiling DirectX shader ${pass}"
    VERBATIM
  )
  list(APPEND DX_SHADER_OUTPUTS "${out}.cso" "${out}.rs")
endforeach()
add_custom_target(openglow_shaders DEPENDS ${DX_SHADER_OUTPUTS})

# The GPU filter entry point (xGPUFilterEntry).
add_library(openglow_gpu OBJECT OpenGlowGPU.cpp OpenGlowParams.h)
target_link_libraries(openglow_gpu PRIVATE openglow_dx)

# CUDA, if the toolkit is installed. Premiere uses it on NVIDIA cards.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(openglow_cuda STATIC
    "${CMAKE_CURRENT_LIST_DIR}/GlowCuda.cu" "${CMAKE_CURRENT_LIST_DIR}/GlowCuda.h")
  # CUDA 13 dropped Maxwell and Pascal (GTX 9xx/10xx); 12.x still builds them.
  # PTX for the newest listed architecture covers later cards.
  if(CUDAToolkit_VERSION VERSION_LESS 13)
    set(_cuda_archs "52;61;75;86;89-virtual")
  else()
    set(_cuda_archs "75;86;89;120-virtual")
  endif()
  set_target_properties(openglow_cuda PROPERTIES CUDA_ARCHITECTURES "${_cuda_archs}")
  target_include_directories(openglow_cuda PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries(openglow_cuda PUBLIC openglow_core CUDA::cudart_static)
  target_link_libraries(openglow_gpu PRIVATE openglow_cuda)
  target_compile_definitions(openglow_gpu PRIVATE OPENGLOW_HAS_CUDA=1)
  message(STATUS "OpenGlow: GPU renderer with DirectX 12 and CUDA ${CUDAToolkit_VERSION}")
else()
  target_compile_definitions(openglow_gpu PRIVATE OPENGLOW_HAS_CUDA=0)
  message(STATUS "OpenGlow: GPU renderer with DirectX 12 (CUDA toolkit not found)")
endif()

target_link_libraries(OpenGlow PRIVATE openglow_gpu openglow_dx)
add_dependencies(OpenGlow openglow_shaders)
add_custom_command(TARGET OpenGlow POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_directory "${DX_ASSETS}"
          "$<TARGET_FILE_DIR:OpenGlow>/DirectX_Assets"
  VERBATIM
)

# GPU vs CPU check on the local DirectX 12 device. Skipped (exit code 77) on
# machines without one.
if(OPENGLOW_BUILD_TESTS)
  add_executable(openglow_gpu_dx_test "${PROJECT_SOURCE_DIR}/tests/gpu_dx_test.cpp")
  target_link_libraries(openglow_gpu_dx_test PRIVATE openglow_dx dxgi)
  target_compile_definitions(openglow_gpu_dx_test PRIVATE NOMINMAX)
  add_dependencies(openglow_gpu_dx_test openglow_shaders)
  add_test(NAME openglow_gpu_dx_test COMMAND openglow_gpu_dx_test "${DX_ASSETS}/")
  set_tests_properties(openglow_gpu_dx_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
