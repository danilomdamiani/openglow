set(AE_SDK_EXAMPLES "${AE_SDK_ROOT}/Examples")
if(NOT EXISTS "${AE_SDK_EXAMPLES}/Headers/AE_Effect.h")
  message(FATAL_ERROR "AE_SDK_ROOT does not look like the After Effects SDK: "
                      "${AE_SDK_EXAMPLES}/Headers/AE_Effect.h not found")
endif()

set(AE_SDK_INCLUDES
  "${AE_SDK_EXAMPLES}/Headers"
  "${AE_SDK_EXAMPLES}/Headers/SP"
  "${AE_SDK_EXAMPLES}/Util"
  "${AE_SDK_EXAMPLES}/Resources"
)

set(PIPL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/OpenGlowPiPL.r")
set(PIPL_OUT "${CMAKE_CURRENT_BINARY_DIR}/pipl")
file(MAKE_DIRECTORY "${PIPL_OUT}")

if(WIN32)
  # Windows: preprocess the .r, convert it with Adobe's PiPLtool, then
  # preprocess again into a .rc that gets linked into the .aex.
  find_program(PIPLTOOL PiPLtool PATHS "${AE_SDK_EXAMPLES}/Resources" NO_DEFAULT_PATH REQUIRED)
  set(_inc_flags "")
  foreach(dir ${AE_SDK_INCLUDES})
    list(APPEND _inc_flags "/I${dir}")
  endforeach()
  add_custom_command(
    OUTPUT "${PIPL_OUT}/OpenGlowPiPL.rc"
    COMMAND "${CMAKE_CXX_COMPILER}" /nologo /EP /P "/Fi${PIPL_OUT}/OpenGlowPiPL.rr"
            ${_inc_flags} /Tc "${PIPL_SOURCE}"
    COMMAND "${PIPLTOOL}" "${PIPL_OUT}/OpenGlowPiPL.rr" "${PIPL_OUT}/OpenGlowPiPL.rrc"
    COMMAND "${CMAKE_CXX_COMPILER}" /nologo /EP /P /DMSWindows "/Fi${PIPL_OUT}/OpenGlowPiPL.rc"
            /Tc "${PIPL_OUT}/OpenGlowPiPL.rrc"
    DEPENDS "${PIPL_SOURCE}"
    COMMENT "Building PiPL resource"
    VERBATIM
  )
  set(PIPL_RESOURCE "${PIPL_OUT}/OpenGlowPiPL.rc")
elseif(APPLE)
  # macOS: Rez compiles the .r into a resource file inside the bundle.
  add_custom_command(
    OUTPUT "${PIPL_OUT}/OpenGlow.rsrc"
    COMMAND xcrun Rez -d AE_OS_MAC -d __MACH__ -useDF
            -i "${AE_SDK_EXAMPLES}/Headers" -i "${AE_SDK_EXAMPLES}/Headers/SP"
            -i "${AE_SDK_EXAMPLES}/Resources"
            -o "${PIPL_OUT}/OpenGlow.rsrc" "${PIPL_SOURCE}"
    DEPENDS "${PIPL_SOURCE}"
    COMMENT "Building PiPL resource"
    VERBATIM
  )
  set(PIPL_RESOURCE "${PIPL_OUT}/OpenGlow.rsrc")
  set_source_files_properties("${PIPL_RESOURCE}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
else()
  message(FATAL_ERROR "The plugin only builds on Windows and macOS")
endif()

add_library(OpenGlow MODULE
  OpenGlow.cpp
  OpenGlow.h
  "${PIPL_RESOURCE}"
)
target_include_directories(OpenGlow PRIVATE ${AE_SDK_INCLUDES})
target_link_libraries(OpenGlow PRIVATE openglow_core)

if(WIN32)
  target_compile_definitions(OpenGlow PRIVATE MSWindows WIN32 _WINDOWS NOMINMAX)
  set_target_properties(OpenGlow PROPERTIES PREFIX "" SUFFIX ".aex")
  target_compile_options(OpenGlow PRIVATE /W3)
else()
  set_target_properties(OpenGlow PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION "plugin"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/mac/Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER "io.github.danilomdamiani.openglow"
    MACOSX_BUNDLE_BUNDLE_NAME "OpenGlow"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    CXX_VISIBILITY_PRESET hidden
  )
  target_compile_options(OpenGlow PRIVATE -Wall)
endif()
