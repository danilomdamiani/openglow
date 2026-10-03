cmake_minimum_required(VERSION 3.21)

project(OpenGlow VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The plugin needs the After Effects SDK, which Adobe does not allow us to
# redistribute. Point AE_SDK_ROOT at your local copy (the folder that contains
# "Examples") to build it. Without it, only the core library and tests build.
set(AE_SDK_ROOT "$ENV{AE_SDK_ROOT}" CACHE PATH "Path to the After Effects SDK root")

if(AE_SDK_ROOT)
  set(_plugin_default ON)
else()
  set(_plugin_default OFF)
endif()
option(OPENGLOW_BUILD_PLUGIN "Build the Premiere/After Effects plugin" ${_plugin_default})
option(OPENGLOW_BUILD_TESTS "Build the core tests" ON)

add_subdirectory(core)

if(OPENGLOW_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(OPENGLOW_BUILD_PLUGIN)
  if(NOT AE_SDK_ROOT)
    message(FATAL_ERROR "OPENGLOW_BUILD_PLUGIN is ON but AE_SDK_ROOT is not set")
  endif()
  add_subdirectory(plugin)
endif()
