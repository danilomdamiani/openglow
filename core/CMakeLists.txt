add_library(openglow_core STATIC
  src/glow.cpp
)
target_include_directories(openglow_core PUBLIC include)
set_target_properties(openglow_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(openglow_core PRIVATE /W4)
else()
  target_compile_options(openglow_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
