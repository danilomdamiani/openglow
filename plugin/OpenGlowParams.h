// Effect identity and parameter layout, shared by the CPU effect (After
// Effects API) and the Premiere GPU filter. No SDK includes on purpose: the two
// sides are compiled against different SDK header sets.
#pragma once

#define OPENGLOW_NAME "OpenGlow"
#define OPENGLOW_MATCH_NAME "OpenGlow Glow"
#define OPENGLOW_CATEGORY "OpenGlow"
#define OPENGLOW_SUPPORT_URL "https://github.com/danilomdamiani/openglow"

// Parameter order in the effect UI. Index 0 is always the input layer.
enum {
  OPENGLOW_INPUT = 0,
  OPENGLOW_EXPOSURE,
  OPENGLOW_RADIUS,
  OPENGLOW_TINT,
  OPENGLOW_TINT_COLOR,
  OPENGLOW_THRESHOLD,
  OPENGLOW_NUM_PARAMS
};

// Disk IDs identify parameters in saved projects: never reuse or renumber.
enum {
  EXPOSURE_DISK_ID = 1,
  RADIUS_DISK_ID = 2,
  TINT_DISK_ID = 3,
  TINT_COLOR_DISK_ID = 4,
  THRESHOLD_DISK_ID = 5,
};
