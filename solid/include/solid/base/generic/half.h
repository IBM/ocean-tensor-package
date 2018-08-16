/* ------------------------------------------------------------------------ */
/* Copyright 2018, IBM Corp.                                                */
/*                                                                          */
/* Licensed under the Apache License, Version 2.0 (the "License");          */
/* you may not use this file except in compliance with the License.         */
/* You may obtain a copy of the License at                                  */
/*                                                                          */
/*    http://www.apache.org/licenses/LICENSE-2.0                            */
/*                                                                          */
/* Unless required by applicable law or agreed to in writing, software      */
/* distributed under the License is distributed on an "AS IS" BASIS,        */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/* See the License for the specific language governing permissions and      */
/* limitations under the License.                                           */
/* ------------------------------------------------------------------------ */

#ifndef __SOLID_CPU_HALF_H__
#define __SOLID_CPU_HALF_H__

#include "solid/base/generic/api.h"
#include "solid/base/generic/types.h"

#include <stdint.h>

#define SD_HALF_EXPONENT_MASK       0x7C00
#define SD_HALF_QUIET_NAN           0x7F00
#define SD_HALF_SIGNAL_NAN          0x7D00
#define SD_HALF_PLUS_INFINITY       0x7C00
#define SD_HALF_NEGATIVE_INFINIT    0xFC00
#define SD_HALF_ZERO                0x0000

#define SD_HALF_IS_ZERO(X)          (((X) & 0x7FFF) == 0)
#define SD_HALF_IS_INF(X)           (((X) & 0x7FFF) == SD_HALF_PLUS_INFINITY)
#define SD_HALF_IS_NAN(X)           (((X) & 0x7FFF) >  SD_HALF_PLUS_INFINITY)

#define SD_FLOAT_EXPONENT_MASK      0x7F800000
#define SD_FLOAT_QUIET_NAN          0x7FD00000
#define SD_FLOAT_SIGNAL_NAN         0x7F900000
#define SD_FLOAT_PLUS_INFINITY      0x7F800000
#define SD_FLOAT_NEGATIVE_INFINITY  0xFF800000


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

SOLID_API solid_float solid_half_to_float(solid_half h);
SOLID_API solid_half  solid_float_to_half(solid_float f);

#endif
