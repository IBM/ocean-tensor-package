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

#ifndef __OC_HALF_H__
#define __OC_HALF_H__

#include "ocean/base/api.h"

#include <stdint.h>

#define OC_HALF_EXPONENT_MASK       0x7C00
#define OC_HALF_QUIET_NAN           0x7F00
#define OC_HALF_SIGNAL_NAN          0x7D00
#define OC_HALF_PLUS_INFINITY       0x7C00
#define OC_HALF_NEGATIVE_INFINIT    0xFC00
#define OC_HALF_ZERO                0x0000

#define OC_HALF_IS_ZERO(X)          (((X) & 0x7FFF) == 0)

#define OC_FLOAT_EXPONENT_MASK      0x7F800000
#define OC_FLOAT_QUIET_NAN          0x7FD00000
#define OC_FLOAT_SIGNAL_NAN         0x7F900000
#define OC_FLOAT_PLUS_INFINITY      0x7F800000
#define OC_FLOAT_NEGATIVE_INFINITY  0xFF800000


/* Define the Ocean half-precision floating-point type */
typedef uint16_t OcHalf;

/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

OC_API float  OcHalfToFloat(OcHalf h);
OC_API OcHalf OcFloatToHalf(float  f);

#endif
