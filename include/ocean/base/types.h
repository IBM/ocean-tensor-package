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

#ifndef __OC_TYPES_H__
#define __OC_TYPES_H__

#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>
#include "ocean/base/half.h"

/* Basic data types */
typedef char                OcBool;
typedef int8_t              OcInt8;
typedef int16_t             OcInt16;
typedef int32_t             OcInt32;
typedef int64_t             OcInt64;
typedef uint8_t             OcUInt8;
typedef uint16_t            OcUInt16;
typedef uint32_t            OcUInt32;
typedef uint64_t            OcUInt64;
typedef float               OcFloat;
typedef double              OcDouble;

/* Complex data types */
typedef struct { OcHalf   real, imag; } OcCHalf;
typedef struct { OcFloat  real, imag; } OcCFloat;
typedef struct { OcDouble real, imag; } OcCDouble;

/* Size and index types */
typedef ptrdiff_t  OcIndex;
typedef ptrdiff_t  OcSize;

/* Integer type for pointers */
typedef uintptr_t  OcUintptr;

/* Limits */
#define OC_BOOL_MIN          0
#define OC_BOOL_MAX          1
#define OC_UINT8_MIN         0
#define OC_UINT8_MAX         255
#define OC_UINT16_MIN        0
#define OC_UINT16_MAX        65535
#define OC_UINT32_MIN        0
#define OC_UINT32_MAX        4294967295
#define OC_UINT64_MIN        0
#define OC_UINT64_MAX        ULLONG_MAX
#define OC_INT8_MIN         -128
#define OC_INT8_MAX          127
#define OC_INT16_MIN        -32768
#define OC_INT16_MAX         32767
#define OC_INT32_MIN        -2147483648L
#define OC_INT32_MAX         2147483647L
#define OC_INT64_MIN         LLONG_MIN
#define OC_INT64_MAX         LLONG_MAX

#define OC_HALF_MAX          65503
#define OC_FLOAT_MAX         FLT_MAX
#define OC_DOUBLE_MAX        DBL_MAX

/* Limits for exact integer representation for floating-point */
#define OC_HALF_INT_MAX      2048
#define OC_FLOAT_INT_MAX     16777216
#define OC_DOUBLE_INT_MAX    9007199254740992L
#define OC_HALF_INT_MIN     -2048
#define OC_FLOAT_INT_MIN    -16777216
#define OC_DOUBLE_INT_MIN   -9007199254740992L

#endif
