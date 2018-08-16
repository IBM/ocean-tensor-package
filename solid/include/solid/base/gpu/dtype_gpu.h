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

#ifndef __SOLID_DTYPE_GPU_H__
#define __SOLID_DTYPE_GPU_H__

#include "solid.h"
#include "solid/base/gpu/solid_cuda.h"

#include <stdint.h>


/* Native types */
typedef uint8_t    solid_gpu_bool;
typedef uint8_t    solid_gpu_uint8;
typedef uint16_t   solid_gpu_uint16;
typedef uint32_t   solid_gpu_uint32;
typedef uint64_t   solid_gpu_uint64;
typedef int8_t     solid_gpu_int8;
typedef int16_t    solid_gpu_int16;
typedef int32_t    solid_gpu_int32;
typedef int64_t    solid_gpu_int64;
typedef half       solid_gpu_half;
typedef float      solid_gpu_float;
typedef double     solid_gpu_double;

/* Complex data types */
typedef struct { solid_gpu_half   real, imag; } solid_gpu_chalf;
typedef struct { solid_gpu_float  real, imag; } solid_gpu_cfloat;
typedef struct { solid_gpu_double real, imag; } solid_gpu_cdouble;


/* -------------------------------------------------------------------- */
/* Corresponding C type - device specific                               */
/* -------------------------------------------------------------------- */
#define SOLID_C_TYPE_B(TYPE) solid_gpu_##TYPE
#define SOLID_C_TYPE(TYPE)   SOLID_C_TYPE_B(TYPE)


/* -------------------------------------------------------------------- */
/* Basic or special type - device specific                              */
/* -------------------------------------------------------------------- */
#define SDTYPE_BASIC_bool      0
#define SDTYPE_BASIC_uint8     1
#define SDTYPE_BASIC_uint16    1
#define SDTYPE_BASIC_uint32    1
#define SDTYPE_BASIC_uint64    1
#define SDTYPE_BASIC_int8      1
#define SDTYPE_BASIC_int16     1
#define SDTYPE_BASIC_int32     1
#define SDTYPE_BASIC_int64     1
#define SDTYPE_BASIC_half      0
#define SDTYPE_BASIC_float     1
#define SDTYPE_BASIC_double    1
#define SDTYPE_BASIC_chalf     0
#define SDTYPE_BASIC_cfloat    1
#define SDTYPE_BASIC_cdouble   1


/* -------------------------------------------------------------------- */
/* Zero values for special types                                        */
/* -------------------------------------------------------------------- */
#define SDTYPE_ZERO_bool        0
#define SDTYPE_ZERO_half        __float2half(0)


/* -------------------------------------------------------------------- */
/* Work types for special data types - device specific                  */
/* -------------------------------------------------------------------- */
#define SOLID_WORKTYPE_bool    bool
#define SOLID_WORKTYPE_half    float
#define SOLID_WORKTYPE_chalf   cfloat


/* -------------------------------------------------------------------- */
/* Conversion from type to worktype and back - device specific          */
/* -------------------------------------------------------------------- */
#define SOLID_SPECIAL_TO_BASIC_bool(TYPE,X)  ((SOLID_C_TYPE(TYPE))X)
#define SOLID_BASIC_TO_SPECIAL_bool(TYPE,X)  (((SOLID_C_TYPE(TYPE))(X) == 0) ? 0 : 1)
#define SOLID_SPECIAL_TO_BASIC_half(TYPE,X)  (SOLID_C_TYPE(TYPE))__half2float(X)
#define SOLID_BASIC_TO_SPECIAL_half(TYPE,X)  __float2half((float)(X))


/* -------------------------------------------------------------------- */
/* Conversion between special types                                     */
/* -------------------------------------------------------------------- */
#define SOLID_CONVERT_bool_TO_bool(X) X
#define SOLID_CONVERT_bool_TO_half(X) __float2half((float)(X))
#define SOLID_CONVERT_half_TO_bool(X) (__half2float(X) == 0 ? 0 : 1)
#define SOLID_CONVERT_half_TO_half(X) X


#endif
