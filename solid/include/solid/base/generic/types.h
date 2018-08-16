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

#ifndef __SOLID_TYPES_H__
#define __SOLID_TYPES_H__

#include <stdint.h>
#include <stddef.h>

#include "solid/base/generic/const.h"

/* Size and index types */
typedef size_t    solid_size;
typedef ptrdiff_t solid_index;

/* Data types */
typedef uint8_t   solid_bool;
typedef uint8_t   solid_uint8;
typedef uint16_t  solid_uint16;
typedef uint32_t  solid_uint32;
typedef uint64_t  solid_uint64;
typedef int8_t    solid_int8;
typedef int16_t   solid_int16;
typedef int32_t   solid_int32;
typedef int64_t   solid_int64;
typedef uint16_t  solid_half;
typedef float     solid_float;
typedef double    solid_double;

typedef struct { solid_half   real; solid_half   imag; }  solid_chalf;
typedef struct { solid_float  real; solid_float  imag; }  solid_cfloat;
typedef struct { solid_double real; solid_double imag; }  solid_cdouble;

/* Tensor layout */
typedef struct
{  int           ndims;
   size_t        size[SOLID_MAX_TENSOR_DIMS];
   ptrdiff_t     strides[SOLID_MAX_TENSOR_DIMS];
   char         *ptr;   
} solid_layout;

typedef struct
{  int           ndims;
   int           size[SOLID_MAX_TENSOR_DIMS];
   unsigned int  strides[SOLID_MAX_TENSOR_DIMS];
   char         *ptr;   
} solid_layout_small;

typedef solid_layout solid_layout_large;

#endif
