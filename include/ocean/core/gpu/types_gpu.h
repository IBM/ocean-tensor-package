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

#ifndef __OC_TYPES_GPU_H__
#define __OC_TYPES_GPU_H__

#include <stddef.h>
#include <stdint.h>
#include "cuda_fp16.h"

/* Basic data types */
typedef char                OcCudaBool;
typedef int8_t              OcCudaInt8;
typedef int16_t             OcCudaInt16;
typedef int32_t             OcCudaInt32;
typedef int64_t             OcCudaInt64;
typedef uint8_t             OcCudaUInt8;
typedef uint16_t            OcCudaUInt16;
typedef uint32_t            OcCudaUInt32;
typedef uint64_t            OcCudaUInt64;
typedef half                OcCudaHalf;
typedef float               OcCudaFloat;
typedef double              OcCudaDouble;

/* Complex data types */
typedef struct { OcCudaHalf   real, imag; } OcCudaCHalf;
typedef struct { OcCudaFloat  real, imag; } OcCudaCFloat;
typedef struct { OcCudaDouble real, imag; } OcCudaCDouble;

#endif
