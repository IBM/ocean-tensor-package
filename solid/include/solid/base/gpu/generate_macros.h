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

#ifndef __SOLID_GPU_GENERATE_MACROS_H__
#define __SOLID_GPU_GENERATE_MACROS_H__

#include "solid/base/generic/generate_macros.h"


#define SOLID_DEVICE gpu

/* ------------------------------------------------------------------------ */
/* GPU Kernel names                                                         */
/* ------------------------------------------------------------------------ */
#define SOLID_KERNEL_B(PREFIX,NDIM,MODE) PREFIX ## _kernel_ ## NDIM ## _ ## MODE
#define SOLID_KERNEL(PREFIX,NDIM,MODE)   SOLID_KERNEL_B(PREFIX,NDIM,MODE)


/* ------------------------------------------------------------------------ */
/* Unrolling flag                                                           */
/* ------------------------------------------------------------------------ */
#define SOLID_FLAG_UNROLLED_UNROLL       1
#define SOLID_FLAG_UNROLLED_NO_UNROLLING 0
#define SOLID_FLAG_UNROLLED(UNROLLING)   SOLID_FLAG_UNROLLED_##UNROLLING


/* ------------------------------------------------------------------------ */
/* Create kernel structure types                                            */
/* ------------------------------------------------------------------------ */
#define SOLID_CREATE_KERNEL_TYPES(PREFIX, FLAG_PARAM, PARAM) \
   SOLID_KERNEL_PARAM_STRUCT_FLAG(PREFIX, FLAG_PARAM, PARAM)


/* ------------------------------------------------------------------------ */
/* Kernel parameters - user defined                                         */
/* ------------------------------------------------------------------------ */
#define SOLID_KERNEL_PARAM_PREFIX_B(PREFIX) PREFIX ## _kernel_param
#define SOLID_KERNEL_PARAM_PREFIX(PREFIX)   SOLID_KERNEL_PARAM_PREFIX_B(PREFIX)
#define SOLID_KERNEL_PARAM(NAME)            SOLID_KERNEL_PARAM_PREFIX(SOLID_FUNCTION_TYPE(NAME,SDXTYPE))

#define SOLID_KERNEL_PARAM_STRUCT_0(PREFIX,PARAM) \
  /* Empty */

#define SOLID_KERNEL_PARAM_STRUCT_1(PREFIX,PARAM) \
  typedef struct \
  PARAM \
  SOLID_KERNEL_PARAM_PREFIX(PREFIX);

#define SOLID_KERNEL_PARAM_STRUCT(PREFIX,BODY) \
   SOLID_KERNEL_PARAM_STRUCT_1(PREFIX,BODY)

#define SOLID_KERNEL_PARAM_STRUCT_FLAG(PREFIX, FLAG_PARAM, PARAM) \
   SOLID_KERNEL_PARAM_STRUCT_ ## FLAG_PARAM(PREFIX, PARAM)


/* ------------------------------------------------------------------------ */
/* Kernel parameter declaration                                             */
/* ------------------------------------------------------------------------ */

#define SOLID_KERNEL_PARAM_DECLARATION_0(PREFIX) \
  /* Empty */

#define SOLID_KERNEL_PARAM_DECLARATION_1(PREFIX) \
  struct SOLID_KERNEL_PARAM_PREFIX(PREFIX);

#define SOLID_KERNEL_PARAM_DECLARATION(PREFIX, FLAG_PARAM) \
   SOLID_KERNEL_PARAM_DECLARATION_ ## FLAG_PARAM(PREFIX)

#endif
