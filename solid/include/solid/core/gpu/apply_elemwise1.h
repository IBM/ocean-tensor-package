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

#ifndef __SOLID_GPU_APPLY_ELEMWISE1_H__
#define __SOLID_GPU_APPLY_ELEMWISE1_H__

#include "solid.h"
#include "solid/base/gpu/generate_macros.h"
#include "solid/core/gpu/apply_elemwise.h"


/* ------------------------------------------------------------------------ */
/* Interface                                                                */
/* ------------------------------------------------------------------------ */
/* The apply elemwise1 launch macros for GPU assume that the following      */
/* variables have been defined:                                             */
/*   int        ndims;                                                      */
/*   size_t    *size;                                                       */
/*   ptrdiff_t *strides;                                                    */
/*   void      *ptr;                                                        */
/*   int        result;             // Updated to reflect success / failure */
/* ------------------------------------------------------------------------ */
/* Exposed variables in kernel:                                             */
/*   solid_c_type(TYPE) *_ptr                                               */
/*   (long) int         *_index                                             */
/* ------------------------------------------------------------------------ */


/* ------------------------------------------------------------------------ */
/* Tensor information structure                                             */
/* ------------------------------------------------------------------------ */

#define SOLID_ELEMWISE1_DATA_VAR_SMALL  __elemwise1_data_small
#define SOLID_ELEMWISE1_DATA_VAR_LARGE  __elemwise1_data_large
#define SOLID_ELEMWISE1_DATA_VAR(SIZE)  SOLID_ELEMWISE1_DATA_VAR_## SIZE

#define SOLID_ELEMWISE1_DATA_SMALL      solid_elemwise1_data_small
#define SOLID_ELEMWISE1_DATA_LARGE      solid_elemwise1_data_large
#define SOLID_ELEMWISE1_DATA(SIZE)      SOLID_ELEMWISE1_DATA_##SIZE

typedef struct
{  unsigned int  size[SOLID_MAX_TENSOR_DIMS];
   int           strides[SOLID_MAX_TENSOR_DIMS];
   char         *ptr;
   size_t        nelem;
   int           ndims;
} solid_elemwise1_data_small;

typedef struct
{  size_t        size[SOLID_MAX_TENSOR_DIMS];
   ptrdiff_t     strides[SOLID_MAX_TENSOR_DIMS];
   char         *ptr;
   size_t        nelem;
   int           ndims;
} solid_elemwise1_data_large;


/* ------------------------------------------------------------------------ */
/* Analysis function declaration                                            */
/* ------------------------------------------------------------------------ */
SOLID_API int solid_gpu_elemwise1_analyze(int ndims, const size_t *size, const ptrdiff_t *strides,
                                          void *ptr, int elemsize, int flag_unroll,
                                          solid_elemwise1_data_small *data_small,
                                          solid_elemwise1_data_large *data_large,
                                          solid_gpu_config *config, int *kernel_index);


/* ------------------------------------------------------------------------ */
/* Name macros                                                              */
/* ------------------------------------------------------------------------ */

#define SOLID_KERNEL_ELEMWISE1_SMALL(PREFIX, NDIMS, UNROLL) \
   PREFIX##_kernel_##NDIMS##_##UNROLL 
#define SOLID_KERNEL_ELEMWISE1_LARGE(PREFIX, NDIMS, UNROLL) \
   PREFIX##_kernel_##NDIMS##_##UNROLL##L
#define SOLID_KERNEL_ELEMWISE1_NAME(PREFIX, NDIMS, UNROLL, SIZE) \
   SOLID_KERNEL_ELEMWISE1_##SIZE(PREFIX, NDIMS, UNROLL)

#define SOLID_KERNEL_ELEMWISE1_ITF_0(PREFIX, NDIMS, UNROLL, SIZE) \
   SOLID_KERNEL_ELEMWISE1_NAME(PREFIX, NDIMS, UNROLL, SIZE)\
                               (SOLID_ELEMWISE1_DATA(SIZE) data)
#define SOLID_KERNEL_ELEMWISE1_ITF_1(PREFIX, NDIMS, UNROLL, SIZE) \
   SOLID_KERNEL_ELEMWISE1_NAME(PREFIX, NDIMS, UNROLL, SIZE)\
                               (SOLID_ELEMWISE1_DATA(SIZE) data, SOLID_KERNEL_PARAM_PREFIX(PREFIX) param)
#define SOLID_KERNEL_ELEMWISE1_ITF(PREFIX, FLAG_PARAM, NDIMS, UNROLL, SIZE) \
   SOLID_KERNEL_ELEMWISE1_ITF_##FLAG_PARAM(PREFIX, NDIMS, UNROLL, SIZE)


/* ------------------------------------------------------------------------ */
/* Create all cuda kernels                                                  */
/* ------------------------------------------------------------------------ */

#define SOLID_CREATE_KERNELS_ELEMWISE1_0(PREFIX, DTYPE, FLAG_PARAM, CODE) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 1, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 2, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 3, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, N, 1, SMALL, 1024) \
   \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 1, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 2, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 3, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, N, 1, LARGE, 1024)

#define SOLID_CREATE_KERNELS_ELEMWISE1_1(PREFIX, DTYPE, FLAG_PARAM, CODE) \
   /* Create all regular kernels */ \
   SOLID_CREATE_KERNELS_ELEMWISE1_0(PREFIX, DTYPE, FLAG_PARAM, CODE) \
   \
   /* Create all unrolled kernels */ \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 2, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 3, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, 3, 4, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, N, 2, LARGE, 1024) \
   SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, N, 4, LARGE, 1024)

#define SOLID_CREATE_KERNELS_ELEMWISE1(PREFIX, FLAG_UNROLLED, FLAG_PARAM, DTYPE, CODE) \
   SOLID_CREATE_KERNELS_ELEMWISE1_## FLAG_UNROLLED(PREFIX, DTYPE, FLAG_PARAM, CODE)


/* Create types and kernels */
#define SOLID_KERNELS_ELEMWISE1_PREFIX_B(PREFIX, FLAG_UNROLLED, DTYPE, FLAG_PARAM, PARAM, CODE) \
   SOLID_CREATE_KERNEL_TYPES(PREFIX, FLAG_PARAM, PARAM) \
   SOLID_CREATE_KERNELS_ELEMWISE1(PREFIX, FLAG_UNROLLED, FLAG_PARAM, DTYPE, CODE)

#define SOLID_KERNELS_ELEMWISE1_PREFIX(PREFIX, FLAG_UNROLLED, DTYPE, FLAG_PARAM, PARAM, CODE) \
   SOLID_KERNELS_ELEMWISE1_PREFIX_B(PREFIX, FLAG_UNROLLED, DTYPE, FLAG_PARAM, PARAM, CODE)


/* Main interfaces */
#define SOLID_KERNELS_ELEMWISE1_FULL(DTYPE, UNROLLING, NAME, PARAM, CODE) \
   SOLID_KERNELS_ELEMWISE1_PREFIX(SOLID_FUNCTION_TYPE(NAME, DTYPE), \
                                  SOLID_FLAG_UNROLLED(UNROLLING), DTYPE, \
                                  1, PARAM, CODE)

#define SOLID_KERNELS_ELEMWISE1_PARAM(UNROLLING, NAME, PARAM, CODE) \
   SOLID_KERNELS_ELEMWISE1_FULL(SDXTYPE, UNROLLING, NAME, PARAM, CODE)

#define SOLID_KERNELS_ELEMWISE1_TYPE(DTYPE, UNROLLING, NAME, CODE) \
   SOLID_KERNELS_ELEMWISE1_PREFIX(SOLID_FUNCTION_TYPE(NAME, DTYPE), \
                                  SOLID_FLAG_UNROLLED(UNROLLING), DTYPE, \
                                  0, { }, CODE)

#define SOLID_KERNELS_ELEMWISE1(UNROLLING, NAME, CODE) \
   SOLID_KERNELS_ELEMWISE1_TYPE(SDXTYPE, UNROLLING, NAME, CODE)


/* ------------------------------------------------------------------------ */
/* Launching the kernels                                                    */
/* ------------------------------------------------------------------------ */

#define SOLID_SUBMIT_ELEMWISE1_PARAM_0(PARAM, SIZE) \
      (SOLID_ELEMWISE1_DATA_VAR(SIZE))
#define SOLID_SUBMIT_ELEMWISE1_PARAM_1(PARAM, SIZE) \
      (SOLID_ELEMWISE1_DATA_VAR(SIZE), PARAM)
#define SOLID_SUBMIT_ELEMWISE1_PARAM(FLAG_PARAM, PARAM, SIZE) \
      SOLID_SUBMIT_ELEMWISE1_PARAM_## FLAG_PARAM(PARAM, SIZE)

#define SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, NDIMS, UNROLL, SIZE) \
      SOLID_KERNEL_ELEMWISE1_NAME(PREFIX, NDIMS, UNROLL, SIZE)\
      <<<(CONFIG)->blocks, (CONFIG)->threads, SHAREDMEM, STREAM>>>\
      SOLID_SUBMIT_ELEMWISE1_PARAM(FLAG_PARAM, PARAM, SIZE)

#define SOLID_LAUNCH_ELEMWISE1_CASES_0(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
      case 0 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, 1, SMALL); break; \
      case 1 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 1, SMALL); break; \
      case 2 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 3, 1, SMALL); break; \
      case 3 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 1, SMALL); break; \
      case 4 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, 1, LARGE); break; \
      case 5 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 1, LARGE); break; \
      case 6 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 3, 1, LARGE); break; \
      case 7 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 1, LARGE); break;

#define SOLID_LAUNCH_ELEMWISE1_CASES_1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
      SOLID_LAUNCH_ELEMWISE1_CASES_0(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
      case 8 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 2, LARGE); break; \
      case 9 : SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 3, 2, LARGE); break; \
      case 10: SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 3, 4, LARGE); break; \
      case 11: SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 2, LARGE); break; \
      case 12: SOLID_SUBMIT_ELEMWISE1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 4, LARGE); break;

#define SOLID_LAUNCH_ELEMWISE1_CASES(PREFIX, FLAG_UNROLLED, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
      SOLID_LAUNCH_ELEMWISE1_CASES_##FLAG_UNROLLED(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM)

#define SOLID_LAUNCH_ELEMWISE1_CODE(PREFIX, FLAG_UNROLLED, DTYPE, \
                                    SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT) \
   {  SOLID_ELEMWISE1_DATA_SMALL SOLID_ELEMWISE1_DATA_VAR_SMALL; \
      SOLID_ELEMWISE1_DATA_LARGE SOLID_ELEMWISE1_DATA_VAR_LARGE; \
      solid_gpu_config           __elemwise1_kernel_config; \
      int                        __elemwise1_kernel_index; \
      cudaError_t                __cuda_status; \
      \
      /* Initialize the data and determine which kernel to use */ \
      (RESULT) = solid_gpu_elemwise1_analyze(ndims, size, strides, ptr, \
                                             sizeof(SOLID_C_TYPE(DTYPE)), FLAG_UNROLLED, \
                                             &(SOLID_ELEMWISE1_DATA_VAR_SMALL), \
                                             &(SOLID_ELEMWISE1_DATA_VAR_LARGE), \
                                             &(__elemwise1_kernel_config), \
                                             &(__elemwise1_kernel_index)); \
      \
      /* Call the appropriate kernel */ \
      if (((RESULT) == 0) && (SOLID_ELEMWISE1_DATA_VAR_SMALL.nelem > 0)) \
      {  switch (__elemwise1_kernel_index) \
         {  SOLID_LAUNCH_ELEMWISE1_CASES(PREFIX, FLAG_UNROLLED, &(__elemwise1_kernel_config), \
                                         SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
            default : SOLID_ERROR_MESSAGE("Unrecognized kernel index (%d)", __elemwise1_kernel_index); \
         } \
         if ((__cuda_status = cudaGetLastError()) != cudaSuccess) \
         {  SOLID_ERROR_MESSAGE("Cuda error: %s", cudaGetErrorString(__cuda_status)); \
            (RESULT) = -1; \
         } \
      } \
   }

#define SOLID_LAUNCH_ELEMWISE1_PREFIX_B(PREFIX, FLAG_UNROLLED, DTYPE, \
                                        SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE1_CODE(PREFIX, FLAG_UNROLLED, DTYPE, \
                               SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT)

#define SOLID_LAUNCH_ELEMWISE1_PREFIX(PREFIX, FLAG_UNROLLED, DTYPE, \
                                      SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE1_PREFIX_B(PREFIX, FLAG_UNROLLED, DTYPE, \
                                   SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT)

/* Main interfaces */
#define SOLID_LAUNCH_ELEMWISE1_FULL(DTYPE, UNROLLING, NAME, \
                                    SHAREDMEM, STREAM, PARAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE1_PREFIX(SOLID_FUNCTION_TYPE(NAME, DTYPE), \
                                 SOLID_FLAG_UNROLLED(UNROLLING), DTYPE, \
                                 SHAREDMEM, STREAM, 1, PARAM, RESULT)

#define SOLID_LAUNCH_ELEMWISE1_PARAM(UNROLLING, NAME, SHAREDMEM, STREAM, PARAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE1_FULL(SDXTYPE, UNROLLING, NAME, SHAREDMEM, STREAM, PARAM, RESULT)

#define SOLID_LAUNCH_ELEMWISE1_TYPE(DTYPE, UNROLLING, NAME, \
                                    SHAREDMEM, STREAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE1_PREFIX(SOLID_FUNCTION_TYPE(NAME, DTYPE), \
                                 SOLID_FLAG_UNROLLED(UNROLLING), DTYPE, \
                                 SHAREDMEM, STREAM, 0, NULL, RESULT)

#define SOLID_LAUNCH_ELEMWISE1(UNROLLING, NAME, SHAREDMEM, STREAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE1_TYPES(SDXTYPE, UNROLLING, NAME, SHAREDMEM, STREAM, RESULT)


/* --------------------------------------------------------------------- */
/* KERNEL OFFSET COMPUTATION                                             */
/* --------------------------------------------------------------------- */
#define SOLID_ELEMWISE1_OFFSET_1(SIZE) \
   _offset = _index * data.strides[0];

#define SOLID_ELEMWISE1_OFFSET_2(SIZE) \
   {  SOLID_ELEMWISE_INDEX_TYPE(SIZE) _idx; \
      SOLID_ELEMWISE_INDEX_TYPE(SIZE) _s; \
      \
      _s       = _index % data.size[0]; \
      _idx     = _index / data.size[0]; \
      _offset  = _s * data.strides[0]; \
      _offset += _idx * data.strides[1]; \
   }

#define SOLID_ELEMWISE1_OFFSET_3(SIZE) \
   {  SOLID_ELEMWISE_INDEX_TYPE(SIZE) _idx; \
      SOLID_ELEMWISE_INDEX_TYPE(SIZE) _s; \
      \
      _s       = _index % data.size[0]; \
      _idx     = _index / data.size[0]; \
      _offset  = _s * data.strides[0]; \
      _s       = _idx % data.size[1]; \
      _idx     = _idx / data.size[1]; \
      _offset += _s * data.strides[1]; \
      _offset += _idx * data.strides[2]; \
   }

#define SOLID_ELEMWISE1_OFFSET_N(SIZE) \
   {  SOLID_ELEMWISE_INDEX_TYPE(SIZE) _idx; \
      SOLID_ELEMWISE_INDEX_TYPE(SIZE) _s; \
      int _i; \
      \
      _idx = _index; \
      _offset = 0; \
      for (_i = 0; _i < data.ndims-1; _i++) \
      {  _s       = _idx % data.size[_i]; \
         _idx     = _idx / data.size[_i]; \
         _offset += _s * data.strides[_i]; \
      } \
      _offset += _idx * data.strides[data.ndims-1]; \
   }

#define SOLID_ELEMWISE1_OFFSET_B(NDIMS, SIZE) \
           SOLID_ELEMWISE1_OFFSET_##NDIMS(SIZE)
#define SOLID_ELEMWISE1_OFFSET(NDIMS, SIZE) \
           SOLID_ELEMWISE1_OFFSET_B(NDIMS, SIZE)


/* --------------------------------------------------------------------- */
/* KERNEL CODE UNROLLING                                                 */
/* --------------------------------------------------------------------- */

#define SOLID_ELEMWISE1_CODE_C1(DTYPE, CODE) \
    /* Determine the pointer */ \
    _ptr = (SOLID_C_TYPE(DTYPE) *)(data.ptr + _offset); \
    \
    /* CODE */ \
    CODE

#define SOLID_ELEMWISE1_CODE_C2(DTYPE, CODE) \
   SOLID_ELEMWISE1_CODE_C1(DTYPE, CODE) \
   \
   /* Next element */ \
   _index ++; \
   _offset += data.strides[0]; \
   \
   SOLID_ELEMWISE1_CODE_C1(DTYPE, CODE)

#define SOLID_ELEMWISE1_CODE_C4(DTYPE, CODE) \
   SOLID_ELEMWISE1_CODE_C2(DTYPE, CODE) \
   \
   /* Next element */ \
   _index ++; \
   _offset += data.strides[0]; \
   \
   SOLID_ELEMWISE1_CODE_C2(DTYPE, CODE)

#define SOLID_ELEMWISE1_CODE_B(DTYPE, UNROLL, CODE) \
   SOLID_ELEMWISE1_CODE_C##UNROLL(DTYPE, CODE)
#define SOLID_ELEMWISE1_CODE(DTYPE, UNROLL, CODE) \
   SOLID_ELEMWISE1_CODE_B(DTYPE, UNROLL, CODE)


/* --------------------------------------------------------------------- */
/* KERNELS                                                               */
/* --------------------------------------------------------------------- */

#define SOLID_CREATE_KERNEL_ELEMWISE1(PREFIX, DTYPE, FLAG_PARAM, CODE, NDIMS, UNROLL, SIZE, BOUNDS) \
   __launch_bounds__(BOUNDS) \
   __global__ void SOLID_KERNEL_ELEMWISE1_ITF(PREFIX, FLAG_PARAM, NDIMS, UNROLL, SIZE) \
   {  SOLID_ELEMWISE_SIZE_TYPE(SIZE)  _index; \
      SOLID_ELEMWISE_INDEX_TYPE(SIZE) _offset; \
      SOLID_C_TYPE(DTYPE)            *_ptr; \
      \
      for (_index = (UNROLL) * (blockIdx.x * blockDim.x + threadIdx.x); \
           _index < data.nelem; \
           _index += (UNROLL) * (gridDim.x * blockDim.x) - ((UNROLL) - 1)) \
      { \
         /* Determine the offsets */ \
         SOLID_ELEMWISE1_OFFSET(NDIMS, SIZE) \
         \
         /* Expand the code */ \
         SOLID_ELEMWISE1_CODE(DTYPE, UNROLL, CODE) \
      } \
   }

#endif

