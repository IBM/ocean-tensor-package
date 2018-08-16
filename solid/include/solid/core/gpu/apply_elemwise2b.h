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

#ifndef __SOLID_GPU_APPLY_ELEMWISE2B_H__
#define __SOLID_GPU_APPLY_ELEMWISE2B_H__

#include "solid.h"
#include "solid/base/gpu/generate_macros.h"
#include "solid/core/gpu/apply_elemwise.h"


/* ------------------------------------------------------------------------ */
/* Interface                                                                */
/* ------------------------------------------------------------------------ */
/* These kernels can be used for two tensors when the sizes match exactly.  */
/* The strides for each dimension can differ. The apply elemwise2b launch   */
/* macros for GPU assume that the following variables have been defined:    */
/*   int        ndims1                                                      */
/*   int        ndims1                                                      */
/*   size_t    *size1                                                       */
/*   size_t    *size2                                                       */
/*   ptrdiff_t *strides1                                                    */
/*   ptrdiff_t *strides2                                                    */
/*   void      *ptr1                                                        */
/*   void      *ptr2                                                        */
/*   int        result;             // Updated to reflect success / failure */
/* ------------------------------------------------------------------------ */
/* Exposed variables in kernel:                                             */
/*   solid_c_type(TYPE1) *_ptr1                                             */
/*   solid_c_type(TYPE2) *_ptr2                                             */
/* ------------------------------------------------------------------------ */


/* ------------------------------------------------------------------------ */
/* Tensor information structure                                             */
/* ------------------------------------------------------------------------ */

#define SOLID_ELEMWISE2B_DATA_VAR_SMALL  __elemwise2b_data_small
#define SOLID_ELEMWISE2B_DATA_VAR_LARGE  __elemwise2b_data_large
#define SOLID_ELEMWISE2B_DATA_VAR(SIZE)  SOLID_ELEMWISE2B_DATA_VAR_## SIZE

#define SOLID_ELEMWISE2B_DATA_SMALL      solid_elemwise2b_data_small
#define SOLID_ELEMWISE2B_DATA_LARGE      solid_elemwise2b_data_large
#define SOLID_ELEMWISE2B_DATA(SIZE)      SOLID_ELEMWISE2B_DATA_##SIZE

typedef struct
{  solid_layout_small layout1;
   solid_layout_small layout2;
   size_t             nelem;
} solid_elemwise2b_data_small;

typedef struct
{  solid_layout_large layout1;
   solid_layout_large layout2;
   size_t             nelem;
} solid_elemwise2b_data_large;


/* ------------------------------------------------------------------------ */
/* Analysis function declaration                                            */
/* ------------------------------------------------------------------------ */
SOLID_API int solid_gpu_elemwise2b_analyze(int ndims1, const size_t *size1, const ptrdiff_t *strides1, void *ptr1,
                                           int ndims2, const size_t *size2, const ptrdiff_t *strides2, void *ptr2,
                                           int elemsize1, int elemsize2, int flag_unroll,
                                           solid_elemwise2b_data_small *data_small,
                                           solid_elemwise2b_data_large *data_large,
                                           solid_gpu_config *config, int *kernel_index);


/* --------------------------------------------------------------------- */
/* Name macros                                                           */
/* --------------------------------------------------------------------- */

#define SOLID_KERNEL_ELEMWISE2B_SMALL(PREFIX, NDIMS1, NDIMS2, UNROLL) \
   PREFIX##_kernel_##NDIMS1##_##NDIMS2##_##UNROLL 
#define SOLID_KERNEL_ELEMWISE2B_LARGE(PREFIX, NDIMS1, NDIMS2, UNROLL) \
   PREFIX##_kernel_##NDIMS1##_##NDIMS2##_##UNROLL##L
#define SOLID_KERNEL_ELEMWISE2B_NAME(PREFIX, NDIMS1, NDIMS2, UNROLL, SIZE) \
   SOLID_KERNEL_ELEMWISE2B_##SIZE(PREFIX, NDIMS1, NDIMS2, UNROLL)


#define SOLID_KERNEL_ELEMWISE2B_ITF_0(PREFIX, NDIMS1, NDIMS2, UNROLL, SIZE) \
   SOLID_KERNEL_ELEMWISE2B_NAME(PREFIX, NDIMS1, NDIMS2, UNROLL, SIZE)\
                               (SOLID_ELEMWISE2B_DATA(SIZE) data)
#define SOLID_KERNEL_ELEMWISE2B_ITF_1(PREFIX, NDIMS1, NDIMS2, UNROLL, SIZE) \
   SOLID_KERNEL_ELEMWISE2B_NAME(PREFIX, NDIMS1, NDIMS2, UNROLL, SIZE)\
                                (SOLID_ELEMWISE2B_DATA(SIZE) data, SOLID_KERNEL_PARAM_PREFIX(PREFIX) param)
#define SOLID_KERNEL_ELEMWISE2B_ITF(PREFIX, FLAG_PARAM, NDIMS1, NDIMS2, UNROLL, SIZE) \
   SOLID_KERNEL_ELEMWISE2B_ITF_##FLAG_PARAM(PREFIX, NDIMS1, NDIMS2, UNROLL, SIZE)


/* ------------------------------------------------------------------------ */
/* Create all cuda kernels                                                  */
/* ------------------------------------------------------------------------ */

#define SOLID_CREATE_KERNELS_ELEMWISE2B_0(PREFIX, FLAG_PARAM, DTYPE1, DTYPE2, CODE) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, 1, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, 2, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, N, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, 1, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, 2, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, N, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, 1, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, 2, 1, SMALL, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, N, 1, SMALL, 512) \
   \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, 1, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, 2, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, N, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, 1, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, 2, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, N, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, 1, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, 2, 1, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, N, 1, LARGE, 512)

#define SOLID_CREATE_KERNELS_ELEMWISE2B_1(PREFIX, FLAG_PARAM, DTYPE1, DTYPE2, CODE) \
   /* Create all regular kernels */ \
   SOLID_CREATE_KERNELS_ELEMWISE2B_0(PREFIX, FLAG_PARAM, DTYPE1, DTYPE2, CODE) \
   \
   /* Create all unrolled kernels */ \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, N, 2, SMALL, 512) \
   \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, 2, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, N, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 1, N, 4, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, 1, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, 2, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, N, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, 2, N, 4, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, 1, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, 1, 4, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, 2, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, 2, 4, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, N, 2, LARGE, 512) \
   SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, N, N, 4, LARGE, 512)

#define SOLID_CREATE_KERNELS_ELEMWISE2B(PREFIX, FLAG_UNROLLED, FLAG_PARAM, DTYPE1, DTYPE2, CODE) \
   SOLID_CREATE_KERNELS_ELEMWISE2B_## FLAG_UNROLLED(PREFIX, FLAG_PARAM, DTYPE1, DTYPE2, CODE)


/* Create types and kernels */
#define SOLID_KERNELS_ELEMWISE2B_PREFIX_B(PREFIX, FLAG_UNROLLED, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, CODE) \
   SOLID_CREATE_KERNEL_TYPES(PREFIX, FLAG_PARAM, PARAM) \
   SOLID_CREATE_KERNELS_ELEMWISE2B(PREFIX, FLAG_UNROLLED, FLAG_PARAM, DTYPE1, DTYPE2, CODE)

#define SOLID_KERNELS_ELEMWISE2B_PREFIX(PREFIX, FLAG_UNROLLED, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, CODE) \
   SOLID_KERNELS_ELEMWISE2B_PREFIX_B(PREFIX, FLAG_UNROLLED, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, CODE)


/* Main interfaces */
#define SOLID_KERNELS_ELEMWISE2B_FULL(DTYPE1, DTYPE2, NAME_TYPE, UNROLLING, NAME, PARAM, CODE) \
   SOLID_KERNELS_ELEMWISE2B_PREFIX(SOLID_FUNCTION2_TYPES(NAME, NAME_TYPE, DTYPE1, DTYPE2), \
                                   SOLID_FLAG_UNROLLED(UNROLLING), DTYPE1, DTYPE2, \
                                   1, PARAM, CODE)

#define SOLID_KERNELS_ELEMWISE2B_PARAM(UNROLLING, NAME, PARAM, CODE) \
   SOLID_KERNELS_ELEMWISE2B_FULL(SDXTYPE, SDXTYPE, 1, UNROLLING, NAME, PARAM, CODE)

#define SOLID_KERNELS_ELEMWISE2B_TYPES(DTYPE1, DTYPE2, NAME_TYPE, UNROLLING, NAME, CODE) \
   SOLID_KERNELS_ELEMWISE2B_PREFIX(SOLID_FUNCTION2_TYPES(NAME, NAME_TYPE, DTYPE1, DTYPE2), \
                                   SOLID_FLAG_UNROLLED(UNROLLING), DTYPE1, DTYPE2, \
                                   0, { }, CODE)

#define SOLID_KERNELS_ELEMWISE2B(UNROLLING, NAME, CODE) \
   SOLID_KERNELS_ELEMWISE2B_TYPES(SDXTYPE, SDXTYPE, 1, UNROLLING, NAME, CODE)


/* ------------------------------------------------------------------------ */
/* Launching the kernels                                                    */
/* ------------------------------------------------------------------------ */

#define SOLID_SUBMIT_ELEMWISE2B_PARAM_0(PARAM, SIZE) \
      (SOLID_ELEMWISE2B_DATA_VAR(SIZE))
#define SOLID_SUBMIT_ELEMWISE2B_PARAM_1(PARAM, SIZE) \
      (SOLID_ELEMWISE2B_DATA_VAR(SIZE), PARAM)
#define SOLID_SUBMIT_ELEMWISE2B_PARAM(FLAG_PARAM, PARAM, SIZE) \
      SOLID_SUBMIT_ELEMWISE2B_PARAM_## FLAG_PARAM(PARAM, SIZE)

#define SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, NDIMS1, NDIMS2, UNROLL, SIZE) \
      SOLID_KERNEL_ELEMWISE2B_NAME(PREFIX, NDIMS1, NDIMS2, UNROLL, SIZE)\
      <<<(CONFIG)->blocks, (CONFIG)->threads, SHAREDMEM, STREAM>>>\
      SOLID_SUBMIT_ELEMWISE2B_PARAM(FLAG_PARAM, PARAM, SIZE)

#define SOLID_LAUNCH_ELEMWISE2B_CASES_0(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
      case  0 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, 1, 1, SMALL); break; \
      case  1 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, 2, 1, SMALL); break; \
      case  2 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, N, 1, SMALL); break; \
      case  3 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 1, 1, SMALL); break; \
      case  4 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 2, 1, SMALL); break; \
      case  5 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, N, 1, SMALL); break; \
      case  6 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 1, 1, SMALL); break; \
      case  7 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 2, 1, SMALL); break; \
      case  8 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, N, 1, SMALL); break; \
      \
      case  9 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, 1, 1, LARGE); break; \
      case 10 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, 2, 1, LARGE); break; \
      case 11 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, N, 1, LARGE); break; \
      case 12 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 1, 1, LARGE); break; \
      case 13 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 2, 1, LARGE); break; \
      case 14 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, N, 1, LARGE); break; \
      case 15 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 1, 1, LARGE); break; \
      case 16 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 2, 1, LARGE); break; \
      case 17 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, N, 1, LARGE); break;

#define SOLID_LAUNCH_ELEMWISE2B_CASES_1(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
      SOLID_LAUNCH_ELEMWISE2B_CASES_0(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
      case 18 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, N, 2, SMALL); break; \
      case 19 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, 2, 2, LARGE); break; \
      case 20 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, N, 2, LARGE); break; \
      case 21 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 1, N, 4, LARGE); break; \
      case 22 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 1, 2, LARGE); break; \
      case 23 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, 2, 2, LARGE); break; \
      case 24 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, N, 2, LARGE); break; \
      case 25 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, 2, N, 4, LARGE); break; \
      case 29 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 1, 2, LARGE); break; \
      case 30 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 1, 4, LARGE); break; \
      case 31 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 2, 2, LARGE); break; \
      case 32 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, 2, 4, LARGE); break; \
      case 33 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, N, 2, LARGE); break; \
      case 34 : SOLID_SUBMIT_ELEMWISE2B(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM, N, N, 4, LARGE); break;

#define SOLID_LAUNCH_ELEMWISE2B_CASES(PREFIX, FLAG_UNROLLED, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
      SOLID_LAUNCH_ELEMWISE2B_CASES_##FLAG_UNROLLED(PREFIX, CONFIG, SHAREDMEM, STREAM, FLAG_PARAM, PARAM)

#define SOLID_LAUNCH_ELEMWISE2B_CODE(PREFIX, FLAG_UNROLLED, DTYPE1, DTYPE2, \
                                     SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT) \
   {  SOLID_ELEMWISE2B_DATA_SMALL SOLID_ELEMWISE2B_DATA_VAR_SMALL; \
      SOLID_ELEMWISE2B_DATA_LARGE SOLID_ELEMWISE2B_DATA_VAR_LARGE; \
      solid_gpu_config            __elemwise2b_kernel_config; \
      int                         __elemwise2b_kernel_index; \
      cudaError_t                 __cuda_status; \
      \
      /* Initialize the data and determine which kernel to use */ \
      (RESULT) = solid_gpu_elemwise2b_analyze(ndims1, size1, strides1, ptr1, \
                                              ndims2, size2, strides2, ptr2, \
                                              sizeof(SOLID_C_TYPE(DTYPE1)), \
                                              sizeof(SOLID_C_TYPE(DTYPE2)), FLAG_UNROLLED, \
                                              &(SOLID_ELEMWISE2B_DATA_VAR_SMALL), \
                                              &(SOLID_ELEMWISE2B_DATA_VAR_LARGE), \
                                              &(__elemwise2b_kernel_config), \
                                              &(__elemwise2b_kernel_index)); \
      \
      /* Call the appropriate kernel */ \
      if (((RESULT) == 0) && (SOLID_ELEMWISE2B_DATA_VAR_SMALL.nelem > 0)) \
      {  switch (__elemwise2b_kernel_index) \
         {  SOLID_LAUNCH_ELEMWISE2B_CASES(PREFIX, FLAG_UNROLLED, &(__elemwise2b_kernel_config), \
                                          SHAREDMEM, STREAM, FLAG_PARAM, PARAM) \
            default : SOLID_ERROR_MESSAGE("Unrecognized kernel index (%d)", __elemwise2b_kernel_index); \
         } \
         if ((__cuda_status = cudaGetLastError()) != cudaSuccess) \
         {  SOLID_ERROR_MESSAGE("Cuda error: %s", cudaGetErrorString(__cuda_status)); \
            (RESULT) = -1; \
         } \
      } \
   }

#define SOLID_LAUNCH_ELEMWISE2B_PREFIX_B(PREFIX, FLAG_UNROLLED, DTYPE1, DTYPE2, \
                                         SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE2B_CODE(PREFIX, FLAG_UNROLLED, DTYPE1, DTYPE2, \
                                SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT)

#define SOLID_LAUNCH_ELEMWISE2B_PREFIX(PREFIX, FLAG_UNROLLED, DTYPE1, DTYPE2, \
                                       SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE2B_PREFIX_B(PREFIX, FLAG_UNROLLED, DTYPE1, DTYPE2, \
                                    SHAREDMEM, STREAM, FLAG_PARAM, PARAM, RESULT)

/* Main interfaces */
#define SOLID_LAUNCH_ELEMWISE2B_FULL(DTYPE1, DTYPE2, NAME_TYPE, UNROLLING, NAME, \
                                     SHAREDMEM, STREAM, PARAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE2B_PREFIX(SOLID_FUNCTION2_TYPES(NAME, NAME_TYPE, DTYPE1, DTYPE2), \
                                  SOLID_FLAG_UNROLLED(UNROLLING), DTYPE1, DTYPE2, \
                                  SHAREDMEM, STREAM, 1, PARAM, RESULT)

#define SOLID_LAUNCH_ELEMWISE2B_PARAM(UNROLLING, NAME, SHAREDMEM, STREAM, PARAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE2B_FULL(SDXTYPE, SDXTYPE, 1, UNROLLING, NAME, SHAREDMEM, STREAM, PARAM, RESULT)

#define SOLID_LAUNCH_ELEMWISE2B_TYPES(DTYPE1, DTYPE2, NAME_TYPE, UNROLLING, NAME, \
                                      SHAREDMEM, STREAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE2B_PREFIX(SOLID_FUNCTION2_TYPES(NAME, NAME_TYPE, DTYPE1, DTYPE2), \
                                  SOLID_FLAG_UNROLLED(UNROLLING), DTYPE1, DTYPE2, \
                                  SHAREDMEM, STREAM, 0, NULL, RESULT)

#define SOLID_LAUNCH_ELEMWISE2B(UNROLLING, NAME, SHAREDMEM, STREAM, RESULT) \
   SOLID_LAUNCH_ELEMWISE2B_TYPES(SDXTYPE, SDXTYPE, 1, UNROLLING, NAME, SHAREDMEM, STREAM, RESULT)


/* --------------------------------------------------------------------- */
/* KERNEL OFFSET COMPUTATION                                             */
/* --------------------------------------------------------------------- */
#define SOLID_ELEMWISE2B_OFFSET_1(OFFSET, INDEX, LAYOUT, SIZE) \
   OFFSET = INDEX * (LAYOUT).strides[0];

#define SOLID_ELEMWISE2B_OFFSET_2(OFFSET, INDEX, LAYOUT, SIZE) \
   {  SOLID_ELEMWISE_SIZE_TYPE(SIZE)  __s; \
      SOLID_ELEMWISE_INDEX_TYPE(SIZE) __index; \
      __s       = INDEX % (LAYOUT).size[0]; \
      __index   = INDEX / (LAYOUT).size[0]; \
      OFFSET    = __s * (LAYOUT).strides[0]; \
      OFFSET   += __index * (LAYOUT).strides[1]; \
   }

#define SOLID_ELEMWISE2B_OFFSET_N(OFFSET, INDEX, LAYOUT, SIZE) \
   {  SOLID_ELEMWISE_SIZE_TYPE(SIZE)  __s; \
      SOLID_ELEMWISE_INDEX_TYPE(SIZE) __index; \
      short int __i; \
      __index = INDEX; \
      OFFSET  = 0; \
      for (__i = 0; __i < (LAYOUT).ndims-1; __i++) \
      {  __s      = __index % (LAYOUT).size[__i]; \
         __index  = __index / (LAYOUT).size[__i]; \
         OFFSET  += __s * (LAYOUT).strides[__i]; \
      } \
      OFFSET += __index * (LAYOUT).strides[(LAYOUT).ndims-1]; \
   }

#define SOLID_ELEMWISE2B_OFFSET_B(OFFSET, INDEX, LAYOUT, NDIMS, SIZE) \
           SOLID_ELEMWISE2B_OFFSET_##NDIMS(OFFSET, INDEX, LAYOUT, SIZE)
#define SOLID_ELEMWISE2B_OFFSET(OFFSET, INDEX, LAYOUT, NDIMS, SIZE) \
           SOLID_ELEMWISE2B_OFFSET_B(OFFSET, INDEX, LAYOUT, NDIMS, SIZE)


/* --------------------------------------------------------------------- */
/* KERNEL CODE UNROLLING                                                 */
/* --------------------------------------------------------------------- */

#define SOLID_ELEMWISE2B_CODE_C1(DTYPE1, DTYPE2, CODE) \
   /* Determine the pointers */ \
   _ptr1 = (SOLID_C_TYPE(DTYPE1) *)(data.layout1.ptr + _offset1); \
   _ptr2 = (SOLID_C_TYPE(DTYPE2) *)(data.layout2.ptr + _offset2); \
   \
   /* CODE */ \
   CODE

#define SOLID_ELEMWISE2B_CODE_C2(DTYPE1, DTYPE2, CODE) \
   SOLID_ELEMWISE2B_CODE_C1(DTYPE1, DTYPE2, CODE) \
   \
   /* Next element */ \
   _offset1 += data.layout1.strides[0]; \
   _offset2 += data.layout2.strides[0]; \
   \
   SOLID_ELEMWISE2B_CODE_C1(DTYPE1, DTYPE2, CODE)

#define SOLID_ELEMWISE2B_CODE_C4(DTYPE1, DTYPE2, CODE) \
   SOLID_ELEMWISE2B_CODE_C2(DTYPE1, DTYPE2, CODE) \
   \
   /* Next element */ \
   _offset1 += data.layout1.strides[0]; \
   _offset2 += data.layout2.strides[0]; \
   \
   SOLID_ELEMWISE2B_CODE_C2(DTYPE1, DTYPE2, CODE)

#define SOLID_ELEMWISE2B_CODE_B(DTYPE1, DTYPE2, UNROLL, CODE) \
   SOLID_ELEMWISE2B_CODE_C##UNROLL(DTYPE1, DTYPE2, CODE)
#define SOLID_ELEMWISE2B_CODE(DTYPE1, DTYPE2, UNROLL, CODE) \
   SOLID_ELEMWISE2B_CODE_B(DTYPE1, DTYPE2, UNROLL, CODE)


/* --------------------------------------------------------------------- */
/* KERNELS                                                               */
/* --------------------------------------------------------------------- */
#define SOLID_CREATE_KERNEL_ELEMWISE2B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE, NDIMS1, NDIMS2, UNROLL, SIZE, BOUNDS) \
   __launch_bounds__(BOUNDS) \
   __global__ void SOLID_KERNEL_ELEMWISE2B_ITF(PREFIX, FLAG_PARAM, NDIMS1, NDIMS2, UNROLL, SIZE) \
   {  SOLID_ELEMWISE_SIZE_TYPE(SIZE)  _idx; \
      SOLID_ELEMWISE_INDEX_TYPE(SIZE) _offset1; \
      SOLID_ELEMWISE_INDEX_TYPE(SIZE) _offset2; \
      \
      SOLID_C_TYPE(DTYPE1) *_ptr1; \
      SOLID_C_TYPE(DTYPE2) *_ptr2; \
      \
      for (_idx = (UNROLL) * (blockIdx.x * blockDim.x + threadIdx.x); \
           _idx < data.nelem; \
           _idx += (UNROLL) * (gridDim.x * blockDim.x)) \
      { \
         /* Determine the offsets */ \
         SOLID_ELEMWISE2B_OFFSET(_offset1, _idx, data.layout1, NDIMS1, SIZE); \
         SOLID_ELEMWISE2B_OFFSET(_offset2, _idx, data.layout2, NDIMS2, SIZE); \
         \
         /* Expand the code */ \
         SOLID_ELEMWISE2B_CODE(DTYPE1, DTYPE2, UNROLL, CODE) \
      } \
   }

#endif
