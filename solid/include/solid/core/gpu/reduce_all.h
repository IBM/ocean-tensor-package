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

#ifndef __SOLID_GPU_REDUCE_ALL_H__
#define __SOLID_GPU_REDUCE_ALL_H__

#include "solid.h"
#include "solid/base/gpu/types_gpu.h"
#include "solid/base/gpu/generate_macros.h"
#include "solid/base/generic/dtype_assign.h"


/* ------------------------------------------------------------------------ */
/* Tensor information structure                                             */
/* ------------------------------------------------------------------------ */

/* ------------------------------------------------------------------------ */
/* Note: To avoid generating too many functions we only provide reduction   */
/* implementations with long int addressing and omit the somewhat faster    */
/* int indexing. If support for int indexing is required, please see the    */
/* elementwise macros for examples.                                         */
/* ------------------------------------------------------------------------ */

typedef struct
{  size_t        size[SOLID_MAX_TENSOR_DIMS];
   ptrdiff_t     strides[SOLID_MAX_TENSOR_DIMS];
   char         *ptrData;
   char         *ptrBuffer;
   char         *ptrOutput;
   size_t        nelem;
   int           ndims;
} solid_reduce_all_data;


/* ------------------------------------------------------------------------ */
/* Analysis function declaration                                            */
/* ------------------------------------------------------------------------ */
SOLID_API int solid_gpu_reduce_all_config     (size_t nelem, int device, solid_gpu_config *config);
SOLID_API int solid_gpu_reduce_all_buffer_size(size_t nelem, int device, size_t *size);
SOLID_API int solid_gpu_reduce_all_analyze    (int ndims, const size_t *size, const ptrdiff_t *strides,
                                               void *ptr, void *buffer,
                                               solid_reduce_all_data *data, solid_gpu_config *config);


/* ------------------------------------------------------------------------ */
/* NAME MACROS                                                              */
/* ------------------------------------------------------------------------ */

#define SOLID_KERNEL_REDUCE_ALL_NAME(PREFIX, INFIX) \
   PREFIX##_kernel_##INFIX

#define SOLID_KERNEL_REDUCE_ALL_NAME_NDIMS_C(PREFIX, NDIMS) \
   PREFIX##_##NDIMS
#define SOLID_KERNEL_REDUCE_ALL_NAME_NDIMS_B(PREFIX, NDIMS) \
   SOLID_KERNEL_REDUCE_ALL_NAME_NDIMS_C(PREFIX, NDIMS)
#define SOLID_KERNEL_REDUCE_ALL_NAME_NDIMS(PREFIX, INFIX, NDIMS) \
    SOLID_KERNEL_REDUCE_ALL_NAME_NDIMS_B(SOLID_KERNEL_REDUCE_ALL_NAME(PREFIX, INFIX),NDIMS)


/* --------------------------------------------------------------------- */
/* KERNEL OFFSET COMPUTATION                                             */
/* --------------------------------------------------------------------- */
#define SOLID_REDUCE_ALL_OFFSET_1 \
   _offset = _index * data.strides[0];

#define SOLID_REDUCE_ALL_OFFSET_2 \
   {  long int _idx; \
      long int _s; \
      \
      _s       = _index % data.size[0]; \
      _idx     = _index / data.size[0]; \
      _offset  = _s * data.strides[0]; \
      _offset += _idx * data.strides[1]; \
   }

#define SOLID_REDUCE_ALL_OFFSET_3 \
   {  long int _idx; \
      long int _s; \
      \
      _s       = _index % data.size[0]; \
      _idx     = _index / data.size[0]; \
      _offset  = _s * data.strides[0]; \
      _s       = _idx % data.size[1]; \
      _idx     = _idx / data.size[1]; \
      _offset += _s * data.strides[1]; \
      _offset += _idx * data.strides[2]; \
   }

#define SOLID_REDUCE_ALL_OFFSET_N \
   {  long int _idx; \
      long int _s; \
      int      _i; \
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

#define SOLID_REDUCE_ALL_OFFSET_B(NDIMS) \
           SOLID_REDUCE_ALL_OFFSET_##NDIMS
#define SOLID_REDUCE_ALL_OFFSET(NDIMS) \
           SOLID_REDUCE_ALL_OFFSET_B(NDIMS)


/* --------------------------------------------------------------------- */
/* KERNELS - ACCUMULATION                                                */
/* --------------------------------------------------------------------- */

/* Kernel interface */
#define SOLID_KERNEL_REDUCE_ALL_ITF_0(PREFIX, INFIX, NDIMS) \
   SOLID_KERNEL_REDUCE_ALL_NAME_NDIMS(PREFIX, INFIX, NDIMS)(solid_reduce_all_data data)
#define SOLID_KERNEL_REDUCE_ALL_ITF_1(PREFIX, INFIX, NDIMS) \
   SOLID_KERNEL_REDUCE_ALL_NAME_NDIMS(PREFIX, INFIX, NDIMS)(solid_reduce_all_data data, SOLID_KERNEL_PARAM_PREFIX(PREFIX) param)
#define SOLID_KERNEL_REDUCE_ALL_ITF(PREFIX, INFIX, FLAG_PARAM, NDIMS) \
   SOLID_KERNEL_REDUCE_ALL_ITF_##FLAG_PARAM(PREFIX, INFIX, NDIMS)

/* Kernel declaration */
#define SOLID_CREATE_KERNEL_REDUCE_ALL_ACCUMULATE(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE_INIT, \
                                                  CODE_ACCUMULATE, CODE_REDUCE, NDIMS) \
   __launch_bounds__(512) \
   __global__ void SOLID_KERNEL_REDUCE_ALL_ITF(PREFIX, accumulate, FLAG_PARAM, NDIMS) \
   {  long int _index; \
      long int _offset; \
      int      _b, _tid; \
      SOLID_C_TYPE(DTYPE1) *_ptr = NULL; \
      volatile __shared__ SOLID_C_WORKTYPE_TYPE(DTYPE2) _intermediate[512]; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_partial; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_result; \
      SOLID_C_WORKTYPE_TYPE(DTYPE2) _accumulate; \
      \
      /* Initialize */ \
      _tid = threadIdx.x; \
      _ptr = (SOLID_C_TYPE(DTYPE1) *)(data.ptrData); \
      _result = &(_intermediate[_tid]); \
      CODE_INIT;  /* Templated initialization of _accumulate     */ \
      (void)_ptr; /* Avoid compiler warning of unused assignment */ \
      \
      /* Accumulate the local result */ \
      for (_index = (blockIdx.x * blockDim.x + _tid); \
           _index < data.nelem; \
           _index += gridDim.x * blockDim.x) \
      { \
         /* Determine the _offset based on _index and data.strides */ \
         SOLID_REDUCE_ALL_OFFSET(NDIMS) \
         \
         /* Determine the pointer */ \
         _ptr = (SOLID_C_TYPE(DTYPE1) *)(data.ptrData + _offset); \
         \
         /* Templated accumulation of *_ptr into _accumulate */ \
         CODE_ACCUMULATE; \
      } \
      \
      /* Assign the intermediate value and synchronize threads */ \
      SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), SOLID_WORKTYPE(DTYPE2), &_accumulate, _result); \
      __syncthreads(); \
      \
      /* Reduce to warp-size - the block dimension is assumed */ \
      /* to be a power of two of size at least 32.            */ \
      for (_b = blockDim.x / 2; _b > 32; _b /= 2) \
      {  if (_tid < _b) \
         {  _partial = &(_intermediate[_tid + _b]); \
            CODE_REDUCE; /* Reduce _partial into _result */ \
         } \
         __syncthreads(); \
      } \
      \
      /* Final block-level reduction in the first warp */ \
      if (_tid < 32) \
      {  /* The first reduction step only applies when blockDim.x exceeds 32 */ \
         if (blockDim.x > 32) \
         {  _partial = &(_intermediate[_tid + 32]); \
            CODE_REDUCE; /* Reduce _partial into _result */ \
         } \
      } \
      if (_tid < 16) \
      {  _partial = &(_intermediate[_tid + 16]);  CODE_REDUCE; \
         _partial = &(_intermediate[_tid + 8 ]);  CODE_REDUCE; \
         _partial = &(_intermediate[_tid + 4 ]);  CODE_REDUCE; \
         _partial = &(_intermediate[_tid + 2 ]);  CODE_REDUCE; \
         _partial = &(_intermediate[_tid + 1 ]);  CODE_REDUCE; \
      } \
      \
      /* Final block-level result */ \
      if (_tid == 0) \
      {  SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), SOLID_WORKTYPE(DTYPE2), _result, \
                      &(((SOLID_C_WORKTYPE_TYPE(DTYPE2) *)(data.ptrBuffer))[blockIdx.x])) \
      } \
   }


/* --------------------------------------------------------------------- */
/* KERNELS - FINALIZATION                                                */
/* --------------------------------------------------------------------- */

/* Kernel interface */
#define SOLID_KERNEL_REDUCE_ALL_FINALIZE_ITF_0(PREFIX, INFIX) \
   SOLID_KERNEL_REDUCE_ALL_NAME(PREFIX, INFIX)(void *buffer, void *output, int nblocks)
#define SOLID_KERNEL_REDUCE_ALL_FINALIZE_ITF_1(PREFIX, INFIX) \
   SOLID_KERNEL_REDUCE_ALL_NAME(PREFIX, INFIX)(void *buffer, void *output, int nblocks, SOLID_KERNEL_PARAM_PREFIX(PREFIX) param)
#define SOLID_KERNEL_REDUCE_ALL_FINALIZE_ITF(PREFIX, INFIX, FLAG_PARAM) \
   SOLID_KERNEL_REDUCE_ALL_FINALIZE_ITF_##FLAG_PARAM(PREFIX, INFIX)

/* Finalization code */
#define SOLID_REDUCE_ALL_FINALIZE_B0(RTYPE, CODE_FINALIZE) /* Empty */
#define SOLID_REDUCE_ALL_FINALIZE_B1(RTYPE, CODE_FINALIZE) \
   {  SOLID_C_WORKTYPE_TYPE(RTYPE) _temp = _intermediate[0]; \
      _partial = &_temp; \
      CODE_FINALIZE; \
   }
#define SOLID_REDUCE_ALL_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_REDUCE_ALL_FINALIZE_B##FLAG_FINALIZE(RTYPE, CODE_FINALIZE)
   
#define SOLID_CREATE_KERNEL_REDUCE_ALL_FINALIZE(PREFIX, DTYPE2, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   __launch_bounds__(32) \
   __global__ void SOLID_KERNEL_REDUCE_ALL_FINALIZE_ITF(PREFIX, finalize, FLAG_PARAM) \
   {  volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_intermediate = (SOLID_C_WORKTYPE_TYPE(DTYPE2) *)buffer; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_result; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_partial; \
      SOLID_C_TYPE(DTYPE2) _value; \
      int _s, _b, _tid; \
      \
      /* ----------------------------------------------------------- */ \
      /* Variable _s gives the number of remaining elements in the   */ \
      /* intermediate array after the reduction; _b gives the number */ \
      /* of elements that are updated. This code can deal with the   */ \
      /* case where the number of blocks used for the accumulation   */ \
      /* is not a power of two. The final result of the reduction is */ \
      /* stored in buffer[0] in the desired data, instead of the     */ \
      /* work type (it is more difficult to do the conversion on the */ \
      /* CPU since it is not guaranteed that the same data types     */ \
      /* require work types, and that these types are the same).     */ \
      /* ----------------------------------------------------------- */ \
      _tid = threadIdx.x; \
      _s = (nblocks + 1) / 2; \
      _b = (nblocks    ) / 2; \
      _result = &(_intermediate[_tid]); \
      \
      for ( ; _b > 0; _b = _s / 2, _s = (_s+1) / 2) \
      {  if (_tid < _b) \
         {  _partial = &(_intermediate[_tid + _s]); \
            CODE_REDUCE /* Reduce _partial into _result */ \
         } \
      } \
      \
      /* Convert the result from the work type to the desired type */ \
      if (_tid == 0) \
      {  /* Apply finalization if needed (the type must remain the same) */ \
         SOLID_REDUCE_ALL_FINALIZE(DTYPE2, FLAG_FINALIZE, CODE_FINALIZE) \
         \
         /* Convert from work type _intermediate[0] to desired type in */ \
         /* buffer[0] since _intermediate points to buffer we use an   */ \
         /* intermediate variable _value for the converstion.          */ \
         SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), DTYPE2, _intermediate, &_value); \
         SOLID_ASSIGN(DTYPE2, DTYPE2, &_value, output); \
      } \
   }



/* ------------------------------------------------------------------------ */
/* CREATE ALL CUDA KERNELS                                                  */
/* ------------------------------------------------------------------------ */

/* Create the actual kernels */
#define SOLID_CREATE_KERNELS_REDUCE_ALL(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_CREATE_KERNEL_REDUCE_ALL_ACCUMULATE(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, 1) \
   SOLID_CREATE_KERNEL_REDUCE_ALL_ACCUMULATE(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, 2) \
   SOLID_CREATE_KERNEL_REDUCE_ALL_ACCUMULATE(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, 3) \
   SOLID_CREATE_KERNEL_REDUCE_ALL_ACCUMULATE(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, N) \
   SOLID_CREATE_KERNEL_REDUCE_ALL_FINALIZE(PREFIX, DTYPE2, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM)

/* Create types, kernels, and launch code */
#define SOLID_CREATE_REDUCE_ALL_FULL_B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_CREATE_KERNEL_TYPES(PREFIX, FLAG_PARAM, PARAM) \
   SOLID_CREATE_KERNELS_REDUCE_ALL(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_CREATE_LAUNCH_REDUCE_ALL(PREFIX, DTYPE2, FLAG_INIT, CODE_INIT, FLAG_PARAM)

#define SOLID_CREATE_REDUCE_ALL_FULL(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_CREATE_REDUCE_ALL_FULL_B(PREFIX, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE)

/* Main interface */
#define SOLID_CREATE_REDUCE_ALL(NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_CREATE_REDUCE_ALL_FULL(SOLID_FUNCTION_TYPE(NAME, DTYPE1), \
                                 DTYPE1, DTYPE2, FLAG_PARAM, PARAM, \
                                 FLAG_INIT, CODE_INIT, CODE_ACCUMULATE, \
                                 CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE)


/* ------------------------------------------------------------------------ */
/* LAUNCHING THE KERNELS                                                    */
/* ------------------------------------------------------------------------ */

/* Launching the accumulation kernels */
#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_C0            (data)
#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_C1            (data, *param)
#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_B(FLAG_PARAM) SOLID_LAUNCH_KERNEL_REDUCE_ALL_C##FLAG_PARAM

#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_A(PREFIX, NDIMS) \
   SOLID_KERNEL_REDUCE_ALL_NAME_NDIMS(PREFIX, accumulate, NDIMS)<<<config.blocks, config.threads, 0, stream>>>

#define SOLID_LAUNCH_KERNEL_REDUCE_ALL(PREFIX, FLAG_PARAM, NDIMS) \
   SOLID_LAUNCH_KERNEL_REDUCE_ALL_A(PREFIX, NDIMS)SOLID_LAUNCH_KERNEL_REDUCE_ALL_B(FLAG_PARAM)

/* Launching the finalization kernel */
#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE_D0(FUNCTION) \
   FUNCTION<<<gridSize,blockSize,0,stream>>>(data.ptrBuffer, data.ptrOutput, config.blocks.x)
#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE_D1(FUNCTION) \
   FUNCTION<<<gridSize,blockSize,0,stream>>>(data.ptrBuffer, data.ptrOutput, config.blocks.x, *param)
#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE_C(FUNCTION, FLAG_PARAM) \
   SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE_D##FLAG_PARAM(FUNCTION)
#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE_B(PREFIX, FLAG_PARAM) \
   SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE_C(SOLID_KERNEL_REDUCE_ALL_NAME(PREFIX,finalize), FLAG_PARAM)                                         
#define SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE(PREFIX, FLAG_PARAM) \
   SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE_B(PREFIX, FLAG_PARAM)


/* Launch function interface */
#define SOLID_LAUNCH_REDUCE_ALL_ITF_0(PREFIX) \
   int SOLID_KERNEL_REDUCE_ALL_NAME(PREFIX, launch)(int ndims, const size_t *size, const ptrdiff_t *strides, \
                                                    void *ptr, void *buffer, void *result, cudaStream_t stream)
#define SOLID_LAUNCH_REDUCE_ALL_ITF_1(PREFIX) \
   int SOLID_KERNEL_REDUCE_ALL_NAME(PREFIX, launch)(int ndims, const size_t *size, const ptrdiff_t *strides, \
                                                    void *ptr, void *buffer, void *result, cudaStream_t stream, \
                                                   SOLID_KERNEL_PARAM_PREFIX(PREFIX) *param)
#define SOLID_LAUNCH_REDUCE_ALL_ITF(PREFIX, FLAG_PARAM) \
   SOLID_LAUNCH_REDUCE_ALL_ITF_##FLAG_PARAM(PREFIX)

/* Launch function initialization (CPU) */
#define SOLID_LAUNCH_REDUCE_ALL_CPU_INIT_B0(PREFIX, DTYPE, CODE_INIT) \
   {  /* Generate an error */ \
      SOLID_ERROR(-1, "Empty initialization is not allowed in function "#PREFIX); \
   }
#define SOLID_LAUNCH_REDUCE_ALL_CPU_INIT_B1(PREFIX, DTYPE2, CODE_INIT) \
   {  SOLID_CPU_C_WORKTYPE(DTYPE2) _accumulate; \
      \
      /* Initialize the result */ \
      CODE_INIT; \
      \
      /* Convert from the work type to the basic type */ \
      SOLID_CPU_ASSIGN_FROM_WORKTYPE(DTYPE2, &_accumulate, result); \
   }
#define SOLID_LAUNCH_REDUCE_ALL_CPU_INIT(PREFIX, DTYPE2, FLAG_INIT, CODE_INIT) \
   SOLID_LAUNCH_REDUCE_ALL_CPU_INIT_B##FLAG_INIT(PREFIX, DTYPE2, CODE_INIT)


/* Launch function definition */
#define SOLID_CREATE_LAUNCH_REDUCE_ALL(PREFIX, DTYPE2, FLAG_INIT, CODE_INIT, FLAG_PARAM) \
   /* --------------------------------------------------------------------------- */ \
   SOLID_LAUNCH_REDUCE_ALL_ITF(PREFIX, FLAG_PARAM) \
   /* --------------------------------------------------------------------------- */ \
   {  solid_reduce_all_data data; \
      solid_gpu_config      config; \
      cudaError_t           status = cudaSuccess; \
      dim3                  gridSize; \
      dim3                  blockSize; \
      \
      /* Analyze and simplify */ \
      if (solid_gpu_reduce_all_analyze(ndims, size, strides, ptr, buffer, &data, &config) != 0) \
         return -1; \
      \
      /* Deal with empty tensors */ \
      if (data.nelem == 0) \
      {  SOLID_LAUNCH_REDUCE_ALL_CPU_INIT(PREFIX, DTYPE2, FLAG_INIT, CODE_INIT) \
      } \
      else \
      {  /* Accumulate the result per block */ \
         ndims = data.ndims; \
         if (ndims > 3) ndims = 4; \
         switch (ndims) \
         {  case 1 : SOLID_LAUNCH_KERNEL_REDUCE_ALL(PREFIX, FLAG_PARAM, 1); break; \
            case 2 : SOLID_LAUNCH_KERNEL_REDUCE_ALL(PREFIX, FLAG_PARAM, 2); break; \
            case 3 : SOLID_LAUNCH_KERNEL_REDUCE_ALL(PREFIX, FLAG_PARAM, 3); break; \
            case 4 : SOLID_LAUNCH_KERNEL_REDUCE_ALL(PREFIX, FLAG_PARAM, N); break; \
         } \
         if ((status = cudaGetLastError()) != cudaSuccess) \
            SOLID_ERROR(-1, "Error launching reduction kernel: %s", cudaGetErrorString(status)); \
         \
         /* Final reduction from the block-level results; since the number   */ \
         /* of blocks is small we use a single warp for this reduction step. */ \
         gridSize.x  = 1;  \
         blockSize.x = 32; \
         SOLID_LAUNCH_KERNEL_REDUCE_ALL_FINALIZE(PREFIX, FLAG_PARAM);\
         status = cudaGetLastError(); \
         if (status != cudaSuccess) SOLID_ERROR(-1, "Error launching reduction kernel: %s", cudaGetErrorString(status)); \
         \
         /* Synchronize the stream  */ \
         status = cudaStreamSynchronize(stream); \
         if (status != cudaSuccess) SOLID_ERROR(-1, "Error synchronizing reduction kernel"); \
         \
         /* Copy the result */ \
         status = cudaMemcpy(result, data.ptrOutput, SDTYPE_SIZE(DTYPE2), cudaMemcpyDeviceToHost); \
         if (status != cudaSuccess) SOLID_ERROR(-1, "Error copying reduction result"); \
      } \
      \
      return 0; \
   }


/* ------------------------------------------------------------------------ */
/* CALLING THE LAUNCH FUNCTION                                              */
/* ------------------------------------------------------------------------ */

#define SOLID_CALL_REDUCE_ALL_TYPE_E0(NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM) \
   (NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM)
#define SOLID_CALL_REDUCE_ALL_TYPE_E1(NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM) \
   (NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM)
#define SOLID_CALL_REDUCE_ALL_TYPE_D(PREFIX) \
   SOLID_KERNEL_REDUCE_ALL_NAME(PREFIX, launch)
#define SOLID_CALL_REDUCE_ALL_FULL_C(PREFIX, NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM, FLAG_PARAM) \
   SOLID_CALL_REDUCE_ALL_TYPE_D(PREFIX)\
   SOLID_CALL_REDUCE_ALL_TYPE_E##FLAG_PARAM(NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM)
#define SOLID_CALL_REDUCE_ALL_FULL_B(NAME, DTYPE, NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM, FLAG_PARAM) \
   SOLID_CALL_REDUCE_ALL_FULL_C(SOLID_FUNCTION_TYPE(NAME, DTYPE), NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM, FLAG_PARAM)
#define SOLID_CALL_REDUCE_ALL_FULL(NAME, DTYPE, NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM, FLAG_PARAM) \
   SOLID_CALL_REDUCE_ALL_FULL_B(NAME, DTYPE, NDIMS, SIZE, STRIDES, PTR, BUFFER, RESULT, STREAM, PARAM, FLAG_PARAM) \

#define SOLID_CALL_REDUCE_ALL_PARAM(NAME, PARAM) \
   SOLID_CALL_REDUCE_ALL_FULL(NAME, SDXTYPE, ndims, size, strides, ptr, buffer, result, stream, PARAM, 1)

#define SOLID_CALL_REDUCE_ALL(NAME) \
   SOLID_CALL_REDUCE_ALL_FULL(NAME, SDXTYPE, ndims, size, strides, ptr, buffer, result, stream, PARAM, 0)

#endif
