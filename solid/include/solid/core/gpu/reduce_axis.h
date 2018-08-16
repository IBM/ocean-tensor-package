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

#ifndef __SOLID_GPU_REDUCE_AXIS_H__
#define __SOLID_GPU_REDUCE_AXIS_H__

#include "solid.h"
#include "solid/base/gpu/types_gpu.h"
#include "solid/base/gpu/solid_cuda.h"
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
   size_t        rsize[SOLID_MAX_TENSOR_DIMS];
   ptrdiff_t     strides1[SOLID_MAX_TENSOR_DIMS];
   ptrdiff_t     strides2[SOLID_MAX_TENSOR_DIMS];
   ptrdiff_t     rstrides[SOLID_MAX_TENSOR_DIMS];
   char         *ptr1;
   char         *ptr2;
   char         *ptrBuffer;
   size_t        nelem;
   size_t        relem; /* Thread-based: excluding rsize[0] */
   int           ndims;
   int           rdims;
} solid_gpu_reduce_axis_data;

typedef struct
{  solid_gpu_reduce_axis_data data;
   int          mode;
   dim3         gridSize;
   dim3         blockSize;
   dim3         gridSizeFinalize;
   dim3         blockSizeFinalize;
   size_t       bufferSize;         /* Used after analysis to allocate the buffer */
   size_t       sharedMem;
   size_t       sharedMemFinalize;
   cudaStream_t stream;
} solid_gpu_reduce_axis_config;


/* --------------------------------------------------------------------- */
/* Macro: SOLID_CREATE_REDUCE_AXIS                                       */
/* --------------------------------------------------------------------- */

/* Create the parameter structure */
#define SOLID_CREATE_REDUCE_AXIS_PARAM_C0(PREFIX, PREFIX_AXIS, PARAM) \
   /* Empty */
#define SOLID_CREATE_REDUCE_AXIS_PARAM_C1(PREFIX, PREFIX_AXIS, PARAM) \
   SOLID_KERNEL_PARAM_STRUCT(PREFIX, PARAM)
#define SOLID_CREATE_REDUCE_AXIS_PARAM_B(PREFIX, PREFIX_AXIS, FLAG_PARAM, PARAM) \
   SOLID_CREATE_REDUCE_AXIS_PARAM_C##FLAG_PARAM(PREFIX, PREFIX_AXIS, PARAM)
#define SOLID_CREATE_REDUCE_AXIS_PARAM(PREFIX, NAME, DTYPE, FLAG_PARAM, PARAM) \
   SOLID_CREATE_REDUCE_AXIS_PARAM_B(PREFIX, SOLID_FUNCTION_TYPE(NAME, DTYPE), FLAG_PARAM, PARAM)


/* Create the parameter structure and entry functions */
#define SOLID_CREATE_REDUCE_AXIS_B(PREFIX, NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, CODE_INIT, \
                                   CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_CREATE_REDUCE_AXIS_PARAM(PREFIX, NAME, DTYPE1, FLAG_PARAM, PARAM) \
   SOLID_CREATE_REDUCE_AXIS_THREAD(PREFIX, NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, \
                                           CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   SOLID_CREATE_REDUCE_AXIS_WARP(PREFIX, NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, \
                                          CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   SOLID_CREATE_REDUCE_AXIS_BLOCK(PREFIX, NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, \
                                           CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   SOLID_CREATE_REDUCE_AXIS_BLOCK_FINALIZE(PREFIX, NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, \
                                           CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   SOLID_CREATE_REDUCE_AXIS_LAUNCH(PREFIX, NAME, DTYPE1, FLAG_PARAM)

#define SOLID_CREATE_REDUCE_AXIS(NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, CODE_INIT, \
                                 CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_CREATE_REDUCE_AXIS_B(SOLID_FUNCTION_TYPE(NAME,DTYPE1), NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, \
                              FLAG_INIT, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE)


/* --------------------------------------------------------------------- */
/* Helper macros                                                         */
/* --------------------------------------------------------------------- */

/* Function name */
#define SOLID_REDUCE_AXIS_FUNCTION_D(PREFIX, MODE) PREFIX##_##MODE
#define SOLID_REDUCE_AXIS_FUNCTION_C(PREFIX, MODE) SOLID_REDUCE_AXIS_FUNCTION_D(PREFIX, MODE)
#define SOLID_REDUCE_AXIS_FUNCTION_B(NAME, DTYPE, MODE) SOLID_REDUCE_AXIS_FUNCTION_C(SOLID_FUNCTION_TYPE(NAME, DTYPE),MODE)
#define SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, MODE) SOLID_REDUCE_AXIS_FUNCTION_B(NAME, DTYPE,MODE)

/* Function interface */
#define SOLID_REDUCE_AXIS_FUNCTION_ITF_D0(FUNCTION, PARAM_TYPE) \
   FUNCTION(solid_gpu_reduce_axis_data data)
#define SOLID_REDUCE_AXIS_FUNCTION_ITF_D1(FUNCTION, PARAM_TYPE) \
   FUNCTION(solid_gpu_reduce_axis_data data, PARAM_TYPE param)

#define SOLID_REDUCE_AXIS_FUNCTION_ITF_C(FUNCTION, PARAM_TYPE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_FUNCTION_ITF_D##FLAG_PARAM(FUNCTION, PARAM_TYPE)
#define SOLID_REDUCE_AXIS_FUNCTION_ITF_B(PREFIX, NAME, DTYPE, MODE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_FUNCTION_ITF_C(SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, MODE), SOLID_KERNEL_PARAM_PREFIX(PREFIX), FLAG_PARAM)
#define SOLID_REDUCE_AXIS_FUNCTION_ITF(PREFIX, NAME, DTYPE, MODE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_FUNCTION_ITF_B(PREFIX, NAME, DTYPE, MODE, FLAG_PARAM)

/* Variable names for shared intermediate arrays - this avoids errors */
/* that the declaration is incompatible with previous declarations.   */
#define SOLID_REDUCE_AXIS_INTERMEDIATE_D(FUNCTION) \
   FUNCTION##_shared_
#define SOLID_REDUCE_AXIS_INTERMEDIATE_C(FUNCTION) \
   SOLID_REDUCE_AXIS_INTERMEDIATE_D(FUNCTION)
#define SOLID_REDUCE_AXIS_INTERMEDIATE_B(NAME, DTYPE, MODE) \
   SOLID_REDUCE_AXIS_INTERMEDIATE_C(SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, MODE))
#define SOLID_REDUCE_AXIS_INTERMEDIATE(NAME, DTYPE, MODE) \
   SOLID_REDUCE_AXIS_INTERMEDIATE_B(NAME, DTYPE, MODE) \



/* --------------------------------------------------------------------- */
/* Macro: SOLID_CREATE_REDUCE_AXIS_THREAD                                */
/* --------------------------------------------------------------------- */

#define SOLID_CREATE_REDUCE_AXIS_THREAD(PREFIX, NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, \
                                        CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   __global__ void SOLID_REDUCE_AXIS_FUNCTION_ITF(PREFIX, NAME, DTYPE1, thread, FLAG_PARAM) \
   {  SOLID_C_TYPE(DTYPE1) *_ptr; \
      SOLID_C_TYPE(DTYPE1) *_ptr1; \
      SOLID_C_TYPE(DTYPE2) *_ptr2; \
      SOLID_C_WORKTYPE_TYPE(DTYPE2) _accumulate; \
      ptrdiff_t _offset1; \
      ptrdiff_t _offset2; \
      size_t    _index; \
      size_t    _rindex; \
      size_t    _idx; \
      size_t    _s; \
      int       _i; \
      \
      for (_index = blockIdx.x * blockDim.x + threadIdx.x; \
           _index < data.nelem; \
           _index += gridDim.x * blockDim.x) \
      {  \
         /* ---------------------------------------------------- */ \
         /* Possible optimization: special kernels for one, two, */ \
         /* and higher-dimensional reductions.                   */ \
         /* ---------------------------------------------------- */ \
         \
         /* Determine the offsets - generic case */ \
         _idx = _index; \
         _offset1 = 0; \
         _offset2 = 0; \
         for (_i = 0; _i < data.ndims-1; _i++) \
         {  _s        = _idx % data.size[_i]; \
            _idx      = _idx / data.size[_i]; \
            _offset1 += _s * data.strides1[_i]; \
            _offset2 += _s * data.strides2[_i]; \
         } \
         _offset1 += _idx * data.strides1[data.ndims-1]; \
         _offset2 += _idx * data.strides2[data.ndims-1]; \
         \
         /* Determine the pointers */ \
         _ptr1 = (SOLID_C_TYPE(DTYPE1) *)(data.ptr1 + _offset1); \
         _ptr2 = (SOLID_C_TYPE(DTYPE2) *)(data.ptr2 + _offset2); \
         _ptr = _ptr1; \
         \
         /* Templated initialization of _accumulate */ \
         CODE_INIT; \
         \
         /* Accumulation loop */ \
         if (data.relem > 0) \
         {  _rindex = 0; \
            while(1) \
            {  for (_s = 0; _s < data.rsize[0]; _s++) \
               {  /* Templated accumulation of *_ptr into _accumulate */ \
                  CODE_ACCUMULATE; \
                  \
                  /* Update the pointer */ \
                  _ptr = (SOLID_C_TYPE(DTYPE1) *)((char *)(_ptr) + data.rstrides[0]); \
               } \
               \
               /* Move to the next element */ \
               _rindex ++; if (_rindex >= data.relem) break; \
               \
               /* Set the pointer - generic case */ \
               if (data.rdims > 1) \
               {  _idx = _rindex; _offset1 = 0; /* Offset wrt _ptr1 */ \
                  for (_i = 1; _i < data.rdims-1; _i++) \
                  {  _s        = _idx % data.rsize[_i]; \
                     _idx      = _idx / data.rsize[_i]; \
                     _offset1 += _s * data.rstrides[_i]; \
                  } \
                  _offset1 += _idx * data.rstrides[data.rdims-1]; \
                  _ptr = (SOLID_C_TYPE(DTYPE1) *)((char *)(_ptr1) + _offset1); \
               } \
            } \
         } \
         \
         /* Finalize based on _accumulate and write _ptr2 */ \
         SOLID_REDUCE_AXIS_THREAD_FINALIZE(DTYPE2, FLAG_FINALIZE, CODE_FINALIZE) \
      } \
   }


/* Finalization for thread-based axis reduction */
#define SOLID_REDUCE_AXIS_THREAD_FINALIZE_B0(DTYPE2, CODE_FINALIZE) \
   SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), DTYPE2, &_accumulate, _ptr2);

#define SOLID_REDUCE_AXIS_THREAD_FINALIZE_B1(DTYPE2, CODE_FINALIZE) \
   {  SOLID_C_WORKTYPE_TYPE(DTYPE2) *_partial; \
      SOLID_C_WORKTYPE_TYPE(DTYPE2) *_result; \
      SOLID_C_WORKTYPE_TYPE(DTYPE2)  _finalized; \
      _partial = &_accumulate; \
      _result  = &_finalized; \
      CODE_FINALIZE \
      SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), DTYPE2, _result, _ptr2); \
   }

#define SOLID_REDUCE_AXIS_THREAD_FINALIZE(DTYPE2, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_REDUCE_AXIS_THREAD_FINALIZE_B##FLAG_FINALIZE(DTYPE2, CODE_FINALIZE)



/* --------------------------------------------------------------------- */
/* Macro: SOLID_CREATE_REDUCE_AXIS_WARP                                  */
/* --------------------------------------------------------------------- */

#define SOLID_CREATE_REDUCE_AXIS_WARP(PREFIX, NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, \
                                      CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   __global__ void SOLID_REDUCE_AXIS_FUNCTION_ITF(PREFIX, NAME, DTYPE1, warp, FLAG_PARAM) \
   {  SOLID_C_TYPE(DTYPE1) *_ptr; \
      SOLID_C_TYPE(DTYPE1) *_ptr1; \
      SOLID_C_TYPE(DTYPE2) *_ptr2; \
      volatile extern __shared__ SOLID_C_WORKTYPE_TYPE(DTYPE2) SOLID_REDUCE_AXIS_INTERMEDIATE(NAME, SOLID_WORKTYPE(DTYPE2), warp)[]; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_partial; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_result; \
      SOLID_C_WORKTYPE_TYPE(DTYPE2) _accumulate; \
      ptrdiff_t _offset1; \
      ptrdiff_t _offset2; \
      size_t    _index; \
      size_t    _rindex; \
      size_t    _idx; \
      size_t    _s; \
      int       _i; \
      \
      /* ----------------------------------------------------- */ \
      /* NOTE: This kernel assumes that the number of elements */ \
      /* in each reduction is at least 32. Launch parameter    */ \
      /* threadIdx.y is the warp index, threadIdx.x is the     */ \
      /* index within the warp [0..31].                        */ \
      /* ----------------------------------------------------- */ \
      for (_index = blockIdx.x * blockDim.y + threadIdx.y; \
           _index < data.nelem; \
           _index += gridDim.x * blockDim.y) \
      {  \
         /* ---------------------------------------------------- */ \
         /* Possible optimization: special kernels for one, two, */ \
         /* and higher-dimensional reductions.                   */ \
         /* ---------------------------------------------------- */ \
         \
         /* Determine the offsets - generic case */ \
         _idx = _index; \
         _offset1 = 0; \
         _offset2 = 0; \
         for (_i = 0; _i < data.ndims-1; _i++) \
         {  _s        = _idx % data.size[_i]; \
            _idx      = _idx / data.size[_i]; \
            _offset1 += _s * data.strides1[_i]; \
            _offset2 += _s * data.strides2[_i]; \
         } \
         _offset1 += _idx * data.strides1[data.ndims-1]; \
         _offset2 += _idx * data.strides2[data.ndims-1]; \
         \
         /* Determine the pointers */ \
         _ptr1 = (SOLID_C_TYPE(DTYPE1) *)(data.ptr1 + _offset1); \
         _ptr2 = (SOLID_C_TYPE(DTYPE2) *)(data.ptr2 + _offset2); \
         \
         /* Accumulation loop */ \
         for (_rindex = threadIdx.x; _rindex < data.relem; _rindex +=32) \
         {  /* Set the pointer - generic case */ \
            if (data.rdims == 1) \
            {  _ptr = (SOLID_C_TYPE(DTYPE1) *)((char *)(_ptr1) + _rindex * data.rstrides[0]); \
            } \
            else \
            {  _idx = _rindex; _offset1 = 0; /* Offset wrt _ptr1 */ \
               for (_i = 0; _i < data.rdims-1; _i++) \
               {  _s        = _idx % data.rsize[_i]; \
                  _idx      = _idx / data.rsize[_i]; \
                  _offset1 += _s * data.rstrides[_i]; \
               } \
               _offset1 += _idx * data.rstrides[data.rdims-1]; \
               _ptr = (SOLID_C_TYPE(DTYPE1) *)((char *)(_ptr1) + _offset1); \
            } \
            \
            if (_rindex < 32) \
            {  /* Templated initialization of _accumulate */ \
               CODE_INIT; \
            } \
            /* Templated accumulation of *_ptr into _accumulate */ \
            CODE_ACCUMULATE; \
         } \
         \
         /* Assign the intermediate value */ \
         _result = SOLID_REDUCE_AXIS_INTERMEDIATE(NAME, SOLID_WORKTYPE(DTYPE2), warp); \
         _result += threadIdx.x + 32 * threadIdx.y; \
         SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), SOLID_WORKTYPE(DTYPE2), &_accumulate, _result); \
         \
         /* Final warp-level reduction */ \
         if (threadIdx.x < 16) \
         {  _partial = _result + 16; CODE_REDUCE; \
            _partial = _result + 8;  CODE_REDUCE; \
            _partial = _result + 4;  CODE_REDUCE; \
            _partial = _result + 2;  CODE_REDUCE; \
            _partial = _result + 1;  CODE_REDUCE; \
         } \
         /* Finalize the value and set the result */ \
         if (threadIdx.x == 0) \
         {  SOLID_REDUCE_AXIS_WARP_FINALIZE(DTYPE2, FLAG_FINALIZE, CODE_FINALIZE) \
         } \
      } \
   }

/* Finalization for warp-based axis reduction */
#define SOLID_REDUCE_AXIS_WARP_FINALIZE_B0(DTYPE2, CODE_FINALIZE) \
   SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), DTYPE2, _result, _ptr2);

#define SOLID_REDUCE_AXIS_WARP_FINALIZE_B1(DTYPE2, CODE_FINALIZE) \
   {  _partial = _result; \
      CODE_FINALIZE \
      SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), DTYPE2, _result, _ptr2); \
   }

#define SOLID_REDUCE_AXIS_WARP_FINALIZE(DTYPE2, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_REDUCE_AXIS_WARP_FINALIZE_B##FLAG_FINALIZE(DTYPE2, CODE_FINALIZE)



/* --------------------------------------------------------------------- */
/* Macro: SOLID_CREATE_REDUCE_AXIS_BLOCK                                 */
/* --------------------------------------------------------------------- */

#define SOLID_CREATE_REDUCE_AXIS_BLOCK(PREFIX, NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, \
                                       CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   __global__ void SOLID_REDUCE_AXIS_FUNCTION_ITF(PREFIX, NAME, DTYPE1, block, FLAG_PARAM) \
   {  SOLID_C_TYPE(DTYPE1) *_ptr; \
      SOLID_C_TYPE(DTYPE1) *_ptr1; \
      SOLID_C_WORKTYPE_TYPE(DTYPE2) *_ptr2; \
      volatile extern __shared__ SOLID_C_WORKTYPE_TYPE(DTYPE2) SOLID_REDUCE_AXIS_INTERMEDIATE(NAME, SOLID_WORKTYPE(DTYPE2), block)[]; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_partial; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_result; \
      SOLID_C_WORKTYPE_TYPE(DTYPE2) _accumulate; \
      ptrdiff_t _offset1; \
      ptrdiff_t _offset2; \
      size_t    _index; \
      size_t    _rindex, _rstep, _rstart; \
      size_t    _idx; \
      size_t    _s; \
      int       _i; \
      \
      /* ------------------------------------------------------- */ \
      /* NOTE:  Launch parameter threadIdx.y is the warp index,  */ \
      /* threadIdx.x is the index within the warp [0..31].       */ \
      /* Parameter blockIdx.y gives the number of blocks working */ \
      /* on a single reduction; blockIdx.x indicates different   */ \
      /* reductions. This kernel assumes that there are at least */ \
      /* 32 * blockDim.y * gridDim.y elements per reduction.     */ \
      /* ------------------------------------------------------- */ \
      \
      /* Initialize */ \
      _result = SOLID_REDUCE_AXIS_INTERMEDIATE(NAME, SOLID_WORKTYPE(DTYPE2), block); \
      _result += threadIdx.x + 32 * threadIdx.y; \
      _rstart = threadIdx.x + 32 * (threadIdx.y + blockDim.y * blockIdx.y); \
      _rstep = 32 * blockDim.y * gridDim.y; \
      \
      for (_index = blockIdx.x; \
           _index < data.nelem; \
           _index += gridDim.x) \
      {  \
         /* ---------------------------------------------------- */ \
         /* Possible optimization: special kernels for one, two, */ \
         /* and higher-dimensional reductions.                   */ \
         /* ---------------------------------------------------- */ \
         \
         /* Determine the offsets of the input and output elements */ \
         _idx = _index; _offset1 = 0; \
         for (_i = 0; _i < data.ndims-1; _i++) \
         {  _s        = _idx % data.size[_i]; \
            _idx      = _idx / data.size[_i]; \
            _offset1 += _s * data.strides1[_i]; \
         } \
         _offset1 += _idx * data.strides1[data.ndims-1]; \
         _offset2 = threadIdx.y + blockDim.y * (blockIdx.y + gridDim.y * blockIdx.x); \
         \
         /* Determine the pointers */ \
         _ptr1 = (SOLID_C_TYPE(DTYPE1) *)(data.ptr1 + _offset1); \
         _ptr2 = (SOLID_C_WORKTYPE_TYPE(DTYPE2) *)(data.ptrBuffer + _offset2 * sizeof(SOLID_C_WORKTYPE_TYPE(DTYPE2))); \
         \
         /* Accumulation loop */ \
         for (_rindex = _rstart; _rindex < data.relem; _rindex += _rstep) \
         {  /* Set the pointer - generic case */ \
            if (data.rdims == 1) \
            {  _ptr = (SOLID_C_TYPE(DTYPE1) *)((char *)(_ptr1) + _rindex * data.rstrides[0]); \
            } \
            else \
            {  _idx = _rindex; _offset1 = 0; /* Offset wrt _ptr1 */ \
               for (_i = 0; _i < data.rdims-1; _i++) \
               {  _s        = _idx % data.rsize[_i]; \
                  _idx      = _idx / data.rsize[_i]; \
                  _offset1 += _s * data.rstrides[_i]; \
               } \
               _offset1 += _idx * data.rstrides[data.rdims-1]; \
               _ptr = (SOLID_C_TYPE(DTYPE1) *)((char *)(_ptr1) + _offset1); \
            } \
            \
            if (_rindex == _rstart) \
            {  /* Templated initialization of _accumulate */ \
               CODE_INIT; \
            } \
            /* Templated accumulation of *_ptr into _accumulate */ \
            CODE_ACCUMULATE; \
         } \
         \
         /* Assign the intermediate value */ \
         SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), SOLID_WORKTYPE(DTYPE2), &_accumulate, _result); \
         \
         /* Warp-level reduction */ \
         if (threadIdx.x < 16) \
         {  _partial = _result + 16; CODE_REDUCE; \
            _partial = _result + 8;  CODE_REDUCE; \
            _partial = _result + 4;  CODE_REDUCE; \
            _partial = _result + 2;  CODE_REDUCE; \
            _partial = _result + 1;  CODE_REDUCE; \
         } \
         \
         /* Write the warp-level final value */ \
         if (threadIdx.x == 0) \
         {  SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), SOLID_WORKTYPE(DTYPE2), _result, _ptr2); \
         } \
      } \
   }


/* Reduction and finalization of block-based axis reduction */
#define SOLID_CREATE_REDUCE_AXIS_BLOCK_FINALIZE(PREFIX, NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, \
                                                CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   __global__ void SOLID_REDUCE_AXIS_FUNCTION_ITF(PREFIX, NAME, DTYPE1, block_finalize, FLAG_PARAM) \
   {  volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_buffer; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_partial; \
      volatile SOLID_C_WORKTYPE_TYPE(DTYPE2) *_result; \
      SOLID_C_TYPE(DTYPE2) *_ptr2; \
      ptrdiff_t _offset2; \
      size_t    _index, _idx; \
      int       _s, _b, _i, _tid; \
      \
      /* Loop over the reductions */ \
      for (_index = blockIdx.x; _index < data.nelem; _index += gridDim.x) \
      { \
         /* Get a pointer to the input buffer (pointer addition is in worktype elements) */ \
         _buffer = ((SOLID_C_WORKTYPE_TYPE(DTYPE2) *)(data.ptrBuffer)) + _index * data.relem; \
         \
         /* Thread-level reduction - overwrite existing buffer */ \
         _result = _buffer + threadIdx.x; \
         for (_idx = threadIdx.x + blockDim.x; _idx < data.relem; _idx += blockDim.x) \
         {  _partial = _buffer + _idx; \
            CODE_REDUCE \
         } \
         \
         /* Block-level reduction - overwrite existing buffer */ \
         _tid = threadIdx.x; \
         _s = (blockDim.x + 1) / 2; /* Elements remaining after combination */ \
         _b = (blockDim.x) / 2;     /* Number of elements to update */ \
         for ( ; _b > 0; _b = _s / 2, _s = (_s+1) / 2) \
         {  if (_s > 16) __syncthreads(); \
            if (_tid < _b) \
            {  _partial = _result + _s; /* Thread index is included in _result pointer */ \
               CODE_REDUCE /* Reduce _partial into _result */ \
            } \
         } \
         \
         /* Convert the result from the work type to the desired type */ \
         if (_tid == 0) \
         {  /* Apply finalization if needed (the type must remain the same) */ \
            SOLID_REDUCE_AXIS_FINALIZE(DTYPE2, FLAG_FINALIZE, CODE_FINALIZE) \
            \
            /* Determine the output offset */ \
            _idx = _index; _offset2 = 0; \
            for (_i = 0; _i < data.ndims-1; _i++) \
            {  _s        = _idx % data.size[_i]; \
               _idx      = _idx / data.size[_i]; \
               _offset2 += _s * data.strides2[_i]; \
            } \
            _offset2 += _idx * data.strides2[data.ndims-1]; \
            \
            /* Determine the pointer */ \
            _ptr2 = (SOLID_C_TYPE(DTYPE2) *)(data.ptr2 + _offset2); \
            \
            /* Convert the worktype value in _result[0] to the */ \
            /* final value and output to _ptr2.                */ \
            SOLID_ASSIGN(SOLID_WORKTYPE(DTYPE2), DTYPE2, _result, _ptr2); \
         } \
      } \
   }


/* Finalization code */
#define SOLID_REDUCE_AXIS_FINALIZE_B0(RTYPE, CODE_FINALIZE) /* Empty */
#define SOLID_REDUCE_AXIS_FINALIZE_B1(RTYPE, CODE_FINALIZE) \
   {  SOLID_C_WORKTYPE_TYPE(RTYPE) _temp = _buffer[0]; \
      _partial = &_temp; \
      CODE_FINALIZE; \
   }
#define SOLID_REDUCE_AXIS_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_REDUCE_AXIS_FINALIZE_B##FLAG_FINALIZE(RTYPE, CODE_FINALIZE)


/* --------------------------------------------------------------------- */
/* Macro: SOLID_CREATE_REDUCE_AXIS_LAUNCH                                */
/* --------------------------------------------------------------------- */

/* Launch function interface */
#define SOLID_REDUCE_AXIS_LAUNCH_ITF_D0(FUNCTION, PARAM_TYPE) \
   FUNCTION(solid_gpu_reduce_axis_config *config)
#define SOLID_REDUCE_AXIS_LAUNCH_ITF_D1(FUNCTION, PARAM_TYPE) \
   FUNCTION(solid_gpu_reduce_axis_config *config, PARAM_TYPE *param)

#define SOLID_REDUCE_AXIS_LAUNCH_ITF_C(FUNCTION, PARAM_TYPE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_LAUNCH_ITF_D##FLAG_PARAM(FUNCTION, PARAM_TYPE)
#define SOLID_REDUCE_AXIS_LAUNCH_ITF_B(PREFIX, NAME, DTYPE, MODE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_LAUNCH_ITF_C(SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, MODE), SOLID_KERNEL_PARAM_PREFIX(PREFIX), FLAG_PARAM)
#define SOLID_REDUCE_AXIS_LAUNCH_ITF(PREFIX, NAME, DTYPE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_LAUNCH_ITF_B(PREFIX, NAME, DTYPE, launch, FLAG_PARAM)


/* Launching the kernel */
#define SOLID_LAUNCH_KERNEL_REDUCE_AXIS_C0(FUNCTION, GRIDSIZE, BLOCKSIZE, SHAREDMEM, STREAM, DATA) \
   FUNCTION<<<GRIDSIZE,BLOCKSIZE,SHAREDMEM,STREAM>>>(DATA)
#define SOLID_LAUNCH_KERNEL_REDUCE_AXIS_C1(FUNCTION, GRIDSIZE, BLOCKSIZE, SHAREDMEM, STREAM, DATA) \
   FUNCTION<<<GRIDSIZE,BLOCKSIZE,SHAREDMEM,STREAM>>>(DATA, *param)
#define SOLID_LAUNCH_KERNEL_REDUCE_AXIS_B(FUNCTION, GRIDSIZE, BLOCKSIZE, SHAREDMEM, STREAM, DATA, FLAG_PARAM) \
   SOLID_LAUNCH_KERNEL_REDUCE_AXIS_C##FLAG_PARAM(FUNCTION, GRIDSIZE, BLOCKSIZE, SHAREDMEM, STREAM, DATA)
#define SOLID_LAUNCH_KERNEL_REDUCE_AXIS(NAME, DTYPE, MODE, GRIDSIZE, BLOCKSIZE, SHAREDMEM, STREAM, DATA, FLAG_PARAM) \
   SOLID_LAUNCH_KERNEL_REDUCE_AXIS_B(SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, MODE), GRIDSIZE, BLOCKSIZE, SHAREDMEM, STREAM, DATA, FLAG_PARAM)


/* Launch function definition */
#define SOLID_CREATE_REDUCE_AXIS_LAUNCH(PREFIX, NAME, DTYPE, FLAG_PARAM) \
   /* ------------------------------------------------------------------ */ \
   int SOLID_REDUCE_AXIS_LAUNCH_ITF(PREFIX, NAME, DTYPE, FLAG_PARAM) \
   /* ------------------------------------------------------------------ */ \
   {  cudaError_t status = cudaSuccess; \
      \
      /* Deal with empty output tensors */ \
      if (config->data.nelem == 0) return 0; \
      \
      /* Launch the kernel */ \
      switch (config->mode) \
      {  case 1 : SOLID_LAUNCH_KERNEL_REDUCE_AXIS(NAME, DTYPE, thread, config->gridSize, config->blockSize, \
                                                  config->sharedMem, config->stream, config->data, FLAG_PARAM); \
                  status = cudaGetLastError(); \
                  break; \
         case 2 : SOLID_LAUNCH_KERNEL_REDUCE_AXIS(NAME, DTYPE, warp, config->gridSize, config->blockSize, \
                                                  config->sharedMem, config->stream, config->data, FLAG_PARAM); \
                  status = cudaGetLastError(); \
                  break; \
         case 3 : SOLID_LAUNCH_KERNEL_REDUCE_AXIS(NAME, DTYPE, block, config->gridSize, config->blockSize, \
                                                   config->sharedMem, config->stream, config->data, FLAG_PARAM); \
                  if ((status = cudaGetLastError()) != cudaSuccess) break; \
                  \
                  /* Update the number of reduction elements */ \
                  config -> data.relem = config -> gridSize.y * config -> blockSize.y; \
                  SOLID_LAUNCH_KERNEL_REDUCE_AXIS(NAME, DTYPE, block_finalize, config->gridSizeFinalize, \
                                                  config->blockSizeFinalize, config->sharedMemFinalize, \
                                                  config->stream, config->data, FLAG_PARAM); \
                  status = cudaGetLastError(); \
                  break; \
      } \
      if (status != cudaSuccess) \
         SOLID_ERROR(-1, "Error launching reduction kernel: %s", cudaGetErrorString(status)); \
      return 0; \
   }


/* ------------------------------------------------------------------------ */
/* CALLING THE LAUNCH FUNCTION                                              */
/* ------------------------------------------------------------------------ */

#define SOLID_CALL_REDUCE_AXIS_TYPE_D0(FUNCTION, CONFIG, PARAM) \
   FUNCTION(CONFIG)
#define SOLID_CALL_REDUCE_AXIS_TYPE_D1(FUNCTION, CONFIG, PARAM) \
   FUNCTION(CONFIG, PARAM)

#define SOLID_CALL_REDUCE_AXIS_FULL_C(FUNCTION, CONFIG, PARAM, FLAG_PARAM) \
   SOLID_CALL_REDUCE_AXIS_TYPE_D##FLAG_PARAM(FUNCTION, CONFIG, PARAM)
#define SOLID_CALL_REDUCE_AXIS_FULL_B(NAME, DTYPE, CONFIG, PARAM, FLAG_PARAM) \
   SOLID_CALL_REDUCE_AXIS_FULL_C(SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, launch), CONFIG, PARAM, FLAG_PARAM)
#define SOLID_CALL_REDUCE_AXIS_FULL(NAME, DTYPE, CONFIG, PARAM, FLAG_PARAM) \
   SOLID_CALL_REDUCE_AXIS_FULL_B(NAME, DTYPE, CONFIG, PARAM, FLAG_PARAM)

#define SOLID_CALL_REDUCE_AXIS_PARAM(NAME, PARAM) \
   SOLID_CALL_REDUCE_AXIS_FULL(NAME, SDXTYPE, config, PARAM, 1)

#define SOLID_CALL_REDUCE_AXIS(NAME) \
   SOLID_CALL_REDUCE_AXIS_FULL(NAME, SDXTYPE, config, {}, 0)

#endif
