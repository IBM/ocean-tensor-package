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

#ifndef __SOLID_REDUCE_ALL_CPU_H__
#define __SOLID_REDUCE_ALL_CPU_H__

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/base/cpu/generate_macros.h"
#include "solid/base/cpu/solid_omp.h"

#include <stdlib.h>

/* ----------------------------------------------- */
/* TYPE     Input data type                        */
/* RTYPE    Accumulate/result data type            */
/* ----------------------------------------------- */

/* --------------------------------------------------------------------- */
/* Macro: SOLID_REDUCE_ALL                                               */
/* --------------------------------------------------------------------- */

/* Finalization */
#define SOLID_REDUCE_ALL_FINALIZE_C0(RTYPE, CODE_FINALIZE, VAR) /* Empty */
#define SOLID_REDUCE_ALL_FINALIZE_C1(RTYPE, CODE_FINALIZE, VAR) \
   {  SOLID_C_WORKTYPE_TYPE(RTYPE)  _temp    = VAR; \
      SOLID_C_WORKTYPE_TYPE(RTYPE) *_partial = &_temp; \
      SOLID_C_WORKTYPE_TYPE(RTYPE) *_result  = &(VAR); \
      CODE_FINALIZE \
   }
#define SOLID_REDUCE_ALL_FINALIZE_B(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, VAR) \
   SOLID_REDUCE_ALL_FINALIZE_C##FLAG_FINALIZE(RTYPE, CODE_FINALIZE, VAR)
#define SOLID_REDUCE_ALL_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, VAR) \
   SOLID_REDUCE_ALL_FINALIZE_B(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, VAR)


/* Main code */
#define SOLID_REDUCE_ALL(CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_REDUCE_ALL_TYPE(SDXTYPE, SDXTYPE, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE)

#define SOLID_REDUCE_ALL_TYPE(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_REDUCE_ALL_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, \
                         FLAG_FINALIZE, CODE_FINALIZE, ndims, size, strides, ptr, result, -1)

#if SOLID_ENABLE_OMP
#define SOLID_REDUCE_ALL_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, \
                              FLAG_FINALIZE, CODE_FINALIZE, NDIMS, SIZE, STRIDES, PTR, RESULTPTR, NTHREADS) \
   {  size_t __nelem = 1; \
      int    __k; \
      int __solid_omp_threads = NTHREADS; \
      if (__solid_omp_threads < 0) __solid_omp_threads = solid_omp_get_max_threads(); \
      \
      /* Determine maximum number of threads based on element count */ \
      for (__k = 0; __k < NDIMS; __k++) __nelem *= SIZE[__k]; \
      if (__nelem < __solid_omp_threads * 512) __solid_omp_threads = __nelem / 512; \
      \
      /* Run the code with or without threads */ \
      if (__solid_omp_threads > 1) \
      {  SOLID_C_WORKTYPE_TYPE(RTYPE) _intermediate[SOLID_OMP_MAX_THREADS], *_result, *_partial; \
         int _j; \
         \
         _Pragma("omp parallel num_threads(__solid_omp_threads)") \
         SOLID_REDUCE_ALL_OMP_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, NDIMS, SIZE, STRIDES, PTR, (_intermediate + _rank)) \
         \
         /* Combine the intermediate results */ \
         _result = &(_intermediate[0]); \
         for (_j = 1; _j < __solid_omp_threads; _j++) \
         {  _partial = &(_intermediate[_j]); \
            CODE_REDUCE \
         } \
         \
         /* Finalize the result */ \
         SOLID_REDUCE_ALL_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, _intermediate[0]) \
         \
         /* Assign the final result */ \
         SOLID_ASSIGN(SOLID_WORKTYPE(RTYPE), RTYPE, _result, RESULTPTR) \
      } \
      else \
      {  SOLID_REDUCE_ALL_CPU_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, FLAG_FINALIZE, CODE_FINALIZE, NDIMS, SIZE, STRIDES, PTR, RESULTPTR) \
      } \
   }
#else
#define SOLID_REDUCE_ALL_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, \
                              FLAG_FINALIZE, CODE_FINALIZE, NDIMS, SIZE, STRIDES, PTR, RESULTPTR, NTHREADS) \
   {  SOLID_REDUCE_ALL_CPU_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, FLAG_FINALIZE, CODE_FINALIZE, NDIMS, SIZE, STRIDES, PTR, RESULTPTR) \
   }
#endif


/* --------------------------------------------------------------------- */
/* Macro: SOLID_REDUCE_ALL_CPU                                           */
/* --------------------------------------------------------------------- */

/* Main code */
#define SOLID_REDUCE_ALL_CPU(CODE_INIT, CODE_ACCUMULATE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_REDUCE_ALL_CPU_TYPE(SDXTYPE, SDXTYPE, CODE_INIT, CODE_ACCUMULATE, FLAG_FINALIZE, CODE_FINALIZE)

#define SOLID_REDUCE_ALL_CPU_TYPE(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, FLAG_FINALIZE, CODE_FINALIZE) \
   SOLID_REDUCE_ALL_CPU_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, FLAG_FINALIZE, CODE_FINALIZE, ndims, size, strides, ptr, result)

#define SOLID_REDUCE_ALL_CPU_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, FLAG_FINALIZE, CODE_FINALIZE, NDIMS, SIZE, STRIDES, PTR, RESULTPTR) \
   {  SOLID_C_TYPE(TYPE)            *_ptr = (SOLID_C_TYPE(TYPE) *)(PTR); \
      SOLID_C_WORKTYPE_TYPE(RTYPE)   _accumulate; \
      size_t    _idx[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _index, _indexStop, _indexMax; \
      size_t    _nelem, _s; \
      int       _i; \
      \
      /* Initialize the indices */ \
      _nelem = 1; \
      if (NDIMS > 0) \
      {  _s = _nelem = SIZE[0]; \
         _idx[0] = 0; \
         _strides[0] = STRIDES[0]; \
         for (_i = NDIMS-1; _i > 0; _i --) \
         {  _nelem *= SIZE[_i]; \
            _idx[_i] = 0; \
            _strides[_i] = STRIDES[_i] - SIZE[_i-1] * STRIDES[_i-1]; \
         } \
      } \
      else \
      {  _s = 1; \
         _strides[0] = 0; \
      } \
      \
      /* Initialize the accumulation (can use _ptr, but only when _nelem > 0)) */ \
      CODE_INIT \
      \
      _index = 0; \
      _indexMax = _nelem; \
      if (_index < _indexMax) \
      {  while (1) \
         {  _indexStop = _index + _s; \
            while (_index < _indexStop) \
            {  CODE_ACCUMULATE \
               _index ++; \
               _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + _strides[0]); \
            } \
            if (_index >= _indexMax) break; \
            \
            _i = 1; \
            _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + _strides[_i]); \
            while (++_idx[_i] == SIZE[_i]) \
            {  _idx[_i] = 0; _i++; \
               _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + _strides[_i]); \
            } \
         } \
      } \
      \
      /* Finalize the result */ \
      SOLID_REDUCE_ALL_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, _accumulate) \
      \
      /* Assign the final result */ \
      SOLID_ASSIGN(SOLID_WORKTYPE(RTYPE), RTYPE, &_accumulate, RESULTPTR) \
   }


/* --------------------------------------------------------------------- */
/* Macro: SOLID_REDUCE_ALL_OMP                                           */
/* --------------------------------------------------------------------- */

/* Main code */
#define SOLID_REDUCE_ALL_OMP(CODE_INIT, CODE_ACCUMULATE) \
   SOLID_REDUCE_ALL_OMP_TYPE(SDXTYPE, SDXTYPE, CODE_INIT, CODE_ACCUMULATE)
           
#define SOLID_REDUCE_ALL_OMP_TYPE(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE) \
   SOLID_REDUCE_ALL_OMP_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, ndims, size, strides, ptr)

#define SOLID_REDUCE_ALL_OMP_FULL(TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, NDIMS, SIZE, STRIDES, PTR, RESULTPTR) \
   {  SOLID_C_TYPE(TYPE)           *_ptr = (SOLID_C_TYPE(TYPE) *)(PTR); \
      SOLID_C_WORKTYPE_TYPE(RTYPE)  _accumulate; \
      size_t    _idx[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _index, _indexStop, _indexMax; \
      size_t    _offset; \
      size_t    _nelem, _s; \
      int       _rank = omp_get_thread_num(); \
      int       _nthreads = omp_get_num_threads(); \
      int       _i; \
      \
      /* Initialize the size and strides */ \
      _nelem = 1; \
      for (_i = 0; _i < NDIMS; _i++) \
      {  _nelem *= SIZE[_i]; \
         _strides[_i] = STRIDES[_i]; \
      } \
      \
      /* Compute the offset */ \
      _offset = _nelem % _nthreads; _nelem /= _nthreads; \
      if (_rank < _offset) \
      {  _nelem ++; \
         _offset = _rank * _nelem; \
      } \
      else \
      {  _offset += _rank * _nelem; \
      } \
      _index    = _offset; \
      _indexMax = _offset + _nelem; \
      \
      /* Initialize the data pointer and indices */ \
      if (_nelem == 0) \
      {  _s = 0; \
      } \
      else if (NDIMS > 0) \
      {  _s = SIZE[0]; \
         for (_i = 0; _i < NDIMS; _i++) \
         {  _idx[_i] = _offset % SIZE[_i]; \
            _offset /= SIZE[_i]; \
            _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + _strides[_i] * _idx[_i]); \
         } \
      } \
      else \
      {  _s = 1; \
         _idx[0] = _offset; \
      } \
      \
      /* Update the strides */ \
      for (_i = NDIMS-1; _i > 0; _i--) \
      {  _strides[_i] -= SIZE[_i-1] * _strides[_i-1]; \
      } \
      \
      /* Initialize the accumulation (can use _ptr, but only when _nelem > 0) */ \
      CODE_INIT \
      \
      /* Loop over the data */ \
      _indexStop = _index + _s - _idx[0]; \
      if (_index < _indexMax) \
      {  while (1) \
         { \
            if (_indexStop > _indexMax) _indexStop = _indexMax; \
            while (_index < _indexStop) \
            {  CODE_ACCUMULATE \
               _index ++; \
               _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + _strides[0]); \
            } \
            if (_index >= _indexMax) break; \
            \
            _i = 1; _indexStop = _index + _s; \
            _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + _strides[_i]); \
            while (++_idx[_i] == SIZE[_i]) \
            {  _idx[_i] = 0; _i++; \
               _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + _strides[_i]); \
            } \
         } \
      } \
      \
      /* Result pointer must have the work data type */ \
      *((SOLID_C_WORKTYPE_TYPE(RTYPE) *)RESULTPTR) = _accumulate; \
   }

#endif
