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

#ifndef __SOLID_APPLY_ELEMWISE1_CPU_H__
#define __SOLID_APPLY_ELEMWISE1_CPU_H__

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/base/cpu/generate_macros.h"
#include "solid/base/cpu/solid_omp.h"


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE1                                          */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE1(CODE) \
   SOLID_APPLY_ELEMWISE1_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, strides, ptr, -1)

#if SOLID_ENABLE_OMP
#define SOLID_APPLY_ELEMWISE1_CTYPE(TYPE, CODE, NDIMS, SIZE, STRIDES, PTR, NTHREADS) \
   {  int __solid_omp_threads = NTHREADS; \
      if (__solid_omp_threads < 0) __solid_omp_threads = solid_omp_get_max_threads(); \
      \
      if (__solid_omp_threads > 1) \
      {   \
         _Pragma("omp parallel num_threads(__solid_omp_threads)") \
         SOLID_APPLY_ELEMWISE1_OMP_CTYPE(TYPE, CODE, NDIMS, SIZE, STRIDES, PTR) \
      } \
      else \
      {  SOLID_APPLY_ELEMWISE1_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, STRIDES, PTR) \
      } \
   }
#else
#define SOLID_APPLY_ELEMWISE1_CTYPE(TYPE, CODE, NDIMS, SIZE, STRIDES, PTR, NTHREADS) \
   {  SOLID_APPLY_ELEMWISE1_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, STRIDES, PTR) \
   }
#endif


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE1_CPU                                      */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE1_CPU(CODE) \
   SOLID_APPLY_ELEMWISE1_CPU_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, strides, ptr)

#define SOLID_APPLY_ELEMWISE1_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, STRIDES, PTR) \
   {  size_t    _idx[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _index, _indexStop; \
      size_t    _nelem, _s; \
      TYPE     *_ptr = (TYPE *)(PTR); \
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
      _index = 0;  \
      if (_index < _nelem) \
      {  while (1) \
         {  _indexStop = _index + _s; \
            while (_index < _indexStop) \
            {  CODE \
               _index ++; \
               _ptr = (TYPE *)(((char *)_ptr) + _strides[0]); \
            } \
            if (_index >= _nelem) break; \
            \
            _i = 1; \
            _ptr = (TYPE *)(((char *)_ptr) + _strides[_i]); \
            while (++_idx[_i] == SIZE[_i]) \
            {  _idx[_i] = 0; _i++; \
               _ptr = (TYPE *)(((char *)_ptr) + _strides[_i]); \
            } \
         } \
      } \
   }


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE1_OMP                                      */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE1_OMP(CODE) \
   SOLID_APPLY_ELEMWISE1_OMP_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, strides, ptr)
           
#define SOLID_APPLY_ELEMWISE1_OMP_CTYPE(TYPE, CODE, NDIMS, SIZE, STRIDES, PTR) \
   {  size_t    _idx[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _index, _indexStop, _indexMax; \
      size_t    _offset; \
      TYPE     *_ptr = (TYPE *)(PTR); \
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
      _index  = _offset; \
      _indexMax = _offset + _nelem; \
      \
      /* Initialize the data pointer and indices */ \
      if ((NDIMS > 0) && (_nelem > 0)) \
      {  _s = SIZE[0]; \
         for (_i = 0; _i < NDIMS; _i++) \
         {  _idx[_i] = _offset % SIZE[_i]; \
            _offset /= SIZE[_i]; \
            _ptr = (TYPE *)(((char *)_ptr) + _strides[_i] * _idx[_i]); \
         } \
      } \
      else \
      {  /* We also just set these values when the number of elements   */ \
         /* is zero. This will give an incorrect _indexStop value below */ \
         /* but since _indexMax will be equal to _index this value is   */ \
         /* then never used.                                            */ \
         _s = 1; \
         _idx[0] = _offset; \
      } \
      \
      /* Update the strides */ \
      for (_i = NDIMS-1; _i > 0; _i--) \
      {  _strides[_i] -= SIZE[_i-1] * _strides[_i-1]; \
      } \
      \
      /* Loop over the data */ \
      _indexStop = _index + _s - _idx[0]; \
      if (_index < _indexMax) \
      {  while (1) \
         { \
            if (_indexStop > _indexMax) _indexStop = _indexMax; \
            while (_index < _indexStop) \
            {  CODE \
               _index ++; \
               _ptr = (TYPE *)(((char *)_ptr) + _strides[0]); \
            } \
            if (_index >= _indexMax) break; \
            \
            _i = 1; _indexStop = _index + _s; \
            _ptr = (TYPE *)(((char *)_ptr) + _strides[_i]); \
            while (++_idx[_i] == SIZE[_i]) \
            {  _idx[_i] = 0; _i++; \
               _ptr = (TYPE *)(((char *)_ptr) + _strides[_i]); \
            } \
         } \
      } \
   }

#endif
