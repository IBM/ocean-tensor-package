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

#ifndef __SOLID_APPLY_ELEMWISE2B_CPU_H__
#define __SOLID_APPLY_ELEMWISE2B_CPU_H__

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/base/cpu/generate_macros.h"
#include "solid/base/cpu/solid_omp.h"

/* These kernels can be used for two tensors when the sizes  */
/* match exactly. The strides for each dimension can differ. */
/* The variables exposed by the macros are: ptr1, ptr2       */


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE2B                                         */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE2B(CODE) \
   SOLID_APPLY_ELEMWISE2B_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), CODE, \
                                ndims1, size1, strides1, ptr1, \
                                ndims2, size2, strides2, ptr2, nthreads)

#if SOLID_ENABLE_OMP
#define SOLID_APPLY_ELEMWISE2B_CTYPE(TYPE1, TYPE2, CODE, NDIMS1, SIZE1, STRIDES1, PTR1, \
                                                         NDIMS2, SIZE2, STRIDES2, PTR2, NTHREADS) \
   {  int __solid_omp_threads = NTHREADS; \
      if (__solid_omp_threads < 0) __solid_omp_threads = solid_omp_get_max_threads(); \
      \
      if (__solid_omp_threads > 1) \
      {   \
         _Pragma("omp parallel num_threads(__solid_omp_threads)") \
         SOLID_APPLY_ELEMWISE2B_OMP_CTYPE(TYPE1, TYPE2, CODE, NDIMS1, SIZE1, STRIDES1, PTR1, \
                                                              NDIMS2, SIZE2, STRIDES2, PTR2) \
      } \
      else \
      {  SOLID_APPLY_ELEMWISE2B_CPU_CTYPE(TYPE1, TYPE2, CODE, NDIMS1, SIZE1, STRIDES1, PTR1, \
                                                              NDIMS2, SIZE2, STRIDES2, PTR2) \
      } \
   }
#else
#define SOLID_APPLY_ELEMWISE2B_CTYPE(TYPE1, TYPE2, CODE, NDIMS1, SIZE1, STRIDES1, PTR1, \
                                                         NDIMS2, SIZE2, STRIDES2, PTR2, NTHREADS) \
   {  SOLID_APPLY_ELEMWISE2B_CPU_CTYPE(TYPE1, TYPE2, CODE, NDIMS1, SIZE1, STRIDES1, PTR1, \
                                                           NDIMS2, SIZE2, STRIDES2, PTR2) \
   }
#endif


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE2B_CPU                                     */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE2B_CPU(CODE) \
   SOLID_APPLY_ELEMWISE2B_CPU_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), CODE, \
                                    ndims1, size1, strides1, ptr1, ndims2, size2, strides2, ptr2)

#define SOLID_APPLY_ELEMWISE2B_CPU_CTYPE(TYPE1, TYPE2, CODE, NDIMS1, SIZE1, STRIDES1, PTR1, \
                                                             NDIMS2, SIZE2, STRIDES2, PTR2) \
   {  size_t    _idx1[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _idx2[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides1[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides2[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _index, _indexStop1, _indexStop2, _indexStop; \
      size_t    _nelem, _s1, _s2; \
      TYPE1    *_ptr1 = (TYPE1 *)(PTR1); \
      TYPE2    *_ptr2 = (TYPE2 *)(PTR2); \
      int       _i; \
      \
      /* Initialize the indices */ \
      _nelem = 1; \
      if ((NDIMS1 > 0) && (NDIMS2 > 0)) \
      {  _s1 = _nelem = SIZE1[0]; \
         _s2 = SIZE2[0]; \
         _idx1[0] = 0; \
         _idx2[0] = 0; \
         _strides1[0] = STRIDES1[0]; \
         _strides2[0] = STRIDES2[0]; \
         for (_i = 1; _i < (NDIMS1); _i++) \
         {  _nelem *= SIZE1[_i]; \
            _idx1[_i] = 0; \
            _strides1[_i] = STRIDES1[_i] - SIZE1[_i-1] * STRIDES1[_i-1]; \
         } \
         for (_i = 1; _i < (NDIMS2); _i++) \
         {  _idx2[_i] = 0; \
            _strides2[_i] = STRIDES2[_i] - SIZE2[_i-1] * STRIDES2[_i-1]; \
         } \
      } \
      else \
      {  _s1 = 1; \
         _s2 = 1; \
         _strides1[0] = 0; \
         _strides2[0] = 0; \
      } \
      \
      _index = 0; \
      _indexStop1 = _s1; \
      _indexStop2 = _s2; \
      if (_index < _nelem) \
      {  while (1) \
         {  \
            _indexStop = (_indexStop1 < _indexStop2) ? _indexStop1 : _indexStop2; \
            while (_index < _indexStop) \
            {  CODE \
               _index ++; \
               _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[0]); \
               _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[0]); \
            } \
            if (_index >= _nelem) break; \
            \
            if (_indexStop == _indexStop1) \
            {  _i = 1; _indexStop1 += _s1; \
               _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i]); \
               while (++_idx1[_i] == SIZE1[_i]) \
               {  _idx1[_i] = 0; _i++; \
                  _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i]); \
               } \
            } \
            if (_indexStop == _indexStop2) \
            {  _i = 1; _indexStop2 += _s2; \
               _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i]); \
               while (++_idx2[_i] == SIZE2[_i]) \
               {  _idx2[_i] = 0; _i++; \
                  _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i]); \
               } \
            } \
         } \
      } \
   }


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE2B_OMP                                     */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE2B_OMP(CODE) \
   SOLID_APPLY_ELEMWISE2B_OMP_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), CODE, \
                                    ndims1, size1, strides1, ptr1, ndims2, size2, strides2, ptr2)

#define SOLID_APPLY_ELEMWISE2B_OMP_CTYPE(TYPE1, TYPE2, CODE, NDIMS1, SIZE1, STRIDES1, PTR1, \
                                                             NDIMS2, SIZE2, STRIDES2, PTR2) \
   {  size_t    _idx1[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _idx2[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides1[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides2[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _index, _indexStop1, _indexStop2, _indexStop, _indexMax; \
      size_t    _offset, _offset2; \
      TYPE1    *_ptr1 = (TYPE1 *)(ptr1); \
      TYPE2    *_ptr2 = (TYPE2 *)(ptr2); \
      size_t    _nelem, _s1, _s2; \
      int       _rank = omp_get_thread_num(); \
      int       _nthreads = omp_get_num_threads(); \
      int       _i; \
      \
      /* Initialize the size and strides */ \
      _nelem = 1; \
      for (_i = 0; _i < NDIMS1; _i++) \
      {  _nelem *= SIZE1[_i]; \
         _strides1[_i] = STRIDES1[_i]; \
      } \
      for (_i = 0; _i < NDIMS2; _i++) \
      {  _strides2[_i] = STRIDES2[_i]; \
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
      _offset2  = _offset; \
      \
      /* Initialize the data pointers and indices */ \
      if ((NDIMS1 > 0) && (NDIMS2 > 0) && (_nelem > 0)) \
      {  _s1 = SIZE1[0]; \
         for (_i = 0; _i < NDIMS1; _i++) \
         {  _idx1[_i] = _offset % SIZE1[_i]; \
            _offset /= SIZE1[_i]; \
            _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i] * _idx1[_i]); \
         } \
         _s2 = SIZE2[0]; \
         for (_i = 0; _i < NDIMS2; _i++) \
         {  _idx2[_i] = _offset2 % SIZE2[_i]; \
            _offset2 /= SIZE2[_i]; \
            _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i] * _idx2[_i]); \
         } \
      } \
      else \
      {  /* We also just set these values when the number of elements   */ \
         /* is zero. This will give an incorrect _indexStop value below */ \
         /* but since _indexMax will be equal to _index this value is   */ \
         /* then never used.                                            */ \
         _s1 = 1; _s2 = 1; \
         _idx1[0] = _offset; \
         _idx2[0] = _offset2; \
      } \
      \
      /* Update the strides */ \
      for (_i = NDIMS1-1; _i > 0; _i--) \
         _strides1[_i] -= SIZE1[_i-1] * _strides1[_i-1]; \
      for (_i = NDIMS2-1; _i > 0; _i--) \
         _strides2[_i] -= SIZE2[_i-1] * _strides2[_i-1]; \
      \
      /* Loop over the data */ \
      _indexStop1 = _index + _s1 - _idx1[0]; \
      _indexStop2 = _index + _s2 - _idx2[0]; \
      if (_index < _indexMax) \
      {  while (1) \
         { \
            _indexStop = (_indexStop1 < _indexStop2) ? _indexStop1 : _indexStop2; \
            if (_indexStop > _indexMax) _indexStop = _indexMax; \
            while (_index < _indexStop) \
            {  CODE \
               _index ++; \
               _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[0]); \
               _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[0]); \
            } \
            if (_index >= _indexMax) break; \
            \
            if (_indexStop == _indexStop1) \
            {  _i = 1; _indexStop1 = _index + _s1; \
               _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i]); \
               while (++_idx1[_i] == SIZE1[_i]) \
               {  _idx1[_i] = 0; _i++; \
                  _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i]); \
               } \
            } \
            if (_indexStop == _indexStop2) \
            {  _i = 1; _indexStop2 = _index + _s2; \
               _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i]); \
               while (++_idx2[_i] == SIZE2[_i]) \
               {  _idx2[_i] = 0; _i++; \
                  _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i]); \
               } \
            } \
         } \
      } \
   }

#endif
