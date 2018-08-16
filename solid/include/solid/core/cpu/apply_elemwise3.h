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

#ifndef __SOLID_APPLY_ELEMWISE3_CPU_H__
#define __SOLID_APPLY_ELEMWISE3_CPU_H__

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/base/cpu/generate_macros.h"
#include "solid/base/cpu/solid_omp.h"

/* These kernels can be used for two tensors when the sizes  */
/* match exactly. The strides for each dimension can differ. */
/* The variables exposed are: _ptr1, _ptr2, _ptr3            */


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE3                                          */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE3(CODE) \
   SOLID_APPLY_ELEMWISE3_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), CODE, \
                               ndims, size, ptr1, strides1, ptr2, strides2, ptr3, strides3, 0)

#define SOLID_APPLY_ELEMWISE3_TYPES(TYPE1, TYPE2, TYPE3, CODE) \
   SOLID_APPLY_ELEMWISE3_CTYPE(SOLID_C_TYPE(TYPE1), SOLID_C_TYPE(TYPE2), SOLID_C_TYPE(TYPE3), CODE, \
                               ndims, size, ptr1, strides1, ptr2, strides2, ptr3, strides3, 0)

#if SOLID_ENABLE_OMP
#define SOLID_APPLY_ELEMWISE3_CTYPE(TYPE1, TYPE2, TYPE3, CODE, NDIMS, SIZE, PTR1, STRIDES1, PTR2, STRIDES2, PTR3, STRIDES3, NTHREADS) \
   {  int __solid_omp_threads = NTHREADS; \
      if (__solid_omp_threads < 0) __solid_omp_threads = solid_omp_get_max_threads(); \
      \
      if (__solid_omp_threads > 1) \
      {   \
         _Pragma("omp parallel num_threads(__solid_omp_threads)") \
         SOLID_APPLY_ELEMWISE3_OMP_CTYPE(TYPE1, TYPE2, TYPE3, CODE, NDIMS, SIZE, PTR1, STRIDES1, PTR2, STRIDES2, PTR3, STRIDES3) \
      } \
      else \
      {  SOLID_APPLY_ELEMWISE3_CPU_CTYPE(TYPE1, TYPE2, TYPE3, CODE, NDIMS, SIZE, PTR1, STRIDES1, PTR2, STRIDES2, PTR3, STRIDES3) \
      } \
   }
#else
#define SOLID_APPLY_ELEMWISE3_CTYPE(TYPE1, TYPE2, TYPE3, CODE, NDIMS, SIZE, PTR1, STRIDES1, PTR2, STRIDES2, PTR3, STRIDES3, NTHREADS) \
   {  SOLID_APPLY_ELEMWISE3_CPU_CTYPE(TYPE1, TYPE2, TYPE3, CODE, NDIMS, SIZE, PTR1, STRIDES1, PTR2, STRIDES2, PTR3, STRIDES3) \
   }
#endif


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE3_CPU                                      */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE3_CPU(CODE) \
   SOLID_APPLY_ELEMWISE3_CPU_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), CODE, \
                                   ndims, size, ptr1, strides1, ptr2, strides2, ptr3, strides3)

#define SOLID_APPLY_ELEMWISE3_CPU_CTYPE(TYPE1, TYPE2, TYPE3, CODE, NDIMS, SIZE, PTR1, STRIDES1, PTR2, STRIDES2, PTR3, STRIDES3) \
   {  size_t    _idx[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides1[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides2[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides3[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _index, _indexStop; \
      size_t    _nelem, _s; \
      TYPE1    *_ptr1 = (TYPE1 *)(PTR1); \
      TYPE2    *_ptr2 = (TYPE2 *)(PTR2); \
      TYPE3    *_ptr3 = (TYPE3 *)(PTR3); \
      int       _i; \
      \
      /* Initialize the indices */ \
      _nelem = 1; \
      if (NDIMS > 0) \
      {  _s = _nelem = SIZE[0]; \
         _idx[0] = 0; \
         _strides1[0] = STRIDES1[0]; \
         _strides2[0] = STRIDES2[0]; \
         _strides3[0] = STRIDES3[0]; \
         for (_i = 1; _i < NDIMS; _i ++) \
         {  _nelem *= SIZE[_i]; \
            _idx[_i] = 0; \
            _strides1[_i] = STRIDES1[_i] - SIZE[_i-1] * STRIDES1[_i-1]; \
            _strides2[_i] = STRIDES2[_i] - SIZE[_i-1] * STRIDES2[_i-1]; \
            _strides3[_i] = STRIDES3[_i] - SIZE[_i-1] * STRIDES3[_i-1]; \
         } \
      } \
      else \
      {  _s = 1; \
         _strides1[0] = 0; \
         _strides2[0] = 0; \
         _strides3[0] = 0; \
      } \
      \
      _index = 0; \
      if (_index < _nelem) \
      {  while (1) \
         {  \
            _indexStop = _index + _s; \
            while (_index < _indexStop) \
            {  CODE \
               _index ++; \
               _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[0]); \
               _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[0]); \
               _ptr3 = (TYPE3 *)(((char *)_ptr3) + _strides3[0]); \
            } \
            if (_index >= _nelem) break; \
            \
            _i = 1; \
            _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i]); \
            _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i]); \
            _ptr3 = (TYPE3 *)(((char *)_ptr3) + _strides3[_i]); \
            while (++_idx[_i] == SIZE[_i]) \
            {  _idx[_i] = 0; _i++; \
               _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i]); \
               _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i]); \
               _ptr3 = (TYPE3 *)(((char *)_ptr3) + _strides3[_i]); \
            } \
         } \
      } \
   }


/* --------------------------------------------------------------------- */
/* Macro: SOLID_APPLY_ELEMWISE3_OMP                                      */
/* --------------------------------------------------------------------- */

#define SOLID_APPLY_ELEMWISE3_OMP(CODE) \
   SOLID_APPLY_ELEMWISE3_OMP_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE), CODE, \
                                   ndims, size, ptr1, strides1, ptr2, strides2, ptr3, strides3)

#define SOLID_APPLY_ELEMWISE3_OMP_CTYPE(TYPE1, TYPE2, TYPE3, CODE, NDIMS, SIZE, PTR1, STRIDES1, PTR2, STRIDES2, PTR3, STRIDES3) \
   {  size_t    _idx[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides1[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides2[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t _strides3[SOLID_MAX_TENSOR_DIMS]; \
      size_t    _index, _indexStop, _indexMax; \
      size_t    _offset; \
      TYPE1    *_ptr1 = (TYPE1 *)(ptr1); \
      TYPE2    *_ptr2 = (TYPE2 *)(ptr2); \
      TYPE3    *_ptr3 = (TYPE3 *)(ptr3); \
      size_t    _nelem, _s; \
      int       _rank = omp_get_thread_num(); \
      int       _nthreads = omp_get_num_threads(); \
      int       _i; \
      \
      /* Initialize the size and strides */ \
      _nelem = 1; \
      for (_i = 0; _i < NDIMS; _i++) \
      {  _nelem *= SIZE[_i]; \
         _strides1[_i] = STRIDES1[_i]; \
         _strides2[_i] = STRIDES2[_i]; \
         _strides3[_i] = STRIDES3[_i]; \
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
      /* Initialize the data pointers and indices */ \
      if ((NDIMS > 0) && (_nelem > 0)) \
      {  _s = SIZE[0]; \
         for (_i = 0; _i < NDIMS; _i++) \
         {  _idx[_i] = _offset % SIZE[_i]; \
            _offset /= SIZE[_i]; \
            _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i] * _idx[_i]); \
            _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i] * _idx[_i]); \
            _ptr3 = (TYPE3 *)(((char *)_ptr3) + _strides3[_i] * _idx[_i]); \
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
      {  _strides1[_i] -= SIZE[_i-1] * _strides1[_i-1]; \
         _strides2[_i] -= SIZE[_i-1] * _strides2[_i-1]; \
         _strides3[_i] -= SIZE[_i-1] * _strides3[_i-1]; \
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
               _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[0]); \
               _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[0]); \
               _ptr3 = (TYPE3 *)(((char *)_ptr3) + _strides3[0]); \
            } \
            if (_index >= _indexMax) break; \
            \
            _i = 1; _indexStop = _index + _s; \
            _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i]); \
            _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i]); \
            _ptr3 = (TYPE3 *)(((char *)_ptr3) + _strides3[_i]); \
            while (++_idx[_i] == SIZE[_i]) \
            {  _idx[_i] = 0; _i++; \
               _ptr1 = (TYPE1 *)(((char *)_ptr1) + _strides1[_i]); \
               _ptr2 = (TYPE2 *)(((char *)_ptr2) + _strides2[_i]); \
               _ptr3 = (TYPE3 *)(((char *)_ptr3) + _strides3[_i]); \
            } \
         } \
      } \
   }

#endif
