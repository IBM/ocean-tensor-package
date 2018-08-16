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

#ifndef __SOLID_INDEX_CPU_H__
#define __SOLID_INDEX_CPU_H__

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/base/cpu/generate_macros.h"
#include "solid/base/cpu/solid_omp.h"



/* --------------------------------------------------------------------- */
/* Macro: SOLID_INDEX1                                                   */
/* --------------------------------------------------------------------- */

#define SOLID_INDEX1(CODE) \
   SOLID_INDEX1_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, offsets, strides, ptr, -1)

#if SOLID_ENABLE_OMP
#define SOLID_INDEX1_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES, PTR, NTHREADS) \
   {  int    __solid_omp_threads = NTHREADS; \
      if (__solid_omp_threads < 0) \
      {  size_t __solid_nelem = 1; \
         int __i; \
         __solid_omp_threads = solid_omp_get_max_threads(); \
         for (__i = 0; __i < NDIMS; __i++) __solid_nelem *= SIZE[__i]; \
         __i = (__solid_nelem / 32); \
         if (__i < __solid_omp_threads) __solid_omp_threads = __i; \
      } \
      \
      if (__solid_omp_threads > 1) \
      {   \
         _Pragma("omp parallel num_threads(__solid_omp_threads)") \
         SOLID_INDEX1_OMP_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES, PTR) \
      } \
      else \
      {  SOLID_INDEX1_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES, PTR) \
      } \
   }
#else
#define SOLID_INDEX1_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES, PTR, NTHREADS) \
   {  SOLID_INDEX1_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES, PTR) \
   }
#endif


/* --------------------------------------------------------------------- */
/* Macro: SOLID_INDEX1_CPU                                               */
/* --------------------------------------------------------------------- */

#define SOLID_INDEX1_CPU(CODE) \
   SOLID_INDEX1_CPU_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, offsets, strides, ptr)

#define SOLID_INDEX1_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES, PTR) \
   {  size_t       _idx[SOLID_MAX_TENSOR_DIMS]; \
      size_t       _nloops, _s, _j; \
      solid_int64 *_offset; \
      solid_int64 *_offsets0 = OFFSETS[0]; \
      ptrdiff_t    _strides0 = STRIDES[0]; \
      ptrdiff_t    _strides[SOLID_MAX_TENSOR_DIMS]; \
      char        *_ptrBase; \
      TYPE        *_ptr = (TYPE *)(PTR); \
      int          _i, _ndims; \
      \
      /* Initialize */ \
      _nloops = 1; \
      _ndims = NDIMS; \
      if (_ndims == 0) \
      {  _strides[0] = 0; \
         _offsets0   = NULL; \
         _s = 1; _ndims = 1; \
      } \
      else \
      {  _s = SIZE[0]; \
         _strides[0] = (_offsets0) ? 0 : STRIDES[0]; \
         for (_i = 1; _i < _ndims; _i++) \
         {  if ((_offset = OFFSETS[_i]) != NULL) \
            {  _strides[_i] = 0; \
               _ptr = (TYPE *)(((char *)_ptr) + _offset[0]); \
            } \
            else \
            { _strides[_i] = STRIDES[_i]; \
            } \
         } \
      } \
      for (_i = _ndims-1; _i > 0; _i--) \
      {  _idx[_i] = 0; \
         _nloops *= SIZE[_i]; \
         _strides[_i] -= SIZE[_i-1] * _strides[_i-1]; \
      } \
      \
      while (1) \
      {  /* Inner-most loop */ \
         if (_offsets0) \
         {  _ptrBase = (char *)(_ptr); \
            for (_i = 0; _i < _s; _i++) \
            {  _ptr = (TYPE *)(_ptrBase + _offsets0[_i]); \
               CODE \
            } \
            _ptr = (TYPE *)_ptrBase; \
         } \
         else \
         {  for (_i = 0; _i < _s; _i++) \
            {  CODE \
               _ptr = (TYPE *)(((char *)_ptr) + _strides0); \
            } \
         } \
         \
         /* Outer loop */ \
         _nloops --; if (_nloops == 0) break; \
         _i = 1; \
         while (1) \
         {  _ptr = (TYPE *)(((char *)_ptr) + _strides[_i]); \
            if ((_offset = OFFSETS[_i]) != NULL) \
            {  _j = _idx[_i]; \
               if (_j + 1 < SIZE[_i]) \
               {  _ptr = (TYPE *)(((char *)_ptr) - _offset[_j] + _offset[_j+1]); \
                  _idx[_i] = _j + 1; break; \
               } \
               else \
               {  _ptr = (TYPE *)(((char *)_ptr) - _offset[_j] + _offset[0]); \
                  _idx[_i] = 0; \
               } \
            } \
            else \
            {  _j = _idx[_i] + 1; \
               if (_j < SIZE[_i]) \
               {  _idx[_i] = _j; break; \
               } \
               else \
               {  _idx[_i] = 0; \
               } \
            } \
            _i ++; \
         } \
      } \
   }


/* --------------------------------------------------------------------- */
/* Macro: SOLID_INDEX1_OMP                                               */
/* --------------------------------------------------------------------- */

#define SOLID_INDEX1_OMP(CODE) \
   SOLID_INDEX1_OMP_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, offsets, strides, ptr)

#define SOLID_INDEX1_OMP_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES, PTR) \
   {  char        *_ptrBase; \
      TYPE        *_ptr = (TYPE *)(PTR); \
      size_t       _idx[SOLID_MAX_TENSOR_DIMS]; \
      size_t       _index, _indexStop; \
      size_t       _indexOffset, _indexMax; \
      size_t       _nelem, _s, _j; \
      solid_int64 *_offset; \
      solid_int64 *_offsets0 = OFFSETS[0]; \
      ptrdiff_t    _strides[SOLID_MAX_TENSOR_DIMS]; \
      int          _rank = omp_get_thread_num(); \
      int          _nthreads = omp_get_num_threads(); \
      int          _i; \
      \
      /* Initialize the size and strides */ \
      _nelem = 1; \
      for (_i = 0; _i < NDIMS; _i++) \
      {  _nelem *= SIZE[_i]; \
         _strides[_i] = OFFSETS[_i] ? 0 : STRIDES[_i]; \
      } \
      \
      /* Compute the offset */ \
      _indexOffset = _nelem % _nthreads; _nelem /= _nthreads; \
      if (_rank < _indexOffset) \
      {  _nelem ++; \
         _indexOffset = _rank * _nelem; \
      } \
      else \
      {  _indexOffset += _rank * _nelem; \
      } \
      _index  = _indexOffset; \
      _indexMax = _indexOffset + _nelem; \
      \
      if (_index < _indexMax) \
      {  /* Initialize the data pointer and indices */ \
         if (NDIMS == 0) \
         {  _strides[0] = 0; \
            _idx[0]     = 0; \
            _offsets0   = NULL; \
         } \
         _s = (NDIMS > 0) ? SIZE[0] : 1; \
         for (_i = 0; _i < NDIMS; _i++) \
         {  _idx[_i] = _j = _indexOffset % SIZE[_i]; \
            _indexOffset /= SIZE[_i]; \
            _ptr = (TYPE *)(((char *)_ptr) + _strides[_i] * _j); \
            if ((_i > 0) && (OFFSETS[_i])) \
               _ptr = (TYPE *)(((char *)_ptr) + OFFSETS[_i][_j]); \
         } \
         \
         /* Update the strides */ \
         for (_i = NDIMS-1; _i > 0; _i--) \
         {  _strides[_i] -= SIZE[_i-1] * _strides[_i-1]; \
         } \
         \
         /* Loop over the data */ \
         _indexStop = _index + _s - _idx[0]; \
         while (1) \
         {  /* Inner-most loop */ \
            if (_indexStop > _indexMax) _indexStop = _indexMax; \
            if (_offsets0) \
            {  _ptrBase = (char *)_ptr; \
               while (_index < _indexStop) \
               {  _ptr = (TYPE *)(_ptrBase + _offsets0[_index % _s]); \
                  CODE \
                  _index ++; \
               } \
               _ptr = (TYPE *)_ptrBase; \
            } \
            else\
            {  while (_index < _indexStop) \
               { \
                  CODE \
                  _index ++; \
                  _ptr = (TYPE *)(((char *)_ptr) + _strides[0]); \
               } \
            } \
            if (_index >= _indexMax) break; \
            \
            /* Outer loops */ \
            _i = 1; _indexStop = _index + _s; \
            while (1) \
            {  _ptr = (TYPE *)(((char *)_ptr) + _strides[_i]); \
               if ((_offset = OFFSETS[_i]) != NULL) \
               {  _j = _idx[_i]; \
                  if (_j + 1 < SIZE[_i]) \
                  {  _ptr = (TYPE *)(((char *)_ptr) - _offset[_j] + _offset[_j+1]); \
                     _idx[_i] = _j + 1; break; \
                  } \
                  else \
                  {  _ptr = (TYPE *)(((char *)_ptr) - _offset[_j] + _offset[0]); \
                     _idx[_i] = 0; \
                  } \
               } \
               else \
               {  _j = _idx[_i] + 1; \
                  if (_j < SIZE[_i]) \
                  {  _idx[_i] = _j; break; \
                  } \
                  else \
                  {  _idx[_i] = 0; \
                  } \
               } \
               _i ++; \
            } \
         } \
      } \
   }


/* --------------------------------------------------------------------- */
/* Macro: SOLID_INDEX2                                                   */
/* --------------------------------------------------------------------- */

#define SOLID_INDEX2(CODE) \
   SOLID_INDEX2_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, offsets, strides1, ptr1, strides2, ptr2, -1)

#if SOLID_ENABLE_OMP
#define SOLID_INDEX2_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES1, PTR1, STRIDES2, PTR2, NTHREADS) \
   {  int    __solid_omp_threads = NTHREADS; \
      if (__solid_omp_threads < 0) \
      {  size_t __solid_nelem = 1; \
         int __i; \
         __solid_omp_threads = solid_omp_get_max_threads(); \
         for (__i = 0; __i < NDIMS; __i++) __solid_nelem *= SIZE[__i]; \
         __i = (__solid_nelem / 32); \
         if (__i < __solid_omp_threads) __solid_omp_threads = __i; \
      } \
      \
      if (__solid_omp_threads > 1) \
      {   \
         _Pragma("omp parallel num_threads(__solid_omp_threads)") \
         SOLID_INDEX2_OMP_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES1, PTR1, STRIDES2, PTR2) \
      } \
      else \
      {  SOLID_INDEX2_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES1, PTR1, STRIDES2, PTR2) \
      } \
   }
#else
#define SOLID_INDEX2_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES1, PTR1, STRIDES2, PTR2, NTHREADS) \
   {  SOLID_INDEX2_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES1, PTR1, STRIDES2, PTR2) \
   }
#endif


/* --------------------------------------------------------------------- */
/* Macro: SOLID_INDEX2_CPU                                               */
/* --------------------------------------------------------------------- */

#define SOLID_INDEX2_CPU(CODE) \
   SOLID_INDEX2_CPU_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, offsets, strides1, ptr1, strides2, ptr2)

#define SOLID_INDEX2_CPU_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES1, PTR1, STRIDES2, PTR2) \
   {  size_t       _idx[SOLID_MAX_TENSOR_DIMS]; \
      size_t       _nloops, _s, _j; \
      solid_int64 *_offset; \
      solid_int64 *_offsets0 = OFFSETS[0]; \
      ptrdiff_t    _strides1[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t    _strides2[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t    _strides1_0 = STRIDES1[0]; \
      ptrdiff_t    _strides2_0 = STRIDES2[0]; \
      char        *_ptr0; \
      TYPE        *_ptr1 = (TYPE *)(PTR1); \
      TYPE        *_ptr2 = (TYPE *)(PTR2); \
      int          _i, _ndims; \
      \
      /* Initialize */ \
      _nloops = 1; \
      _ndims = NDIMS; \
      if (_ndims == 0) \
      {  _strides1[0] = 0; \
         _strides2[0] = 0; \
         _offsets0    = NULL; \
         _s = 1; _ndims = 1; \
      } \
      else \
      {  _s = SIZE[0]; \
         _strides1[0] = (_offsets0) ? 0 : STRIDES1[0]; \
         for (_i = 1; _i < NDIMS; _i++) \
         {  if ((_offset = OFFSETS[_i]) != NULL) \
            {  _strides1[_i] = 0; \
               _ptr1 = (TYPE *)(((char *)_ptr1) + _offset[0]); \
            } \
            else \
            { _strides1[_i] = STRIDES1[_i]; \
            } \
         } \
      } \
      for (_i = _ndims-1; _i > 0; _i--) \
      {  _idx[_i] = 0; \
         _nloops *= SIZE[_i]; \
         _strides1[_i] -= SIZE[_i-1] * _strides1[_i-1]; \
         _strides2[_i] = STRIDES2[_i] - SIZE[_i-1] * STRIDES2[_i-1]; \
      } \
      \
      while (1) \
      {  /* Inner-most loop */ \
         if (_offsets0) \
         {  _ptr0 = (char *)(_ptr1); \
            for (_i = 0; _i < _s; _i++) \
            {  _ptr1 = (TYPE *)(_ptr0 + _offsets0[_i]); \
               CODE \
               _ptr2 = (TYPE *)(((char *)_ptr2) + _strides2_0); \
            } \
            _ptr1 = (TYPE *)_ptr0; \
         } \
         else \
         {  for (_i = 0; _i < _s; _i++) \
            {  CODE \
               _ptr1 = (TYPE *)(((char *)_ptr1) + _strides1_0); \
               _ptr2 = (TYPE *)(((char *)_ptr2) + _strides2_0); \
            } \
         } \
         \
         /* Outer loop */ \
         _nloops --; if (_nloops == 0) break; \
         _i = 1; \
         while (1) \
         {  _ptr1 = (TYPE *)(((char *)_ptr1) + _strides1[_i]); \
            _ptr2 = (TYPE *)(((char *)_ptr2) + _strides2[_i]); \
            if ((_offset = OFFSETS[_i]) != NULL) \
            {  _j = _idx[_i]; \
               if (_j + 1 < SIZE[_i]) \
               {  _ptr1 = (TYPE *)(((char *)_ptr1) - _offset[_j] + _offset[_j+1]); \
                  _idx[_i] = _j + 1; break; \
               } \
               else \
               {  _ptr1 = (TYPE *)(((char *)_ptr1) - _offset[_j] + _offset[0]); \
                  _idx[_i] = 0; \
               } \
            } \
            else \
            {  _j = _idx[_i] + 1; \
               if (_j < SIZE[_i]) \
               {  _idx[_i] = _j; break; \
               } \
               else \
               {  _idx[_i] = 0; \
               } \
            } \
            _i ++; \
         } \
      } \
   }


/* --------------------------------------------------------------------- */
/* Macro: SOLID_INDEX2_OMP                                               */
/* --------------------------------------------------------------------- */

#define SOLID_INDEX2_OMP(CODE) \
   SOLID_INDEX2_OMP_CTYPE(SOLID_C_TYPE(SDXTYPE), CODE, ndims, size, offsets, strides1, ptr1, strides2, ptr2)

#define SOLID_INDEX2_OMP_CTYPE(TYPE, CODE, NDIMS, SIZE, OFFSETS, STRIDES1, PTR1, STRIDES2, PTR2) \
   {  char        *_ptr0; \
      TYPE        *_ptr1 = (TYPE *)(PTR1); \
      TYPE        *_ptr2 = (TYPE *)(PTR2); \
      size_t       _idx[SOLID_MAX_TENSOR_DIMS]; \
      size_t       _index, _indexStop; \
      size_t       _indexOffset, _indexMax; \
      size_t       _nelem, _s, _j; \
      solid_int64 *_offset; \
      solid_int64 *_offsets0 = OFFSETS[0]; \
      ptrdiff_t    _strides1[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t    _strides2[SOLID_MAX_TENSOR_DIMS]; \
      int          _rank = omp_get_thread_num(); \
      int          _nthreads = omp_get_num_threads(); \
      int          _i; \
      \
      /* Initialize the size and strides */ \
      _nelem = 1; \
      for (_i = 0; _i < NDIMS; _i++) \
      {  _nelem *= SIZE[_i]; \
         _strides1[_i] = OFFSETS[_i] ? 0 : STRIDES1[_i]; \
         _strides2[_i] = STRIDES2[_i]; \
      } \
      \
      /* Compute the offset */ \
      _indexOffset = _nelem % _nthreads; _nelem /= _nthreads; \
      if (_rank < _indexOffset) \
      {  _nelem ++; \
         _indexOffset = _rank * _nelem; \
      } \
      else \
      {  _indexOffset += _rank * _nelem; \
      } \
      _index  = _indexOffset; \
      _indexMax = _indexOffset + _nelem; \
      \
      if (_index < _indexMax) \
      {  /* Initialize the data pointer and indices */ \
         if (NDIMS == 0) \
         {  _strides1[0] = 0; \
            _strides2[0] = 0; \
            _idx[0]      = 0; \
            _offsets0    = NULL; \
         } \
         _s = (NDIMS > 0) ? SIZE[0] : 1; \
         for (_i = 0; _i < NDIMS; _i++) \
         {  _idx[_i] = _j = _indexOffset % SIZE[_i]; \
            _indexOffset /= SIZE[_i]; \
            _ptr1 = (TYPE *)(((char *)_ptr1) + _strides1[_i] * _j); \
            _ptr2 = (TYPE *)(((char *)_ptr2) + _strides2[_i] * _j); \
            if ((_i > 0) && (OFFSETS[_i])) \
               _ptr1 = (TYPE *)(((char *)_ptr1) + OFFSETS[_i][_j]); \
         } \
         \
         /* Update the strides */ \
         for (_i = NDIMS-1; _i > 0; _i--) \
         {  _strides1[_i] -= SIZE[_i-1] * _strides1[_i-1]; \
            _strides2[_i] -= SIZE[_i-1] * _strides2[_i-1]; \
         } \
         \
         /* Loop over the data */ \
         _indexStop = _index + _s - _idx[0]; \
         while (1) \
         {  /* Inner-most loop */ \
            if (_indexStop > _indexMax) _indexStop = _indexMax; \
            if (_offsets0) \
            {  _ptr0 = (char *)_ptr1; \
               while (_index < _indexStop) \
               {  _ptr1 = (TYPE *)(_ptr0 + _offsets0[_index % _s]); \
                  CODE \
                  _index ++; \
                  _ptr2 = (TYPE *)(((char *)_ptr2) + _strides2[0]); \
               } \
               _ptr1 = (TYPE *)_ptr0; \
            } \
            else\
            {  while (_index < _indexStop) \
               { \
                  CODE \
                  _index ++; \
                  _ptr1 = (TYPE *)(((char *)_ptr1) + _strides1[0]); \
                  _ptr2 = (TYPE *)(((char *)_ptr2) + _strides2[0]); \
               } \
            } \
            if (_index >= _indexMax) break; \
            \
            /* Outer loops */ \
            _i = 1; _indexStop = _index + _s; \
            while (1) \
            {  _ptr1 = (TYPE *)(((char *)_ptr1) + _strides1[_i]); \
               _ptr2 = (TYPE *)(((char *)_ptr2) + _strides2[_i]); \
               if ((_offset = OFFSETS[_i]) != NULL) \
               {  _j = _idx[_i]; \
                  if (_j + 1 < SIZE[_i]) \
                  {  _ptr1 = (TYPE *)(((char *)_ptr1) - _offset[_j] + _offset[_j+1]); \
                     _idx[_i] = _j + 1; break; \
                  } \
                  else \
                  {  _ptr1 = (TYPE *)(((char *)_ptr1) - _offset[_j] + _offset[0]); \
                     _idx[_i] = 0; \
                  } \
               } \
               else \
               {  _j = _idx[_i] + 1; \
                  if (_j < SIZE[_i]) \
                  {  _idx[_i] = _j; break; \
                  } \
                  else \
                  {  _idx[_i] = 0; \
                  } \
               } \
               _i ++; \
            } \
         } \
      } \
   }

#endif
