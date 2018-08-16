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

#ifndef __SOLID_REDUCE_AXIS_CPU_H__
#define __SOLID_REDUCE_AXIS_CPU_H__

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
/* Macro: SOLID_CREATE_REDUCE_AXIS                                       */
/* --------------------------------------------------------------------- */

/* Create the parameter structure and entry functions */
#if SOLID_ENABLE_OMP
#define SOLID_CREATE_REDUCE_AXIS(NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE,\
                                 FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM, PARAM) \
   SOLID_PARAM_STRUCT_FLAG(SOLID_FUNCTION_TYPE(NAME,DTYPE1), FLAG_PARAM, PARAM) \
   SOLID_REDUCE_AXIS_CPU(NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_OMP1(NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_OMP2(NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM)
#else
#define SOLID_CREATE_REDUCE_AXIS(NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE,\
                                 FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM, PARAM) \
   SOLID_PARAM_STRUCT_FLAG(SOLID_FUNCTION_TYPE(NAME,DTYPE1), FLAG_PARAM, PARAM) \
   SOLID_REDUCE_AXIS_CPU(NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM)
#endif

/* --------------------------------------------------------------------- */
/* Macro: SOLID_CALL_REDUCE_AXIS                                         */
/* --------------------------------------------------------------------- */

#define SOLID_CALL_REDUCE_AXIS_FUN_B0(NAME, DTYPE, MODE, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM) \
   SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, MODE)(NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES)
#define SOLID_CALL_REDUCE_AXIS_FUN_B1(NAME, DTYPE, MODE, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM) \
   SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, MODE)(NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM)
#define SOLID_CALL_REDUCE_AXIS_FUN(NAME, DTYPE, MODE, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM, FLAG_PARAM) \
   SOLID_CALL_REDUCE_AXIS_FUN_B##FLAG_PARAM(NAME, DTYPE, MODE, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM)

#if SOLID_ENABLE_OMP
#define SOLID_CALL_REDUCE_AXIS_FULL(NAME, DTYPE, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM, FLAG_PARAM) \
   {  size_t __solid_nelem = 1; \
      size_t __solid_relem = 1; \
      int     __solid_omp_threads; \
      \
      /* Determine the maximum number of threads */ \
      __solid_omp_threads = solid_omp_get_max_threads(); \
      \
      /* Compute the number of elements */ \
      if (__solid_omp_threads > 1) \
      {  int __i; \
         for (__i = 0; __i < NDIMS; __i++) __solid_nelem *= SIZE[__i]; \
         for (__i = 0; __i < RDIMS; __i++) __solid_relem *= RSIZE[__i]; \
      } \
      \
      /* Decide between single or multiple thread reduction    */ \
      /* Use single-thread reduction when the total number     */ \
      /* of operations is small. For multiple-thread reduction */ \
      /* we use the first variation (omp1) when the number of  */ \
      /* outputs is sufficiently large to partition then among */ \
      /* the threads without a large load imbalance. Otherwise */ \
      /* we use the second multi-thread reduction (omp2), in   */ \
      /* which all threads cooperate to compute the reduction  */ \
      /* for each output element; this is useful in settings   */ \
      /* where the number of elements in the reduction is large*/ \
      /* and the number of outputs is small.                   */ \
      if ((__solid_omp_threads <= 1) || (__solid_nelem * __solid_relem <= 256)) \
      {  SOLID_CALL_REDUCE_AXIS_FUN(NAME, DTYPE, single, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM, FLAG_PARAM); \
      } \
      else \
      {  if (__solid_nelem >= 4 * __solid_omp_threads) \
         {  /* Thread compute full reductions for their set of outputs */ \
            SOLID_CALL_REDUCE_AXIS_FUN(NAME, DTYPE, omp1, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM, FLAG_PARAM); \
         } \
         else \
         {  /* Each reduction is done collaboratively with all threads */ \
            SOLID_CALL_REDUCE_AXIS_FUN(NAME, DTYPE, omp2, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM, FLAG_PARAM); \
         } \
      } \
   }
#else
#define SOLID_CALL_REDUCE_AXIS_FULL(NAME, DTYPE, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM, FLAG_PARAM) \
   {  SOLID_CALL_REDUCE_AXIS_FUN(NAME, DTYPE, single, NDIMS, SIZE, STRIDES1, PTR1, STRIDES2, PTR2, RDIMS, RSIZE, RSTRIDES, PARAM, FLAG_PARAM); \
   }
#endif

#define SOLID_CALL_REDUCE_AXIS_PARAM(NAME, PARAM) \
   SOLID_CALL_REDUCE_AXIS_FULL(NAME, SDXTYPE, ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides, PARAM, 1)

#define SOLID_CALL_REDUCE_AXIS(NAME) \
   SOLID_CALL_REDUCE_AXIS_FULL(NAME, SDXTYPE, ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides, { }, 0)


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
   FUNCTION(int ndims, const size_t *size, \
            const ptrdiff_t *param_strides1, void *param_ptr1, \
            const ptrdiff_t *param_strides2, void *param_ptr2, \
            int rdims, const size_t *rsize, const ptrdiff_t *param_rstrides)

#define SOLID_REDUCE_AXIS_FUNCTION_ITF_D1(FUNCTION, PARAM_TYPE) \
   FUNCTION(int ndims, const size_t *size, \
            const ptrdiff_t *param_strides1, void *param_ptr1, \
            const ptrdiff_t *param_strides2, void *param_ptr2, \
            int rdims, const size_t *rsize, const ptrdiff_t *param_rstrides, \
            PARAM_TYPE *ptr_param)
#define SOLID_REDUCE_AXIS_FUNCTION_ITF_C(FUNCTION, PARAM_TYPE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_FUNCTION_ITF_D##FLAG_PARAM(FUNCTION, PARAM_TYPE)
#define SOLID_REDUCE_AXIS_FUNCTION_ITF_B(NAME, DTYPE, MODE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_FUNCTION_ITF_C(SOLID_REDUCE_AXIS_FUNCTION(NAME, DTYPE, MODE), SOLID_PARAM_TYPE(NAME, DTYPE), FLAG_PARAM)
#define SOLID_REDUCE_AXIS_FUNCTION_ITF(NAME, DTYPE, MODE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_FUNCTION_ITF_B(NAME, DTYPE, MODE, FLAG_PARAM)

/* Finalization of the result */
#define SOLID_REDUCE_AXIS_FINALIZE_C0(RTYPE, CODE_FINALIZE, VAR) /* Empty */
#define SOLID_REDUCE_AXIS_FINALIZE_C1(RTYPE, CODE_FINALIZE, VAR) \
   {  SOLID_C_WORKTYPE_TYPE(RTYPE)  _temp    = VAR; \
      SOLID_C_WORKTYPE_TYPE(RTYPE) *_partial = &_temp; \
      SOLID_C_WORKTYPE_TYPE(RTYPE) *_result  = &(VAR); \
      CODE_FINALIZE \
   }
#define SOLID_REDUCE_AXIS_FINALIZE_B(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, VAR) \
   SOLID_REDUCE_AXIS_FINALIZE_C##FLAG_FINALIZE(RTYPE, CODE_FINALIZE, VAR)
#define SOLID_REDUCE_AXIS_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, VAR) \
   SOLID_REDUCE_AXIS_FINALIZE_B(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, VAR)

/* Declaration of the local param variable */
#define SOLID_REDUCE_AXIS_CREATE_PARAM_C0(PARAM_TYPE) /* Empty */
#define SOLID_REDUCE_AXIS_CREATE_PARAM_C1(PARAM_TYPE) PARAM_TYPE param = *ptr_param;
#define SOLID_REDUCE_AXIS_CREATE_PARAM_B(NAME, DTYPE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_CREATE_PARAM_C##FLAG_PARAM(SOLID_PARAM_TYPE(NAME, DTYPE))
#define SOLID_REDUCE_AXIS_CREATE_PARAM(NAME, DTYPE, FLAG_PARAM) \
   SOLID_REDUCE_AXIS_CREATE_PARAM_B(NAME, DTYPE, FLAG_PARAM)


/* --------------------------------------------------------------------- */
/* Macro: SOLID_REDUCE_AXIS_CPU                                          */
/* --------------------------------------------------------------------- */

#define SOLID_REDUCE_AXIS_CPU(NAME, TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
void SOLID_REDUCE_AXIS_FUNCTION_ITF(NAME, TYPE, single, FLAG_PARAM) \
{  SOLID_REDUCE_AXIS_CREATE_PARAM(NAME, TYPE, FLAG_PARAM); /* param */ \
   SOLID_C_WORKTYPE_TYPE(RTYPE) _accumulate; \
   SOLID_C_TYPE(TYPE)          *_ptr; \
   char     *ptr1 = (char *)(param_ptr1); \
   char     *ptr2 = (char *)(param_ptr2); \
   size_t    idx[SOLID_MAX_TENSOR_DIMS] = {0}; \
   size_t    ridx[SOLID_MAX_TENSOR_DIMS] = {0}; \
   ptrdiff_t strides1[SOLID_MAX_TENSOR_DIMS]; \
   ptrdiff_t strides2[SOLID_MAX_TENSOR_DIMS]; \
   ptrdiff_t rstrides[SOLID_MAX_TENSOR_DIMS], r; \
   ptrdiff_t data_strides1, data_strides2, data_rstrides; \
   size_t    data_size, data_rsize; \
   size_t    nelem, relem, s, k1, k2, j; \
   int       i; \
   \
   /* ---------------------------------------------------------- */ \
   /* Make sure that the number of dimensions is at least one    */ \
   /* ---------------------------------------------------------- */ \
   if (ndims == 0) \
   {  ndims = 1; \
      data_size = 1; size = &(data_size ); \
      data_strides1 = 0; param_strides1 = &(data_strides1); \
      data_strides2 = 0; param_strides2 = &(data_strides2); \
   } \
   if (rdims == 0) \
   {  rdims = 1; \
      data_rsize = 1; rsize = &(data_rsize); \
      data_rstrides = 0; param_rstrides = &(data_rstrides); \
   } \
   \
   /* ---------------------------------------------------------- */ \
   /* Initialize the outer indices                               */ \
   /* ---------------------------------------------------------- */ \
   nelem = size[0]; \
   strides1[0] = param_strides1[0]; \
   strides2[0] = param_strides2[0]; \
   for (i = 1; i < ndims; i++) \
   {  nelem *= size[i]; \
      strides1[i] = param_strides1[i] - size[i-1] * param_strides1[i-1]; \
      strides2[i] = param_strides2[i] - size[i-1] * param_strides2[i-1]; \
   } \
   \
   /* ---------------------------------------------------------- */ \
   /* Initialize the inner problem - exclude the first dimension */ \
   /* from the number of elements to determine the number of     */ \
   /* loops; the first dimension has and explicit loop.          */ \
   /* ---------------------------------------------------------- */ \
   s = rsize[0]; \
   r = param_rstrides[0]; \
   for (i = 1, relem = 1; i < rdims; i++) \
   {  relem *= rsize[i]; \
      rstrides[i] = param_rstrides[i] - rsize[i-1] * param_rstrides[i-1]; \
   } \
   \
   /* ---------------------------------------------------------- */ \
   /* Outer iteration                                            */ \
   /* ---------------------------------------------------------- */ \
   if (nelem > 0) \
   {  k1 = 0; \
      while (1) \
      {  /* Initialize the accumulation (can use _ptr, but only when relem > 0)) */ \
         _ptr = (SOLID_C_TYPE(TYPE) *)(ptr1); \
         CODE_INIT \
         \
         /* Inner iteration */ \
         if (relem > 0) \
         {  /* Initialize the indices */ \
            for (i = 1; i < rdims; i++) ridx[i] = 0; \
            \
            k2 = 0; \
            while (1) \
            {  /* Accumulate along the first dimension - increment the  */ \
               /* inner element index k2 to enable short-cut evaluation */ \
               /* by setting k2 to relem and breaking from the loop from*/ \
               /* the option SOLID_OP_REDUCE_BREAK statement in the     */ \
               /* CODE_ACCUMULATE macro.                                */ \
               k2 ++; \
               for (j = 0; j < s; j++) \
               {  CODE_ACCUMULATE \
                  _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + r); \
               } \
               if (k2 == relem) break; \
               \
               /* Update the higher-dimension inner indices */ \
               i = 1; \
               _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + rstrides[i]); \
               while (++ridx[i] == rsize[i]) \
               {  ridx[i] = 0; i++; \
                  _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + rstrides[i]); \
               } \
            } \
         } \
         \
         /* Finalize the result */ \
         SOLID_REDUCE_AXIS_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, _accumulate) \
         \
         /* Ouput the result */ \
         SOLID_ASSIGN(SOLID_WORKTYPE(RTYPE), RTYPE, &_accumulate, ptr2) \
         \
         /* Move to the next outer element */ \
         k1 ++; if (k1 == nelem) break; \
         \
         /* Update the higher-dimension outer indices */ \
         i = 0; \
         ptr1 = ptr1 + strides1[i]; \
         ptr2 = ptr2 + strides2[i]; \
         while (++idx[i] == size[i]) \
         {  idx[i] = 0; i++; \
            ptr1 = ptr1 + strides1[i]; \
            ptr2 = ptr2 + strides2[i]; \
         } \
      } \
   } \
}



/* --------------------------------------------------------------------- */
/* Macro: SOLID_REDUCE_AXIS_OMP1                                         */
/* The outer tensor is equally divided over all threads, each thread is  */
/* responsible for the full reduction for each element.                  */
/* --------------------------------------------------------------------- */

#define SOLID_REDUCE_AXIS_OMP1(NAME, TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
void SOLID_REDUCE_AXIS_FUNCTION_ITF(NAME, TYPE, omp1, FLAG_PARAM) \
{  int max_threads = solid_omp_get_max_threads(); \
   ptrdiff_t data_strides1, data_strides2, data_rstrides; \
   size_t data_size, data_rsize; \
   \
   /* ---------------------------------------------------------- */ \
   /* Make sure that the number of dimensions is at least one    */ \
   /* ---------------------------------------------------------- */ \
   if (ndims == 0) \
   {  ndims = 1; \
      data_size = 1; size = &(data_size ); \
      data_strides1 = 0; param_strides1 = &(data_strides1); \
      data_strides2 = 0; param_strides2 = &(data_strides2); \
   } \
   if (rdims == 0) \
   {  rdims = 1; \
      data_rsize = 1; rsize = &(data_rsize); \
      data_rstrides = 0; param_rstrides = &(data_rstrides); \
   } \
   \
   _Pragma("omp parallel num_threads(max_threads)") \
   {  SOLID_REDUCE_AXIS_CREATE_PARAM(NAME, TYPE, FLAG_PARAM); /* param */ \
      SOLID_C_WORKTYPE_TYPE(RTYPE) _accumulate; \
      SOLID_C_TYPE(TYPE)          *_ptr; \
      char      *ptr1 = (char *)(param_ptr1); \
      char      *ptr2 = (char *)(param_ptr2); \
      size_t     idx[SOLID_MAX_TENSOR_DIMS] = {0}; \
      size_t     ridx[SOLID_MAX_TENSOR_DIMS] = {0}; \
      ptrdiff_t  strides1[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t  strides2[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t  rstrides[SOLID_MAX_TENSOR_DIMS], r; \
      size_t     offset, nelem, relem, s, k1, k2, j; \
      int        rank = omp_get_thread_num(); \
      int        nthreads = omp_get_num_threads(); \
      int        i; \
      \
      /* ---------------------------------------------------------- */ \
      /* Determine the number of outer indices                      */ \
      /* ---------------------------------------------------------- */ \
      for (i = 0, nelem = 1; i < ndims; i++) nelem *= size[i]; \
      \
      /* ---------------------------------------------------------- */ \
      /* Compute the offset                                         */ \
      /* ---------------------------------------------------------- */ \
      offset = nelem % nthreads; nelem /= nthreads; \
      if (rank < offset) \
      {  nelem ++; \
         offset = rank * nelem; \
      } \
      else \
      {  offset += rank * nelem; \
      } \
      \
      /* ---------------------------------------------------------- */ \
      /* Initialize the data pointer and indices                    */ \
      /* ---------------------------------------------------------- */ \
      if (nelem > 0) \
      {  for (i = 0; i < ndims; i++) \
         {  idx[i] = offset % size[i]; \
            offset /= size[i]; \
            ptr1 += param_strides1[i] * idx[i]; \
            ptr2 += param_strides2[i] * idx[i]; \
         } \
      } \
      \
      /* ---------------------------------------------------------- */ \
      /* Update the strides                                         */ \
      /* ---------------------------------------------------------- */ \
      strides1[0] = param_strides1[0]; \
      strides2[0] = param_strides2[0]; \
      for (i = 1; i < ndims; i++) \
      {  strides1[i] = param_strides1[i] - size[i-1] * param_strides1[i-1]; \
         strides2[i] = param_strides2[i] - size[i-1] * param_strides2[i-1]; \
      } \
      \
      /* ---------------------------------------------------------- */ \
      /* Initialize the inner problem - exclude the first dimension */ \
      /* from the number of elements to determine the number of     */ \
      /* loops; the first dimension has an explicit loop.           */ \
      /* ---------------------------------------------------------- */ \
      s = rsize[0]; \
      r  = param_rstrides[0]; \
      for (i = 1, relem = 1; i < rdims; i++) \
      {  relem *= rsize[i]; \
         rstrides[i] = param_rstrides[i] - rsize[i-1] * param_rstrides[i-1]; \
      } \
      \
      /* Loop over the data */ \
      if (nelem > 0) \
      {  k1 = 0; \
         while (1) \
         {   /* Initialize the accumulation (can use _ptr, but only when relem > 0)) */ \
            _ptr = (SOLID_C_TYPE(TYPE) *)(ptr1); \
            CODE_INIT \
            \
            /* Inner iteration */ \
            if (relem > 0) \
            {  /* Initialize the indices */ \
               for (i = 1; i < rdims; i++) ridx[i] = 0; \
               \
               k2 = 0; \
               while (1) \
               {  /* Accumulate along the first dimension - increment the  */ \
                  /* inner element index k2 to enable short-cut evaluation */ \
                  /* by setting k2 to relem and breaking from the loop from*/ \
                  /* the option SOLID_OP_REDUCE_BREAK statement in the     */ \
                  /* CODE_ACCUMULATE macro.                                */ \
                  k2 ++; \
                  for (j = 0; j < s; j++) \
                  {  CODE_ACCUMULATE \
                     _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + r); \
                  } \
                  if (k2 >= relem) break; \
                  \
                  /* Update the higher-dimension inner indices */ \
                  i = 1; \
                  _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + rstrides[i]); \
                  while (++ridx[i] == rsize[i]) \
                  {  ridx[i] = 0; i++; \
                     _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + rstrides[i]); \
                  } \
               } \
            } \
            \
            /* Finalize the result */ \
            SOLID_REDUCE_AXIS_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, _accumulate) \
            \
            /* Ouput the result */ \
            SOLID_ASSIGN(SOLID_WORKTYPE(RTYPE), RTYPE, &_accumulate, ptr2) \
            \
            /* Move to the next outer element */ \
            k1 ++; if (k1 == nelem) break; \
            i = 0; \
            ptr1 += strides1[i]; \
            ptr2 += strides2[i]; \
            while (++idx[i] == size[i]) \
            {  idx[i] = 0; i++; \
               ptr1 += strides1[i]; \
               ptr2 += strides2[i]; \
            } \
         } \
      } \
   } \
}



/* --------------------------------------------------------------------- */
/* Macro: SOLID_REDUCE_AXIS_OMP2                                         */
/* For each element in the outer tensor, all threads are used to compute */
/* the reduction. This function can be used when the number of elements  */
/* in the reduction is large while the number of outer elements is small */
/* compared to the number of threads.                                    */
/* --------------------------------------------------------------------- */

#define SOLID_REDUCE_AXIS_OMP2(NAME, TYPE, RTYPE, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM) \
void SOLID_REDUCE_AXIS_FUNCTION_ITF(NAME, TYPE, omp2, FLAG_PARAM) \
{  SOLID_C_WORKTYPE_TYPE(RTYPE) _intermediate[SOLID_OMP_MAX_THREADS], *_result, *_partial; \
   int reduction_elements[SOLID_OMP_MAX_THREADS]; \
   int max_threads = solid_omp_get_max_threads(); \
   ptrdiff_t data_strides1, data_strides2, data_rstrides; \
   size_t data_size, data_rsize; \
   \
   /* ---------------------------------------------------------- */ \
   /* Make sure that the number of dimensions is at least one    */ \
   /* ---------------------------------------------------------- */ \
   if (ndims == 0) \
   {  ndims = 1; \
      data_size = 1; size = &(data_size ); \
      data_strides1 = 0; param_strides1 = &(data_strides1); \
      data_strides2 = 0; param_strides2 = &(data_strides2); \
   } \
   if (rdims == 0) \
   {  rdims = 1; \
      data_rsize = 1; rsize = &(data_rsize); \
      data_rstrides = 0; param_rstrides = &(data_rstrides); \
   } \
   \
   _Pragma("omp parallel num_threads(max_threads)") \
   {  SOLID_REDUCE_AXIS_CREATE_PARAM(NAME, TYPE, FLAG_PARAM); /* param */ \
      SOLID_C_WORKTYPE_TYPE(RTYPE) _accumulate; \
      SOLID_C_TYPE(TYPE)          *_ptr; \
      char      *ptr1 = (char *)(param_ptr1); \
      char      *ptr2 = (char *)(param_ptr2); \
      size_t     idx[SOLID_MAX_TENSOR_DIMS] = {0}; \
      size_t     ridx[SOLID_MAX_TENSOR_DIMS] = {0}; \
      size_t     base_idx[SOLID_MAX_TENSOR_DIMS] = {0}; \
      ptrdiff_t  ptr_offset = 0; \
      ptrdiff_t  strides1[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t  strides2[SOLID_MAX_TENSOR_DIMS]; \
      ptrdiff_t  rstrides[SOLID_MAX_TENSOR_DIMS], r; \
      size_t     offset, nelem, relem, s, k1, k2, j, n; \
      int        rank = omp_get_thread_num(); \
      int        nthreads = omp_get_num_threads(); \
      int        i; \
      \
      /* ---------------------------------------------------------- */ \
      /* Determine the outer strides and number of elements         */ \
      /* ---------------------------------------------------------- */ \
      nelem = size[0]; \
      strides1[0] = param_strides1[0]; \
      strides2[0] = param_strides2[0]; \
      for (i = 1; i < ndims; i++)\
      {  nelem *= size[i]; \
         strides1[i] = param_strides1[i] - size[i-1] * param_strides1[i-1]; \
         strides2[i] = param_strides2[i] - size[i-1] * param_strides2[i-1]; \
      } \
      \
      /* ---------------------------------------------------------- */ \
      /* Determine the outer strides and number of elements         */ \
      /* ---------------------------------------------------------- */ \
      relem = s = rsize[0]; \
      r = param_rstrides[0]; \
      for (i = 1; i < rdims; i++) \
      {  relem *= rsize[i]; \
         rstrides[i] = param_rstrides[i] - rsize[i-1] * param_rstrides[i-1]; \
      } \
      \
      /* ---------------------------------------------------------- */ \
      /* Compute the thread offset and number of reduction elements */ \
      /* ---------------------------------------------------------- */ \
      offset = relem % nthreads; relem /= nthreads; \
      if (rank < offset) \
      {  relem ++; \
         offset = rank * relem; \
      } \
      else \
      {  offset += rank * relem; \
      } \
      \
      /* ---------------------------------------------------------- */ \
      /* Initialize the data pointer offset and base indices        */ \
      /* ---------------------------------------------------------- */ \
      ptr_offset = 0; \
      if (relem > 0) \
      {  for (i = 0; i < rdims; i++) \
         {  base_idx[i] = offset % rsize[i]; \
            offset /= rsize[i]; \
            ptr_offset += param_rstrides[i] * base_idx[i]; \
         } \
      } \
      reduction_elements[rank] = relem; \
      \
      /* ---------------------------------------------------------- */ \
      /* Loop over the outer elements                               */ \
      /* ---------------------------------------------------------- */ \
      if (nelem > 0) \
      {  k1 = 0; \
         while (1) \
         {  /* === Perform a single reduction === */ \
            \
            /* Initialize the accumulator */ \
            _ptr = (SOLID_C_TYPE(TYPE) *)(ptr1 + ptr_offset); \
            if ((relem > 0) || (rank == 0)) \
            {  CODE_INIT \
               if (relem == 0)_intermediate[rank] = _accumulate; \
            } \
            \
            /* Loop over the data */ \
            if (relem > 0) \
            {  /* Initialize the indices */ \
               for (i = 0; i < rdims; i++) ridx[i] = base_idx[i]; \
               \
               k2 = 0; \
               while (1) \
               {  /* Accumulate along the first dimension - update the     */ \
                  /* inner element index k2 to enable short-cut evaluation */ \
                  /* by setting k2 to relem and breaking from the loop from*/ \
                  /* the option SOLID_OP_REDUCE_BREAK statement in the     */ \
                  /* CODE_ACCUMULATE macro.                                */ \
                  n = s - ridx[0]; if (k2 + n > relem) n = relem - k2; \
                  k2 += n ; \
                  for (j = 0; j < n; j++) \
                  {  CODE_ACCUMULATE \
                     _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + r); \
                  } \
                  if (k2 >= relem) break; \
                  \
                  i = 1; \
                  _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + rstrides[1]); \
                  while (++ridx[i] == rsize[i]) \
                  {  ridx[i] = 0; i++; \
                     _ptr = (SOLID_C_TYPE(TYPE) *)(((char *)_ptr) + rstrides[i]); \
                  } \
               } \
               \
               /* Output the intermediate result */ \
               _intermediate[rank] = _accumulate; \
            } \
            \
            /* Finalize the reduction element */ \
            _Pragma("omp barrier") \
            if (rank == 0) \
            {  _result = &(_intermediate[0]); \
               for (i = 1; i < nthreads; i++) \
               {  if (reduction_elements[i] == 0) continue; \
                  _partial = &(_intermediate[i]); \
                  CODE_REDUCE \
               } \
               \
               /* Finalize the result */ \
               SOLID_REDUCE_AXIS_FINALIZE(RTYPE, FLAG_FINALIZE, CODE_FINALIZE, _intermediate[0]) \
               \
               /* Ouput the result */ \
               SOLID_ASSIGN(SOLID_WORKTYPE(RTYPE), RTYPE, _result, ptr2) \
            } \
            _Pragma("omp barrier") \
            \
            /* Move to the next outer element */ \
            k1 ++; if (k1 == nelem) break; \
            i = 0; \
            ptr1 += strides1[i]; \
            ptr2 += strides2[i]; \
            while (++idx[i] == size[i]) \
            {  idx[i] = 0; i++; \
               ptr1 += strides1[i]; \
               ptr2 += strides2[i]; \
            } \
         } \
      } \
   } \
}

#endif
