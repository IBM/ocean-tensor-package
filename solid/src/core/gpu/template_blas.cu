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

#ifndef SD_TEMPLATE_FILE
#define SD_TEMPLATE_FILE "core/gpu/template_blas.cu"

#include "solid.h"
#include "solid_gpu.h"
#include "solid_core_gpu.h"
#include "solid/base/generic/dtype_macros.h"
#include "solid/base/gpu/generate_macros.h"
#include "solid/base/gpu/dtype_gpu.h"


/* -------------------------------------------------------------------- */
SOLID_API int solid_gpu_gemm_create_plan(char transA, char transB,
                                         size_t M, size_t N, size_t K,
                                         int elemsize, int device,
                                         solid_gpu_gemm_plan *plan)
/* -------------------------------------------------------------------- */
{  solid_gpu_properties *prop;
   int tilesize;
   int memsize;

   /* Get the current device properties */
   prop = solid_gpu_get_current_device_properties();
   if (prop == NULL) return -1;

   /* Initialize the basic parameters */
   plan -> transA   = transA;
   plan -> transB   = transB;
   plan -> M        = M;
   plan -> N        = N;
   plan -> K        = K;

   /* Determine the tile size */
   for (tilesize = 32; tilesize > 0; tilesize /= 2)
   {  memsize = tilesize * tilesize * 2 * elemsize;
      if ((memsize * (prop -> max_blocks_per_multiprocessor) <= (prop -> max_shared_mem_per_multiprocessor)) &&
          (memsize <= (prop -> max_shared_mem_per_threadblock)) &&
          (tilesize * tilesize <= (prop -> max_threads_per_block)))
         break;
   }

   /* Make sure the tile size is supported */   
   if (tilesize < 8) SOLID_ERROR(-1, "Insufficient GPU capability");

   /* Determine the kernel parameters */
   plan -> gridsize.x  = ((M + tilesize - 1) / tilesize);
   plan -> gridsize.y  = ((N + tilesize - 1) / tilesize);
   plan -> gridsize.z  = 1;
   plan -> blocksize.x = tilesize;
   plan -> blocksize.y = tilesize;
   plan -> blocksize.z = 1;
   plan -> tilesize    = tilesize;

   return 0;
}


/* Import templated code */
#include "solid/base/generic/generate_all_types.h"
#else


#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
/* -------------------------------------------------------------------- */
/* void SOLID_FUNCTION(gemm_NN)(parameters)                             */
/* void SOLID_FUNCTION(gemm_NT)(parameters)                             */
/* void SOLID_FUNCTION(gemm_TN)(parameters)                             */
/* void SOLID_FUNCTION(gemm_TT)(parameters)                             */
/* -------------------------------------------------------------------- */

#define SD_INDEX_N(ROW,COL,LD)       ((ROW) + (COL) * LD)
#define SD_INDEX_T(ROW,COL,LD)       ((COL) + (ROW) * LD)
#define SD_INDEX(ROW,COL,LD,TRANS)   SD_INDEX_##TRANS(ROW,COL,LD)

#if SDTYPE_IS_BOOL(SDXTYPE)
   #define SD_PROD(A,B)     A &= B
   #define SD_ADDMUL(A,B,C) A |= (B) & (C)
#else
   #define SD_PROD(A,B)     A *= B;
   #define SD_ADDMUL(A,B,C) A += (B) * (C)
#endif

#define SD_TEMPLATE(TRANSA,TRANSB,TILESIZE)\
__global__ \
void SOLID_FUNCTION(gemm_## TRANSA ## TRANSB ##_##TILESIZE)(size_t M, size_t N, size_t K, \
                    SOLID_C_TYPE(SDXTYPE) alpha, SOLID_C_TYPE(SDXTYPE) *ptrAlpha, \
                    SOLID_C_TYPE(SDXTYPE) beta,  SOLID_C_TYPE(SDXTYPE) *ptrBeta, \
                    SOLID_C_TYPE(SDXTYPE) *ptrA, ptrdiff_t ldA, \
                    SOLID_C_TYPE(SDXTYPE) *ptrB, ptrdiff_t ldB, \
                    SOLID_C_TYPE(SDXTYPE) *ptrC, ptrdiff_t ldC) \
{ \
   __shared__ SOLID_C_TYPE(SDXTYPE) A[TILESIZE][TILESIZE]; \
   __shared__ SOLID_C_TYPE(SDXTYPE) B[TILESIZE][TILESIZE]; \
   SOLID_C_TYPE(SDXTYPE) accumulator = 0; \
   int row, col, offset, k; \
   int i = threadIdx.x; \
   int j = threadIdx.y; \
   \
   /* Override parameter if device pointer is given */ \
   if (ptrAlpha) alpha = *ptrAlpha; \
   if (ptrBeta)  beta  = *ptrBeta; \
   \
   row = blockIdx.x * TILESIZE + i; \
   col = blockIdx.y * TILESIZE + j; \
   \
   \
   if (alpha != 0) \
   for (offset = 0; offset < K; offset += TILESIZE) \
   { \
      /* Load the local data - doubling the tile size in one dimension and      */ \
      /* loading two elements of A and B per thread give a slight performance   */ \
      /* improvement, but not enough to warrant doubling the number of kernels. */ \
      if ((row < M) && (offset + j < K)) A[j][i] = ptrA[SD_INDEX(row, (offset + j), ldA, TRANSA)]; else A[j][i] = 0; \
      if ((col < N) && (offset + i < K)) B[i][j] = ptrB[SD_INDEX((offset + i), col, ldB, TRANSB)]; else B[i][j] = 0;\
      __syncthreads(); \
      \
      if ((row < M ) && (col < N)) \
      {  for (k = 0; k < TILESIZE; k++) \
         {  SD_ADDMUL(accumulator, A[k][i], B[k][j]); \
         } \
      } \
      __syncthreads(); \
   } \
   \
   if ((row < M) && (col < N)) \
   {  SD_PROD(accumulator, alpha); \
      SD_ADDMUL(accumulator, beta, ptrC[row + col * ldC]); \
      ptrC[row + col * ldC] = accumulator; \
   } \
}

/* Generate functions with multiple tile sizes */
SD_TEMPLATE(N, N, 8)
SD_TEMPLATE(N, T, 8)
SD_TEMPLATE(T, N, 8)
SD_TEMPLATE(T, T, 8)
SD_TEMPLATE(N, N, 16)
SD_TEMPLATE(N, T, 16)
SD_TEMPLATE(T, N, 16)
SD_TEMPLATE(T, T, 16)
SD_TEMPLATE(N, N, 32)
SD_TEMPLATE(N, T, 32)
SD_TEMPLATE(T, N, 32)
SD_TEMPLATE(T, T, 32)

#undef SD_PROD
#undef SD_ADDMUL
#undef SD_INDEX_N
#undef SD_INDEX_T
#undef SD_INDEX
#undef SD_TEMPLATE
#endif



#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(gemm_with_plan)(solid_gpu_gemm_plan *plan,
                                             void *ptrAlpha, void *ptrA, ptrdiff_t ldA,
                                                             void *ptrB, ptrdiff_t ldB,
                                             void *ptrBeta,  void *ptrC, ptrdiff_t ldC,
                                             int scalarModeHost, cudaStream_t stream)
/* -------------------------------------------------------------------- */
{  void (*funptr)(size_t M, size_t N, size_t K,
                  SOLID_C_TYPE(SDXTYPE) alpha, SOLID_C_TYPE(SDXTYPE) *ptrAlpha,
                  SOLID_C_TYPE(SDXTYPE) beta,  SOLID_C_TYPE(SDXTYPE) *ptrBeta,
                  SOLID_C_TYPE(SDXTYPE) *ptrA, ptrdiff_t ldA,
                  SOLID_C_TYPE(SDXTYPE) *ptrB, ptrdiff_t ldB,
                  SOLID_C_TYPE(SDXTYPE) *ptrC, ptrdiff_t ldC) = 0;
   SOLID_C_TYPE(SDXTYPE) alpha, beta;
   int tilesize = plan -> tilesize;

   if (scalarModeHost)
   {  alpha = *((SOLID_C_TYPE(SDXTYPE) *)ptrAlpha); ptrAlpha = NULL;
      beta  = *((SOLID_C_TYPE(SDXTYPE) *)ptrBeta);  ptrBeta  = NULL;
   } 
   else
   {  alpha = 0;
      beta  = 0;
   }

   /* Get the function pointer */
   if (plan -> transA == 'N')
   {  if (plan -> transB == 'N')
      {  if (tilesize == 8)       funptr = SOLID_FUNCTION(gemm_NN_8);
         else if (tilesize == 16) funptr = SOLID_FUNCTION(gemm_NN_16);
         else if (tilesize == 32) funptr = SOLID_FUNCTION(gemm_NN_32);
      }
      else
      {  if (tilesize == 8)       funptr = SOLID_FUNCTION(gemm_NT_8);
         else if (tilesize == 16) funptr = SOLID_FUNCTION(gemm_NT_16);
         else if (tilesize == 32) funptr = SOLID_FUNCTION(gemm_NT_32);
      }
   }
   else
   {  if (plan -> transB == 'N')
      {  if (tilesize == 8)       funptr = SOLID_FUNCTION(gemm_TN_8);
         else if (tilesize == 16) funptr = SOLID_FUNCTION(gemm_TN_16);
         else if (tilesize == 32) funptr = SOLID_FUNCTION(gemm_TN_32);
      }
      else
      {  if (tilesize == 8)       funptr = SOLID_FUNCTION(gemm_TT_8);
         else if (tilesize == 16) funptr = SOLID_FUNCTION(gemm_TT_16);
         else if (tilesize == 32) funptr = SOLID_FUNCTION(gemm_TT_32);
      }
   }
   if (funptr == 0)
      SOLID_ERROR(0, "Internal error: unsupported parameters in gemm");

   /* Launch the kernel */
   funptr<<<plan -> gridsize, plan -> blocksize, plan -> sharedmem, stream>>>\
         (plan -> M, plan -> N, plan -> K,
          alpha, (SOLID_C_TYPE(SDXTYPE) *)ptrAlpha,
          beta, (SOLID_C_TYPE(SDXTYPE) *)ptrBeta,
          (SOLID_C_TYPE(SDXTYPE) *)ptrA, ldA,\
          (SOLID_C_TYPE(SDXTYPE) *)ptrB, ldB,\
          (SOLID_C_TYPE(SDXTYPE) *)ptrC, ldC);

   /* Check status */
   return solid_gpu_check_status();
}
#endif



#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(gemm)(char transA, char transB, size_t M, size_t N, size_t K,
                                   void *ptrAlpha, void *ptrA, ptrdiff_t ldA, void *ptrB, ptrdiff_t ldB,
                                   void *ptrBeta, void *ptrC, ptrdiff_t ldC,
                                   int scalarModeHost, cudaStream_t stream)
/* -------------------------------------------------------------------- */
{  solid_gpu_gemm_plan plan;
   int device;

   if (solid_gpu_get_current_device(&device) != 0) return -1;
   if (solid_gpu_gemm_create_plan(transA, transB, M, N, K, SDTYPE_SIZE(SDXTYPE), device, &plan) != 0) return -1;

   return SOLID_FUNCTION(gemm_with_plan)(&plan, ptrAlpha, ptrA, ldA, ptrB, ldB, ptrBeta, ptrC, ldC, scalarModeHost, stream);
}
#endif

#endif
