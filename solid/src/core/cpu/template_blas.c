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
#define SD_TEMPLATE_FILE "core/cpu/template_blas.c"

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/cpu/generate_macros.h"
#include "solid/base/cpu/dtype_cpu.h"
#include <stddef.h>

#include "solid/base/generic/generate_all_types.h"
#else

#if SDTYPE_IS_INT(SDXTYPE)
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(gemm)(char transA, char transB, size_t M, size_t N, size_t K,
                         void *ptrAlpha, void *ptrA, ptrdiff_t ldA, void *ptrB, ptrdiff_t ldB,
                         void *ptrBeta,  void *ptrC, ptrdiff_t ldC)
/* -------------------------------------------------------------------- */
{  SOLID_C_TYPE(SDXTYPE) alpha, beta, s;
   SOLID_C_TYPE(SDXTYPE) *A, *B, *C;
   size_t i,j,k;

   alpha = (ptrAlpha) ? *((SOLID_C_TYPE(SDXTYPE) *)ptrAlpha) : 1;
   beta  = (ptrBeta)  ? *((SOLID_C_TYPE(SDXTYPE) *)ptrBeta)  : 0;
   A     = (SOLID_C_TYPE(SDXTYPE) *)ptrA;
   B     = (SOLID_C_TYPE(SDXTYPE) *)ptrB;
   C     = (SOLID_C_TYPE(SDXTYPE) *)ptrC;

   /* Check the modes */
   if ((transA == 'n') || (transA == 'N')) transA = 'N';
   else if ((transA == 't') || (transA == 'T')) transA = 'T';
   else SOLID_ERROR(-1, "Invalid mode for matrix A in gemm");

   if ((transB == 'n') || (transB == 'N')) transB = 'N';
   else if ((transB == 't') || (transB == 'T')) transB = 'T';
   else SOLID_ERROR(-1, "Invalid mode for matrix B in gemm");

   /* Special case: identity operation */
   if ((alpha == 0) && (beta == 1)) return 0;

   /* Special case: scale output */
   if (alpha == 0)
   {  if (beta == 0)
      {  for (i = 0; i < N; i++)
         {  for (j = 0; j < M; j++)
            {  C[j] = 0;
            }
            C += ldC;
         }
      }
      else
      {  for (i = 0; i < N; i++)
         {  for (j = 0; j < M; j++)
            {  C[j] *= beta;
            }
            C += ldC;
         }
      }
      return 0;
   }

   if (transA == 'N')
   {  if (transB == 'N')
      {  /* ---------------------------------- */
         /*  Case 1 - Normal A, normal B       */
         /* ---------------------------------- */
         for (i = 0; i < N; i++)
         {  /* Scale C(i,:) */
            if (beta == 0)
            {  for (j = 0; j < M; j++) C[j] = 0;
            }
            else if (beta != 1)
            {  for (j = 0; j < M; j++) C[j] *= beta;
            }

            /* Update C(i,:) += alpha * sum_k B(k,i) * A(:,k) */ 
            for (k = 0; k < K; k++)
            {  s = alpha * B[k];
               for (j = 0; j < M; j++)
               {  C[j] += s * A[j];
               }
               A += ldA;
            } 
            A -= K * ldA;
            B += ldB;
            C += ldC;
         }
      }
      else
      {  /* ---------------------------------- */
         /*  Case 2 - Normal A, tranpose B     */
         /* ---------------------------------- */
         for (i = 0; i < N; i++)
         {  /* Scale C(i,:) */
            if (beta == 0)
            {  for (j = 0; j < M; j++) C[j] = 0;
            }
            else if (beta != 1)
            {  for (j = 0; j < M; j++) C[j] *= beta;
            }

            /* Update C(i,:) += alpha * sum_k B(k,i) * A(:,k) */ 
            for (k = 0; k < K; k++)
            {  s = alpha * B[k*ldB];
               for (j = 0; j < M; j++)
               {  C[j] += s * A[j];
               }
               A += ldA;
            } 
            A -= K * ldA;
            B ++;
            C += ldC;
         }
      }
   }
   else
   {  if (transB == 'N')
      {  /* ---------------------------------- */
         /* Case 3 - Transpose A, normal B     */
         /* ---------------------------------- */
         for (i = 0; i < N; i++)
         {  for (j = 0; j < M; j++)
            {  s = 0;
               for (k = 0; k < K; k++)
               {  s += A[k] * B[k];
               }
               C[j] = beta * C[j] + alpha * s;
               A += ldA;
            }
            A -= M * ldA;
            B += ldB;
            C += ldC;
         }
      }
      else
      {  /* ---------------------------------- */
         /* Case 4 - Transpose A, transpose B  */
         /* ---------------------------------- */
         for (i = 0; i < N; i++)
         {  for (j = 0; j < M; j++)
            {  s = 0;
               for (k = 0; k < K; k++)
               {  s += A[k] * B[k*ldB];
               }
               C[j] = beta * C[j] + alpha * s;
               A += ldA;
            }
            A -= M * ldA;
            B ++;
            C += ldC;
         }
      }
   }

   return 0;
}
#endif


#if SDTYPE_IS_BOOL(SDXTYPE)
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(gemm)(char transA, char transB, size_t M, size_t N, size_t K,
                         void *ptrAlpha, void *ptrA, ptrdiff_t ldA, void *ptrB, ptrdiff_t ldB,
                         void *ptrBeta,  void *ptrC, ptrdiff_t ldC)
/* -------------------------------------------------------------------- */
{  SOLID_C_TYPE(SDXTYPE) alpha, beta, s;
   SOLID_C_TYPE(SDXTYPE) *A, *B, *C;
   size_t i,j,k;

   /* ----------------------------------------------------------------- */
   /* In the Boolean mode the entries in all tensors are either 0 or 1. */
   /* We take advantage of this, and also ensure that the output of the */
   /* multiplication results in 0 or 1 entries.                         */
   /* ----------------------------------------------------------------- */
   alpha = (ptrAlpha) ? *((SOLID_C_TYPE(SDXTYPE) *)ptrAlpha) : 1;
   beta  = (ptrBeta)  ? *((SOLID_C_TYPE(SDXTYPE) *)ptrBeta)  : 0;
   A     = (SOLID_C_TYPE(SDXTYPE) *)ptrA;
   B     = (SOLID_C_TYPE(SDXTYPE) *)ptrB;
   C     = (SOLID_C_TYPE(SDXTYPE) *)ptrC;

   /* Check the modes */
   if ((transA == 'n') || (transA == 'N')) transA = 'N';
   else if ((transA == 't') || (transA == 'T')) transA = 'T';
   else SOLID_ERROR(-1, "Invalid mode for matrix A in gemm");

   if ((transB == 'n') || (transB == 'N')) transB = 'N';
   else if ((transB == 't') || (transB == 'T')) transB = 'T';
   else SOLID_ERROR(-1, "Invalid mode for matrix B in gemm");

   /* Special case: scale output */
   if (alpha == 0)
   {  if (beta == 0)
      {  for (i = 0; i < N; i++)
         {  for (j = 0; j < M; j++)
            {  C[j] = 0;
            }
            C += ldC;
         }
      }
      else
      {  /* Beta is 1 - do nothing */
      }
      return 0;
   }

   if (transA == 'N')
   {  if (transB == 'N')
      {  /* ---------------------------------- */
         /*  Case 1 - Normal A, normal B       */
         /* ---------------------------------- */
         for (i = 0; i < N; i++)
         {  /* Scale C(i,:) */
            if (beta == 0)
            {  for (j = 0; j < M; j++) C[j] = 0;
            }

            /* Update C(i,:) += 1 * sum_k B(k,i) * A(:,k) */ 
            for (k = 0; k < K; k++)
            {  s = 1 * B[k];
               for (j = 0; j < M; j++)
               {  C[j] |= (s & A[j]);
               }
               A += ldA;
            } 
            A -= K * ldA;
            B += ldB;
            C += ldC;
         }
      }
      else
      {  /* ---------------------------------- */
         /*  Case 2 - Normal A, tranpose B     */
         /* ---------------------------------- */
         for (i = 0; i < N; i++)
         {  /* Scale C(i,:) */
            if (beta == 0)
            {  for (j = 0; j < M; j++) C[j] = 0;
            }

            /* Update C(i,:) += 1 * sum_k B(k,i) * A(:,k) */ 
            for (k = 0; k < K; k++)
            {  s = 1 * B[k*ldB];
               for (j = 0; j < M; j++)
               {  C[j] |= (s & A[j]);
               }
               A += ldA;
            } 
            A -= K * ldA;
            B ++;
            C += ldC;
         }
      }
   }
   else
   {  if (transB == 'N')
      {  /* ---------------------------------- */
         /* Case 3 - Transpose A, normal B     */
         /* ---------------------------------- */
         for (i = 0; i < N; i++)
         {  for (j = 0; j < M; j++)
            {  s = 0;
               for (k = 0; k < K; k++)
               {  s |= (A[k] & B[k]);
               }
               C[j] = (beta & C[j]) | (1 * s);
               A += ldA;
            }
            A -= M * ldA;
            B += ldB;
            C += ldC;
         }
      }
      else
      {  /* ---------------------------------- */
         /* Case 4 - Transpose A, transpose B  */
         /* ---------------------------------- */
         for (i = 0; i < N; i++)
         {  for (j = 0; j < M; j++)
            {  s = 0;
               for (k = 0; k < K; k++)
               {  s |= (A[k] & B[k*ldB]);
               }
               C[j] = (beta & C[j]) | (1 * s);
               A += ldA;
            }
            A -= M * ldA;
            B ++;
            C += ldC;
         }
      }
   }

   return 0;
}
#endif

#endif


