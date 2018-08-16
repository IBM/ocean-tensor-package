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

#include "ocean/core/interface/tensor_itf.h"
#include "ocean/core/cpu/op/tensor_blas_cpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"

/* Blas library */
#include "ocean/external/ocean-blas/ocean_blas.h"

#include <stdio.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorCPU_gemm(OcSize M, OcSize N, OcSize K, char transA, char transB,
                            OcTensor *alpha, OcTensor *A, OcIndex ldA,
                                             OcTensor *B, OcIndex ldB,
                            OcTensor *beta,  OcTensor *C, OcIndex ldC);

OC_API int OCTensorCPU_gemmSupportedOn(OcDevice *device, OcDType dtype);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeBlasOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Blas operations */
   module -> Tensor_gemm            = OcTensorCPU_gemm;
   module -> Tensor_gemmSupportedOn = OCTensorCPU_gemmSupportedOn;
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OC_API int OcTensorCPU_gemm(OcSize M, OcSize N, OcSize K, char transA, char transB,
                            OcTensor *alpha, OcTensor *A, OcIndex ldA,
                                             OcTensor *B, OcIndex ldB,
                            OcTensor *beta,  OcTensor *C, OcIndex ldC)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_gemm funptr = 0;
   OcBlas_Transpose modeA = 0, modeB = 0;
   OcIndex  index[OC_TENSOR_MAX_DIMS];
   OcSize  *size = A -> size;
   OcDType  dtype = A -> dtype;
   char    *ptrAlpha, *ptrBeta;
   char    *ptrA, *ptrB, *ptrC;
   int      ndims = A -> ndims;
   int      flagReal;
   int      i, result;

   /* ---------------------------------------------------------------- */
   /* Call guarantees: all tensors have the same number of dimensions  */
   /* (at least 2) and the sizes are all compatible. The data types of */
   /* all tensors are the same and their data is memory aligned and in */
   /* native byte order. All tensors are are on the CPU.               */
   /* ---------------------------------------------------------------- */

   /* Initialize the higher-dimensional loop indices */
   for (i = 2; i < ndims; i++) index[i] = 0;

   /* Get the tensor information */
   ptrAlpha  = OcTensor_data(alpha);
   ptrBeta   = OcTensor_data(beta);
   ptrA      = OcTensor_data(A);
   ptrB      = OcTensor_data(B);
   ptrC      = OcTensor_data(C);

   /* Special cases: empty result or empty K */
   if (C -> nelem == 0) return 0;
   if (K == 0) { ldA = N; ldB = M; }

   /* Look up the function pointer */ 
   if (!OcDType_isFloat(dtype))
   {  OC_SOLID_FUNPTR("gemm", solid_cpu_gemm, funptr, dtype, "CPU");
      if (funptr == 0) return -1;
   }
   else
   {  /* Convert the modes to OcBlas format */
      if (transA == 'N')
           modeA = OcBlasNoTrans;
      else modeA = (transA == 'T') ? OcBlasTrans : OcBlasConjTrans;

      if (transB == 'N')
           modeB = OcBlasNoTrans;
      else modeB = (transB == 'T') ? OcBlasTrans : OcBlasConjTrans;
   }

   /* Make sure the data type is supported */
   flagReal = OcDType_isReal(dtype);
   if (OcDType_isFloat(dtype)   &&
       (dtype != OcDTypeFloat)  && (dtype != OcDTypeDouble) &&
       (dtype != OcDTypeCFloat) && (dtype != OcDTypeCDouble)) 
   {  OcError(-1, "Unsupported data type in function cpu gemm (%s)", OcDType_name(dtype)); 
   }

   /* Loop over all GEMM operations */
   while (1)
   {
      /* Compute a single GEMM operation */
      if (funptr)
      {  result = funptr(transA, transB, M, N, K, (void *)ptrAlpha, (void *)ptrA, ldA,
                                                                    (void *)ptrB, ldB,
                                                  (void *)ptrBeta,  (void *)ptrC, ldC);
         if (result != 0) { OC_SOLID_ERRMSG(); return result; }
      }
      else if (flagReal)
      {  /* Real floating point */
         if (dtype == OcDTypeFloat)
         {  ocblas_sgemm(OcBlasColMajor, modeA, modeB, M, N, K,
                         *((float *)ptrAlpha), (float *)ptrA, ldA, (float *)ptrB, ldB,
                         *((float *)ptrBeta),  (float *)ptrC, ldC);
         }
         else /* (dtype == OcDTypeDouble) */
         {  ocblas_dgemm(OcBlasColMajor, modeA, modeB, M, N, K,
                         *((double *)ptrAlpha), (double *)ptrA, ldA, (double *)ptrB, ldB,
                         *((double *)ptrBeta),  (double *)ptrC, ldC);
         }
      }
      else
      {  /* Complex floating point */
         if (dtype == OcDTypeCFloat)
         {  ocblas_cgemm(OcBlasColMajor, modeA, modeB, M, N, K,
                         ptrAlpha, ptrA, ldA, ptrB, ldB, ptrBeta, ptrC, ldC);
         }
         else /* (dtype == OcDTypeCDouble) */
         {  ocblas_zgemm(OcBlasColMajor, modeA, modeB, M, N, K,
                         ptrAlpha, ptrA, ldA, ptrB, ldB, ptrBeta, ptrC, ldC);
         }
      }

      /* Proceed to the next index */
      for (i = 2; i < ndims; i++)
      {  index[i] ++;
         if (i < alpha -> ndims) ptrAlpha += alpha -> strides[i];
         if (i < beta  -> ndims) ptrBeta  += beta -> strides[i];
         ptrA += A -> strides[i];
         ptrB += B -> strides[i];
         ptrC += C -> strides[i];

         if (index[i] < size[i]) break;

         if (i < alpha -> ndims) ptrAlpha -= size[i] * (alpha -> strides[i]);
         if (i < beta -> ndims) ptrBeta  -= size[i] * (beta  -> strides[i]);
         ptrA -= size[i] * (A -> strides[i]);
         ptrB -= size[i] * (B -> strides[i]);
         ptrC -= size[i] * (C -> strides[i]);
      }
      if (i >= ndims) break;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OCTensorCPU_gemmSupportedOn(OcDevice *device, OcDType dtype)
/* -------------------------------------------------------------------- */
{
   if ((dtype == OcDTypeHalf) || (dtype == OcDTypeCHalf))
        return 0;
   else return 1;
}
