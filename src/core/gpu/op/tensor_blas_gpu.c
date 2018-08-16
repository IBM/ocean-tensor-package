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
#include "ocean/core/gpu/op/tensor_blas_gpu.h"
#include "ocean/core/gpu/module_core_gpu.h"
#include "ocean/core/gpu/tensor_gpu.h"
#include "ocean/core/cpu/device_cpu.h"
#include "ocean/base/error.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_gpu.h"

#include "cublas_v2.h"
#include <stdio.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorGPU_gemm(OcSize M, OcSize N, OcSize K, char transA, char transB,
                            OcTensor *alpha, OcTensor *A, OcIndex ldA,
                                             OcTensor *B, OcIndex ldB,
                            OcTensor *beta,  OcTensor *C, OcIndex ldC);

OC_API int OCTensorGPU_gemmSupportedOn(OcDevice *device, OcDType dtype);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorGPU_initializeBlasOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Blas operations */
   module -> Tensor_gemm            = OcTensorGPU_gemm;
   module -> Tensor_gemmSupportedOn = OCTensorGPU_gemmSupportedOn;
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OC_API int OcTensorGPU_gemm(OcSize M, OcSize N, OcSize K, char transA, char transB,
                            OcTensor *alpha, OcTensor *A, OcIndex ldA,
                                             OcTensor *B, OcIndex ldB,
                            OcTensor *beta,  OcTensor *C, OcIndex ldC)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_gemm_with_plan funptr = 0;
   solid_gpu_gemm_plan  plan;
   cublasHandle_t       handle;
   cublasStatus_t       status;
   cublasOperation_t    opA = 0, opB = 0;
   cublasPointerMode_t  ptrMode;
   cudaStream_t         stream;

   OcIndex   index[OC_TENSOR_MAX_DIMS];
   OcDevice *device = A -> device;
   OcSize   *size = A -> size;
   OcDType   dtype = A -> dtype;
   char     *ptrAlpha, *ptrBeta;
   char     *ptrA, *ptrB, *ptrC;
   int       ndims = A -> ndims;
   int       flagScalarHost;
   int       flagReal, flag3m = 0;
   int       i, result;

   /* ---------------------------------------------------------------- */
   /* Call guarantees: all tensors have the same number of dimensions  */
   /* (at least 2) and the sizes are all compatible. The data types of */
   /* all tensors are the same and their data is memory aligned and in */
   /* native byte order. Tensors A, B, and C are on the same device    */
   /* and alpha and beta are either both on that device too or both on */
   /* the CPU.                                                         */
   /* ---------------------------------------------------------------- */

   /* Special cases: empty result or empty K */
   if (C -> nelem == 0) return 0;
   if (K == 0) { ldA = N; ldB = M; }

   /* Actvate the device */
   if (OcCuda_setDevice(device -> index) != 0) return -1;

   /* Get the cuBLAS handle for the current device */
   result = OcModuleCoreGPU_cublasHandle(device, &handle);
   if (result != 0) return -1;

   /* Set the scalar mode */
   flagScalarHost = (alpha -> device == OcCPU);

   /* Initialize the higher-dimensional loop indices */
   for (i = 2; i < ndims; i++) index[i] = 0;

   /* Get the tensor information */
   ptrAlpha  = OcTensor_data(alpha);
   ptrBeta   = OcTensor_data(beta);
   ptrA      = OcTensor_data(A);
   ptrB      = OcTensor_data(B);
   ptrC      = OcTensor_data(C);
   stream    = OcTensorGPU_cudaStream(A);

   /* Look up the function pointer */ 
   if (!OcDType_isFloat(dtype))
   {  OC_SOLID_FUNPTR("gemm_with_plan", solid_gpu_gemm_with_plan, funptr, dtype, "GPU");
      if (funptr == 0) return -1;

      /* Create a plan */
      result = solid_gpu_gemm_create_plan(transA, transB, M, N, K, OcDType_size(dtype), \
                                          device -> index, &plan);
      if (result != 0) { OC_SOLID_ERRMSG(); return result; }
   }
   else
   {  /* Convert the modes to CBlas format */
      if (transA == 'N')
           opA = CUBLAS_OP_N;
      else opA = (transA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C;

      if (transB == 'N')
           opB = CUBLAS_OP_N;
      else opB = (transB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C;

      if (flagScalarHost)
           ptrMode = CUBLAS_POINTER_MODE_HOST;
      else ptrMode = CUBLAS_POINTER_MODE_DEVICE;
      status = cublasSetPointerMode(handle, ptrMode);
      if (status != CUBLAS_STATUS_SUCCESS) OcError(-1, "Error setting scalar mode in cuBLAS");

      /* Set the stream */
      status = cublasSetStream(handle, stream);
      if (status != CUBLAS_STATUS_SUCCESS) OcError(-1, "Error setting the cuBLAS stream");

      /* Set the math mode */
      #if (CUDART_VERSION >= 9000)
      status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
      if (status != CUBLAS_STATUS_SUCCESS) OcError(-1, "Error setting the cuBLAS math mode");
      #endif

      /* Check whether we can use gemm3m (cuBlas 8.0 and higher) */
      #if (CUDART_VERSION >= 8000)
      flag3m = (((OcDeviceGPU *)device) -> properties.major >= 5);
      #else
      (void)flag3m;
      #endif
   }

   /* Make sure the data type is supported */
   flagReal = OcDType_isReal(dtype);
   if (!OCTensorGPU_gemmSupportedOn(A -> device, dtype))
      OcError(-1, "Unsupported data type in function cpu gemm (%s)", OcDType_name(dtype)); 

   /* Loop over all GEMM operations */
   while (1)
   {
      /* Compute a single GEMM operation */
      if (funptr)
      {  result = funptr(&plan, (void *)ptrAlpha, (void *)ptrA, ldA, (void *)ptrB, ldB,
                                (void *)ptrBeta,  (void *)ptrC, ldC, flagScalarHost, stream);
         if (result != 0) { OC_SOLID_ERRMSG(); return result; }
      }
      else
      {  if (flagReal)
         {  /* Real floating point */
            if (dtype == OcDTypeFloat)
            {  status = cublasSgemm(handle, opA, opB, (int)M, (int)N, (int)K,
                                    (float *)ptrAlpha, (float *)ptrA, (int)ldA, (float *)ptrB, (int)ldB,
                                    (float *)ptrBeta,  (float *)ptrC, (int)ldC);
            }
            else if (dtype == OcDTypeDouble)
            {  status = cublasDgemm(handle, opA, opB, (int)M, (int)N, (int)K,
                                    (double *)ptrAlpha, (double *)ptrA, (int)ldA, (double *)ptrB, (int)ldB,
                                    (double *)ptrBeta,  (double *)ptrC, (int)ldC);
            }
            #if (CUDART_VERSION >= 7050)
            else if (dtype == OcDTypeHalf)
            {  status = cublasHgemm(handle, opA, opB, (int)M, (int)N, (int)K,
                                    (void *)ptrAlpha, (void *)ptrA, (int)ldA, (void *)ptrB, (int)ldB,
                                    (void *)ptrBeta,  (void *)ptrC, (int)ldC);
            }
            #endif
         }
         else
         {  /* Complex floating point */
            #if (CUDART_VERSION >= 8000)
            if (flag3m)
            {  if (dtype == OcDTypeCFloat)
               {  status = cublasCgemm3m(handle, opA, opB, (int)M, (int)N, (int)K,
                                         (cuComplex *)ptrAlpha, (cuComplex *)ptrA, (int)ldA, (cuComplex *)ptrB, (int)ldB,
                                         (cuComplex *)ptrBeta,  (cuComplex *)ptrC, (int)ldC);
               }
               else if (dtype == OcDTypeCDouble)
               {  status = cublasZgemm3m(handle, opA, opB, (int)M, (int)N, (int)K,
                                         (cuDoubleComplex *)ptrAlpha, (cuDoubleComplex *)ptrA, (int)ldA, (cuDoubleComplex *)ptrB, (int)ldB,
                                         (cuDoubleComplex *)ptrBeta,  (cuDoubleComplex *)ptrC, (int)ldC);
               }
            }
            else
            #endif
            {  if (dtype == OcDTypeCFloat)
               {  status = cublasCgemm(handle, opA, opB, (int)M, (int)N, (int)K,
                                        (cuComplex *)ptrAlpha, (cuComplex *)ptrA, (int)ldA, (cuComplex *)ptrB, (int)ldB,
                                        (cuComplex *)ptrBeta,  (cuComplex *)ptrC, (int)ldC);
               }
               else if (dtype == OcDTypeCDouble)
               {  status = cublasZgemm(handle, opA, opB, (int)M, (int)N, (int)K,
                                       (cuDoubleComplex *)ptrAlpha, (cuDoubleComplex *)ptrA, (int)ldA, (cuDoubleComplex *)ptrB, (int)ldB,
                                       (cuDoubleComplex *)ptrBeta,  (cuDoubleComplex *)ptrC, (int)ldC);
               }
            }
         }

         /* Check the status */
         if (status != CUBLAS_STATUS_SUCCESS) OcError(-1, "Error executing the cublas gemm function");
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
int OCTensorGPU_gemmSupportedOn(OcDevice *device, OcDType dtype)
/* -------------------------------------------------------------------- */
{  int major, minor;
 
   /* Complex-half is not yet supported in cuBlas */
   if (dtype == OcDTypeCHalf) return 0;

   /* Types other than half are supported */
   if (dtype != OcDTypeHalf) return 1;

   /* ------------------------------------- */
   /* Check support for half-precision gemm */
   /* ------------------------------------- */

   #if (CUDART_VERSION < 7050)
      /* The cublasHgemm function is not provided */
      return 0;
   #else
      /* The cublasHgemm function is available, but may return a */
      /* CUBLAS_STATUS_ARCH_MISMATCH error when called when the  */
      /* device compute capability is below 5.3.                 */
      major = ((OcDeviceGPU *)device) -> properties.major;
      minor = ((OcDeviceGPU *)device) -> properties.minor;
      if ((major < 5) || ((major == 5) && (minor < 3)))
         return 0;
      else return 1;
   #endif
}
