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

#ifndef __OC_MODULE_CORE_H__
#define __OC_MODULE_CORE_H__

#include "ocean/base/api.h"
#include "ocean/base/module.h"
#include "ocean/base/device.h"
#include "ocean/base/device_module.h"
#include "ocean/base/scalar.h"
#include "ocean/base/storage.h"
#include "ocean/base/tensor.h"
#include "ocean/base/index.h"
#include "ocean/base/generate_macros.h"


/* Macro for module function access */
#define OC_GET_CORE_FUNCTION(device, function) OC_GET_FUNCTION(OcModuleCore, oc_module_core, device, function)


/* ===================================================================== */
/* Definition of the module structure                                    */
/* ===================================================================== */

typedef int (*OcUnaryFunc)(OcTensor *);
typedef int (*OcBinaryFunc)(OcTensor *, OcTensor *);


typedef struct
{  
   /* --- Device module base structure -------------------------- */
   OcDeviceModule HEAD;

   /* -- Device functions --------------------------------------- */
   int         (*Device_formatInfo      )(OcDevice *device, char **str, const char *header, const char *footer);

   /* --- Buffer functions -------------------------------------- */
   int         (*Buffer_copy            )(OcStorage *srcStorage, void *srcData,
                                          OcStorage *dstStorage, void *dstData, OcDType dtype, OcSize n);
   int         (*Buffer_copyFromCPU     )(OcStorage *srcStorage, void *srcData,
                                          OcStorage *dstStorage, void *dstData, OcDType dtype, OcSize n);
   int         (*Buffer_copyToCPU       )(OcStorage *srcStorage, void *srcData,
                                          OcStorage *dstStorage, void *dstData, OcDType dtype, OcSize n);
   int         (*Buffer_copyHostStorage )(OcStorage *storage, OcSize offset, OcSize nbytes, void *ptr);
   int         (*Buffer_copyStorageHost )(OcStorage *storage, OcSize offset, OcSize nbytes, void *ptr);
   int         (*Buffer_zero            )(OcStorage *storage, void *data, OcDType dtype, OcSize size);

   /* --- Storage functions ------------------------------------- */
   OcStorage  *(*Storage_create         )(OcSize nelem, OcDType dtype, OcDevice *device, OcStream *stream);
   OcStorage  *(*Storage_fromObject     )(OcSize nelem, OcDType dtype, OcDevice *device,
                                          void *object, void *data, int byteswapped,
                                          void (*free)(void *object, void *data), OcStream *stream);

   /* --- Tensor functions -------------------------------------- */
   int         (*Tensor_copy            )(OcTensor *src, OcTensor *dst);
   int         (*Tensor_copyFromCPU     )(OcTensor *src, OcTensor *dst);
   int         (*Tensor_copyToCPU       )(OcTensor *src, OcTensor *dst);
   int         (*Tensor_byteswapNoFlag  )(OcTensor *tensor);
   int         (*Tensor_fill            )(OcTensor *tensor, OcScalar *scalar);
   int         (*Tensor_fillNaN         )(OcTensor *tensor, OcScalar *scalar);
   int         (*Tensor_maskedFill      )(OcTensor *tensor, OcTensor *mask, OcScalar *scalar);
   int         (*Tensor_stepsInt64      )(OcTensor *tensor, OcInt64 offset, OcInt64 step);
   int         (*Tensor_stepsDouble     )(OcTensor *tensor, OcDouble offset, OcDouble step);
   int         (*Tensor_stepsCDouble    )(OcTensor *tensor, OcCDouble offset, OcCDouble step);

   /* --- Index functions --------------------------------------- */
   OcTensor   *(*Tensor_find            )(OcTensor *tensor);
   OcTensor   *(*Tensor_maskToOffset    )(OcTensor *tensor, OcIndex *strides);
   int         (*Tensor_indexToOffset   )(OcTensor *tensor, OcInt64 *strides, OcTensor *offsets);
   int         (*Tensor_addIfNegative   )(OcTensor *tensor, OcScalar *scalar);
   int         (*Tensor_getIndex        )(OcTensorIndexView *view, OcTensor *dst);
   int         (*Tensor_setIndex        )(OcTensorIndexView *view, OcTensor *src);
   int         (*Tensor_fillIndex       )(OcTensorIndexView *view, OcScalar *scalar);

   /* --- Elementwise unary functions --------------------------- */
   #define OC_TEMPLATE(NAME,X,Y,Z) \
   int         (*Tensor_##NAME          )(OcTensor *src, OcTensor *dst);
   #include "ocean/core/generic/generate_tensor_unary.h"
   #undef OC_TEMPLATE

   /* --- Elementwise binary functions -------------------------- */
   #define OC_TEMPLATE(NAME,X,Y,Z) \
   int         (*Tensor_##NAME          )(OcTensor *tensor1, OcTensor *tensor2, OcTensor *tensor3);
   #include "ocean/core/generic/generate_tensor_binary.h"
   #undef OC_TEMPLATE

   /* --- Domain checks ----------------------------------------- */
   int         (*Tensor_allGELE         )(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
   int         (*Tensor_allGELT         )(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
   int         (*Tensor_allGTLE         )(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
   int         (*Tensor_allGTLT         )(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
   int         (*Tensor_allLE           )(OcTensor *tensor, OcScalar *bound, int *result);
   int         (*Tensor_allLT           )(OcTensor *tensor, OcScalar *bound, int *result);
   int         (*Tensor_allGE           )(OcTensor *tensor, OcScalar *bound, int *result);
   int         (*Tensor_allGT           )(OcTensor *tensor, OcScalar *bound, int *result);

   /* --- Global reduction operations --------------------------- */
   int         (*Tensor_any             )(OcTensor *tensor, int *result);
   int         (*Tensor_all             )(OcTensor *tensor, int *result);
   int         (*Tensor_allFinite       )(OcTensor *tensor, int *result);
   int         (*Tensor_anyInf          )(OcTensor *tensor, int *result);
   int         (*Tensor_anyNaN          )(OcTensor *tensor, int *result);
   int         (*Tensor_nnz             )(OcTensor *tensor, OcUInt64 *result);
   int         (*Tensor_nnzNaN          )(OcTensor *tensor, OcUInt64 *result);
   int         (*Tensor_sum             )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_prod            )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_sumNaN          )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_prodNaN         )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_sumAbs          )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_sumAbsNaN       )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_maximum         )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_minimum         )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_maximumAbs      )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_minimumAbs      )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_norm2           )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_norm2NaN        )(OcTensor *tensor, OcScalar *result);
   int         (*Tensor_norm            )(OcTensor *tensor, double p, OcScalar *result);
   int         (*Tensor_normNaN         )(OcTensor *tensor, double p, OcScalar *result);

   /* --- Axis reduction operations ----------------------------- */
   int         (*Tensor_axisAny         )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisAll         )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisAllFinite   )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisAnyInf      )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisAnyNaN      )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisNnz         )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisNnzNaN      )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisSum         )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisProd        )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisSumNaN      )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisProdNaN     )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisSumAbs      )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisSumAbsNaN   )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisMinimum     )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisMaximum     )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisMinimumAbs  )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisMaximumAbs  )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisNorm2       )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisNorm2NaN    )(OcTensor *src, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisNorm        )(OcTensor *src, double p, int n, int *axes, OcTensor *dst);
   int         (*Tensor_axisNormNaN     )(OcTensor *src, double p, int n, int *axes, OcTensor *dst);

   /* --- Linear algebra functions ------------------------------ */
   int         (*Tensor_gemm            )(OcSize M, OcSize N, OcSize K, char transA, char transB, 
                                          OcTensor *alpha, OcTensor *A, OcIndex ldA,
                                                           OcTensor *B, OcIndex ldB,
                                          OcTensor *beta,  OcTensor *C, OcIndex ldC);
   int         (*Tensor_gemmSupportedOn )(OcDevice *device, OcDType dtype);

} OcModuleCore;


typedef struct
{  OcStorage **bufferList;                    /* List of temporary storage buffers    */
   int         bufferCount;                   /* Number of elements in the list       */
   int         bufferIndex;                   /* Index of least-recently used buffer  */
   OcSize      bufferMaxSize;                 /* Maximum size of each storage buffer  */
   OcTensor   *scalarList[OC_DTYPE_COUNT][8]; /* Scalar tensors for each type         */
   int         scalarIndex[OC_DTYPE_COUNT];   /* Index of least-recently used scalar  */
   int         scalarCount;                   /* Initialized to match scalarList size */
} OcModuleCore_Context;


/* ===================================================================== */
/* Module initialization                                                 */
/* ===================================================================== */

OC_API int                   OcModuleCore_initialize(void);
OC_API OcModuleCore_Context *OcModuleCore_createContext(OcDevice *device, size_t size);
OC_API void                  OcModuleCore_freeContext(OcModuleCore_Context *context);
OC_API OcModuleCore_Context *OcModuleCore_getContext(OcDevice *device);


/* ===================================================================== */
/* Definition of the module for implementations                          */
/* ===================================================================== */

extern OcModule oc_module_core;

#endif
