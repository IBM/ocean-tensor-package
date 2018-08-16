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

#include "ocean/core/gpu/op/tensor_index_gpu.h"
#include "ocean/core/gpu/tensor_gpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_gpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorGPU_indexToOffset(OcTensor *tensor, OcInt64 *strides, OcTensor *offsets);
OC_API int OcTensorGPU_addIfNegative(OcTensor *tensor, OcScalar *scalar);
OC_API int OcTensorGPU_getIndex     (OcTensorIndexView *view, OcTensor *dst);
OC_API int OcTensorGPU_setIndex     (OcTensorIndexView *view, OcTensor *src);
OC_API int OcTensorGPU_fillIndex    (OcTensorIndexView *view, OcScalar *scalar);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorGPU_initializeIndexOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set the module functions */
   module -> Tensor_indexToOffset = OcTensorGPU_indexToOffset;
   module -> Tensor_addIfNegative = OcTensorGPU_addIfNegative;
   module -> Tensor_getIndex      = OcTensorGPU_getIndex;
   module -> Tensor_setIndex      = OcTensorGPU_setIndex;
   module -> Tensor_fillIndex     = OcTensorGPU_fillIndex;
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorGPU_addIfNegative(OcTensor *tensor, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcSolidElemwise1                 config;
   solid_funptr_gpu_add_if_negative funptr;
   solid_scalar                     value;
   int                              result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("addIfNegative", solid_gpu_add_if_negative, funptr, tensor -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(tensor -> device -> index) != 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr, value,
                 OcTensorGPU_cudaStream(tensor));

   /* Synchronization */
   if (result == 0) result = OcTensor_update(tensor);

   return result;
}


/* ------------------------------------------------------------------------------ */
int OcTensorGPU_indexToOffset(OcTensor *tensor, OcInt64 *strides, OcTensor *offsets)
/* ------------------------------------------------------------------------------ */
{  solid_funptr_gpu_index_to_offset funptr;
   ptrdiff_t                        stride1;
   ptrdiff_t                        stride2 = offsets -> strides[0];
   ptrdiff_t                        strideReduce;
   size_t                           nelem;
   int                              nstrides;
   int                              result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("indexToOffset", solid_gpu_index_to_offset, funptr, tensor -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Prepare */
   if (tensor -> ndims == 1)
   {  stride1      = tensor -> strides[0];
      strideReduce = 0;
      nstrides     = 1;
      nelem        = tensor -> size[0];
   }
   else
   {  stride1      = tensor -> strides[1];
      strideReduce = tensor -> strides[0];
      nstrides     = tensor -> size[0];
      nelem        = tensor -> size[1];
   }

   /* Synchronize */
   if (OcTensor_startRead(offsets, tensor) != 0) return -1;

   /* Call the function */
   OC_SOLID_CALL(result, funptr, nstrides, (solid_int64 *)strides, strideReduce, nelem,
                                 stride1, (void *)(OcTensor_data(tensor)),
                                 stride2, (void *)(OcTensor_data(offsets)),
                                 OcTensorGPU_cudaStream(offsets));

   /* Synchronize */
   if (result != 0) return result;
   if (OcTensor_update(offsets) != 0) return -1;
   if (OcTensor_finishRead(offsets, tensor) != 0) return -1;

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_getIndex(OcTensorIndexView *view, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_get_index funptr;
   OcSolidIndex2 config;
   int i, result;

   /* Look up the function pointer */
   OC_SOLID_FUNPTR("getIndex", solid_gpu_get_index, funptr, dst -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(dst -> device -> index) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeIndex2(&config, view -> view, view -> offsets, view -> offsetStrides, 0, dst);

   /* Synchronize */
   if (OcTensor_startRead(dst, view -> view) != 0) return -1;
   for (i = 0; i < view -> ndims; i++)
   {  if ((view -> offsets[i]) && (OcTensor_startRead(dst, view -> offsets[i]) != 0))
         return -1;
   }

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.offsets,
                 config.strides1, config.ptr1,
                 config.strides2, config.ptr2, OcTensorGPU_cudaStream(dst));

   /* Synchronize */
   if (result != 0) return result;
   if (OcTensor_update(dst) != 0) return -1;
   if (OcTensor_finishRead(dst, view -> view) != 0) return -1;
   for (i = 0; i <  view -> ndims; i++)
   {  if ((view -> offsets[i]) && (OcTensor_finishRead(dst, view -> offsets[i]) != 0))
         return -1;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_setIndex(OcTensorIndexView *view, OcTensor *src)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_get_index funptr;
   OcSolidIndex2 config;
   OcTensor *dst = view -> view;
   int i, result;

   /* Look up the function pointer */
   OC_SOLID_FUNPTR("setIndex", solid_gpu_set_index, funptr, dst -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(dst -> device -> index) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeIndex2(&config, dst, view -> offsets, view -> offsetStrides, 1, src);

   /* Synchronize */
   if (OcTensor_startRead(dst, src) != 0) return -1;
   for (i = 0; i < view -> ndims; i++)
   {  if ((view -> offsets[i]) && (OcTensor_startRead(dst, view -> offsets[i]) != 0))
         return -1;
   }

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.offsets,
                 config.strides1, config.ptr1,
                 config.strides2, config.ptr2, OcTensorGPU_cudaStream(dst));

   /* Synchronize */
   if (result != 0) return result;
   if (OcTensor_update(dst) != 0) return -1;
   if (OcTensor_finishRead(dst, src) != 0) return -1;
   for (i = 0; i <  view -> ndims; i++)
   {  if ((view -> offsets[i]) && (OcTensor_finishRead(dst, view -> offsets[i]) != 0))
         return -1;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_fillIndex(OcTensorIndexView *view, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_fill_index funptr;
   solid_scalar   value;
   OcSolidIndex1  config;
   OcTensor      *dst = view -> view;
   int            i, result;

   /* Look up the function pointer */
   OC_SOLID_FUNPTR("fillIndex", solid_gpu_fill_index, funptr, dst -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(dst -> device -> index) != 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeIndex1(&config, dst, view -> offsets, view -> offsetStrides);

   /* Synchronize */
   for (i = 0; i < view -> ndims; i++)
   {  if ((view -> offsets[i]) && (OcTensor_startRead(dst, view -> offsets[i]) != 0))
         return -1;
   }

   /* Call the function */
   OC_SOLID_CALL(result, funptr, config.ndims, config.size, config.offsets,
                 config.strides, config.ptr, value, OcTensorGPU_cudaStream(dst));

   /* Synchronize */
   if (result != 0) return result;
   if (OcTensor_update(dst) != 0) return -1;
   for (i = 0; i <  view -> ndims; i++)
   {  if ((view -> offsets[i]) && (OcTensor_finishRead(dst, view -> offsets[i]) != 0))
         return -1;
   }

   return 0;
}
