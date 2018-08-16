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

#include "ocean/core/cpu/op/tensor_index_cpu.h"
#include "ocean/core/interface/tensor_itf.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API OcTensor *OcTensorCPU_find         (OcTensor *tensor);
OC_API OcTensor *OcTensorCPU_maskToOffset (OcTensor *tensor, OcIndex *strides);
OC_API int       OcTensorCPU_indexToOffset(OcTensor *tensor, OcInt64 *strides, OcTensor *offsets);
OC_API int       OcTensorCPU_addIfNegative(OcTensor *tensor, OcScalar *scalar);
OC_API int       OcTensorCPU_getIndex     (OcTensorIndexView *view, OcTensor *dst);
OC_API int       OcTensorCPU_setIndex     (OcTensorIndexView *view, OcTensor *src);
OC_API int       OcTensorCPU_fillIndex    (OcTensorIndexView *view, OcScalar *scalar);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeIndexOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set the module functions */
   module -> Tensor_find          = OcTensorCPU_find;
   module -> Tensor_maskToOffset  = OcTensorCPU_maskToOffset;
   module -> Tensor_indexToOffset = OcTensorCPU_indexToOffset;
   module -> Tensor_addIfNegative = OcTensorCPU_addIfNegative;
   module -> Tensor_getIndex      = OcTensorCPU_getIndex;
   module -> Tensor_setIndex      = OcTensorCPU_setIndex;
   module -> Tensor_fillIndex     = OcTensorCPU_fillIndex;
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* ------------------------------------------------------------------------------ */
OcTensor *OcTensorCPU_find(OcTensor *tensor)
/* ------------------------------------------------------------------------------ */
{  solid_funptr_cpu_find_prepare  funptr1;
   solid_funptr_cpu_find_apply    funptr2;
   solid_funptr_cpu_find_finalize funptr3;
   ptrdiff_t   strides[OC_TENSOR_MAX_DIMS];
   size_t      size[OC_TENSOR_MAX_DIMS];
   OcSize      result_size[2];
   OcTensor   *result = NULL;
   size_t      nnz;
   void       *data;
   int         i, status;

   /* Look up the function pointers */ 
   OC_SOLID_FUNPTR("findPrepare",  solid_cpu_find_prepare,  funptr1, tensor -> dtype, "CPU");
   OC_SOLID_FUNPTR("findApply",    solid_cpu_find_apply,    funptr2, tensor -> dtype, "CPU");
   OC_SOLID_FUNPTR("findFinalize", solid_cpu_find_finalize, funptr3, tensor -> dtype, "CPU");
   if ((funptr1 == 0) || (funptr2 == 0) || (funptr3 == 0)) return NULL;

   /* Convert the strides and dimensions */
   for (i = 0; i < tensor -> ndims; i++)
   {  size[i]    = (size_t)tensor -> size[i];
      strides[i] = (ptrdiff_t)tensor -> strides[i];
   }

   /* Prepare the operation */
   status = funptr1(tensor -> ndims, size, strides, OcTensor_data(tensor), &nnz, &data);
   if (status != 0) OcError(NULL, "Error initializing the find operation");

   /* Allocate the output tensor */
   result_size[0] = tensor -> ndims;
   result_size[1] = nnz;
   result = OcTensor_create(2, result_size, NULL, OcDTypeInt64, OcCPU);
   if (result == NULL) goto final;

   /* Locate the non-zero entries */
   status = funptr2(data, (solid_int64 *)(OcTensor_data(result)), NULL);
   if (status != 0) { OcDecrefTensor(result); result = NULL; }

final : ;
   /* Finalize */
   funptr3(data);
   return result;
}


/* ------------------------------------------------------------------------------ */
OcTensor *OcTensorCPU_maskToOffset(OcTensor *tensor, OcIndex *stridesIdx)
/* ------------------------------------------------------------------------------ */
{  solid_funptr_cpu_find_prepare  funptr1;
   solid_funptr_cpu_find_apply    funptr2;
   solid_funptr_cpu_find_finalize funptr3;
   solid_int64 multiplier[OC_TENSOR_MAX_DIMS];
   ptrdiff_t   strides[OC_TENSOR_MAX_DIMS];
   size_t      size[OC_TENSOR_MAX_DIMS];
   OcSize      result_size;
   OcTensor   *result = NULL;
   size_t      nnz;
   void       *data;
   int         i, status;

   /* Look up the function pointers */ 
   OC_SOLID_FUNPTR("findPrepare",  solid_cpu_find_prepare,  funptr1, tensor -> dtype, "CPU");
   OC_SOLID_FUNPTR("findApply",    solid_cpu_find_apply,    funptr2, tensor -> dtype, "CPU");
   OC_SOLID_FUNPTR("findFinalize", solid_cpu_find_finalize, funptr3, tensor -> dtype, "CPU");
   if ((funptr1 == 0) || (funptr2 == 0) || (funptr3 == 0)) return NULL;

   /* Convert the strides and dimensions */
   for (i = 0; i < tensor -> ndims; i++)
   {  size[i]       = (size_t)tensor -> size[i];
      strides[i]    = (ptrdiff_t)tensor -> strides[i];
      multiplier[i] = stridesIdx[i];
   }

   /* Prepare the operation */
   status = funptr1(tensor -> ndims, size, strides, OcTensor_data(tensor), &nnz, &data);
   if (status != 0) OcError(NULL, "Error initializing the find operation");

   /* Allocate the output tensor */
   result_size = nnz;
   result = OcTensor_create(1, &result_size, NULL, OcDTypeInt64, OcCPU);
   if (result == NULL) goto final;

   /* Locate the non-zero entries */
   status = funptr2(data, (solid_int64 *)(OcTensor_data(result)), multiplier);
   if (status != 0) { OcDecrefTensor(result); result = NULL; }

final : ;
   /* Finalize */
   funptr3(data);
   return result;
}


/* ------------------------------------------------------------------------------ */
int OcTensorCPU_indexToOffset(OcTensor *tensor, OcInt64 *strides, OcTensor *offsets)
/* ------------------------------------------------------------------------------ */
{  solid_funptr_cpu_index_to_offset funptr;
   ptrdiff_t                        stride1;
   ptrdiff_t                        stride2 = offsets -> strides[0];
   ptrdiff_t                        strideReduce;
   size_t                           nelem;
   int                              nstrides;
   int                              result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("indexToOffset", solid_cpu_index_to_offset, funptr, tensor -> dtype, "CPU");
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

   /* Call the function */
   OC_SOLID_CALL(result, funptr, nstrides, (solid_int64 *)strides, strideReduce, nelem,
                                 stride1, (void *)(OcTensor_data(tensor)),
                                 stride2, (void *)(OcTensor_data(offsets)));

   return result;
}


/* ------------------------------------------------------------------------------ */
int OcTensorCPU_addIfNegative(OcTensor *tensor, OcScalar *scalar)
/* ------------------------------------------------------------------------------ */
{  OcSolidElemwise1                 config;
   solid_funptr_cpu_add_if_negative funptr;
   solid_scalar                     value;
   int                              result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("addIfNegative", solid_cpu_add_if_negative, funptr, tensor -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr, config.ndims, config.size, config.strides, config.ptr, value);

   return result;
}


/* ------------------------------------------------------------------------------ */
int OcTensorCPU_getIndex(OcTensorIndexView *view, OcTensor *dst)
/* ------------------------------------------------------------------------------ */
{  solid_funptr_cpu_get_index funptr;
   OcSolidIndex2 config;
   int result;

   /* -------------------------------------------- */
   /* Data type, device, and byte-order must match */
   /* -------------------------------------------- */

   /* Look up the function pointers */ 
   OC_SOLID_FUNPTR("getIndex",  solid_cpu_get_index, funptr, dst -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeIndex2(&config, view -> view, view -> offsets, view -> offsetStrides, 0, dst);

   /* Call the function */
   OC_SOLID_CALL(result, funptr, config.ndims, config.size, config.offsets,
                 config.strides1, config.ptr1, config.strides2, config.ptr2);

   return result;
}


/* ------------------------------------------------------------------------------ */
int OcTensorCPU_setIndex(OcTensorIndexView *view, OcTensor *src)
/* ------------------------------------------------------------------------------ */
{  solid_funptr_cpu_set_index funptr;
   OcSolidIndex2 config;
   OcTensor     *dst = view -> view;
   int           result;

   /* -------------------------------------------- */
   /* Data type, device, and byte-order must match */
   /* -------------------------------------------- */

   /* Look up the function pointers */ 
   OC_SOLID_FUNPTR("setIndex",  solid_cpu_set_index, funptr, src -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeIndex2(&config, dst, view -> offsets, view -> offsetStrides, 1, src);

   /* Call the function */
   OC_SOLID_CALL(result, funptr, config.ndims, config.size, config.offsets,
                 config.strides1, config.ptr1, config.strides2, config.ptr2);

   return result;
}


/* ------------------------------------------------------------------------------ */
int OcTensorCPU_fillIndex(OcTensorIndexView *view, OcScalar *scalar)
/* ------------------------------------------------------------------------------ */
{  solid_funptr_cpu_fill_index funptr;
   solid_scalar   value;
   OcSolidIndex1  config;
   OcTensor      *dst = view -> view;
   int            result;

   /* -------------------------------------------- */
   /* Data type, device, and byte-order must match */
   /* -------------------------------------------- */

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Look up the function pointers */ 
   OC_SOLID_FUNPTR("fillIndex",  solid_cpu_fill_index, funptr, view -> view -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeIndex1(&config, dst, view -> offsets, view -> offsetStrides);

   /* Call the function */
   OC_SOLID_CALL(result, funptr, config.ndims, config.size, config.offsets,
                                 config.strides, config.ptr, value);

   return result;
}
