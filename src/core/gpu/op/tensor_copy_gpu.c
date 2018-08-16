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

#include "ocean/core/gpu/op/tensor_copy_gpu.h"
#include "ocean/core/gpu/storage_gpu.h"
#include "ocean/core/gpu/tensor_gpu.h"
#include "ocean/core/interface/tensor_itf.h"
#include "ocean/base/shape.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_gpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorGPU_copy       (OcTensor *src, OcTensor *dst);
OC_API int OcTensorGPU_copyFromCPU(OcTensor *src, OcTensor *dst);
OC_API int OcTensorGPU_copyToCPU  (OcTensor *src, OcTensor *dst);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorGPU_initializeCopyOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Tensor copy */
   module -> Tensor_copy        = OcTensorGPU_copy;
   module -> Tensor_copyFromCPU = OcTensorGPU_copyFromCPU;
   module -> Tensor_copyToCPU   = OcTensorGPU_copyToCPU;
}


/* ===================================================================== */
/* Function implementations - copy operations                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorGPU_intrnlCopy(OcTensor *src, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_copy funptr;
   OcSolidElemwise2b config;
   int result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR2("copy", solid_gpu_copy, funptr, src -> dtype, dst -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise2b(&config, src, dst);

   /* Synchronization */
   if (OcTensor_startRead(dst, src) != 0) return -1;

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims1, config.size1, config.strides1, config.ptr1,
                 config.ndims2, config.size2, config.strides2, config.ptr2,
                 OcTensorGPU_cudaStream(dst));

   /* Synchronization */
   if (result != 0) return result;
   if (OcTensor_update(dst) != 0) return -1;
   if (OcTensor_finishRead(dst, src) != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_copy(OcTensor *src, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  OcSize tensorOffset1, tensorOffset2, tensorExtent;

   /* ---------------------------------------------------------------- */
   /* The copy function can assume that device types match, and that   */
   /* the destination tensor is writable, has no self overlaps, and    */
   /* has no zero strides. The source and destination tensors have the */
   /* same shape or number of elements, and do not have any overlap.   */
   /* The source and destination data type, byte order and device      */
   /* instance may differ.                                             */
   /* ---------------------------------------------------------------- */

   if (OcTensor_isByteswapped(src) || OcTensor_isByteswapped(dst))
   {  OcError(-1, "Copying of byte-swapped data is not supported on device %s",
              src -> device -> type -> name);
   }

   /* Directly copy the data (possibly between device instances) */
   if ((src -> dtype == dst -> dtype) && 
       (OcTensors_haveCompatibleLayout(src, dst)) &&
       (OcTensor_isContiguous(src)))
   {
      /* Compute the offset and extent */
      OcShape_extent(src -> ndims, src -> size, src -> strides, src -> elemsize, &tensorOffset1, &tensorExtent);
      OcShape_extent(dst -> ndims, dst -> size, dst -> strides, dst -> elemsize, &tensorOffset2, NULL);

      /* Directly copy the data - use the effective number of elements */
      /* instead of the total number of elements to deal with the case */
      /* of zero strides. */
      return OcBufferGPU_copy(src -> storage, OcTensor_data(src) - tensorOffset1,
                              dst -> storage, OcTensor_data(dst) - tensorOffset2,
                              OcDTypeInt8, tensorExtent);
   }

   /* Copy within the device or between devices */
   if (src -> device == dst -> device)
   {  return OcTensorGPU_intrnlCopy(src, dst);
   }
   else
   {  /* Call the generic function for copying between devices */
      return OcTensor_intrnlCopyDevices(src, dst);
   }
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_copyFromCPU(OcTensor *src, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  OcSize tensorOffset1, tensorOffset2, tensorExtent;

   /* Use direct copy whenever possible */
   if ((src -> dtype == dst -> dtype) &&
       (OcTensor_isContiguous(src)) &&
       (OcTensors_haveSameByteOrder(src, dst)) &&
       (OcTensors_haveCompatibleLayout(src, dst)))
   {
      /* Compute the offset and extent */
      OcShape_extent(src -> ndims, src -> size, src -> strides, src -> elemsize, &tensorOffset1, &tensorExtent);
      OcShape_extent(dst -> ndims, dst -> size, dst -> strides, dst -> elemsize, &tensorOffset2, NULL);

      /* Directly copy the data - use the effective number of */
      /* elements instead of the total number of elements to  */
      /* deal with the case of zero strides.                  */
      return OcBufferGPU_copyFromCPU(src -> storage, OcTensor_data(src) - tensorOffset1,
                                     dst -> storage, OcTensor_data(dst) - tensorOffset2,
                                     OcDTypeInt8, tensorExtent);
   }
   else
   {  /* Generic case */
      return OcTensor_intrnlCopyDevices(src, dst);
   }
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_copyToCPU(OcTensor *src, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  OcSize tensorOffset1, tensorOffset2, tensorExtent;
   int    result;

   /* Use direct copy whenever possible */
   if ((src -> dtype == dst -> dtype) &&
       (OcTensor_isContiguous(src)) &&
       (OcTensors_haveIdenticalLayout(src, dst)))
   {
      /* Compute the offset and extent */
      OcShape_extent(src -> ndims, src -> size, src -> strides, src -> elemsize, &tensorOffset1, &tensorExtent);
      OcShape_extent(dst -> ndims, dst -> size, dst -> strides, dst -> elemsize, &tensorOffset2, NULL);

      /* Directly copy the data - use the effective number of */
      /* elements instead of the total number of elements to  */
      /* deal with the case of zero strides.                  */
      result = OcBufferGPU_copyToCPU(src -> storage, OcTensor_data(src) - tensorOffset1,
                                     dst -> storage, OcTensor_data(dst) - tensorOffset2,
                                     OcDTypeInt8, tensorExtent);

      /* Byteswap the desgination if needed */
      if ((result == 0) && (!OcTensors_haveSameByteOrder(src, dst)))
      {  result = OcTensor_byteswapNoFlag(dst);
      }
   }
   else
   {  /* Generic case */
      result = OcTensor_intrnlCopyDevices(src, dst);
   }

   return result;
}
