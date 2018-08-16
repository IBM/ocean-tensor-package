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
#include "ocean/core/cpu/op/tensor_copy_cpu.h"
#include "ocean/core/cpu/op/tensor_byteswap_cpu.h"
#include "ocean/core/cpu/storage_cpu.h"
#include "ocean/base/shape.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int  OcTensorCPU_copy(OcTensor *src, OcTensor *dst);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeCopyOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Tensor copy */
   module -> Tensor_copy = OcTensorCPU_copy;
}


/* ===================================================================== */
/* Function implementations - copy operations                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorCPU_intrnlCopy(OcTensor *src, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_copy funptr;
   OcSolidElemwise2b config;
   int result;
   
   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR2("copy", solid_cpu_copy, funptr, src -> dtype, dst -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise2b(&config, src, dst);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims1, config.size1, config.strides1, config.ptr1,
                 config.ndims2, config.size2, config.strides2, config.ptr2);

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorCPU_intrnlByteswapCopy(OcTensor *src, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_byteswap_copy funptr;
   OcSolidElemwise2b config;
   int result;
   
   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR2("copy", solid_cpu_byteswap_copy, funptr, src -> dtype, dst -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise2b(&config, src, dst);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims1, config.size1, config.strides1, config.ptr1,
                 config.ndims2, config.size2, config.strides2, config.ptr2);

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorCPU_copy(OcTensor *src, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  OcSize tensorOffset1, tensorOffset2, tensorExtent;
   int    result = 0;

   /* ---------------------------------------------------------------- */
   /* The copy function can assume that device types match, and that   */
   /* the destination tensor is writable, has no self overlaps, and    */
   /* has no zero strides. The source and destination tensors have the */
   /* same shape or number of elements, and do not have any overlap.   */
   /* The source and destination data type, byte order and device      */
   /* instance may differ.                                             */
   /* ---------------------------------------------------------------- */

   /* Check if source and destination are the same */
   if (OcTensors_match(src, dst))
   {  /* Add synchronization (effective only if the storage differs) */
      OcTensor_startRead(dst, src);
      OcTensor_update(dst);
      OcTensor_finishRead(dst, src);
      return 0;
   }

   /* Directly copy the data */
   if ((src -> dtype == dst -> dtype) && 
       (OcTensor_isByteswapped(src) == OcTensor_isByteswapped(dst)) &&
       (OcTensors_haveCompatibleLayout(src, dst)) &&
       (OcTensor_isContiguous(src)))
   {
      /* Compute the offset and extent */
      OcShape_extent(src -> ndims, src -> size, src -> strides, src -> elemsize, &tensorOffset1, &tensorExtent);
      OcShape_extent(dst -> ndims, dst -> size, dst -> strides, dst -> elemsize, &tensorOffset2, NULL);

      /* Directly copy the data - use the effective number of elements */
      /* instead of the total number of elements to deal with the case */
      /* of zero strides. */
      return OcBufferCPU_copy(src -> storage, OcTensor_data(src) - tensorOffset1,
                              dst -> storage, OcTensor_data(dst) - tensorOffset2,
                              OcDTypeInt8, tensorExtent);
   }

   /* Copy non-contiguous or byte-swapped data */
   if (((src -> dtype != dst -> dtype) && OcTensor_isByteswapped(src)) ||
       ((src -> dtype == dst -> dtype) && (OcTensor_isByteswapped(src) != OcTensor_isByteswapped(dst))))
   {  /* Copy byte-swapped data */
      result = OcTensorCPU_intrnlByteswapCopy(src, dst);
   }
   else
   {  /* Copy non-contiguous data */
      result = OcTensorCPU_intrnlCopy(src, dst);
   }

   /* Byte-swap the data if needed */
   if ((result == 0) && (src -> dtype != dst -> dtype) && OcTensor_isByteswapped(dst))
   {  result = OcTensorCPU_byteswapNoFlag(dst);
   }

   /* Return the result */
   return result;
}
