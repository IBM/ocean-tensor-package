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
#include "ocean/core/generic/tensor_op.h"
#include "ocean/base/shape.h"
#include "ocean/base/error.h"


/* --------------------------------------------------------------------- */
void OcTensorOp_initialize(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{  info -> n      = 0;
   info -> device = NULL;
   info -> dtype  = OcDTypeNone;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_finalize(OcTensorOp *info, int result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;
   int i;

   /* Copy result to existing destination tensors */
   for (i = 0; i < info -> n; i++)
   {  tensor = *(info -> tensorPtr[i]);
      if ((result == 0) && (info -> flags[i] & OC_OP_WRITE) &&
          (info -> init[i]) && (tensor != NULL) && (info -> init[i] != tensor))
      {  /* Copy the tensor */
         result = OcTensor_copy(tensor, info -> init[i]);
      }
   }

   /* Clean up and assign destination tensors */
   for (i = 0; i < info -> n; i++)
   {  
      if ((info -> init[i]) || (result != 0))
      {  /* Restore the original pointer value */
         OcXDecrefTensor(*(info -> tensorPtr[i]));
         *(info -> tensorPtr[i]) = info -> init[i];
      }
      else
      {  /* Destination tensor is already set */
      }
   }

   /* Reset the number of tensors */
   info -> n = 0;

   /* Return the result */
   return result;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_initialize1(OcTensorOp *info, OcTensor **tensor1, unsigned short flags1)
/* --------------------------------------------------------------------- */
{
   OcTensorOp_initialize(info);
   if (OcTensorOp_addTensor(info, tensor1, flags1) < 0) return -1;
   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_initialize2(OcTensorOp *info, OcTensor **tensor1, unsigned short flags1,
                                             OcTensor **tensor2, unsigned short flags2)
/* --------------------------------------------------------------------- */
{
   OcTensorOp_initialize(info);
   if (OcTensorOp_addTensor(info, tensor1, flags1) < 0) return -1;
   if (OcTensorOp_addTensor(info, tensor2, flags2) < 0) return -1;
   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_initialize3(OcTensorOp *info, OcTensor **tensor1, unsigned short flags1,
                                             OcTensor **tensor2, unsigned short flags2,
                                             OcTensor **tensor3, unsigned short flags3)
/* --------------------------------------------------------------------- */
{
   OcTensorOp_initialize(info);
   if (OcTensorOp_addTensor(info, tensor1, flags1) < 0) return -1;
   if (OcTensorOp_addTensor(info, tensor2, flags2) < 0) return -1;
   if (OcTensorOp_addTensor(info, tensor3, flags3) < 0) return -1;
   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_addTensor(OcTensorOp *info, OcTensor **tensor, unsigned short flags)
/* --------------------------------------------------------------------- */
{  int idx;

   /* Basic checks on source tensors */
   if ((flags & OC_OP_READ) && (*tensor == NULL))
      OcError(-1, "Source tensors cannot be NULL");

   /* Basic checks on destination tensors */
   if ((flags & OC_OP_WRITE) && (*tensor != NULL))
   {  /* Make sure that the destination tensors are not read only */
      if (OcTensor_isReadOnly(*tensor))
         OcError(-1, "Attempting to write to a read-only tensor");

      /* Make sure that the ensor does not have any self overlap */
      if (OcTensor_hasZeroStrides(*tensor) || OcTensor_isSelfOverlapping(*tensor))
         OcError(-1, "Attempting to write to a self-overlapping tensor");
   }

   /* Check if there is sufficient space for a new tensor */
   if (info -> n == OC_OP_MAX_TENSORS)
      OcError(-1, "Number of tensors in the tensor operation exceeds the maximum");

   /* Add the tensor */
   idx = info -> n;
   info -> init[idx]      = OcXIncrefTensor(*tensor);
   info -> tensorPtr[idx] = tensor;
   info -> flags[idx]     = flags;
   info -> n              = idx + 1;

   return idx;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_commonType(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;
   OcDevice *device = NULL;
   OcDType   dtype = OcDTypeNone;
   int       i;

   /* Add the tensors */
   for (i = 0; i < info -> n; i++)
   {  if ((tensor = *(info -> tensorPtr[i])) == NULL) continue;
      if (device == NULL) device = tensor -> device;
      dtype = OcDType_getCommonType(dtype, tensor -> dtype);
   }

   /* Set the values */
   info -> device = device;
   info -> dtype  = dtype;

   /* Check device and data type */
   if (device == NULL)
      OcError(-1, "Could not determine the common device");
   if (dtype == OcDTypeNone)
      OcError(-1, "Cound not determine the common data type");

   return 0;   
}


/* --------------------------------------------------------------------- */
int OcTensorOp_applyTypes(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;
   int       flagTemporary;
   int       i, result;

   for (i = 0; i < info -> n; i++)
   {  if ((tensor = *(info -> tensorPtr[i])) != NULL)
      {  if (OcTensor_checkAutoTypecast(*(info -> tensorPtr[i]), info -> dtype, info -> device) != 0) return -1;
         flagTemporary = (info -> init[i]) ? 1 : 0;
         result = OcTensor_ensureFlags(info -> tensorPtr[i], info -> dtype, info -> device, NULL, flagTemporary);
         if (result != 0) return result;
      }
   }

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_ensureDType(OcTensorOp *info, int idx, OcDType dtype)
/* --------------------------------------------------------------------- */
{  int flagTemporary;

   /* Make sure the tensor is valid */
   if ((idx < 0) || (idx >= info -> n))
      OcError(-1, "Internal error: tensor index out of range");
   if (*(info -> tensorPtr[idx]) == NULL) return 0;   

   flagTemporary = (info -> init[idx]) ? 1 : 0;
   return OcTensor_ensureFlags(info -> tensorPtr[idx], dtype, NULL, NULL, flagTemporary);
}


/* --------------------------------------------------------------------- */
int OcTensorOp_ensureDevice(OcTensorOp *info, int idx, OcDevice *device)
/* --------------------------------------------------------------------- */
{  int flagTemporary;

   /* Make sure the tensor is valid */
   if ((idx < 0) || (idx >= info -> n))
      OcError(-1, "Internal error: tensor index out of range");
   if (*(info -> tensorPtr[idx]) == NULL) return 0;   

   flagTemporary = (info -> init[idx]) ? 1 : 0;
   return OcTensor_ensureFlags(info -> tensorPtr[idx], OcDTypeNone, device, NULL, flagTemporary);
}


/* --------------------------------------------------------------------- */
int OcTensorOp_ensure(OcTensorOp *info, int idx, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{    int flagTemporary;

   /* Make sure the tensor is valid */
   if ((idx < 0) || (idx >= info -> n))
      OcError(-1, "Internal error: tensor index out of range");
   if (*(info -> tensorPtr[idx]) == NULL) return 0;   

   flagTemporary = (info -> init[idx]) ? 1 : 0;
   return OcTensor_ensureFlags(info -> tensorPtr[idx], dtype, device, NULL, flagTemporary);
}


/* --------------------------------------------------------------------- */
int OcTensorOp_broadcastElemwise(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;
   OcSize    size[OC_TENSOR_MAX_DIMS];
   int       ndims = 0;
   int       i, flag, result = 0;

   /* Determine the final shape */
   for (i = 0; i < info -> n; i++)
   {  if ((tensor = *(info -> tensorPtr[i])) != NULL)
      {  result = OcShape_broadcastRight(&ndims, size, tensor -> ndims, tensor -> size);
         if (result != 0) return result;
      }
   }

   /* Check existing size */
   flag = OcTensor_getAutoBroadcastMode();
   for (i = 0; i < info -> n; i++)
   {  if ((tensor = *(info -> tensorPtr[i])) == NULL) continue;

      if (((flag == 0) || (info -> flags[i] & OC_OP_WRITE)) &&
          (!OcShapes_match(ndims, size, tensor -> ndims, tensor -> size)))
      {  /* Mismatch in tensor dimensions */
         if (info -> flags[i] & OC_OP_WRITE)
            OcError(-1, "Mismatch in tensor dimensions (destination tensors cannot be broadcast)");
         else if ((ndims > 0) && (tensor -> ndims > 0))
            OcError(-1, "Mismatch in tensor dimensions (automatic broadcasting of tensor dimensions is disabled)");
      }
   }

   /* Broadcast source tensors */
   for (i = 0; i < info -> n; i++)
   {  if ((*(info -> tensorPtr[i]) != NULL) &&
          ((info -> flags[i] & OC_OP_WRITE) == 0))
      {  result = OcTensor_broadcastTo(info -> tensorPtr[i], ndims, size, 0, NULL);
         if (result != 0) return result;
      }
   }

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_allocElemwise(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor, *ref = NULL;
   int i;

   /* Determine the reference tensor */
   for (i = 0; i < info -> n; i++)
   {  if ((ref = *(info -> tensorPtr[i])) != NULL) break;
   }
   if (ref == NULL) OcError(-1, "Could not determine the destination tensor size (reference tensor missing)");

   /* Create all missing tensors */
   for (i = 0; i < info -> n; i++)
   {  if (*(info -> tensorPtr[i])) continue;

      /* Create a new tensor */
      tensor = OcTensor_create(ref -> ndims, ref -> size, NULL, ref -> dtype, ref -> device);
      if (tensor == NULL) return -1;

      /* Assign the value */
      *(info -> tensorPtr[i]) = tensor;
   }

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_allocElemwiseIdx(OcTensorOp *info, int idx, OcDType dtype)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor, *ref = NULL;
   int i;

   /* Make sure the index is a valid destination tensor */
   if ((idx < 0) || (idx >= info -> n))
      OcError(-1, "Internal error: tensor index out of range");
   if ((info -> flags[idx] & OC_OP_WRITE) == 0)
      OcError(-1, "Internal error: destination tensor index expected");

   /* Determine the reference tensor */
   for (i = 0; i < info -> n; i++)
   {  if ((ref = *(info -> tensorPtr[i])) != NULL) break;
   }
   if (ref == NULL) OcError(-1, "Could not determine the destination tensor size (reference tensor missing)");

   /* Check if the tensor exists and has the desired data type */
   tensor = *(info -> tensorPtr[idx]);
   if ((tensor != NULL) && (tensor -> dtype == dtype)) return 0;

   /* Update the data type or create a new tensor */
   if (tensor)
   {  return OcTensor_ensureDType(info -> tensorPtr[idx], dtype, NULL);
   }
   else
   {  /* Create a new tensor */
      tensor = OcTensor_create(ref -> ndims, ref -> size, NULL, dtype, ref -> device);
      if (tensor == NULL) return -1;

      /* Assign the value */
      *(info -> tensorPtr[idx]) = tensor;
   }

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_alignTensors(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;
   int flagTemporary;
   int i;

   for (i = 0; i < info -> n; i++)
   {  tensor = *(info -> tensorPtr[i]);
      if (!OcTensor_isAligned(tensor) || OcTensor_isByteswapped(tensor))
      {  /* Clone the tensor - do not preserve byteswap */
         flagTemporary = (info -> init[i]) ? 1 : 0;
         tensor = OcTensor_cloneFlags(tensor, OcDTypeNone, NULL, 0, flagTemporary);
         if (tensor == NULL) return -1;

         /* Assign the tensor */
         OcDecrefTensor(*(info -> tensorPtr[i]));
         *(info -> tensorPtr[i]) = tensor;
      }
   }

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_overlapElemwise(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;
   int flagTemporary;
   int flagReplace;
   int i, j;

   for (i = 0; i < info -> n; i++)
   {  if (((info -> flags[i]) & OC_OP_WRITE) == 0) continue;

      tensor = *(info -> tensorPtr[i]);
      flagReplace = 0;
      for (j = 0; j < i; j++)
      {  if ((info -> flags[j]) & OC_OP_WRITE)
         {  /* Do not allow any overlap with destination tensors */
            flagReplace = OcTensors_overlap(tensor, *(info -> tensorPtr[j])) ? 1 : 0;
         }
         else
         {  /* Allow only matching tensors */
            flagReplace = OcTensors_match(tensor, *(info -> tensorPtr[j])) ? 0 : 1;
         }

         if (flagReplace) break;
      }

      /* Replace tensor if needed */
      if (flagReplace)
      {  /* Create a new tensor - maintain byteswap flag */
         flagTemporary = (info -> init[i]) ? 1 : 0;
         tensor = OcTensor_cloneFlags(tensor, OcDTypeNone, NULL, 1, flagTemporary);
         if (tensor == NULL) return -1;

         /* Assign the tensor */
         OcDecrefTensor(*(info -> tensorPtr[i]));
         *(info -> tensorPtr[i]) = tensor;
      }
   }
   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_prepareElemwise(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{
   if ((OcTensorOp_applyTypes(info)        == 0) &&
       (OcTensorOp_broadcastElemwise(info) == 0) &&
       (OcTensorOp_allocElemwise(info)     == 0) &&
       (OcTensorOp_alignTensors(info)      == 0) &&
       (OcTensorOp_overlapElemwise(info)   == 0)) return 0;

   return -1;
}


/* --------------------------------------------------------------------- */
int OcTensorOp_removeOverlap(OcTensorOp *info)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor, *ref;
   int i, j;

   for (i = 0; i < info -> n; i++)
   {  tensor = *(info -> tensorPtr[i]);
      if ((tensor == NULL) || ((info -> flags[i] & OC_OP_WRITE) == 0)) continue;

      for (j = 0; j < i; j++)
      {  if ((ref = *(info -> tensorPtr[j])) == NULL) continue;
         if (OcTensors_overlap(tensor, ref))
         {  /* Replace the tensor */
            tensor = OcTensor_clone(tensor);
            if (tensor == NULL) return -1;
            OcDecrefTensor(*(info -> tensorPtr[i]));
            *(info -> tensorPtr[i]) = tensor;
            break;
         }
      }
   }

   return 0;
}
