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

#ifndef __OCEAN_TENSOR_OP_H__
#define __OCEAN_TENSOR_OP_H__

#include "ocean/base/tensor.h"
#include "ocean/base/device.h"
#include "ocean/base/dtype.h"
#include "ocean/base/api.h"

#define OC_OP_READ        0x01
#define OC_OP_WRITE       0x02
#define OC_OP_READWRITE   OC_FLAG_READ | OC_FLAG_WRITE

#define OC_OP_MAX_TENSORS 8

typedef struct
{  OcTensor        *init[OC_OP_MAX_TENSORS];      /* Initial tensors     */
   OcTensor       **tensorPtr[OC_OP_MAX_TENSORS]; /* Pointers to tensors */
   unsigned short   flags[OC_OP_MAX_TENSORS];     /* Tensor flags        */
   OcDevice        *device;                       /* Selected device     */
   OcDType          dtype;                        /* Selected data type  */
   int              n;                            /* Number of tensors   */
} OcTensorOp;



OC_API void OcTensorOp_initialize (OcTensorOp *info);
OC_API int  OcTensorOp_finalize   (OcTensorOp *info, int result);
OC_API int  OcTensorOp_initialize1(OcTensorOp *info, OcTensor **tensor1, unsigned short flags1);
OC_API int  OcTensorOp_initialize2(OcTensorOp *info, OcTensor **tensor1, unsigned short flags1,
                                                      OcTensor **tensor2, unsigned short flags2);
OC_API int  OcTensorOp_initialize3(OcTensorOp *info, OcTensor **tensor1, unsigned short flags1,
                                                     OcTensor **tensor2, unsigned short flags2,
                                                     OcTensor **tensor3, unsigned short flags3);

/* The OcTensorOp_addTensor adds a tensor to the info structure and  */
/* returns the index of the addition, or -1 in case of an error.     */
OC_API int  OcTensorOp_addTensor  (OcTensorOp *info, OcTensor **tensor, unsigned short flags);

/* The OcTensorOp_commonType function determines the device and least */
/* commont data type for all tensors in the structure.                */
OC_API int  OcTensorOp_commonType       (OcTensorOp *info);

/* The OcTensorOp_applyTypes function applies the current device and  */
/* data type to all tensors in the structure.                         */
OC_API int  OcTensorOp_applyTypes       (OcTensorOp *info);

/* The OcTensorOp_ensure functions make sure that that the given      */
/* tensor has the correct data type or device.                        */
OC_API int  OcTensorOp_ensureDType (OcTensorOp *info, int idx, OcDType dtype);
OC_API int  OcTensorOp_ensureDevice(OcTensorOp *info, int idx, OcDevice *device);
OC_API int  OcTensorOp_ensure      (OcTensorOp *info, int idx, OcDType dtype, OcDevice *device);

/* The OcTensorOp_broadcastElemwise function broadcasts all tensors   */
/* in the structure to have the same size or returns an error when    */
/* the tensor dimensions are incompatible.                            */
OC_API int  OcTensorOp_broadcastElemwise(OcTensorOp *info);

/* The OcTensorOp_allocElemwise function allocates all uninitialized  */
/* destination tensors to the same size, data type, and device as the */
/* first initialized tensor.                                          */
OC_API int  OcTensorOp_allocElemwise    (OcTensorOp *info);

/* The OcTensorOp_allocElemwiseIdx function checks the indexed tensor */
/* and makes sure it is a destination tensor. When the tensor has not */
/* yet been initialized a new tensor is created with the given data   */
/* type and with the same size and device as the first initialized    */
/* tensor. When the tensor already exists we only ensure that it has  */
/* the given data type (it is assumed that the size and device of the */
/* tensor are correct).                                               */
OC_API int  OcTensorOp_allocElemwiseIdx (OcTensorOp *info, int idx, OcDType dtype);

/* The OcTensorOp_alignTensors function ensures that all tensors in   */
/* the structure satisfy the memory alignment requirements of the     */
/* respective devices they reside on.                                 */
OC_API int  OcTensorOp_alignTensors     (OcTensorOp *info);

/* The OcTensorOp_overlapElemwise function removes overlap between    */
/* destination tensors with other desination tensors, as well as      */
/* partial overlap with source tensors (matching tensors are allowed  */
/* in this case).                                                     */
OC_API int  OcTensorOp_overlapElemwise  (OcTensorOp *info);

/* The OcTensorOp_prepareElemwise function combines commonly used     */
/* functions for elementwise operations; calling in order:            */
/* 1. OcTensorOp_applyTypes                                           */
/* 2. OcTensorOp_broadcastElemwise                                    */
/* 3. OcTensorOp_allocElemwise                                        */
/* 4. OcTensorOp_alignTensors                                         */
/* 5. OcTensorOp_overlapElemwise                                      */
OC_API int  OcTensorOp_prepareElemwise  (OcTensorOp *info);

/* The OcTensorOp_removeOverlap function removes overlap between      */
/* destination tensors and others.                                    */
OC_API int  OcTensorOp_removeOverlap    (OcTensorOp *info);

#endif
