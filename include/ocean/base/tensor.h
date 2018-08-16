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

#ifndef __OC_TENSOR_H__
#define __OC_TENSOR_H__

#include "ocean/base/api.h"
#include "ocean/base/dtype.h"
#include "ocean/base/types.h"
#include "ocean/base/device.h"
#include "ocean/base/storage.h"
#include "ocean/base/config.h"


/* ===================================================================== */
/* Tensor flags                                                          */
/* ===================================================================== */

/* Shape information */
#define OC_TENSOR_LINEAR                0x0001 /* Fortran-order contiguous */
#define OC_TENSOR_ZERO_STRIDES          0x0002
#define OC_TENSOR_CONTIGUOUS            0x0004 /* Contiguous after normalization */
#define OC_TENSOR_CONTIGUOUS_SET        0x0008
#define OC_TENSOR_SELF_OVERLAPPING      0x0010
#define OC_TENSOR_SELF_OVERLAPPING_SET  0x0020
#define OC_TENSOR_SHAPE_MASK            0x003F
#define OC_TENSOR_EXTENT                0x0040 /* Extent information is up-to-date */
#define OC_TENSOR_RESERVED_FLAG_1       0x0080
#define OC_TENSOR_RESERVED_FLAG_2       0x0100
#define OC_TENSOR_RESERVED_FLAG_3       0x0200
#define OC_TENSOR_RESERVED_FLAG_4       0x0400
#define OC_TENSOR_RESERVED_FLAG_5       0x0800

/* Byte-order information */
#define OC_TENSOR_BYTESWAPPED           0x1000

/* Read-only information */
#define OC_TENSOR_READONLY              0x2000

/* Reserved */
#define OC_TENSOR_RESERVED_FLAG_6       0x4000
#define OC_TENSOR_RESERVED_FLAG_7       0x8000


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef uint32_t  OcTensorFlags;

typedef struct
{  int            ndims;         /* Number of dimensions                  */
   int            capacity;      /* Size and strides buffer capacity      */
   OcSize        *size;          /* Dimensions of the tensor              */
   OcIndex       *strides;       /* Strides between elements in bytes     */
   OcSize         offset;        /* Offset within storage in bytes        */
   OcStorage     *storage;       /* Tensor storage                        */
   OcDevice      *device;        /* Borrowed: storage -> stream -> device */
   OcDType        dtype;         /* Data type of the tensor               */
   int            elemsize;      /* Element size                          */
   OcSize         nelem;         /* Number of elements                    */
   OcSize         blockOffset;   /* Offset within data block in bytes     */
   OcSize         blockExtent;   /* Extent of data block in bytes         */
   long int       refcount;      /* Number of references to the tensor    */
   OcTensorFlags  flags;         /* Flags to indicate tensor properties   */
} OcTensor;


/* ===================================================================== */
/* Macros                                                                */
/* ===================================================================== */

/* Reference count macros allowing NULL tensors; */
/* these may be replaced by functions later.     */
#define OcXIncrefTensor(tensor)      OcIncrefTensor(tensor)
#define OcXDecrefTensor(tensor)      OcDecrefTensor(tensor)

/* Properties */
#define OcTensor_ndims(tensor)       ((tensor) -> ndims)
#define OcTensor_nelem(tensor)       ((tensor) -> nelem)
#define OcTensor_device(tensor)      ((tensor) -> device)
#define OcTensor_deviceType(tensor)  ((tensor) -> device -> type)
#define OcTensor_deviceName(tensor)  ((tensor) -> device -> name)
#define OcTensor_data(tensor)        ((tensor) -> storage -> data + (tensor) -> offset)
#define OcTensor_storage(tensor)     ((tensor) -> storage)
#define OcTensor_dtype(tensor)       ((tensor) -> dtype)
#define OcTensor_stream(tensor)      ((tensor) -> storage -> stream)

/* Synchronization macros */
#define OcTensor_startRead(T,R)      OcStorage_startRead((T) -> storage, (R) -> storage)
#define OcTensor_finishRead(T,R)     OcStorage_finishRead((T) -> storage, (R) -> storage)
#define OcTensor_startWrite(T,W)     OcStorage_startWrite((T) -> storage, (W) -> storage)
#define OcTensor_finishWrite(T,W)    OcStorage_finishWrite((T) -> storage, (W) -> storage)
#define OcTensor_synchronize(T)      OcStorage_synchronize((T) -> storage)
#define OcTensor_update(T)           OcStorage_update((T) -> storage)


/* ===================================================================== */
/* Additional macros                                                     */
/* ===================================================================== */

/* Data type queries */
#define OcTensor_isReal(tensor)    (!(OcDType_isComplex((tensor) -> dtype)))
#define OcTensor_isComplex(tensor) (OcDType_isComplex((tensor) -> dtype))

/* Miscellaneous */
#define OcTensor_isValidSource(tensor)   1
#define OcTensor_isEmpty(tensor)         ((tensor) -> nelem == 0)


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Functions to manipulate tensor reference counts */
OC_API OcTensor *OcIncrefTensor(OcTensor *tensor);
OC_API void      OcDecrefTensor(OcTensor *tensor);

/* Allocate and initialize a tensor object */
OC_API OcTensor *OcAllocateTensor(int ndims, OcDType dtype);

OC_API OcTensor *OcTensor_shallowCopy          (OcTensor *tensor);
OC_API void      OcTensor_setReadOnly          (OcTensor *tensor, int readonly);

/* Tensor shapes */
OC_API int       OcTensor_allocDims            (OcTensor *tensor, int ndims);
OC_API int       OcTensor_updateShape          (OcTensor *tensor, int ndims, OcSize *size, OcIndex *strides,
                                                int updateFlags, int updateSelfOverlap, int updateExtent);
OC_API void      OcTensor_updateShapeFlags     (OcTensor *tensor, int updateSelfOverlap);
OC_API int       OcTensor_updateExtent         (OcTensor *tensor);
OC_API int       OcTensor_extent               (OcTensor *tensor, OcSize *offset, OcSize *extent);
OC_API OcTensor *OcTensor_removeMatchingRepeats(OcTensor *tensor, OcTensor *reference);
OC_API OcTensor *OcTensor_removeRepeats        (OcTensor *tensor);

/* Query functions */
OC_API int       OcTensor_isScalar             (OcTensor *tensor);
OC_API int       OcTensor_isLinear             (OcTensor *tensor);
OC_API int       OcTensor_isContiguous         (OcTensor *tensor);
OC_API int       OcTensor_isSelfOverlapping    (OcTensor *tensor);
OC_API int       OcTensor_isByteswapped        (OcTensor *tensor);
OC_API int       OcTensor_isReadOnly           (OcTensor *tensor);
OC_API int       OcTensor_isAligned            (OcTensor *tensor);
OC_API int       OcTensor_isDetached           (OcTensor *tensor, int flagStorage);
OC_API int       OcTensor_isValidDest          (OcTensor *tensor, int allowZeroStrides);
OC_API int       OcTensor_hasOrder             (OcTensor *tensor, char type);
OC_API int       OcTensor_hasValidAlignment    (OcTensor *tensor);
OC_API int       OcTensor_hasZeroStrides       (OcTensor *tensor);
OC_API OcSize    OcTensor_repeatCount          (OcTensor *tensor);

/* Tensor functions */
OC_API int       OcTensors_match               (OcTensor *tensor1, OcTensor *tensor2);
OC_API int       OcTensors_overlap             (OcTensor *tensor1, OcTensor *tensor2);
OC_API int       OcTensors_haveSameSize        (OcTensor *tensor1, OcTensor *tensor2);
OC_API int       OcTensors_haveSameByteOrder   (OcTensor *tensor1, OcTensor *tensor2);
OC_API int       OcTensors_haveIdenticalLayout (OcTensor *tensor1, OcTensor *tensor2);
OC_API int       OcTensors_haveCompatibleLayout(OcTensor *tensor1, OcTensor *tensor2);

#endif
