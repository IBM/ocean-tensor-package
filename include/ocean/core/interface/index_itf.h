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

#ifndef __OC_MODULE_CORE_ITF_INDEX_H__
#define __OC_MODULE_CORE_ITF_INDEX_H__

#include "ocean/base/scalar.h"
#include "ocean/base/tensor.h"
#include "ocean/base/index.h"
#include "ocean/base/api.h"


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Tensor index creation and deletion */
OC_API OcTensorIndex *OcTensorIndex_create            (void);
OC_API OcTensorIndex *OcTensorIndex_createWithCapacity(int capacity);
OC_API OcTensorIndex *OcTensorIndex_shallowCopy       (OcTensorIndex *index);
OC_API int            OcTensorIndex_ensureCapacity    (OcTensorIndex *index, int capacity);

/* Reference counting */
OC_API OcTensorIndex *OcIncrefTensorIndex             (OcTensorIndex *index);
OC_API void           OcDecrefTensorIndex             (OcTensorIndex *index);

/* Addition of index elements - return element index */
OC_API int            OcTensorIndex_addScalar         (OcTensorIndex *index, OcScalar *scalar);
OC_API int            OcTensorIndex_addTensor         (OcTensorIndex *index, OcTensor *tensor);
OC_API int            OcTensorIndex_addMask           (OcTensorIndex *index, OcTensor *tensor); /* boolean tensor */
OC_API int            OcTensorIndex_addIndices        (OcTensorIndex *index, OcTensor *tensor); /* integer tensor */
OC_API int            OcTensorIndex_addInsert         (OcTensorIndex *index, int ndims);
OC_API int            OcTensorIndex_addAll            (OcTensorIndex *index, int ndims);
OC_API int            OcTensorIndex_addEllipsis       (OcTensorIndex *index);
OC_API int            OcTensorIndex_addRange          (OcTensorIndex *index, OcScalar *start, OcScalar *stop, OcScalar *step);
OC_API int            OcTensorIndex_addSteps          (OcTensorIndex *index, OcScalar *start, OcScalar *step, OcScalar *nelem);
OC_API int            OcTensorIndex_addIndex          (OcTensorIndex *index, OcTensorIndex *index2);
OC_API int            OcTensorIndex_getNumInputDims   (OcTensorIndex *index);
OC_API int            OcTensorIndex_getNumOutputDims  (OcTensorIndex *index);
OC_API int            OcTensorIndex_getInputDims      (OcTensorIndex *index, OcSize *size, int *ndims);
OC_API int            OcTensorIndex_getOutputDims     (OcTensorIndex *index, OcSize *size, int *ndims);
OC_API int            OcTensorIndex_getInputStrides   (OcTensorIndex *index, OcIndex *strides, int *ndims);
OC_API int            OcTensorIndex_bind              (OcTensorIndex **index, int flagAutoExtend, int ndims,
                                                       OcSize *size, OcIndex *strides, OcTensorIndex **result);
OC_API int            OcTensorIndex_createView        (OcTensorIndex *index, OcTensor *tensor, OcTensorIndexView **view);
OC_API void           OcTensorIndex_deleteView        (OcTensorIndexView *view);
OC_API int            OcTensorIndex_detach            (OcTensorIndex **index, OcTensorIndex **result);
OC_API int            OcTensorIndex_setDevice         (OcTensorIndex **index, OcDevice *device, OcTensorIndex **result);
OC_API int            OcTensorIndex_clear             (OcTensorIndex *index);
OC_API int            OcTensorIndex_isScalar          (OcTensorIndex *index);
OC_API int            OcTensorIndex_isView            (OcTensorIndex *index);
OC_API int            OcTensorIndex_isBound           (OcTensorIndex *index, int flagStrides);

/* Formatting */
OC_API int            OcTensorIndex_format            (OcTensorIndex *index, char **str, const char *header, const char *footer);
OC_API int            OcTensorIndex_display           (OcTensorIndex *index);

#endif
