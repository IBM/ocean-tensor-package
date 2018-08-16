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

#ifndef __OC_SHAPE_H__
#define __OC_SHAPE_H__

#include "ocean/base/api.h"
#include "ocean/base/dtype.h"
#include "ocean/base/types.h"


/* Determine number of elements, tensor extent, or strides */
/* these routines check for integer arithmetic overflows.  */
OC_API int  OcShape_nelem            (int ndims, OcSize *size, OcSize *nelem);
OC_API int  OcShape_extent           (int ndims, OcSize *size, OcIndex *strides, OcSize elemsize,
                                      OcSize *dataOffset, OcSize *dataExtent);
OC_API int  OcShape_getStrides       (int ndims, OcSize *size, int elemsize, OcIndex *strides, char type);

/* Normalize strides to be positive */
OC_API void OcShape_normalizeStrides (int ndims, OcSize *size, OcIndex *strides, OcIndex *offset);
OC_API void OcShape_normalizeStrides2(int ndims, OcSize *size, OcIndex *strides1, OcIndex *strides2,
                                      OcIndex *offset1, OcIndex *offset2);

/* Sorting */
OC_API void OcShape_sortStrides      (int ndims, OcSize *size, OcIndex *strides);
OC_API void OcShape_sortStrides2     (int ndims, OcSize *size, OcIndex *strides1, OcIndex *strides2);
OC_API void OcShape_sortStrides3     (int ndims, OcSize *size, OcIndex *strides1, OcIndex *strides2, OcIndex *strides3);

/* Simplification */
OC_API void OcShape_mergeBlocks      (int *ndims, OcSize *size, OcIndex *strides);
OC_API void OcShape_mergeBlocks2     (int *ndims, OcSize *size, OcIndex *strides1, OcIndex *strides2);
OC_API void OcShape_mergeBlocks3     (int *ndims, OcSize *size, OcIndex *strides1, OcIndex *strides2, OcIndex *strides3);

/* Broadcasting */
OC_API int  OcShape_broadcastLeft    (int *ndims, OcSize *size, int ndimsRef, OcSize *sizeRef);
OC_API int  OcShape_broadcastRight   (int *ndims, OcSize *size, int ndimsRef, OcSize *sizeRef);

/* Shape properties */
OC_API int  OcShape_isValidSize      (int ndims, OcSize *size);
OC_API int  OcShape_isContiguous     (int ndims, OcSize *size, OcIndex *strides, int elemsize);
OC_API int  OcShape_isSelfOverlapping(int ndims, OcSize *size, OcIndex *strides, int elemsize);
OC_API int  OcShapes_overlap         (int ndims1, OcSize *size1, OcIndex *strides1, OcIndex offset1, int elemsize1,
                                      int ndims2, OcSize *size2, OcIndex *strides2, OcIndex offset2, int elemsize2);
OC_API int  OcShapes_match           (int ndims1, OcSize *size1, int ndims2, OcSize *size2);

/* Display shape information (debugging) */
OC_API void OcShape_display(int ndims, OcSize *size, OcIndex *strides);

#endif
