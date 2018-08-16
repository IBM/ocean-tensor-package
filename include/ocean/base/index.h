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

#ifndef __OC_INDEX_H__
#define __OC_INDEX_H__

#include "ocean/base/tensor.h"
#include "ocean/base/types.h"


/* ===================================================================== */
/* Type definitions for indexing                                         */
/* ===================================================================== */

typedef enum {OC_IDX_SCALAR,       /* A[3]                                             */
              OC_IDX_INDICES,      /* Index array                                      */
              OC_IDX_MASK,         /* Boolean mask                                     */
              OC_IDX_OFFSET,       /* Internal usage - list of offsets                 */
              OC_IDX_INSERT,       /* A[NONE]                                          */
              OC_IDX_ALL,          /* A[:]                                             */
              OC_IDX_ELLIPSIS,     /* A[...]                                           */
              OC_IDX_RANGE,        /* A[1:5:2] - Use OcDTypeNone to use default        */
              OC_IDX_STEPS         /* Indices start + idx * step for idx = 0...nelem-1 */
             } OcTensorIndexType;

typedef enum {OC_IDX_MODE_BASIC,   /* Indices must be in the range [0,n) where n is    */
                                   /* size of the relevant dimension. For the range it */
                                   /* is possible to specify ranges that are out of    */
                                   /* bound and indexing is done by intersecting the   */
                                   /* range with the domain [0,n). When A = [0,1,2,3], */
                                   /* taking A[4::-2] generates the range 4,2,0,-2,... */
                                   /* and its intersection with 0,1,2,3 gives indices  */
                                   /* A[[2,0]] = [2,0].                                */

              OC_IDX_MODE_PYTHON   /* Wrapped indices for scalars, ranges, and indices */
                                   /* for ranges this mode additionaly has the feature */
                                   /* of mapping out-of-bound start indices to the     */
                                   /* nearest bound. For example for A = [0,1,2,3] we  */
                                   /* A[-100:2] = [0,1], moreover A[4::-2] does not    */
                                   /* generate A[[2,0]] but instead gives A[[3,1]], as */
                                   /* the start index 4 is first changes to 3.         */
             } OcTensorIndexMode;


/* Base class for index elements */
typedef struct
{  OcTensorIndexType  type;
   OcSize            *size;
   OcIndex           *strides;
   int                nInputDims;
   int                nOutputDims;
   long int           refcount;
} OcTensorIndexElem;

typedef struct
{  OcTensorIndexElem HEAD;
   OcIndex           index;
} OcTensorIndexElem_Scalar;

typedef struct
{  OcTensorIndexElem  HEAD;
   OcTensor          *tensor;
   OcIndex            indexMin[OC_TENSOR_MAX_DIMS];
   OcIndex            indexMax[OC_TENSOR_MAX_DIMS];
} OcTensorIndexElem_Indices;

typedef struct
{  OcTensorIndexElem  HEAD;
   OcTensor          *tensor;
   OcSize             nnz;
   int                flagNnz;
} OcTensorIndexElem_Mask;

typedef struct
{  OcTensorIndexElem  HEAD;
   OcTensor          *tensor;
   OcIndex            range; /* Memory range, used in lieu of stride when simplifying the index */
} OcTensorIndexElem_Offset;

typedef struct
{  OcTensorIndexElem  HEAD;
} OcTensorIndexElem_Insert;

typedef struct
{  OcTensorIndexElem  HEAD;
} OcTensorIndexElem_All;

typedef struct
{  OcTensorIndexElem  HEAD;
} OcTensorIndexElem_Ellipsis;

typedef struct
{  OcTensorIndexElem  HEAD;
   OcIndex            start, step, stop;
   int                flagStart;
   int                flagStop;
} OcTensorIndexElem_Range;

typedef struct
{  OcTensorIndexElem  HEAD;
   OcIndex            start, step;
   OcSize             nelem;
} OcTensorIndexElem_Steps;

typedef struct
{  OcTensorIndexElem **elem;
   int                 capacity;
   int                 n;
   long int            refcount;
} OcTensorIndex;

typedef struct
{  int        ndims;
   OcTensor  *view;
   OcTensor **offsets;                           /* Elements can be NULL */
   OcIndex    offsetStrides[OC_TENSOR_MAX_DIMS]; /* Strides corresponding to offsets, used for sorting dimensions */
} OcTensorIndexView;


#endif
