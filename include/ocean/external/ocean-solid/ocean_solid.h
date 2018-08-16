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

#ifndef __OC_EXTERNAL_SOLID_H__
#define __OC_EXTERNAL_SOLID_H__

#include "ocean.h"
#include "solid.h"


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{  int          ndims;
   solid_size   size[OC_TENSOR_MAX_DIMS];
   solid_index  strides[OC_TENSOR_MAX_DIMS];
   void        *ptr;
}  OcSolidElemwise1;

typedef struct
{  int          ndims;
   solid_size   size[OC_TENSOR_MAX_DIMS];
   solid_index  strides1[OC_TENSOR_MAX_DIMS];
   solid_index  strides2[OC_TENSOR_MAX_DIMS];
   void        *ptr1;
   void        *ptr2;
}  OcSolidElemwise2;

typedef struct
{  int          ndims1;
   int          ndims2;
   solid_size   size1[OC_TENSOR_MAX_DIMS];
   solid_size   size2[OC_TENSOR_MAX_DIMS];
   solid_index  strides1[OC_TENSOR_MAX_DIMS];
   solid_index  strides2[OC_TENSOR_MAX_DIMS];
   void        *ptr1;
   void        *ptr2;
}  OcSolidElemwise2b;

typedef struct
{  int          ndims;
   solid_size   size[OC_TENSOR_MAX_DIMS];
   solid_index  strides1[OC_TENSOR_MAX_DIMS];
   solid_index  strides2[OC_TENSOR_MAX_DIMS];
   solid_index  strides3[OC_TENSOR_MAX_DIMS];
   void        *ptr1;
   void        *ptr2;
   void        *ptr3;
}  OcSolidElemwise3;

typedef struct
{  int          ndims;
   solid_size   size[OC_TENSOR_MAX_DIMS];
   solid_index  strides[OC_TENSOR_MAX_DIMS];
   void        *ptr;
}  OcSolidReduce;

typedef struct
{  int          ndims;
   int          rdims;
   solid_size   size[OC_TENSOR_MAX_DIMS];
   solid_size   rsize[OC_TENSOR_MAX_DIMS];
   solid_index  strides1[OC_TENSOR_MAX_DIMS];
   solid_index  strides2[OC_TENSOR_MAX_DIMS];
   solid_index  rstrides[OC_TENSOR_MAX_DIMS];
   void        *ptr1;
   void        *ptr2;
}  OcSolidReduceAxis;

typedef struct
{  int          ndims;
   solid_int64 *offsets[OC_TENSOR_MAX_DIMS];
   solid_size   size[OC_TENSOR_MAX_DIMS];
   solid_index  strides[OC_TENSOR_MAX_DIMS];
   void        *ptr;
}  OcSolidIndex1;

typedef struct
{  int          ndims;
   solid_int64 *offsets[OC_TENSOR_MAX_DIMS];
   solid_size   size[OC_TENSOR_MAX_DIMS];
   solid_index  strides1[OC_TENSOR_MAX_DIMS];
   solid_index  strides2[OC_TENSOR_MAX_DIMS];
   void        *ptr1;
   void        *ptr2;
}  OcSolidIndex2;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Convert Ocean to Solid data type */
OC_API int OcSolid_getType(OcDType dtype);

/* Convert Ocean to Solid scalar */
OC_API int OcSolid_getScalar(OcScalar *scalar, solid_scalar *value);

/* Convert size and stride information */
OC_API void OcSolid_analyzeElemwise1 (OcSolidElemwise1  *config, OcTensor *tensor);
OC_API void OcSolid_analyzeElemwise2 (OcSolidElemwise2  *config, OcTensor *tensor1, OcTensor *tensor2);
OC_API void OcSolid_analyzeElemwise2b(OcSolidElemwise2b *config, OcTensor *tensor1, OcTensor *tensor2);
OC_API void OcSolid_analyzeElemwise3 (OcSolidElemwise3  *config, OcTensor *tensor1, OcTensor *tensor2, OcTensor *tensor3);
OC_API void OcSolid_analyzeReduce    (OcSolidReduce     *config, OcTensor *tensor, int flagRemoveRepeats);
OC_API void OcSolid_analyzeReduceAxis(OcSolidReduceAxis *config, OcTensor *src, OcTensor *dst, int n, int *axes, int flagRemoveRepeats);
OC_API void OcSolid_analyzeIndex1    (OcSolidIndex1     *config, OcTensor *view, OcTensor **offsets, OcIndex *offsetStrides);
OC_API void OcSolid_analyzeIndex2    (OcSolidIndex2     *config, OcTensor *view, OcTensor **offsets, OcIndex *offsetStrides,
                                      int flagSetIndex, OcTensor *tensor);


/* ===================================================================== */
/* Macros                                                                */
/* ===================================================================== */

/* Get function pointers - one type */
#define OC_SOLID_FUNPTR(OCEAN_FUNSTR, SOLID_LUT, FUNPTR, TYPE, DEVICE) \
   {  int __index; \
      FUNPTR = 0; \
      __index = OcSolid_getType(TYPE); \
      if ((__index == -1) || \
          ((FUNPTR = SOLID_LUT[__index]) == 0)) \
      {  OcErrorMessage("The %s function is not available for type %s on %s", \
                        OCEAN_FUNSTR, OcDType_name(TYPE), DEVICE); \
      } \
   }

/* Get function pointers - two types */
#define OC_SOLID_FUNPTR2(OCEAN_FUNSTR, SOLID_LUT, FUNPTR, TYPE1, TYPE2, DEVICE) \
   {  int __index1, __index2; \
      FUNPTR = 0; \
      __index1 = OcSolid_getType(TYPE1); \
      __index2 = OcSolid_getType(TYPE2); \
      if ((__index1 == -1) || (__index2 == -1) || \
          ((FUNPTR = SOLID_LUT[__index1][__index2]) == 0)) \
      {  OcErrorMessage("The %s function is not available for types %s and %s on %s", \
                        OCEAN_FUNSTR, OcDType_name(TYPE1), OcDType_name(TYPE2), DEVICE); \
      } \
   }

/* Import error message */
#define OC_SOLID_ERRMSG() OcErrorMessage("%s", solid_errmsg)

/* Call a function */
#define OC_SOLID_CALL(RESULT, FUNPTR, ...) \
   if (((RESULT) = FUNPTR(__VA_ARGS__)) != 0) OC_SOLID_ERRMSG()

#else
#error ALREADY INCLUDED
#endif
