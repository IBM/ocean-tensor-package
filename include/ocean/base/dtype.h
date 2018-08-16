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

#ifndef __OC_DTYPE_H__
#define __OC_DTYPE_H__

#include "ocean/base/api.h"
#include "ocean/base/types.h"

#include <limits.h>


/* -------------------------------------------------------------------- */
/* Enumeration of all supported data types                              */
/* -------------------------------------------------------------------- */
typedef enum
{  OcDTypeBool = 0,
   OcDTypeUInt8, OcDTypeUInt16, OcDTypeUInt32, OcDTypeUInt64,
   OcDTypeInt8,  OcDTypeInt16,  OcDTypeInt32,  OcDTypeInt64,
   OcDTypeHalf,  OcDTypeFloat,  OcDTypeDouble,
   OcDTypeCHalf, OcDTypeCFloat, OcDTypeCDouble,
   OcDTypeNone  /* Internal usage only! */   
} OcDType;

/* Number of user data types */
#define OC_DTYPE_COUNT 15    /* Excludes OcDTypeNone */
#define OC_DTYPE_MASK  0x1F  /* Includes OcDTypeNone */


/* -------------------------------------------------------------------- */
/* Data type information structure                                      */
/* -------------------------------------------------------------------- */
typedef struct
{  int      size;
   int      parts;
   int      isNumber;
   int      isSigned;
   int      isFloat;
   int      isComplex;
   char    *name;
   char    *uname;
} OcDTypeInfo;

typedef struct
{  OcDType  signedType;
   OcDType  modulusType;
   int      nbits;
} OcDTypeInfo2;


/* -------------------------------------------------------------------- */
/* Table with information about all suppored data types                 */
/* -------------------------------------------------------------------- */
extern OcDTypeInfo  oc_dtype_info[];
extern OcDTypeInfo2 oc_dtype_info2[];


/* -------------------------------------------------------------------- */
/* Function declarations                                                */
/* -------------------------------------------------------------------- */
OC_API OcDType OcDType_getCommonType(OcDType dtype1, OcDType dtype2);
OC_API OcDType OcDType_getBaseType(OcDType dtype); /* Complex base types */
OC_API OcDType OcDType_getFloatType(OcDType dtype); /* Floating-point type */
OC_API OcDType OcDType_getComplexType(OcDType dtype); /* Complex types */

/* Default data type */
OC_API void    OcDType_setDefault(OcDType dtype);
OC_API OcDType OcDType_getDefault(void);
OC_API OcDType OcDType_applyDefault(OcDType dtype);

/* Range checks */
OC_API int     OcDType_inRangeInt64  (OcInt64 value, OcDType dtype);
OC_API int     OcDType_inRangeUInt64 (OcUInt64 value, OcDType dtype);
OC_API int     OcDType_inRangeDouble (OcDouble value, OcDType dtype);
OC_API int     OcDType_inRangeCDouble(OcCDouble value, OcDType dtype);

OC_API OcDType OcDType_getTypeInt64         (OcInt64 value);
OC_API OcDType OcDType_getTypeUInt64        (OcUInt64 value);
OC_API OcDType OcDType_getTypeDouble        (OcDouble value);


/* -------------------------------------------------------------------- */
/* Macros to access data type properties                                */
/* -------------------------------------------------------------------- */
#define OcDType_size(dtype)             oc_dtype_info[dtype].size
#define OcDType_nbits(dtype)            oc_dtype_info2[dtype].nbits
#define OcDType_parts(dtype)            oc_dtype_info[dtype].parts
#define OcDType_isBool(dtype)           (dtype == OcDTypeBool)
#define OcDType_isUInt(dtype)           (((dtype) >= OcDTypeUInt8) && ((dtype) <= OcDTypeUInt64))
#define OcDType_isInt(dtype)            (((dtype) >= OcDTypeInt8) && ((dtype) <= OcDTypeInt64))

#define OcDType_isNumber(dtype)         oc_dtype_info[dtype].isNumber
#define OcDType_isSigned(dtype)         oc_dtype_info[dtype].isSigned
#define OcDType_isUnsigned(dtype)       (oc_dtype_info[dtype].isSigned == 0)
#define OcDType_isInteger(dtype)        (oc_dtype_info[dtype].isFloat == 0)
#define OcDType_isFloat(dtype)          oc_dtype_info[dtype].isFloat
#define OcDType_isComplex(dtype)        oc_dtype_info[dtype].isComplex
#define OcDType_isReal(dtype)           (OcDType_isComplex(dtype) == 0)
#define OcDType_name(dtype)             oc_dtype_info[dtype].name
#define OcDType_uname(dtype)            oc_dtype_info[dtype].uname
#define OcDType_baseSize(dtype)         (oc_dtype_info[dtype].size / oc_dtype_info[dtype].parts)
#define OcDType_getType(dtype)          (dtype)
#define OcDType_getSignedType(dtype)    oc_dtype_info2[dtype].signedType
#define OcDType_getModulusType(dtype)   oc_dtype_info2[dtype].modulusType

#endif
