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

#ifndef __OC_DTYPES_GPU_H__
#define __OC_DTYPES_GPU_H__


/* -------------------------------------------------------------------- */
/* Corresponding C type - device specific                               */
/* -------------------------------------------------------------------- */
#define OCTYPE_C_TYPE_B(TYPE) OcCuda ## TYPE
#define OCTYPE_C_TYPE(TYPE) OCTYPE_C_TYPE_B(TYPE)


/* -------------------------------------------------------------------- */
/* Basic or special type - device specific                              */
/* -------------------------------------------------------------------- */
#define OCTYPE_BASIC_Bool      0
#define OCTYPE_BASIC_Int8      1
#define OCTYPE_BASIC_Int16     1
#define OCTYPE_BASIC_Int32     1
#define OCTYPE_BASIC_Int64     1
#define OCTYPE_BASIC_UInt8     1
#define OCTYPE_BASIC_UInt16    1
#define OCTYPE_BASIC_UInt32    1
#define OCTYPE_BASIC_UInt64    1
#define OCTYPE_BASIC_Half      0
#define OCTYPE_BASIC_Float     1
#define OCTYPE_BASIC_Double    1
#define OCTYPE_BASIC_CHalf     0
#define OCTYPE_BASIC_CFloat    1
#define OCTYPE_BASIC_CDouble   1


/* -------------------------------------------------------------------- */
/* Zero values for special types                                        */
/* -------------------------------------------------------------------- */
#define OCTYPE_ZERO_Bool        0
#define OCTYPE_ZERO_Half        __float2half(0)


/* -------------------------------------------------------------------- */
/* Work types for special data types - device specific                  */
/* -------------------------------------------------------------------- */
#define OCTYPE_WORKTYPE_Bool    Bool
#define OCTYPE_WORKTYPE_Half    Float


/* -------------------------------------------------------------------- */
/* Conversion from type to worktype and back - device specific          */
/* -------------------------------------------------------------------- */
#define OCTYPE_SPECIAL_TO_BASIC_Bool(TYPE,X)  ((OCTYPE_C_TYPE(TYPE))X)
#define OCTYPE_BASIC_TO_SPECIAL_Bool(TYPE,X)  (((OCTYPE_C_TYPE(TYPE))(X) == 0) ? 0 : 1)
#define OCTYPE_SPECIAL_TO_BASIC_Half(TYPE,X)  (OCTYPE_C_TYPE(TYPE))__half2float(X)
#define OCTYPE_BASIC_TO_SPECIAL_Half(TYPE,X)  __float2half((float)(X))

/* -------------------------------------------------------------------- */
/* Conversion between special types                                     */
/* -------------------------------------------------------------------- */
#define OCTYPE_CONVERT_Bool_TO_Bool(X) X
#define OCTYPE_CONVERT_Bool_TO_Half(X) __float2half((float)(X))
#define OCTYPE_CONVERT_Half_TO_Bool(X) (__half2float(X) == 0 ? 0 : 1)
#define OCTYPE_CONVERT_Half_TO_Half(X) X

#endif
