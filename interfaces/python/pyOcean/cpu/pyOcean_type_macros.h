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

#ifndef __PYOCEAN_TYPE_MACROS_H__
#define __PYOCEAN_TYPE_MACROS_H__

/* Macros to apply the PYOC_DECLARE_TYPE macro for different data types */

#define PYOC_DECLARE_BOOL_TYPE \
   PYOC_DECLARE_TYPE(Bool,      bool,     OcDTypeBool,    OcBool  ) \

#define PYOC_DECLARE_INT_TYPES \
   PYOC_DECLARE_TYPE(Int8,      int8,     OcDTypeInt8,    OcInt8  ) \
   PYOC_DECLARE_TYPE(Int16,     int16,    OcDTypeInt16,   OcInt16 ) \
   PYOC_DECLARE_TYPE(Int32,     int32,    OcDTypeInt32,   OcInt32 ) \
   PYOC_DECLARE_TYPE(Int64,     int64,    OcDTypeInt64,   OcInt64 ) \
   PYOC_DECLARE_TYPE(UInt8,     uint8,    OcDTypeUInt8,   OcUInt8 ) \
   PYOC_DECLARE_TYPE(UInt16,    uint16,   OcDTypeUInt16,  OcUInt16) \
   PYOC_DECLARE_TYPE(UInt32,    uint32,   OcDTypeUInt32,  OcUInt32) \
   PYOC_DECLARE_TYPE(UInt64,    uint64,   OcDTypeUInt64,  OcUInt64)

#define PYOC_DECLARE_FLOAT_TYPES \
   PYOC_DECLARE_TYPE(Half,      half,     OcDTypeHalf,    OcHalf  ) \
   PYOC_DECLARE_TYPE(Float,     float,    OcDTypeFloat,   OcFloat ) \
   PYOC_DECLARE_TYPE(Double,    double,   OcDTypeDouble,  OcDouble)

#define PYOC_DECLARE_COMPLEX_TYPES \
   PYOC_DECLARE_TYPE(CHalf,     chalf,    OcDTypeCHalf,   OcCHalf  ) \
   PYOC_DECLARE_TYPE(CFloat,    cfloat,   OcDTypeCFloat,  OcCFloat ) \
   PYOC_DECLARE_TYPE(CDouble,   cdouble,  OcDTypeCDouble, OcCDouble)

#define PYOC_DECLARE_ALL_TYPES \
   PYOC_DECLARE_BOOL_TYPE \
   PYOC_DECLARE_INT_TYPES \
   PYOC_DECLARE_FLOAT_TYPES \
   PYOC_DECLARE_COMPLEX_TYPES

#endif
