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

#ifndef __OC_GENERATE_MACROS_H__
#define __OC_GENERATE_MACROS_H__

#include "ocean/base/macros.h"
#include "ocean/base/dtype.h"


/* ==================================================================== */
/* Macros to apply the OC_TEMPLATE macro for different data types       */
/* ==================================================================== */

/* Basic types */
#define OC_GENERATE_Bool          OC_TEMPLATE(Bool)
#define OC_GENERATE_UInt8         OC_TEMPLATE(UInt8)
#define OC_GENERATE_UInt16        OC_TEMPLATE(UInt16)
#define OC_GENERATE_UInt32        OC_TEMPLATE(UInt32)
#define OC_GENERATE_UInt64        OC_TEMPLATE(UInt64)
#define OC_GENERATE_Int8          OC_TEMPLATE(Int8)
#define OC_GENERATE_Int16         OC_TEMPLATE(Int16)
#define OC_GENERATE_Int32         OC_TEMPLATE(Int32)
#define OC_GENERATE_Int64         OC_TEMPLATE(Int64)
#define OC_GENERATE_Half          OC_TEMPLATE(Half)
#define OC_GENERATE_Float         OC_TEMPLATE(Float)
#define OC_GENERATE_Double        OC_TEMPLATE(Double)
#define OC_GENERATE_CHalf         OC_TEMPLATE(CHalf)
#define OC_GENERATE_CFloat        OC_TEMPLATE(CFloat)
#define OC_GENERATE_CDouble       OC_TEMPLATE(CDouble)

/* Type classes */
#define OC_GENERATE_BOOL          OC_GENERATE_Bool

#define OC_GENERATE_UINT          OC_GENERATE_UInt8 \
                                  OC_GENERATE_UInt16 \
                                  OC_GENERATE_UInt32 \
                                  OC_GENERATE_UInt64


#define OC_GENERATE_INT           OC_GENERATE_Int8 \
                                  OC_GENERATE_Int16 \
                                  OC_GENERATE_Int32 \
                                  OC_GENERATE_Int64

#define OC_GENERATE_INT_TYPES     OC_GENERATE_UINT \
                                  OC_GENERATE_INT

#define OC_GENERATE_FLOAT_TYPES   OC_GENERATE_Half \
                                  OC_GENERATE_Float \
                                  OC_GENERATE_Double

#define OC_GENERATE_CFLOAT_TYPES  OC_GENERATE_CHalf \
                                  OC_GENERATE_CFloat \
                                  OC_GENERATE_CDouble

#define OC_GENERATE_ALL_TYPES     OC_GENERATE_BOOL \
                                  OC_GENERATE_INT_TYPES \
                                  OC_GENERATE_FLOAT_TYPES \
                                  OC_GENERATE_CFLOAT_TYPES

#endif
