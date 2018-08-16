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

#ifndef __OC_MODULE_CORE_ITF_SCALAR_H__
#define __OC_MODULE_CORE_ITF_SCALAR_H__

#include "ocean/base/scalar.h"
#include "ocean/base/api.h"


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* General functions */
OC_API OcScalar     *OcScalar_create       (OcDType dtype);
OC_API OcScalar     *OcScalar_clone        (OcScalar *scalar);
OC_API OcScalar     *OcScalar_castDType    (OcScalar *scalar, OcDType dtype);
OC_API void          OcScalar_free         (OcScalar *scalar);
OC_API void          OcScalar_castTo       (OcScalar *scalar, OcDType dtype, OcScalar *result);
OC_API void          OcScalar_copy         (OcScalar *src, OcScalar *dst);
OC_API void          OcScalar_byteswap     (OcScalar *scalar);
OC_API void          OcScalar_getReal      (OcScalar *scalar, OcScalar *result);
OC_API void          OcScalar_getImag      (OcScalar *scalar, OcScalar *result);
OC_API int           OcScalar_setReal      (OcScalar *scalar, OcScalar *value);
OC_API int           OcScalar_setImag      (OcScalar *scalar, OcScalar *value);

/* Constants and bounds */
OC_API void          OcScalar_setZero      (OcScalar *scalar, OcDType dtype);
OC_API void          OcScalar_setOne       (OcScalar *scalar, OcDType dtype);
OC_API void          OcScalar_setMin       (OcScalar *scalar, OcDType dtype);
OC_API void          OcScalar_setMax       (OcScalar *scalar, OcDType dtype);
OC_API void          OcScalar_setEps       (OcScalar *scalar, OcDType dtype);

/* Infinity and not-a-number */
OC_API void          OcScalar_floatInf     (OcScalar *scalar);
OC_API void          OcScalar_floatNaN     (OcScalar *scalar);
OC_API void          OcScalar_doubleInf    (OcScalar *scalar);
OC_API void          OcScalar_doubleNaN    (OcScalar *scalar);

/* Conversion without range checks */
OC_API void          OcScalar_fromInt64    (OcScalar *scalar, OcInt64 value);
OC_API void          OcScalar_fromUInt64   (OcScalar *scalar, OcUInt64 value);
OC_API void          OcScalar_fromDouble   (OcScalar *scalar, OcDouble value);
OC_API void          OcScalar_fromCDouble  (OcScalar *scalar, OcCDouble value);
OC_API void          OcScalar_fromComplex  (OcScalar *scalar, OcDouble real, OcDouble imag);

/* Conversion without range checks */
OC_API int           OcScalar_asBool       (OcScalar *scalar);
OC_API OcInt64       OcScalar_asInt64      (OcScalar *scalar);
OC_API OcUInt64      OcScalar_asUInt64     (OcScalar *scalar);
OC_API OcDouble      OcScalar_asDouble     (OcScalar *scalar);
OC_API OcCDouble     OcScalar_asCDouble    (OcScalar *scalar);

/* Direct copy of data */
OC_API void          OcScalar_exportData   (OcScalar *scalar, void *dstData, int byteswapped);
OC_API void          OcScalar_importData   (OcScalar *scalar, void *srcData, int byteswapped);
OC_API void          OcScalar_copyRaw      (void *src, OcDType srcType, void *dst, OcDType dstType);

/* Range checks */
OC_API int           OcScalar_inRange      (OcScalar *scalar, OcDType dtype);
OC_API OcDType       OcScalar_getCommonType(OcScalar *scalar, OcDType dtype);
OC_API OcDType       OcScalar_getFloatType (OcScalar *scalar);

/* Unary comparisons */
OC_API int           OcScalar_isZero       (OcScalar *scalar);
OC_API int           OcScalar_isInf        (OcScalar *scalar);
OC_API int           OcScalar_isNaN        (OcScalar *scalar);
OC_API int           OcScalar_isFinite     (OcScalar *scalar);
OC_API int           OcScalar_isLTZero     (OcScalar *scalar);
OC_API int           OcScalar_isLEZero     (OcScalar *scalar);
OC_API int           OcScalar_isGEZero     (OcScalar *scalar);
OC_API int           OcScalar_isGTZero     (OcScalar *scalar);
OC_API int           OcScalar_isEQZero     (OcScalar *scalar);
OC_API int           OcScalar_isLTOne      (OcScalar *scalar);
OC_API int           OcScalar_isEQOne      (OcScalar *scalar);
OC_API int           OcScalar_isGTOneAbs   (OcScalar *scalar);
OC_API int           OcScalar_isLTNegOne   (OcScalar *scalar);

/* Binary comparisons */
OC_API int           OcScalar_isLT         (OcScalar *scalar1, OcScalar *scalar2);
OC_API int           OcScalar_isLE         (OcScalar *scalar1, OcScalar *scalar2);
OC_API int           OcScalar_isEQ         (OcScalar *scalar1, OcScalar *scalar2);
OC_API int           OcScalar_isNE         (OcScalar *scalar1, OcScalar *scalar2);
OC_API int           OcScalar_isGE         (OcScalar *scalar1, OcScalar *scalar2);
OC_API int           OcScalar_isGT         (OcScalar *scalar1, OcScalar *scalar2);

/* Unary functions */
#define OC_TEMPLATE(NAME, X, Y, Z) \
OC_API int OcScalar_ ## NAME (OcScalar *scalar, OcScalar *result);
#include "ocean/core/generic/generate_tensor_unary.h"
#undef OC_TEMPLATE

/* Binary functions */
#define OC_TEMPLATE(NAME, X, Y) \
OC_API int OcScalar_ ## NAME (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result);
#include "ocean/core/generic/generate_scalar_binary.h"
#undef OC_TEMPLATE

/* Configuration */
OC_API int           OcScalar_getCastMode (void);
OC_API void          OcScalar_setCastMode (int mode);

/* Formatting */
OC_API int           OcScalar_format      (OcScalar *scalar, char **str, const char *header, const char *footer);
OC_API int           OcScalar_display     (OcScalar *scalar);

/* Module initialization */
OC_API int OcModuleCore_initializeScalarItf(void);

/* Macros */
#define OcScalar_isNegative(scalar) OcScalar_isLTZero(scalar)

#endif
