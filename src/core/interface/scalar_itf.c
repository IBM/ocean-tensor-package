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

#include "ocean/core/interface/scalar_itf.h"

#include "ocean/base/scalar.h"
#include "ocean/base/byteswap.h"
#include "ocean/base/malloc.h"
#include "ocean/base/format.h"
#include "ocean/base/warning.h"
#include "ocean/base/error.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"

#include <stdint.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


/* ===================================================================== */
/* Global variables                                                      */
/* ===================================================================== */

static int oc_scalar_cast_mode = 2; /* 0 = no auto casting, 1 = upcast to double, 2 = upcast to complex */

/* Warning types - unary */
static int oc_warning_scalar_sqrt;
static int oc_warning_scalar_reciprocal;
static int oc_warning_scalar_arcsin;
static int oc_warning_scalar_arccos;
static int oc_warning_scalar_arccosh;
static int oc_warning_scalar_arctanh;
static int oc_warning_scalar_log;
static int oc_warning_scalar_log2;
static int oc_warning_scalar_log10;
static int oc_warning_scalar_log1p;

/* Warning types - binary */
int oc_warning_scalar_divide;
int oc_warning_scalar_modulo;


/* ===================================================================== */
/* Module initialization                                                 */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcModuleCore_initializeScalarItf(void)
/* --------------------------------------------------------------------- */
{
   /* Register warning types:                               */
   /* Warning message -----------------------------------+  */
   /* Raise only once -------------------------------+   |  */
   /* Variable for unique identifier ---+            |   |  */
   /*                                   |            |   |  */
   OcWarning_register(&oc_warning_scalar_sqrt,       1, "Input argument to square-root must be nonnegative");
   OcWarning_register(&oc_warning_scalar_reciprocal, 1, "Input argument to reciprocal cannot be zero");
   OcWarning_register(&oc_warning_scalar_arcsin,     1, "Input argument to arcsin must be in [-1,1]");
   OcWarning_register(&oc_warning_scalar_arccos,     1, "Input argument to arccos must be in [-1,1]");
   OcWarning_register(&oc_warning_scalar_arccosh,    1, "Input argument to arccosh must be >= 1");
   OcWarning_register(&oc_warning_scalar_arctanh,    1, "Input argument to arctanh must be in [-1,1]");
   OcWarning_register(&oc_warning_scalar_log,        1, "Input argument to log must be >= 0");
   OcWarning_register(&oc_warning_scalar_log2,       1, "Input argument to log2 must be >= 0");
   OcWarning_register(&oc_warning_scalar_log10,      1, "Input argument to log10 must be >= 0");
   OcWarning_register(&oc_warning_scalar_log1p,      1, "Input argument to log1p must be >= -1");

   OcWarning_register(&oc_warning_scalar_divide,     1, "Input argument to scalar division cannot be zero");
   OcWarning_register(&oc_warning_scalar_modulo,     1, "Input argument to scalar modulo cannot be zero");

   return 0;
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcScalar *OcScalar_create(OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcScalar *result = NULL;

   result = (OcScalar *)OcMalloc(sizeof(OcScalar));
   if (result == NULL) OcError(NULL, "Error allocating memory for OcScalar object");

   result -> dtype = dtype;

   return result;
}


/* -------------------------------------------------------------------- */
OcScalar *OcScalar_clone(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcScalar *result = NULL;

   result = (OcScalar *)OcMalloc(sizeof(OcScalar));
   if (result == NULL) OcError(NULL, "Error allocating memory for OcScalar object");

   /* Copy the data type and value */
   *result = *scalar;

   return result;
}


/* -------------------------------------------------------------------- */
OcScalar *OcScalar_castDType(OcScalar *scalar, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcScalar *result = NULL;

   result = (OcScalar *)OcMalloc(sizeof(OcScalar));
   if (result == NULL) OcError(NULL, "Error allocating memory for OcScalar object");

   /* Cast the scalar */
   result -> dtype = dtype;
   OcScalar_copy(scalar, result);

   return result;
}


/* -------------------------------------------------------------------- */
void OcScalar_free(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{
   if (scalar) OcFree(scalar);
}


/* -------------------------------------------------------------------- */
void OcScalar_castTo(OcScalar *scalar, OcDType dtype, OcScalar *result)
/* -------------------------------------------------------------------- */
{
   if ((result == NULL) || (result == scalar))
   {  if (scalar -> dtype != dtype)
      {  OcScalar s;
         s.dtype = dtype;
         OcScalar_copy(scalar, &s);
         *scalar = s;
      }
   }
   else
   {  result -> dtype = dtype;
      OcScalar_copy(scalar, result);
   }
}


/* -------------------------------------------------------------------- */
void OcScalar_copy(OcScalar *src, OcScalar *dst)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_scalar_copy funptr = 0;

   if (src == dst) return ;
   if (src -> dtype == dst -> dtype)
   {  /* This may copy too much data, but avoids a function call */
      dst -> value = src -> value;
   }
   else
   {  /* Look up the function pointer. As an elementary function it */
      /* is assumed to exist, and inadvertent omission will cause a */
      /* segmentation fault, which forces appropriate recompilation.*/
      OC_SOLID_FUNPTR2("copy", solid_cpu_scalar_copy, funptr,
                       src -> dtype, dst -> dtype, "CPU");

      /* Apply the function */
      funptr((void *)&(src -> value), (void *)&(dst -> value));
   }
}


/* -------------------------------------------------------------------- */
void OcScalar_getReal(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{
   switch(scalar -> dtype)
   {
      case OcDTypeCHalf :
         result -> dtype = OcDTypeHalf;
         result -> value.sHalf = scalar -> value.sCHalf.real;
         break;
         
      case OcDTypeCFloat :
         result -> dtype = OcDTypeFloat;
         result -> value.sFloat = scalar -> value.sCFloat.real;
         break;
      
      case OcDTypeCDouble :
         result -> dtype = OcDTypeDouble;
         result -> value.sDouble = scalar -> value.sCDouble.real;
         break;
         
      default :
         result -> dtype = scalar -> dtype;
         result -> value = scalar -> value;
   }
}


/* -------------------------------------------------------------------- */
void OcScalar_getImag(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{
   switch(scalar -> dtype)
   {
      case OcDTypeCHalf :
         result -> dtype = OcDTypeHalf;
         result -> value.sHalf = scalar -> value.sCHalf.imag;
         break;
         
      case OcDTypeCFloat :
         result -> dtype = OcDTypeFloat;
         result -> value.sFloat = scalar -> value.sCFloat.imag;
         break;
      
      case OcDTypeCDouble :
         result -> dtype = OcDTypeDouble;
         result -> value.sDouble = scalar -> value.sCDouble.imag;
         break;
         
      default :
         /* Zero out the result value */
         result -> dtype = scalar -> dtype;
         OcScalar_fromUInt64(result, 0);
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_setReal(OcScalar *scalar, OcScalar *value)
/* -------------------------------------------------------------------- */
{  OcScalar v;
   OcDType basetype;

   if (!OcDType_isReal(value -> dtype))
      OcError(-1, "Cannot assign a complex scalar to the real part");

   /* Make sure the value is floating-point */
   if (!OcDType_isFloat(value -> dtype))
   {  OcScalar_castTo(value, OcDTypeDouble, &v);
      value = &v;
   }

   /* Get the common base type */
   basetype = OcDType_getBaseType(OcDType_getCommonType(scalar -> dtype, value -> dtype));

   /* Cast the data types */
   OcScalar_castTo(scalar, OcDType_getComplexType(basetype), NULL);
   OcScalar_castTo(value, basetype, &v);

   /* Update the real part */
   switch (basetype)
   {  case OcDTypeHalf   : scalar -> value.sCHalf.real   = v.value.sHalf;   break;
      case OcDTypeFloat  : scalar -> value.sCFloat.real  = v.value.sFloat;  break;
      case OcDTypeDouble : scalar -> value.sCDouble.real = v.value.sDouble; break;
      default :
         OcError(-1, "Internal error: invalid basetype in OcScalar_setReal");
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcScalar_setImag(OcScalar *scalar, OcScalar *value)
/* -------------------------------------------------------------------- */
{  OcScalar v;
   OcDType basetype;

   if (!OcDType_isReal(value -> dtype))
      OcError(-1, "Cannot assign a complex scalar to the imaginary part");

   /* Make sure the value is floating-point */
   if (!OcDType_isFloat(value -> dtype))
   {  OcScalar_castTo(value, OcDTypeDouble, &v);
      value = &v;
   }

   /* Get the common base type */
   basetype = OcDType_getBaseType(OcDType_getCommonType(scalar -> dtype, value -> dtype));

   /* Cast the data types */
   OcScalar_castTo(scalar, OcDType_getComplexType(basetype), NULL);
   OcScalar_castTo(value, basetype, &v);

   /* Update the imaginary part */
   switch (basetype)
   {  case OcDTypeHalf   : scalar -> value.sCHalf.imag   = v.value.sHalf;   break;
      case OcDTypeFloat  : scalar -> value.sCFloat.imag  = v.value.sFloat;  break;
      case OcDTypeDouble : scalar -> value.sCDouble.imag = v.value.sDouble; break;
      default :
         OcError(-1, "Internal error: invalid basetype in OcScalar_setImag");
  }

  return 0;
}


/* -------------------------------------------------------------------- */
void OcScalar_setZero(OcScalar *scalar, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   /* Initialize the value */
   v.dtype = OcDTypeInt64;
   v.value.sInt64 = 0;

   scalar -> dtype = dtype;
   OcScalar_copy(&v, scalar);
}


/* -------------------------------------------------------------------- */
void OcScalar_setOne(OcScalar *scalar, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   /* Initialize the value */
   v.dtype = OcDTypeInt64;
   v.value.sInt64 = 1;

   scalar -> dtype = dtype;
   OcScalar_copy(&v, scalar);
}


/* -------------------------------------------------------------------- */
void OcScalar_setMin(OcScalar *scalar, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcDType basetype = OcDType_getBaseType(dtype);

   /* Set the data type */
   scalar -> dtype = dtype;

   switch (basetype)
   {  case OcDTypeBool   : scalar -> value.sBool   = 0;         break;
      case OcDTypeInt8   : scalar -> value.sInt8   = INT8_MIN;  break;
      case OcDTypeInt16  : scalar -> value.sInt16  = INT16_MIN; break;
      case OcDTypeInt32  : scalar -> value.sInt32  = INT32_MIN; break;
      case OcDTypeInt64  : scalar -> value.sInt64  = INT64_MIN; break;
      case OcDTypeUInt8  : scalar -> value.sUInt8  = 0;         break;
      case OcDTypeUInt16 : scalar -> value.sUInt16 = 0;         break;
      case OcDTypeUInt32 : scalar -> value.sUInt32 = 0;         break;
      case OcDTypeUInt64 : scalar -> value.sUInt64 = 0;         break;
      case OcDTypeHalf   : scalar -> value.sHalf   = 0xFBFF;    break;
      case OcDTypeFloat  : scalar -> value.sFloat  = -FLT_MAX;  break;
      case OcDTypeDouble : scalar -> value.sDouble = -DBL_MAX;  break;
      default : ; /* Empty */
   }

   /* Zero out the imaginary part */
   if (OcDType_isComplex(dtype))
   {  if (dtype == OcDTypeHalf)
           scalar -> value.sHalf = 0x0000;
      else if (dtype == OcDTypeFloat)
           scalar -> value.sFloat = 0;
      else scalar -> value.sDouble = 0;
   }
}


/* -------------------------------------------------------------------- */
void OcScalar_setMax(OcScalar *scalar, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcDType basetype = OcDType_getBaseType(dtype);

   /* Set the data type */
   scalar -> dtype = dtype;

   switch (basetype)
   {  case OcDTypeBool   : scalar -> value.sBool   = 1;          break;
      case OcDTypeInt8   : scalar -> value.sInt8   = INT8_MAX;   break;
      case OcDTypeInt16  : scalar -> value.sInt16  = INT16_MAX;  break;
      case OcDTypeInt32  : scalar -> value.sInt32  = INT32_MAX;  break;
      case OcDTypeInt64  : scalar -> value.sInt64  = INT64_MAX;  break;
      case OcDTypeUInt8  : scalar -> value.sUInt8  = UINT8_MAX;  break;
      case OcDTypeUInt16 : scalar -> value.sUInt16 = UINT16_MAX; break;
      case OcDTypeUInt32 : scalar -> value.sUInt32 = UINT32_MAX; break;
      case OcDTypeUInt64 : scalar -> value.sUInt64 = UINT64_MAX; break;
      case OcDTypeHalf   : scalar -> value.sHalf   = 0x7BFF;     break;
      case OcDTypeFloat  : scalar -> value.sFloat  = FLT_MAX;    break;
      case OcDTypeDouble : scalar -> value.sDouble = DBL_MAX;    break;
      default : ; /* Empty */
   }

   /* Zero out the imaginary part */
   if (OcDType_isComplex(dtype))
   {  if (dtype == OcDTypeHalf)
           scalar -> value.sHalf = 0x0000;
      else if (dtype == OcDTypeFloat)
           scalar -> value.sFloat = 0;
      else scalar -> value.sDouble = 0;
   }
}


/* -------------------------------------------------------------------- */
void OcScalar_setEps(OcScalar *scalar, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcDType basetype;

   /* Epsilon for integer types is equal to one */
   if (OcDType_isInteger(dtype))
   {  OcScalar_setOne(scalar, dtype);
      return ;
   }

   /* Floating-point types */
   scalar -> dtype = dtype;
   basetype = OcDType_getBaseType(dtype);

   switch (basetype)
   {  case OcDTypeHalf   : scalar -> value.sHalf   = 0x1400;      break;
      case OcDTypeFloat  : scalar -> value.sFloat  = FLT_EPSILON; break;
      case OcDTypeDouble : scalar -> value.sDouble = DBL_EPSILON; break;
      default : ; /* Empty */
   }

   /* Zero out the imaginary part - we rely on memory alignment to    */
   /* ensure that real part coincides with the data of the base type. */
   if (OcDType_isComplex(dtype))
   {  if (dtype == OcDTypeCHalf)
           scalar -> value.sCHalf.imag = 0x0000;
      else if (dtype == OcDTypeCFloat)
           scalar -> value.sCFloat.imag = 0;
      else scalar -> value.sCDouble.imag = 0;
   }   
}


/* -------------------------------------------------------------------- */
void OcScalar_byteswap(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_byteswap funptr = 0;

   /* Look up the function pointer. As an elementary function it */
   /* is assumed to exist, and inadvertent omission will cause a */
   /* segmentation fault and required appropriate recompilation. */
   OC_SOLID_FUNPTR("byteswap", solid_cpu_byteswap, funptr, scalar -> dtype, "CPU");

   /* Apply the function */
   funptr(0, NULL, NULL, (void *)&(scalar -> value));
}


/* -------------------------------------------------------------------- */
void OcScalar_floatInf(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  double d;
   d = 1. / 0.; /* Using INF generates a warning */
   scalar -> dtype = OcDTypeFloat,
   OcScalar_fromDouble(scalar, d);
}


/* -------------------------------------------------------------------- */
void OcScalar_floatNaN(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  double d;
   d = 0. / 0.; /* Using NAN generates a warning */
   scalar -> dtype = OcDTypeFloat,
   OcScalar_fromDouble(scalar, d);
}


/* -------------------------------------------------------------------- */
void OcScalar_doubleInf(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  double d;
   d = 1. / 0.; /* Using INF generates a warning */
   scalar -> dtype = OcDTypeDouble,
   OcScalar_fromDouble(scalar, d);
}


/* -------------------------------------------------------------------- */
void OcScalar_doubleNaN(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  double d;
   d = 0. / 0.; /* Using NAN generates a warning */
   scalar -> dtype = OcDTypeDouble,
   OcScalar_fromDouble(scalar, d);
}


/* -------------------------------------------------------------------- */
void OcScalar_fromInt64(OcScalar *scalar, OcInt64 value)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   /* Initialize the value */
   v.dtype = OcDTypeInt64;
   v.value.sInt64 = value;

   /* Copy the value */
   OcScalar_copy(&v, scalar);
}


/* -------------------------------------------------------------------- */
void OcScalar_fromUInt64(OcScalar *scalar, OcUInt64 value)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   /* Initialize the value */
   v.dtype = OcDTypeUInt64;
   v.value.sUInt64 = value;

   /* Copy the value */
   OcScalar_copy(&v, scalar);
}


/* -------------------------------------------------------------------- */
void OcScalar_fromDouble(OcScalar *scalar, OcDouble value)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   /* Initialize the value */
   v.dtype = OcDTypeDouble;
   v.value.sDouble = value;

   /* Copy the value */
   OcScalar_copy(&v, scalar);
}


/* -------------------------------------------------------------------- */
void OcScalar_fromCDouble(OcScalar *scalar, OcCDouble value)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   /* Initialize the value */
   v.dtype = OcDTypeCDouble;
   v.value.sCDouble = value;

   /* Copy the value */
   OcScalar_copy(&v, scalar);
}


/* -------------------------------------------------------------------- */
void OcScalar_fromComplex(OcScalar *scalar, OcDouble real, OcDouble imag)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   /* Initialize the value */
   v.dtype = OcDTypeCDouble;
   v.value.sCDouble.real = real;
   v.value.sCDouble.imag = imag;

   /* Copy the value */
   OcScalar_copy(&v, scalar);
}


/* -------------------------------------------------------------------- */
int OcScalar_asBool(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   v.dtype = OcDTypeBool;
   OcScalar_copy(scalar, &v);

   return (v.value.sBool == 0) ? 0 : 1;
}


/* -------------------------------------------------------------------- */
OcInt64 OcScalar_asInt64(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   if (scalar -> dtype == OcDTypeInt64) return scalar -> value.sInt64;

   v.dtype = OcDTypeInt64;
   OcScalar_copy(scalar, &v);
   return v.value.sInt64;
}


/* -------------------------------------------------------------------- */
OcUInt64 OcScalar_asUInt64(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   if (scalar -> dtype == OcDTypeUInt64) return scalar -> value.sUInt64;

   v.dtype = OcDTypeUInt64;
   OcScalar_copy(scalar, &v);
   return v.value.sUInt64;
}


/* -------------------------------------------------------------------- */
OcDouble OcScalar_asDouble(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   if (scalar -> dtype == OcDTypeDouble) return scalar -> value.sDouble;

   v.dtype = OcDTypeDouble;
   OcScalar_copy(scalar, &v);
   return v.value.sDouble;
}


/* -------------------------------------------------------------------- */
OcCDouble OcScalar_asCDouble(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcScalar v;

   if (scalar -> dtype == OcDTypeCDouble) return scalar -> value.sCDouble;

   v.dtype = OcDTypeCDouble;
   OcScalar_copy(scalar, &v);
   return v.value.sCDouble;
}


/* -------------------------------------------------------------------- */
void OcScalar_exportData(OcScalar *scalar, void *dstData, int byteswapped)
/* -------------------------------------------------------------------- */
{  OcScalar s, *ptr;

   /* Create a scalar */
   if (byteswapped)
   {  s.dtype = scalar -> dtype;
      s.value = scalar -> value;
      OcScalar_byteswap(&s);
      ptr = &s;
   }
   else
   {  ptr = scalar;
   }

   /* Copy the data */
   switch (ptr -> dtype)
   {
      #define OC_TEMPLATE(TYPE) \
      case OcDType##TYPE: \
          *((Oc##TYPE *)dstData) = ptr -> value.s##TYPE; \
          break;

      OC_GENERATE_ALL_TYPES
      #undef OC_TEMPLATE

      case OcDTypeNone :
         break;
   }
}


/* -------------------------------------------------------------------- */
void OcScalar_importData(OcScalar *scalar, void *srcData, int byteswapped)
/* -------------------------------------------------------------------- */
{
   /* Copy the data */
   switch (scalar -> dtype)
   {
      #define OC_TEMPLATE(TYPE) \
      case OcDType##TYPE: \
          scalar -> value.s##TYPE = *((Oc##TYPE *)srcData); \
          break;

      OC_GENERATE_ALL_TYPES
      #undef OC_TEMPLATE

      case OcDTypeNone :
         break;
   }

   /* Byteswap if needed */
   if (byteswapped) OcScalar_byteswap(scalar);
}


/* -------------------------------------------------------------------- */
void OcScalar_copyRaw(void *src, OcDType srcType, void *dst, OcDType dstType)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_copy funptr = 0;
   
   /* Look up the function pointer. As an elementary function it */
   /* is assumed to exist, and inadvertent omission will cause a */
   /* segmentation fault, which forces appropriate recompilation.*/
   OC_SOLID_FUNPTR2("copy", solid_cpu_copy, funptr, srcType, dstType, "CPU");

   /* Apply the function */
   funptr(0, NULL, NULL, src, 0, NULL, NULL, dst);
}


/* -------------------------------------------------------------------- */
int OcScalar_inRange(OcScalar *scalar, OcDType dtype)
/* -------------------------------------------------------------------- */
{
   if (!OcDType_isSigned(scalar -> dtype))
   {   /* Boolean, unsigned integer types */
      return OcDType_inRangeUInt64(OcScalar_asUInt64(scalar), dtype);
   }
   else if (!OcDType_isFloat(scalar -> dtype))
   {  /* Signed integer types */
      return OcDType_inRangeInt64(OcScalar_asInt64(scalar), dtype);
   }
   else if (!OcDType_isComplex(scalar -> dtype))
   {  /* Floating-point types */
      return OcDType_inRangeDouble(OcScalar_asDouble(scalar), dtype);
   }
   else
   {  /* Complex types */
      return OcDType_inRangeCDouble(OcScalar_asCDouble(scalar), dtype);
   }
}


/* -------------------------------------------------------------------- */
OcDType OcScalar_getCommonType(OcScalar *scalar, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcDType stype;

   if (OcDType_isFloat(scalar -> dtype) || OcDType_isFloat(dtype))
   {  /* Floating-point scalar */
      if (dtype == OcDTypeNone) return scalar -> dtype;

      /* Minimum required float type */
      stype = OcScalar_getFloatType(scalar);
   }
   else
   {  /* Integer scalar and dtype */
      if (OcDType_isSigned(scalar -> dtype))
      {  OcInt64 value = OcScalar_asInt64(scalar);
         if ((value <= 0) || (OcDType_isSigned(dtype)))
              stype = OcDType_getTypeInt64(value);
         else stype = OcDType_getTypeUInt64((OcUInt64)value);
      }
      else
      {  OcUInt64 value = OcScalar_asUInt64(scalar);
         stype = OcDType_getTypeUInt64(value);
      }
   }

   return OcDType_getCommonType(stype, dtype);
}


/* -------------------------------------------------------------------- */
OcDType OcScalar_getFloatType(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcDType dtype = scalar -> dtype;

   if (!OcDType_isFloat(dtype))
   {  /* Return data type based on exact representation range */
      double value = OcScalar_asDouble(scalar);
      if (value < 0) value *= -1;
      if (value <= 2048) return OcDTypeHalf;
      if (value <= 16777216L) return OcDTypeFloat;
      return OcDTypeDouble;
   }
   if (!OcDType_isComplex(dtype))
   {  /* Return based on the float range */
      return OcDType_getTypeDouble(OcScalar_asDouble(scalar));
   }
   else
   {  OcCDouble value;
      OcDType   dtype1, dtype2;

      value = OcScalar_asCDouble(scalar);
      dtype1 = OcDType_getTypeDouble(value.real);
      dtype2 = OcDType_getTypeDouble(value.imag);
      dtype  = OcDType_getCommonType(dtype1, dtype2);
      return OcDType_getComplexType(dtype);
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isZero(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double    d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      return ((c.real == 0) && (c.imag == 0)) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d == 0) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isInf(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double    d;

   if (!OcDType_isFloat(scalar -> dtype)) return 0;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      return (isinf(c.real) || isinf(c.imag)) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (isinf(d)) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isNaN(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double    d;

   if (!OcDType_isFloat(scalar -> dtype)) return 0;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      return (isnan(c.real) || isnan(c.imag)) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (isnan(d)) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isFinite(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double    d;

   if (!OcDType_isFloat(scalar -> dtype)) return 1;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      return (isfinite(c.real) && isfinite(c.imag)) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (isfinite(d)) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isLTZero(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      if (c.real < 0) return 1;
      if (c.real > 0) return 0;
      return (c.imag < 0) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d < 0) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isLEZero(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      if (c.real < 0) return 1;
      if (c.real > 0) return 0;
      return (c.imag <= 0) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d <= 0) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isGEZero(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      if (c.real > 0) return 1;
      if (c.real < 0) return 0;
      return (c.imag >= 0) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d >= 0) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isGTZero(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      if (c.real > 0) return 1;
      if (c.real < 0) return 0;
      return (c.imag > 0) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d > 0) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isEQZero(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double    d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      return ((c.real == 0) && (c.imag == 0)) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d == 0) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isLTOne(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      if (c.real < 1) return 1;
      if (c.real > 1) return 0;
      return (c.imag < 0) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d < 1) ? 1 : 0;
   }
}

/* -------------------------------------------------------------------- */
int OcScalar_isEQOne(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double    d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      return ((c.real == 1) && (c.imag == 0)) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d == 1) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isGTOneAbs(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double    d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      return ((c.real * c.real + c.imag * c.imag) > 1) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return ((d < -1) || (d > 1)) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
int OcScalar_isLTNegOne(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcCDouble c;
   double d;

   if (OcDType_isComplex(scalar -> dtype))
   {  c = OcScalar_asCDouble(scalar);
      if (c.real < -1) return 1;
      if (c.real > -1) return 0;
      return (c.imag < 0) ? 1 : 0;
   }
   else
   {  d = OcScalar_asDouble(scalar);
      return (d < -1) ? 1 : 0;
   }
}


/* -------------------------------------------------------------------- */
/* int OcScalar_isLT(OcScalar *scalar1, OcScalar *scalar2)              */
/* int OcScalar_isLE(OcScalar *scalar1, OcScalar *scalar2)              */
/* int OcScalar_isEQ(OcScalar *scalar1, OcScalar *scalar2)              */
/* int OcScalar_isNE(OcScalar *scalar1, OcScalar *scalar2)              */
/* int OcScalar_isGE(OcScalar *scalar1, OcScalar *scalar2)              */
/* int OcScalar_isGT(OcScalar *scalar1, OcScalar *scalar2)              */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME, SOLID_OP, DESC) \
   int OC_CONCAT_2(OcScalar_is,OPNAME)(OcScalar *scalar1, OcScalar *scalar2) \
   {  solid_funptr_cpu_##SOLID_OP funptr; \
      OcDType dtype; \
      OcScalar cast1, cast2, result; \
      int      status;\
      \
      dtype = OcDType_getCommonType(scalar1 -> dtype, scalar2 -> dtype); \
      if (dtype == OcDTypeHalf) dtype = OcDTypeFloat; \
      else if (dtype == OcDTypeCHalf) dtype = OcDTypeCFloat; \
      \
      /* Make sure scalar1 has the desired data type */ \
      result.dtype = OcDTypeBool; \
      OcScalar_castTo(scalar1, dtype, &cast1); \
      OcScalar_castTo(scalar2, dtype, &cast2); \
      \
      /* Look up the function pointer */ \
      OC_SOLID_FUNPTR(DESC, solid_cpu_##SOLID_OP, funptr, dtype, "CPU"); \
      if (funptr == 0) return -1; \
      \
      /* Call the function */ \
      OC_SOLID_CALL(status, funptr, \
                    0, NULL, NULL, (void *)&(cast1.value), \
                             NULL, (void *)&(cast2.value), \
                             NULL, (void *)&(result.value)); \
      \
      return (status == 0) ? result.value.sBool : -1; \
   }

OC_TEMPLATE(LT, lt, "less-than"       )
OC_TEMPLATE(LE, le, "less-or-equal"   )
OC_TEMPLATE(EQ, eq, "equal"           )
OC_TEMPLATE(NE, ne, "not-equal"       )
OC_TEMPLATE(GE, ge, "greater-or-equal")
OC_TEMPLATE(GT, gt, "greater-than"    )
#undef OC_TEMPLATE



/* ===================================================================== */
/* Unary functions                                                       */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcScalar_negative(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_negative funptr;
   OcScalar cast;
   OcDType  dtype;
   int      status;

   /* Check if we should cast the type */
   if (OcScalar_getCastMode() != 0)
        dtype = OcDType_getSignedType(scalar -> dtype);
   else dtype = scalar -> dtype;

   /* Get the function */
   OC_SOLID_FUNPTR("negative", solid_cpu_negative, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Cast the data */
   result -> dtype = dtype;
   OcScalar_castTo(scalar, dtype, &cast);

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(cast.value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
int OcScalar_bitwiseNot(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_bitwise_not funptr;
   OcDType  dtype = scalar -> dtype;
   int      status;

   /* Get the function */
   OC_SOLID_FUNPTR("bitwise-NOT", solid_cpu_bitwise_not, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Set the data type */
   result -> dtype = dtype;

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(scalar -> value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
int OcScalar_logicalNot(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_logical_not funptr;
   OcScalar cast;
   OcDType  dtype = OcDTypeBool;
   int      status;

   /* Get the function */
   OC_SOLID_FUNPTR("logical-NOT", solid_cpu_logical_not, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Cast the data */
   result -> dtype = dtype;
   OcScalar_castTo(scalar, dtype, &cast);

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(cast.value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
int OcScalar_conj(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_conj funptr;
   OcDType dtype = scalar -> dtype;
   int status;

   /* Non-complex scalars */
   if (!OcDType_isComplex(dtype)) { *result = *scalar; return 0; }

   /* Get the function */
   OC_SOLID_FUNPTR("conj", solid_cpu_conj, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Set the data type */
   result -> dtype = dtype;

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(scalar -> value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
int OcScalar_reciprocal(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_reciprocal funptr;
   OcDType  dtype = scalar -> dtype;
   OcScalar cast;
   int      status;

   /* Check for zero, if warning system active */
   if (OcWarning_enabled(oc_warning_scalar_reciprocal))
   {  if (OcScalar_isZero(scalar))
      {  if (OcWarning_raise(oc_warning_scalar_reciprocal) != 0) return -1;
      }
   }

   /* Type-cast integers */
   if ((OcScalar_getCastMode() != 0) && (!OcDType_isFloat(dtype)))
      dtype = OcDTypeDouble;

   /* Get the function */
   OC_SOLID_FUNPTR("reciprocal", solid_cpu_reciprocal, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Cast the data */
   result -> dtype = dtype;
   OcScalar_castTo(scalar, dtype, &cast);

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(cast.value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
int OcScalar_sqrt(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_sqrt funptr;
   OcScalar cast;
   OcDType  dtype = scalar -> dtype;
   int      mode = OcScalar_getCastMode();
   int      status;

   if (!OcDType_isComplex(dtype))
   {  /* Check for negative, if warning system active */
      if (OcWarning_enabled(oc_warning_scalar_sqrt) || (mode == 2))
      {  if (OcScalar_isLTZero(scalar))
         {  if (mode == 2)
            {  dtype = OcDTypeCDouble;
            }
            else
            {  if (OcWarning_raise(oc_warning_scalar_sqrt) != 0) return -1;
            }
         }
      }

      /* Type-cast integers */
      if ((mode != 0) && (!OcDType_isFloat(dtype)))
         dtype = OcDTypeDouble;
   }

   /* Get the function */
   OC_SOLID_FUNPTR("sqrt", solid_cpu_sqrt, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Prepare the data */
   result -> dtype = dtype;
   OcScalar_castTo(scalar, dtype, &cast);

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(cast.value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
int OcScalar_cbrt(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_cbrt funptr;
   OcScalar cast;
   OcDType  dtype = scalar -> dtype;
   int      status;

   /* Type-cast integers */
   if ((OcScalar_getCastMode() != 0) && (!OcDType_isFloat(dtype))) dtype = OcDTypeDouble;

   /* Get the function */
   OC_SOLID_FUNPTR("cbrt", solid_cpu_cbrt, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Prepare the data */
   result -> dtype = dtype;
   OcScalar_castTo(scalar, dtype, &cast);

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(cast.value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
int OcScalar_square(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_square funptr;
   OcDType dtype = scalar -> dtype;
   int status;

   /* Get the function */
   OC_SOLID_FUNPTR("square", solid_cpu_square, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Set the data type */
   result -> dtype = dtype;

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(scalar -> value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
/* int OcScalar_absolute(OcScalar *scalar, OcScalar *result)            */
/* int OcScalar_fabs(OcScalar *scalar, OcScalar *result)                */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(NAME, DESC) \
int OcScalar_ ## NAME(OcScalar *scalar, OcScalar *result) \
{  solid_funptr_cpu_fabs funptr; \
   OcDType dtype = scalar -> dtype; \
   int status; \
   \
   /* Unsigned integers */ \
   if (!OcDType_isSigned(dtype)) { *result = *scalar; return 0; } \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##NAME, funptr, dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Set the data type */ \
   result -> dtype = OcDType_getBaseType(dtype); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(scalar -> value), NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(absolute, "absolute")
OC_TEMPLATE(fabs,     "fabs"    )
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
int OcScalar_sign(OcScalar *scalar, OcScalar *result)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_sign funptr;
   OcDType  dtype = scalar -> dtype;
   int      status;

   /* Get the function */
   OC_SOLID_FUNPTR("sign", solid_cpu_sign, funptr, dtype, "CPU");
   if (funptr == 0) return -1;

   /* Set the data type */
   result -> dtype = dtype;

   /* Call the function */
   OC_SOLID_CALL(status, funptr,
                 0, NULL, NULL, (void *)&(scalar -> value), NULL, (void *)&(result -> value));

   return status;
}


/* -------------------------------------------------------------------- */
/* int OcScalar_sin    (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_cos    (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_tan    (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_sinh   (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_cosh   (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_tanh   (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_arctan (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_arcsinh(OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_exp    (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_exp2   (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_exp10  (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_expm1  (OcScalar *scalar, OcScalar *result)             */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(NAME, DESC) \
int OcScalar_ ## NAME(OcScalar *scalar, OcScalar *result) \
{  solid_funptr_cpu_##NAME funptr; \
   OcDType  dtype = scalar -> dtype; \
   OcScalar cast; \
   int status; \
   \
   /* Type-cast integers */ \
   if ((OcScalar_getCastMode() != 0) && (!OcDType_isFloat(dtype))) dtype = OcDTypeDouble; \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##NAME, funptr, dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Convert the data */ \
   result -> dtype = dtype; \
   OcScalar_castTo(scalar, dtype, &cast); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(cast.value), NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(sin,     "sine"               )
OC_TEMPLATE(cos,     "cosine"             )
OC_TEMPLATE(tan,     "tangent"            )
OC_TEMPLATE(sinh,    "hyperbolic sine"    )
OC_TEMPLATE(cosh,    "hyperbolic cosine"  )
OC_TEMPLATE(tanh,    "hyperbolic tangent" )
OC_TEMPLATE(arctan,  "arc tangent"        )
OC_TEMPLATE(arcsinh, "hyperbolic arc sine")
OC_TEMPLATE(exp,     "exponent"           )
OC_TEMPLATE(exp2,    "exponent base-2"    )
OC_TEMPLATE(exp10,   "exponent base-10"   )
OC_TEMPLATE(expm1,   "exponent minus one" )
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
/* int OcScalar_arcsin (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_arccos (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_arccosh(OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_arctanh(OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_log    (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_log2   (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_log10  (OcScalar *scalar, OcScalar *result)             */
/* int OcScalar_log1p  (OcScalar *scalar, OcScalar *result)             */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(NAME, CHECK, DESC) \
int OcScalar_ ## NAME(OcScalar *scalar, OcScalar *result) \
{  solid_funptr_cpu_##NAME funptr; \
   OcScalar cast; \
   OcDType  dtype = scalar -> dtype; \
   int      mode = OcScalar_getCastMode(); \
   int      status; \
   \
   if (!OcDType_isComplex(dtype)) \
   {  if (OcWarning_enabled(oc_warning_scalar_ ## NAME) || (mode == 2)) \
      {  if (CHECK(scalar)) \
         {  if (mode == 2) \
            {  dtype = OcDTypeCDouble; \
            } \
            else \
            {  if (OcWarning_raise(oc_warning_scalar_ ## NAME) != 0) return -1; \
            } \
         } \
      } \
      \
      /* Type-cast integers */ \
      if ((mode != 0) && (!OcDType_isFloat(dtype))) dtype = OcDTypeDouble; \
   } \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##NAME, funptr, dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Convert the data */ \
   result -> dtype = dtype; \
   OcScalar_castTo(scalar, dtype, &cast); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(cast.value), NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(arcsin,  OcScalar_isGTOneAbs, "arc sine"              )
OC_TEMPLATE(arccos,  OcScalar_isGTOneAbs, "arc cosine"            )
OC_TEMPLATE(arccosh, OcScalar_isLTOne,    "hyperbolic arc cosine" )
OC_TEMPLATE(arctanh, OcScalar_isGTOneAbs, "hyperbolic arc tangent")
OC_TEMPLATE(log,     OcScalar_isLTZero,   "logarithm"             )
OC_TEMPLATE(log2,    OcScalar_isLTZero,   "logarithm base-2"      )
OC_TEMPLATE(log10,   OcScalar_isLTZero,   "logarithm base-10"     )
OC_TEMPLATE(log1p,   OcScalar_isLTNegOne, "logarithm one plus"    )
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
/* int OcScalar_ceil (OcScalar *scalar, OcScalar *result)               */
/* int OcScalar_floor(OcScalar *scalar, OcScalar *result)               */
/* int OcScalar_trunc(OcScalar *scalar, OcScalar *result)               */
/* int OcScalar_round(OcScalar *scalar, OcScalar *result)               */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(NAME, DESC) \
int OcScalar_ ## NAME(OcScalar *scalar, OcScalar *result) \
{  solid_funptr_cpu_##NAME funptr; \
   OcDType dtype = scalar -> dtype; \
   int status; \
   \
   /* Return immediately for integers */ \
   if (!OcDType_isFloat(dtype)) { *result = *scalar; return 0; } \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##NAME, funptr, dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Set the data type */ \
   result -> dtype = dtype; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(scalar -> value), NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(ceil,  "ceil" )
OC_TEMPLATE(floor, "floor")
OC_TEMPLATE(trunc, "trunc")
OC_TEMPLATE(round, "round")
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
/* int OcScalar_isinf   (OcScalar *scalar, OcScalar *result)            */
/* int OcScalar_isnan   (OcScalar *scalar, OcScalar *result)            */
/* int OcScalar_isfinite(OcScalar *scalar, OcScalar *result)            */
/* int OcScalar_isposinf(OcScalar *scalar, OcScalar *result)            */
/* int OcScalar_isneginf(OcScalar *scalar, OcScalar *result)            */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(NAME, DESC) \
int OcScalar_ ## NAME(OcScalar *scalar, OcScalar *result) \
{  solid_funptr_cpu_##NAME funptr; \
   int status; \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##NAME, funptr, scalar -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Set the data type */ \
   result -> dtype = OcDTypeBool; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(scalar -> value), NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(isinf,    "isinf"   )
OC_TEMPLATE(isnan,    "isnan"   )
OC_TEMPLATE(isfinite, "isfinite")
OC_TEMPLATE(isposinf, "isposinf")
OC_TEMPLATE(isneginf, "isneginf")
#undef OC_TEMPLATE



/* ===================================================================== */
/* Binary functions                                                      */
/* ===================================================================== */

/* ----------------------------------------------------------------------------- */
/* int OcScalar_add     (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_subtract(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_multiply(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_min     (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_max     (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_fmin    (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_fmax    (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* ----------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcScalar_##OP(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) \
{  solid_funptr_cpu_##OP funptr; \
   OcScalar cast1, cast2; \
   OcDType  dtype; \
   int      status; \
   \
   /* Check if we should cast the type */ \
   dtype = OcDType_getCommonType(scalar1 -> dtype, scalar2 -> dtype); \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##OP, funptr, dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Cast the data */ \
   OcScalar_castTo(scalar1, dtype, &cast1); \
   OcScalar_castTo(scalar2, dtype, &cast2); \
   \
   /* Set the result type */ \
   result -> dtype = dtype; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(cast1.value), \
                          NULL, (void *)&(cast2.value), \
                          NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(add,      "add"     )
OC_TEMPLATE(subtract, "subtract")
OC_TEMPLATE(multiply, "multiply")
OC_TEMPLATE(min,      "min"     )
OC_TEMPLATE(max,      "max"     )
OC_TEMPLATE(fmin,     "fmin"    )
OC_TEMPLATE(fmax,     "fmax"    )
#undef OC_TEMPLATE


/* -------------------------------------------------------------------------------- */
/* int OcScalar_divide     (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_trueDivide (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_floorDivide(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_mod        (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_fmod       (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* -------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, OPNAME, DESC, WARNING) \
int OcScalar_##OP(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) \
{  solid_funptr_cpu_divide funptr; \
   OcScalar cast1, cast2; \
   OcDType  dtype; \
   int      status; \
   \
   /* Check for zero, if warning system active */ \
   if (OcWarning_enabled(WARNING)) \
   {  if (OcScalar_isZero(scalar2)) \
      {  if (OcWarning_raise(WARNING) != 0) return -1; \
      } \
   } \
   \
   /* Get the common data type */ \
   dtype = OcDType_getCommonType(scalar1 -> dtype, scalar2 -> dtype); \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##OPNAME, funptr, dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Cast the data */ \
   OcScalar_castTo(scalar1, dtype, &cast1); \
   OcScalar_castTo(scalar2, dtype, &cast2); \
   \
   /* Set the result type */ \
   result -> dtype = dtype; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(cast1.value), \
                          NULL, (void *)&(cast2.value), \
                          NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(divide,      divide,       "divide",       oc_warning_scalar_divide)
OC_TEMPLATE(trueDivide,  true_divide,  "true-divide",  oc_warning_scalar_divide)
OC_TEMPLATE(floorDivide, floor_divide, "floor-divide", oc_warning_scalar_divide)
OC_TEMPLATE(mod,         mod,          "mod",          oc_warning_scalar_modulo)
OC_TEMPLATE(fmod,        fmod,         "fmod",         oc_warning_scalar_modulo)
#undef OC_TEMPLATE


/* -------------------------------------------------------------------------- */
/* int OcScalar_power(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* -------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcScalar_##OP(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) \
{  solid_funptr_cpu_##OP funptr; \
   OcScalar cast1, cast2; \
   OcDType  dtype1, dtype2; \
   int      status; \
   \
   /* Determine the data types */ \
   dtype1 = scalar1 -> dtype; \
   dtype2 = scalar2 -> dtype; \
   if (OcDType_isInteger(dtype1) && OcDType_isInteger(dtype2)) \
   {  dtype2 = OcDTypeInt16; \
      if (dtype1 != OcDTypeBool) \
         dtype1 = OcDType_isSigned(dtype1) ? OcDTypeInt64 : OcDTypeUInt64; \
   } \
   else \
   {  dtype2 = dtype1 = OcDType_getCommonType(dtype1, dtype2); \
   } \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##OP, funptr, dtype1, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Cast the data */ \
   OcScalar_castTo(scalar1, dtype1, &cast1); \
   OcScalar_castTo(scalar2, dtype2, &cast2); \
   \
   /* Set the result type */ \
   result -> dtype = dtype1; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(cast1.value), \
                          NULL, (void *)&(cast2.value), \
                          NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(power, "power")
#undef OC_TEMPLATE


/* ------------------------------------------------------------------------------- */
/* int OcScalar_bitwiseAnd(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_bitwiseOr (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_bitwiseXor(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* ------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcScalar_##OP(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) \
{  solid_funptr_cpu_##SOLID_OP funptr; \
   OcScalar cast1, cast2; \
   OcDType  dtype; \
   int      status; \
   \
   /* Determine the data types */ \
   if (OcDType_isFloat(scalar1 -> dtype) || OcDType_isFloat(scalar2 -> dtype)) \
      OcError(-1, "Scalar " #DESC " is only supported for integer types"); \
   \
   if (OcDType_nbits(scalar2 -> dtype) > OcDType_nbits(scalar1 -> dtype)) \
        dtype = scalar2 -> dtype; \
   else dtype = scalar1 -> dtype; \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##SOLID_OP, funptr, dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Cast the data */ \
   OcScalar_castTo(scalar1, dtype, &cast1); \
   OcScalar_castTo(scalar2, dtype, &cast2); \
   \
   /* Set the result type */ \
   result -> dtype = dtype; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(cast1.value), \
                          NULL, (void *)&(cast2.value), \
                          NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(bitwiseAnd, bitwise_and, "bitwise AND")
OC_TEMPLATE(bitwiseOr,  bitwise_or,  "bitwise OR" )
OC_TEMPLATE(bitwiseXor, bitwise_xor, "bitwise XOR")
#undef OC_TEMPLATE


/* ------------------------------------------------------------------------------- */
/* int OcScalar_logicalAnd(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_logicalOr (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_logicalXor(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* ------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcScalar_##OP(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) \
{  solid_funptr_cpu_##SOLID_OP funptr; \
   OcScalar cast1, cast2; \
   int      status; \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##SOLID_OP, funptr, OcDTypeBool, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Cast the data */ \
   OcScalar_castTo(scalar1, OcDTypeBool, &cast1); \
   OcScalar_castTo(scalar2, OcDTypeBool, &cast2); \
   \
   /* Set the result type (this has to be done after casting */ \
   /* as the result could point to scalar1 or scalar2).      */ \
   result -> dtype = OcDTypeBool; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(cast1.value), \
                          NULL, (void *)&(cast2.value), \
                          NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(logicalAnd, logical_and, "logical-AND")
OC_TEMPLATE(logicalOr,  logical_or,  "logical-OR" )
OC_TEMPLATE(logicalXor, logical_xor, "logical-XOR")
#undef OC_TEMPLATE


/* ---------------------------------------------------------------------------------- */
/* int OcScalar_bitshiftLeft (OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* int OcScalar_bitshiftRight(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) */
/* ---------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcScalar_##OP(OcScalar *scalar1, OcScalar *scalar2, OcScalar *result) \
{  solid_funptr_cpu_##SOLID_OP funptr; \
   OcScalar shift; \
   int      status; \
   \
   /* Determine the data types */ \
   if (OcDType_isFloat(scalar1 -> dtype) || OcDType_isFloat(scalar2 -> dtype)) \
      OcError(-1, "Scalar " #DESC " is only supported for integer types"); \
   \
   /* Check the shift value */ \
   if (!OcScalar_inRange(scalar2, OcDTypeInt8)) \
   {  result -> dtype = scalar1 -> dtype; \
      OcScalar_fromInt64(result, 0); \
      return 0; \
   } \
   \
   /* Get the function */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##SOLID_OP, funptr, scalar1 -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Cast the data */ \
   OcScalar_castTo(scalar2, OcDTypeInt8, &shift); \
   \
   /* Set the result type */ \
   result -> dtype = scalar1 -> dtype; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 0, NULL, NULL, (void *)&(scalar1 -> value), \
                          NULL, (void *)&(shift.value), \
                          NULL, (void *)&(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(bitshiftLeft,  bitshift_left,  "bitshift left" )
OC_TEMPLATE(bitshiftRight, bitshift_right, "bitshift right")
#undef OC_TEMPLATE



/* ===================================================================== */
/* Configuration functions                                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcScalar_getCastMode(void)
/* -------------------------------------------------------------------- */
{
   return oc_scalar_cast_mode;
}


/* -------------------------------------------------------------------- */
void OcScalar_setCastMode(int mode)
/* -------------------------------------------------------------------- */
{
   if (mode <= 0)
        oc_scalar_cast_mode = 0;
   else oc_scalar_cast_mode = (mode == 1) ? 1 : 2;
}



/* ===================================================================== */
/* Formatting functions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcScalar_format(OcScalar *scalar, char **str, const char *header, const char *footer)
/* -------------------------------------------------------------------- */
{  OcFormatAnalyze_funptr funptrAnalyze;
   OcFormatOutput_funptr  funptrOutput;
   OcFormat  *format = NULL;
   OcSize     slen;
   char      *s = NULL, *buffer = NULL;
   int        k, mode;
   int        result = -1;

   /* Create and initialize a new format structure */
   if ((format = OcFormatCreate(scalar -> dtype)) == NULL)
   {  OcErrorMessage("Could not allocate the formatting information");
      goto final;
   }

   /* Determine the function handle to use */
   funptrAnalyze = OcFormatAnalyze_function(scalar -> dtype);
   funptrOutput  = OcFormatOutput_function(scalar -> dtype);

   if ((funptrAnalyze == 0) || (funptrOutput == 0))
   {  OcErrorMessage("Could not find formatting analysis and output functions");
      goto final;
   }

   /* Determine the format */
   funptrAnalyze((const char *)(&(scalar -> value)), format, 0);

   /* Finalize formatting */
   OcFormatFinalize(format);

   /* Format the scalar */
   for (mode = 0; mode < 2; mode ++)
   {  slen = 0;

      /* Output the header */
      if (header != NULL)
      {  k = strlen(header)+1; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", header);
      }

      /* Output the element */
      k = format -> width; slen += k;
      if (mode == 1)
      {  funptrOutput((const char *)(&(scalar -> value)), format, 0, s);
         s += k;
      }

      /* Output the footer */
      if (footer != NULL)
      {  k = strlen(footer); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", footer);
      }

      /* Allocate memory for the string */
      if (mode == 0)
      {
         /* ------------------------------------------------------------- */
         /* Allocate the memory for the string. We use a regular malloc   */
         /* here instead of OcMalloc to ensure that the library can be    */
         /* recompiled with new memory allocation routines without having */
         /* to recompile any language bindings.                           */
         /* ------------------------------------------------------------- */
         buffer = (char *)malloc(sizeof(char) * (slen + 1));
         s = buffer;
         if (buffer == NULL)
         {  OcErrorMessage("Insufficient memory for output string");
            goto final;
         }
      }

   } /* Mode */

   /* Ensure the string is zero terminated */
   *s = '\0';

   /* Success */
   *str = buffer;
   result = 0;

final : ;
   /* -------------------------------------------------------- */
   /* Clean up. Note that freeing of the buffer has to be done */
   /* using the regular free function to match its allocation  */
   /* above using malloc.                                      */
   /* -------------------------------------------------------- */
   if (format != NULL) OcFormatFree(format);
   if ((result != 0) && (buffer != NULL)) { free(buffer); }

   return result;
}


/* -------------------------------------------------------------------- */
int OcScalar_display(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  char *str = NULL;
   int   result;

   /* Format and display the scalar */
   result = OcScalar_format(scalar, &str, NULL, NULL);
   if (result == 0)
   {  printf("%s", str);
   }

   /* Deallocate memory */
   if (str) free(str);

   return result;
}

