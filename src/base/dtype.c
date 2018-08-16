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

#include "ocean/base/dtype.h"
#include "ocean/base/types.h"
#include "ocean/base/error.h"

#include <math.h>


/* Table with information about all supported data types */

/* Field: mixed-case name ----------------------------------------------------------+         */
/*        name ----------------------------------------------------+                |         */
/*        isComplex --------------------------------------------+  |                |         */
/*        isFloat -------------------------------------------+  |  |                |         */
/*        isSigned ---------------------------------------+  |  |  |                |         */
/*        isNumber ------------------------------------+  |  |  |  |                |         */
/*        parts ------------------------------------+  |  |  |  |  |                |         */
/*        size ------------------+                  |  |  |  |  |  |                |         */
/*                               |                  |  |  |  |  |  |                |         */
OcDTypeInfo oc_dtype_info[] = {{ sizeof(OcBool),    1, 0, 0, 0, 0, "bool"          ,"Bool"    },
                               { sizeof(OcUInt8),   1, 1, 0, 0, 0, "uint8"         ,"UInt8"   },
                               { sizeof(OcUInt16),  1, 1, 0, 0, 0, "uint16"        ,"UInt16"  },
                               { sizeof(OcUInt32),  1, 1, 0, 0, 0, "uint32"        ,"UInt32"  },
                               { sizeof(OcUInt64),  1, 1, 0, 0, 0, "uint64"        ,"UInt64"  },
                               { sizeof(OcInt8),    1, 1, 1, 0, 0, "int8"          ,"Int8"    },
                               { sizeof(OcInt16),   1, 1, 1, 0, 0, "int16"         ,"Int16"   },
                               { sizeof(OcInt32),   1, 1, 1, 0, 0, "int32"         ,"Int32"   },
                               { sizeof(OcInt64),   1, 1, 1, 0, 0, "int64"         ,"Int64"   },
                               { sizeof(OcHalf),    1, 1, 1, 1, 0, "half"          ,"Half"    },
                               { sizeof(OcFloat),   1, 1, 1, 1, 0, "float"         ,"Float"   },
                               { sizeof(OcDouble),  1, 1, 1, 1, 0, "double"        ,"Double"  },
                               { sizeof(OcCHalf),   2, 1, 1, 1, 1, "complex-half"  ,"CHalf"   },
                               { sizeof(OcCFloat),  2, 1, 1, 1, 1, "complex-float" ,"CFloat"  },
                               { sizeof(OcCDouble), 2, 1, 1, 1, 1, "complex-double","CDouble" },
                               { 0,                 0, 0, 0, 0, 0, "--none--"      ,"--None--"}};


/* Field: nbits --------------------------------------------------+  */
/*        modulus type ---------------------------+               |  */
/*        signed type ------------+               |               |  */
/*                                |               |               |  */
OcDTypeInfo2 oc_dtype_info2[] = {{OcDTypeInt16,   OcDTypeBool,    1  },
                                 {OcDTypeInt16,   OcDTypeUInt8,   8  },
                                 {OcDTypeInt32,   OcDTypeUInt16,  16 },
                                 {OcDTypeInt64,   OcDTypeUInt32,  32 },
                                 {OcDTypeDouble,  OcDTypeUInt64,  64 },
                                 {OcDTypeInt8,    OcDTypeUInt8,   8  },
                                 {OcDTypeInt16,   OcDTypeUInt16,  16 },
                                 {OcDTypeInt32,   OcDTypeUInt32,  32 },
                                 {OcDTypeInt64,   OcDTypeUInt64,  64 },
                                 {OcDTypeHalf,    OcDTypeHalf,    16 },
                                 {OcDTypeFloat,   OcDTypeFloat,   32 },
                                 {OcDTypeDouble,  OcDTypeDouble,  64 },
                                 {OcDTypeCHalf,   OcDTypeHalf,    32 },
                                 {OcDTypeCFloat,  OcDTypeFloat,   64 },
                                 {OcDTypeCDouble, OcDTypeDouble,  128},
                                 {OcDTypeNone,    OcDTypeNone,    0  }};


/* ===================================================================== */
/* Local variables                                                       */
/* ===================================================================== */

/* Default data type */
static OcDType oc_default_dtype = OcDTypeFloat;


/* ===================================================================== */
/* Function implementation                                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcDType OcDType_getCommonType(OcDType dtype1, OcDType dtype2)
/* -------------------------------------------------------------------- */
{  OcDType dtype;

   /* Make sure that dtype1 has a larger value than dtype2 */
   if (dtype1 < dtype2) { dtype = dtype1; dtype1 = dtype2; dtype2 = dtype; }
   if (dtype1 == dtype2) return dtype1;
   if (dtype1 == OcDTypeNone) return dtype2;

   /* Boolean */
   if (dtype1 == OcDTypeBool) return dtype1;

   /* Unsigned integers */
   if (!OcDType_isSigned(dtype1)) return dtype1;

   /* Signed integers */
   if (!OcDType_isFloat(dtype1))
   {  int size2 = OcDType_size(dtype2);
      if (dtype2 == OcDTypeBool) return dtype1;
      if (OcDType_isSigned(dtype2)) return dtype1;
      if (size2 < OcDType_size(dtype1)) return dtype1;
      if (size2 == 1) return OcDTypeInt16;
      if (size2 == 2) return OcDTypeInt32;
      if (size2 == 4) return OcDTypeInt64;

      /* Combination of signed integer and unsigned 64-bit integer */
      return OcDTypeDouble; /* Incompatible data types */
   }

   /* Regular floating point */
   if (!OcDType_isComplex(dtype1))
   {
      /* Floating point dtype2 */
      if (OcDType_isFloat(dtype2)) return dtype1;

      /* Integer dtype2 */
      if (dtype1 == OcDTypeHalf)
      {  if (OcDType_size(dtype2) * 8 <= 11) return dtype1;
      }
      else if (dtype1 == OcDTypeFloat)
      {  if (OcDType_size(dtype2) * 8 <= 24) return dtype1;
      }
      else /* OcDTypeDouble */
      {  if (OcDType_size(dtype2) * 8 <= 52) return dtype1;
      }

      /* Integer range exceeds the mantissa */
      if (OcDType_size(dtype2) * 8 <= 24)
           return OcDTypeFloat;
      else if (OcDType_size(dtype2) * 8 <= 52)
           return OcDTypeDouble;
      else
           return OcDTypeDouble; /* Incompatible data types */
   }

   /* Complex numbers */
   if (OcDType_isComplex(dtype1))
   {
      /* Complex dtype2 */
      if (OcDType_isComplex(dtype2)) return dtype1;

      /* Floating point dtype2 */
      if (OcDType_isFloat(dtype2))
      {  if (OcDType_size(dtype2) <= OcDType_baseSize(dtype1)) return dtype1;
         if (dtype2 == OcDTypeFloat)
              return OcDTypeCFloat;
         else return OcDTypeCDouble;
      }

      /* Integer dtype2 */
      if (dtype1 == OcDTypeCHalf)
      {  if (OcDType_size(dtype2) * 8 <= 11) return dtype1;
      }
      else if (dtype1 == OcDTypeCFloat)
      {  if (OcDType_size(dtype2) * 8 <= 24) return dtype1;
      }
      else /* OcDTypeCDouble */
      {  if (OcDType_size(dtype2) * 8 <= 52) return dtype1;
      }

      /* Integer range exceeds the mantissa */
      if (OcDType_size(dtype2) * 8 <= 24)
           return OcDTypeCFloat;
      else if (OcDType_size(dtype2) * 8 <= 52)
           return OcDTypeCDouble;
      else
           return OcDTypeCDouble; /* Incompatible data types */
   }

   return OcDTypeNone; /* This should never be reached */
}


/* -------------------------------------------------------------------- */
OcDType OcDType_getBaseType(OcDType dtype)
/* -------------------------------------------------------------------- */
{
   switch (dtype)
   {  case OcDTypeCHalf   : return OcDTypeHalf;
      case OcDTypeCFloat  : return OcDTypeFloat;
      case OcDTypeCDouble : return OcDTypeDouble;
      default             : return dtype;
   }
}


/* -------------------------------------------------------------------- */
OcDType OcDType_getFloatType(OcDType dtype)
/* -------------------------------------------------------------------- */
{
   if (OcDType_isFloat(dtype)) return dtype;

   switch(dtype)
   {  case OcDTypeBool  :
      case OcDTypeInt8  :
      case OcDTypeUInt8 : return OcDTypeHalf;
      case OcDTypeInt16 : 
      case OcDTypeUInt16: return OcDTypeFloat;
      default           : return OcDTypeDouble;
   }
}


/* -------------------------------------------------------------------- */
OcDType OcDType_getComplexType(OcDType dtype)
/* -------------------------------------------------------------------- */
{
   switch(dtype)
   {  case OcDTypeHalf    : return OcDTypeCHalf;
      case OcDTypeFloat   : return OcDTypeCFloat;
      case OcDTypeDouble  : return OcDTypeCDouble;
      case OcDTypeCHalf   : return dtype;
      case OcDTypeCFloat  : return dtype;
      case OcDTypeCDouble : return dtype;
      default             : return OcDTypeNone;
   }
}


/* ===================================================================== */
/* Default datatype functions                                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcDType_setDefault(OcDType dtype)
/* -------------------------------------------------------------------- */
{
   oc_default_dtype = dtype;
}


/* -------------------------------------------------------------------- */
OcDType OcDType_getDefault(void)
/* -------------------------------------------------------------------- */
{
   return oc_default_dtype;
}


/* -------------------------------------------------------------------- */
OcDType OcDType_applyDefault(OcDType dtype)
/* -------------------------------------------------------------------- */
{
   if (dtype != OcDTypeNone) return dtype;
   if (oc_default_dtype != OcDTypeNone) return oc_default_dtype;
   OcError(OcDTypeNone, "A data type must be specified if no default is set");
}


/* -------------------------------------------------------------------- */
int OcDType_inRangeUInt64(OcUInt64 value, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcInt64 max;

   switch (dtype)
   {  case OcDTypeBool    : max = OC_BOOL_MAX;   break;
      case OcDTypeUInt8   : max = OC_UINT8_MAX;  break;
      case OcDTypeUInt16  : max = OC_UINT16_MAX; break;
      case OcDTypeUInt32  : max = OC_UINT32_MAX; break;
      case OcDTypeUInt64  : return 1;
      case OcDTypeInt8    : max = OC_INT8_MAX;  break;
      case OcDTypeInt16   : max = OC_INT16_MAX; break;
      case OcDTypeInt32   : max = OC_INT32_MAX; break;
      case OcDTypeInt64   : max = OC_INT64_MAX; break;
      case OcDTypeHalf    :
      case OcDTypeCHalf   : max = OC_HALF_MAX; break;
      case OcDTypeFloat   :
      case OcDTypeCFloat  :
      case OcDTypeDouble  :
      case OcDTypeCDouble : return 1;
      default :
         OcError(-1, "Internal error: unexpected data type: %d", (int)dtype);
   }

   return (value <= max) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
int OcDType_inRangeInt64(OcInt64 value, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcInt64 min;

   if (value >= 0)
   {  return OcDType_inRangeUInt64((OcUInt64)value, dtype);
   }
   else if (!OcDType_isSigned(dtype))
   {  return 0;
   }

   /* Negative values only */
   switch (dtype)
   {  case OcDTypeInt8    : min = OC_INT8_MIN;  break;
      case OcDTypeInt16   : min = OC_INT16_MIN; break;
      case OcDTypeInt32   : min = OC_INT32_MIN; break;
      case OcDTypeInt64   : min = OC_INT64_MIN; break;
      case OcDTypeHalf    :
      case OcDTypeCHalf   : min =-OC_HALF_MAX; break;
      case OcDTypeFloat   :
      case OcDTypeCFloat  :
      case OcDTypeDouble  :
      case OcDTypeCDouble : return 1;
      default :
         OcError(-1, "Internal error: unexpected data type: %d", (int)dtype);
   }

   return (value >= min) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
int OcDType_inRangeDouble(OcDouble value, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcUInt64  umax;
   OcInt64   imin, imax;
   OcDouble  dmax;

   if ((!OcDType_isFloat(dtype)) && (isnan(value))) return 0;

   if ((!OcDType_isFloat(dtype)) && (!OcDType_isSigned(dtype)))
   {  /* Boolean, unsigned integer types */
      if (value < 0) return 0;
      switch (dtype)
      {  case OcDTypeBool    : umax = OC_BOOL_MAX;   break;
         case OcDTypeUInt8   : umax = OC_UINT8_MAX;  break;
         case OcDTypeUInt16  : umax = OC_UINT16_MAX; break;
         case OcDTypeUInt32  : umax = OC_UINT32_MAX; break;
         case OcDTypeUInt64  : umax = OC_UINT64_MAX; break;
         default :
            OcError(-1, "Internal error: unexpected data type: %d", (int)dtype);
      }
      return (value - 1 >= umax) ? 0 : 1;
   }
   else if (!OcDType_isFloat(dtype))
   {
      /* Signed integer types */
      switch (dtype)
      {  case OcDTypeInt8  : imin = OC_INT8_MIN;  imax = OC_INT8_MAX;  break;
         case OcDTypeInt16 : imin = OC_INT16_MIN; imax = OC_INT16_MAX; break;
         case OcDTypeInt32 : imin = OC_INT32_MIN; imax = OC_INT32_MAX; break;
         case OcDTypeInt64 : imin = OC_INT64_MIN; imax = OC_INT64_MAX; break;
         default:
            OcError(-1, "Internal error: unexpected data type: %d", (int)dtype);
      }
      if (value < 0)
           return (value + 1 <= imin) ? 0 : 1;
      else return (value - 1 >= imax) ? 0 : 1;
   }
   else
   {  /* Floating-point or complex */
      if (!isfinite(value)) return 1;
   
      switch (dtype)
      {  case OcDTypeHalf    : case OcDTypeCHalf   : dmax = OC_HALF_MAX; break;
         case OcDTypeFloat   : case OcDTypeCFloat  : dmax = OC_FLOAT_MAX; break;
         case OcDTypeDouble  : case OcDTypeCDouble : return 1;
         default:
            OcError(-1, "Internal error: unexpected data type: %d", (int)dtype);
      }
   
      if (value < 0)
           return (value < -dmax) ? 0 : 1;
      else return (value >  dmax) ? 0 : 1;
   }
}


/* -------------------------------------------------------------------- */
int OcDType_inRangeCDouble(OcCDouble value, OcDType dtype)
/* -------------------------------------------------------------------- */
{
   if (OcDType_isComplex(dtype))
   {  return (OcDType_inRangeDouble(value.real, dtype) &&
              OcDType_inRangeDouble(value.imag, dtype)) ? 1 : 0;
   }
   else
   {  return OcDType_inRangeDouble(value.real, dtype);
   }
}


/* -------------------------------------------------------------------- */
OcDType OcDType_getTypeInt64(OcInt64 value)
/* -------------------------------------------------------------------- */
{
   if (value < 0)
   {  if (value >= OC_INT16_MIN)
           return (value >= OC_INT8_MIN)  ? OcDTypeInt8  : OcDTypeInt16;
      else return (value >= OC_INT32_MIN) ? OcDTypeInt32 : OcDTypeInt64;
   }
   else
   {  if (value <= OC_INT16_MAX)
           return (value <= OC_INT8_MAX)  ? OcDTypeInt8  : OcDTypeInt16;
      else return (value <= OC_INT32_MAX) ? OcDTypeInt32 : OcDTypeInt64;
   }
}


/* -------------------------------------------------------------------- */
OcDType OcDType_getTypeUInt64(OcUInt64 value)
/* -------------------------------------------------------------------- */
{
   if (value <= OC_UINT16_MAX)
        return (value <= OC_UINT8_MAX) ? OcDTypeUInt8 : OcDTypeUInt16;
   else return (value <= OC_UINT32_MAX) ? OcDTypeUInt32 : OcDTypeUInt64;
}


/* -------------------------------------------------------------------- */
OcDType OcDType_getTypeDouble(OcDouble value)
/* -------------------------------------------------------------------- */
{
   if (!isfinite(value)) return OcDTypeHalf;

   if (value < 0) value *= -1;
   if (value <= OC_HALF_MAX) return OcDTypeHalf;
   if (value <= OC_FLOAT_MAX) return OcDTypeFloat;

   return OcDTypeDouble;
}
