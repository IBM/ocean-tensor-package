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

#ifndef __SOLID_DTYPE_MACROS_H__
#define __SOLID_DTYPE_MACROS_H__

#include "solid/base/generic/half.h"


/* -------------------------------------------------------------------- */
/* Macros                                                               */
/* -------------------------------------------------------------------- */
/* SOLID_LOGIC_NOT(B)                                                   */
/*    Negates Boolean parameter B                                       */
/*                                                                      */
/* SOLID_LOGIC_AND(B1,B2)                                               */
/*    Applies the logic AND on Boolean parameters B1 and B2             */
/*                                                                      */
/* SOLID_LOGIC_OR(B1,B2)                                                */
/*    Applies the logic OR on Boolean parameters B1 and B2              */
/*                                                                      */
/* SDTYPE_MIXED_CASE(TYPE)                                              */
/*    Returns the mixed case formatting of the given TYPE               */
/*                                                                      */
/* SDTYPE_INDEX(TYPE)                                                   */
/*    Returns the index of the given TYPE                               */
/*                                                                      */
/* SDTYPE_SIZE(TYPE)                                                    */
/*    Returns the size of the TYPE in bytes                             */
/*                                                                      */
/* SDTYPE_ELEMSIZE(TYPE)                                                */
/*    Returns the element size of the TYPE in bytes (complex types)     */
/*                                                                      */
/* SDTYPE_EQUAL_ELEMSIZE(TYPE1, TYPE2)                                  */
/*    Returns 1 if TYPE1 and TYPE2 have the same elements size, and     */
/*    0 otherwise.                                                      */
/*                                                                      */
/* SDTYPE_CLASS(TYPE)                                                   */
/*    Returns the class index of the TYPE: 1 = Boolean, 2 = unsigned    */
/*    integer, 3 = signed integer, 4 = float, 5 = complex.              */
/*                                                                      */
/* SDTYPE_CLASS_EQUAL(TYPE1, TYPE2)                                     */
/*    Returns 1 if the class of TYPE1 and TYPE2 match, 0 otherwise      */
/*                                                                      */
/* SDTYPE_IS_BOOL(TYPE)          Checks if class if 1                   */
/* SDTYPE_IS_UNSIGNED_INT(TYPE)  Checks if class is 2                   */
/* SDTYPE_IS_SIGNED_INT(TYPE)    Checks if class is 3                   */
/* SDTYPE_IS_INT(TYPE)           Checks if class is 2 or 3              */
/* SDTYPE_IS_FLOAT(TYPE)         Checks if class is 4                   */
/* SDTYPE_IS_COMPLEX(TYPE)       Checks if class is 5                   */
/* SDTYPE_IS_REAL(TYPE)          Checks if class is 1, 2, 3, or 4       */
/*                                                                      */
/* SDTYPE_EQUAL(TYPE1, TYPE2)                                           */
/*    Returns 1 if TYPE1 is equal to TYPE2, 0 otherwise                 */
/* -------------------------------------------------------------------- */

/* -------------------------------------------------------------------- */
/* Note: The SDTYPE macros are meant to be device independent, whereas  */
/* the SOLID macros (aside from the boolean operations) are intended    */
/* for device specific operations. For example SDTYPE_ELEMTYPE for type */
/* chalf is always half, but SOLID_WORKTYPE for half might be half or   */
/* float, depending on the device type.                                 */
/* -------------------------------------------------------------------- */


/* ==================================================================== */
/* DEVICE-INDEPENDENT MACROS                                            */
/* ==================================================================== */

/* -------------------------------------------------------------------- */
/* Boolean macros                                                       */
/* -------------------------------------------------------------------- */

#define SOLID_LOGIC_NOT_0  1
#define SOLID_LOGIC_NOT_1  0
#define SOLID_LOGIC_NOT_B(B) SOLID_LOGIC_NOT_##B
#define SOLID_LOGIC_NOT(B) SOLID_LOGIC_NOT_B(B)

#define SOLID_LOGIC_AND_0_0  0
#define SOLID_LOGIC_AND_0_1  0
#define SOLID_LOGIC_AND_1_0  0
#define SOLID_LOGIC_AND_1_1  1
#define SOLID_LOGIC_AND_B(B1,B2) SOLID_LOGIC_AND_##B1##_##B2
#define SOLID_LOGIC_AND(B1,B2) SOLID_LOGIC_AND_B(B1,B2)

#define SOLID_LOGIC_OR_0_0  0
#define SOLID_LOGIC_OR_0_1  1
#define SOLID_LOGIC_OR_1_0  1
#define SOLID_LOGIC_OR_1_1  1
#define SOLID_LOGIC_OR_B(B1,B2) SOLID_LOGIC_OR_##B1##_##B2
#define SOLID_LOGIC_OR(B1,B2) SOLID_LOGIC_OR_B(B1,B2)


/* -------------------------------------------------------------------- */
/* Mixed case                                                           */
/* -------------------------------------------------------------------- */
#define SDTYPE_MIXED_bool     Bool
#define SDTYPE_MIXED_uint8    UInt8
#define SDTYPE_MIXED_uint16   UInt16
#define SDTYPE_MIXED_uit32    UInt32
#define SDTYPE_MIXED_uin64    UInt64
#define SDTYPE_MIXED_int8     Int8
#define SDTYPE_MIXED_int16    Int16
#define SDTYPE_MIXED_int32    Int32
#define SDTYPE_MIXED_int64    Int64
#define SDTYPE_MIXED_half     Half
#define SDTYPE_MIXED_float    Float
#define SDTYPE_MIXED_double   Double
#define SDTYPE_MIXED_chalf    CHalf
#define SDTYPE_MIXED_cfloat   CFloat
#define SDTYPE_MIXED_cdouble  CDouble

#define SDTYPE_MIXED_CASE_B(TYPE)  SDTYPE_MIXED_##TYPE
#define SDTYPE_MIXED_CASE(TYPE)    SDTYPE_MIXED_B(TYPE)


/* -------------------------------------------------------------------- */
/* Data type index                                                      */
/* -------------------------------------------------------------------- */
#define SDTYPE_INDEX_bool      0
#define SDTYPE_INDEX_uint8     1
#define SDTYPE_INDEX_uint16    2
#define SDTYPE_INDEX_uint32    3
#define SDTYPE_INDEX_uint64    4
#define SDTYPE_INDEX_int8      5
#define SDTYPE_INDEX_int16     6
#define SDTYPE_INDEX_int32     7
#define SDTYPE_INDEX_int64     8
#define SDTYPE_INDEX_half      9
#define SDTYPE_INDEX_float    10
#define SDTYPE_INDEX_double   11
#define SDTYPE_INDEX_chalf    12
#define SDTYPE_INDEX_cfloat   13
#define SDTYPE_INDEX_cdouble  14

/* Macro to match types */
#define SDTYPE_INDEX_B(TYPE)        SDTYPE_INDEX_##TYPE
#define SDTYPE_INDEX(TYPE)          SDTYPE_INDEX_B(TYPE)
#define SDTYPE_MATCHES(TYPE1,TYPE2) (SDTYPE_INDEX(TYPE1) == SDTYPE_INDEX(TYPE2))


/* -------------------------------------------------------------------- */
/* Element size in bytes                                                */
/* -------------------------------------------------------------------- */

#define SDTYPE_SIZE_bool       1
#define SDTYPE_SIZE_uint8      1
#define SDTYPE_SIZE_uint16     2
#define SDTYPE_SIZE_uint32     4
#define SDTYPE_SIZE_uint64     8
#define SDTYPE_SIZE_int8       1
#define SDTYPE_SIZE_int16      2
#define SDTYPE_SIZE_int32      4
#define SDTYPE_SIZE_int64      8
#define SDTYPE_SIZE_half       2
#define SDTYPE_SIZE_float      4
#define SDTYPE_SIZE_double     8
#define SDTYPE_SIZE_chalf      4
#define SDTYPE_SIZE_cfloat     8
#define SDTYPE_SIZE_cdouble   16

#define SDTYPE_SIZE_B(TYPE) SDTYPE_SIZE_##TYPE
#define SDTYPE_SIZE(TYPE) SDTYPE_SIZE_B(TYPE)

#define SDTYPE_ELEMSIZE_B(TYPE) SDTYPE_SIZE(TYPE)
#define SDTYPE_ELEMSIZE(TYPE) SDTYPE_ELEMSIZE_B(SDTYPE_ELEMTYPE(TYPE))


/* -------------------------------------------------------------------- */
/* Comparision of element size                                          */
/* -------------------------------------------------------------------- */

#define SDTYPE_EQUAL_ELEMSIZE_1_1   1
#define SDTYPE_EQUAL_ELEMSIZE_1_2   0
#define SDTYPE_EQUAL_ELEMSIZE_1_4   0
#define SDTYPE_EQUAL_ELEMSIZE_1_8   0

#define SDTYPE_EQUAL_ELEMSIZE_2_1   0
#define SDTYPE_EQUAL_ELEMSIZE_2_2   1
#define SDTYPE_EQUAL_ELEMSIZE_2_4   0
#define SDTYPE_EQUAL_ELEMSIZE_2_8   0

#define SDTYPE_EQUAL_ELEMSIZE_4_1   0
#define SDTYPE_EQUAL_ELEMSIZE_4_2   0
#define SDTYPE_EQUAL_ELEMSIZE_4_4   1
#define SDTYPE_EQUAL_ELEMSIZE_4_8   0

#define SDTYPE_EQUAL_ELEMSIZE_8_1   0
#define SDTYPE_EQUAL_ELEMSIZE_8_2   0
#define SDTYPE_EQUAL_ELEMSIZE_8_4   0
#define SDTYPE_EQUAL_ELEMSIZE_8_8   1

#define SDTYPE_EQUAL_ELEMSIZE_C(SIZE1, SIZE2) SDTYPE_EQUAL_ELEMSIZE_##SIZE1##_##SIZE2
#define SDTYPE_EQUAL_ELEMSIZE_B(SIZE1, SIZE2) SDTYPE_EQUAL_ELEMSIZE_C(SIZE1, SIZE2)
#define SDTYPE_EQUAL_ELEMSIZE(TYPE1, TYPE2) \
   SDTYPE_EQUAL_ELEMSIZE_B(SDTYPE_ELEMSIZE(TYPE1), SDTYPE_ELEMSIZE(TYPE2))


/* -------------------------------------------------------------------- */
/* Element size in bits                                                 */
/* -------------------------------------------------------------------- */

#define SDTYPE_BITSIZE_1    8
#define SDTYPE_BITSIZE_2   16
#define SDTYPE_BITSIZE_4   32
#define SDTYPE_BITSIZE_8   64
#define SDTYPE_BITSIZE_16 128

#define SDTYPE_BITSIZE_C(SIZE) SDTYPE_BITSIZE_##SIZE
#define SDTYPE_BITSIZE_B(SIZE) SDTYPE_BITSIZE_C(SIZE)
#define SDTYPE_BITSIZE(TYPE) SDTYPE_BITSIZE_B(SDTYPE_ELEMSIZE(TYPE))


/* -------------------------------------------------------------------- */
/* Class type                                                           */
/* -------------------------------------------------------------------- */
#define SDTYPE_CLASS_bool       1
#define SDTYPE_CLASS_uint8      2
#define SDTYPE_CLASS_uint16     2
#define SDTYPE_CLASS_uint32     2
#define SDTYPE_CLASS_uint64     2
#define SDTYPE_CLASS_int8       3
#define SDTYPE_CLASS_int16      3
#define SDTYPE_CLASS_int32      3
#define SDTYPE_CLASS_int64      3
#define SDTYPE_CLASS_half       4
#define SDTYPE_CLASS_float      4
#define SDTYPE_CLASS_double     4
#define SDTYPE_CLASS_chalf      5
#define SDTYPE_CLASS_cfloat     5
#define SDTYPE_CLASS_cdouble    5

#define SDTYPE_CLASS_B(TYPE) SDTYPE_CLASS_##TYPE
#define SDTYPE_CLASS(TYPE) SDTYPE_CLASS_B(TYPE)


/* -------------------------------------------------------------------- */
/* Class comparison                                                     */
/* SDTYPE_CLASS_EQUAL(TYPE1, TYPE2)                                     */
/* -------------------------------------------------------------------- */

#define SDTYPE_CLASS_EQUAL_1_1  1
#define SDTYPE_CLASS_EQUAL_1_2  0
#define SDTYPE_CLASS_EQUAL_1_3  0
#define SDTYPE_CLASS_EQUAL_1_4  0
#define SDTYPE_CLASS_EQUAL_1_5  0

#define SDTYPE_CLASS_EQUAL_2_1  0
#define SDTYPE_CLASS_EQUAL_2_2  1
#define SDTYPE_CLASS_EQUAL_2_3  0
#define SDTYPE_CLASS_EQUAL_2_4  0
#define SDTYPE_CLASS_EQUAL_2_5  0

#define SDTYPE_CLASS_EQUAL_3_1  0
#define SDTYPE_CLASS_EQUAL_3_2  0
#define SDTYPE_CLASS_EQUAL_3_3  1
#define SDTYPE_CLASS_EQUAL_3_4  0
#define SDTYPE_CLASS_EQUAL_3_5  0

#define SDTYPE_CLASS_EQUAL_4_1  0
#define SDTYPE_CLASS_EQUAL_4_2  0
#define SDTYPE_CLASS_EQUAL_4_3  0
#define SDTYPE_CLASS_EQUAL_4_4  1
#define SDTYPE_CLASS_EQUAL_4_5  0

#define SDTYPE_CLASS_EQUAL_5_1  0
#define SDTYPE_CLASS_EQUAL_5_2  0
#define SDTYPE_CLASS_EQUAL_5_3  0
#define SDTYPE_CLASS_EQUAL_5_4  0
#define SDTYPE_CLASS_EQUAL_5_5  1

#define SDTYPE_CLASS_EQUAL_C(CLASS1, CLASS2) SDTYPE_CLASS_EQUAL_##CLASS1##_##CLASS2
#define SDTYPE_CLASS_EQUAL_B(CLASS1, CLASS2) SDTYPE_CLASS_EQUAL_C(CLASS1, CLASS2)
#define SDTYPE_CLASS_EQUAL(TYPE1, TYPE2) SDTYPE_CLASS_EQUAL_B(SDTYPE_CLASS(TYPE1), SDTYPE_CLASS(TYPE2))

#define SDTYPE_IS_BOOL(TYPE)          SDTYPE_CLASS_EQUAL_B(SDTYPE_CLASS(TYPE), 1)
#define SDTYPE_IS_UNSIGNED_INT(TYPE)  SDTYPE_CLASS_EQUAL_B(SDTYPE_CLASS(TYPE), 2)
#define SDTYPE_IS_SIGNED_INT(TYPE)    SDTYPE_CLASS_EQUAL_B(SDTYPE_CLASS(TYPE), 3)
#define SDTYPE_IS_INT(TYPE)           SOLID_LOGIC_OR(SDTYPE_IS_UNSIGNED_INT(TYPE), SDTYPE_IS_SIGNED_INT(TYPE))
#define SDTYPE_IS_FLOAT(TYPE)         SDTYPE_CLASS_EQUAL_B(SDTYPE_CLASS(TYPE), 4)
#define SDTYPE_IS_COMPLEX(TYPE)       SDTYPE_CLASS_EQUAL_B(SDTYPE_CLASS(TYPE), 5)
#define SDTYPE_IS_REAL(TYPE)          SOLID_LOGIC_NOT(SDTYPE_IS_COMPLEX(TYPE))


/* -------------------------------------------------------------------- */
/* Type comparison                                                      */
/* SDTYPE_EQUAL(TYPE1, TYPE2)                                           */
/* -------------------------------------------------------------------- */

#define SDTYPE_EQUAL(TYPE1, TYPE2) \
   SOLID_LOGIC_AND(SDTYPE_EQUAL_ELEMSIZE(TYPE1, TYPE2), SDTYPE_CLASS_EQUAL(TYPE1, TYPE2))


/* -------------------------------------------------------------------- */
/* Real or complex                                                      */
/* -------------------------------------------------------------------- */

#define SDTYPE_REAL_STR_1     REAL
#define SDTYPE_REAL_STR_0     COMPLEX

#define SDTYPE_REAL_STR_C(FLAG) SDTYPE_REAL_STR_##FLAG
#define SDTYPE_REAL_STR_B(FLAG) SDTYPE_REAL_STR_C(FLAG)
#define SDTYPE_REAL_STR(TYPE)   SDTYPE_REAL_STR_B(SDTYPE_IS_REAL(TYPE))


/* -------------------------------------------------------------------- */
/* Element type for complex types                                       */
/* -------------------------------------------------------------------- */
#define SDTYPE_ELEMTYPE_chalf    half
#define SDTYPE_ELEMTYPE_cfloat   float
#define SDTYPE_ELEMTYPE_cdouble  double

#define SDTYPE_ELEMTYPE_COMPLEX_B(TYPE) SDTYPE_ELEMTYPE_##TYPE
#define SDTYPE_ELEMTYPE_COMPLEX(TYPE) SDTYPE_ELEMTYPE_COMPLEX_B(TYPE)
#define SDTYPE_ELEMTYPE_REAL(TYPE) TYPE

#define SDTYPE_ELEMTYPE_D(TYPE, CLASS) SDTYPE_ELEMTYPE_##CLASS(TYPE)
#define SDTYPE_ELEMTYPE_C(TYPE, CLASS) SDTYPE_ELEMTYPE_D(TYPE,CLASS) 
#define SDTYPE_ELEMTYPE_B(TYPE) SDTYPE_ELEMTYPE_C(TYPE, SDTYPE_REAL_STR(TYPE))
#define SDTYPE_ELEMTYPE(TYPE) SDTYPE_ELEMTYPE_B(TYPE)


/* -------------------------------------------------------------------- */
/* Data type for magnitude                                              */
/* -------------------------------------------------------------------- */

#define SDTYPE_ABSTYPE_int8   uint8
#define SDTYPE_ABSTYPE_int16  uint16
#define SDTYPE_ABSTYPE_int32  uint32
#define SDTYPE_ABSTYPE_int64  uint64

#define SDTYPE_ABSTYPE_D00(TYPE) TYPE
#define SDTYPE_ABSTYPE_D01(TYPE) SDTYPE_ELEMTYPE(TYPE)
#define SDTYPE_ABSTYPE_D10(TYPE) SDTYPE_ABSTYPE_##TYPE
#define SDTYPE_ABSTYPE_C(TYPE, SIGNED_INT, COMPLEX) SDTYPE_ABSTYPE_D##SIGNED_INT##COMPLEX(TYPE)
#define SDTYPE_ABSTYPE_B(TYPE, SIGNED_INT, COMPLEX) SDTYPE_ABSTYPE_C(TYPE, SIGNED_INT, COMPLEX)
#define SDTYPE_ABSTYPE(TYPE) SDTYPE_ABSTYPE_B(TYPE, SDTYPE_IS_SIGNED_INT(TYPE), SDTYPE_IS_COMPLEX(TYPE))


/* ==================================================================== */
/* DEVICE-DEPENDENT MACROS - ABSTRACT                                   */
/* ==================================================================== */

/* -------------------------------------------------------------------- */
/* C types (device specific)                                            */
/* -------------------------------------------------------------------- */

/* Device specific functions for all types: */
/* SOLID_C_TYPE(TYPE) */

/* C type of element type */
#define SOLID_C_ELEMTYPE_TYPE(TYPE) SOLID_C_TYPE(SDTYPE_ELEMTYPE(TYPE))
#define SOLID_C_ELEMTYPE SOLID_C_ELEMTYPE_TYPE(SDXTYPE)


/* -------------------------------------------------------------------- */
/* Basic or special type (device specific)                              */
/* -------------------------------------------------------------------- */

/* Device specific functions for all types: */
/* #define SDTYPE_BASIC_<type>  0 */
/* #define SDTYPE_BASIC_<type>  1 */

#define SDTYPE_BASIC_STR_1     BASIC
#define SDTYPE_BASIC_STR_0     SPECIAL

/* Macro to check if type is basic or special */
#define SDTYPE_IS_BASIC_B(TYPE) SDTYPE_BASIC_##TYPE
#define SDTYPE_IS_BASIC(TYPE) SDTYPE_IS_BASIC_B(TYPE)

#define SDTYPE_BASIC_STR_C(FLAG) SDTYPE_BASIC_STR_##FLAG
#define SDTYPE_BASIC_STR_B(FLAG) SDTYPE_BASIC_STR_C(FLAG)
#define SDTYPE_BASIC_STR(TYPE)  SDTYPE_BASIC_STR_B(SDTYPE_IS_BASIC(TYPE))


/* -------------------------------------------------------------------- */
/* Construction of scalar zero (device specific)                        */
/* -------------------------------------------------------------------- */

/* Device specific functions for special types: */
/* SDTYPE_ZERO_<type> */

#define SDTYPE_ZERO_SPECIAL_B(TYPE) SDTYPE_ZERO_##TYPE
#define SDTYPE_ZERO_SPECIAL(TYPE) SDTYPE_ZERO_SPECIAL_B(TYPE)
#define SDTYPE_ZERO_BASIC(TYPE)  0

#define SDTYPE_ZERO_C(TYPE, BASIC) SDTYPE_ZERO_##BASIC(TYPE)
#define SDTYPE_ZERO_B(TYPE, BASIC) SDTYPE_ZERO_C(TYPE, BASIC)
#define SDTYPE_ZERO(TYPE) SDTYPE_ZERO_B(TYPE, SDTYPE_BASIC_STR(TYPE))


/* -------------------------------------------------------------------- */
/* Work data types                                                      */
/* -------------------------------------------------------------------- */

/* Device specific work type for special types: */
/* SOLID_WORKTYPE_<TYPE> */

/* Determine the work type (device specific) */
#define SOLID_WORKTYPE_D_0(TYPE) SOLID_WORKTYPE_##TYPE
#define SOLID_WORKTYPE_D_1(TYPE) TYPE
#define SOLID_WORKTYPE_C(TYPE, BASIC) SOLID_WORKTYPE_D_##BASIC(TYPE)
#define SOLID_WORKTYPE_B(TYPE, BASIC) SOLID_WORKTYPE_C(TYPE, BASIC)
#define SOLID_WORKTYPE(TYPE) SOLID_WORKTYPE_B(TYPE, SDTYPE_IS_BASIC(TYPE))

/* Determine the element work type (device specific) */
#define SOLID_ELEMWORKTYPE(TYPE) SOLID_WORKTYPE(SDTYPE_ELEMTYPE(TYPE))

/* C type of work type */
#define SOLID_C_WORKTYPE_TYPE_B(TYPE) SOLID_C_TYPE(TYPE)
#define SOLID_C_WORKTYPE_TYPE(TYPE) SOLID_C_WORKTYPE_TYPE_B(SOLID_WORKTYPE(TYPE))
#define SOLID_C_WORKTYPE SOLID_C_WORKTYPE_TYPE(SDXTYPE)

/* C type of element work type */
#define SOLID_C_ELEMWORKTYPE_TYPE(TYPE) SOLID_C_WORKTYPE_TYPE(SDTYPE_ELEMTYPE(TYPE))
#define SOLID_C_ELEMWORKTYPE SOLID_C_ELEMWORKTYPE_TYPE(SDXTYPE)


/* ==================================================================== */
/* CPU-SPECIFIC MACROS                                                  */
/* ==================================================================== */

/* -------------------------------------------------------------------- */
/* C types (CPU - for use in device interfaces)                         */
/* -------------------------------------------------------------------- */

/* Define basic types */
#define SOLID_CPU_BASIC_bool        0
#define SOLID_CPU_BASIC_uint8       1
#define SOLID_CPU_BASIC_uint16      1
#define SOLID_CPU_BASIC_uint32      1
#define SOLID_CPU_BASIC_uint64      1
#define SOLID_CPU_BASIC_int8        1
#define SOLID_CPU_BASIC_int16       1
#define SOLID_CPU_BASIC_int32       1
#define SOLID_CPU_BASIC_int64       1
#define SOLID_CPU_BASIC_half        0
#define SOLID_CPU_BASIC_float       1
#define SOLID_CPU_BASIC_double      1
#define SOLID_CPU_BASIC_chalf       0
#define SOLID_CPU_BASIC_cfloat      1
#define SOLID_CPU_BASIC_cdouble     1

#define SOLID_CPU_BASIC_B(TYPE)     SOLID_CPU_BASIC_##TYPE
#define SOLID_CPU_BASIC(TYPE)       SOLID_CPU_BASIC_B(TYPE)

/* Work types for CPU */
#define SOLID_CPU_WORKTYPE_bool            bool
#define SOLID_CPU_WORKTYPE_half            float
#define SOLID_CPU_WORKTYPE_chalf           cfloat

#define SOLID_CPU_WORKTYPE_D0(TYPE)        SOLID_CPU_WORKTYPE_##TYPE
#define SOLID_CPU_WORKTYPE_D1(TYPE)        TYPE
#define SOLID_CPU_WORKTYPE_C(TYPE, BASIC)  SOLID_CPU_WORKTYPE_D##BASIC(TYPE)
#define SOLID_CPU_WORKTYPE_B(TYPE, BASIC)  SOLID_CPU_WORKTYPE_C(TYPE,BASIC)
#define SOLID_CPU_WORKTYPE(TYPE)           SOLID_CPU_WORKTYPE_B(TYPE, SOLID_CPU_BASIC(TYPE))
#define SOLID_CPU_ELEMWORKTYPE(TYPE)       SOLID_CPU_WORKTYPE(SDTYPE_ELEMTYPE(TYPE))

/* C types */
#define SOLID_CPU_C_TYPE_B(TYPE)           solid_##TYPE
#define SOLID_CPU_C_TYPE(TYPE)             SOLID_CPU_C_TYPE_B(TYPE)

#define SOLID_CPU_C_WORKTYPE(TYPE)         SOLID_CPU_C_TYPE(SOLID_CPU_WORKTYPE(TYPE))
#define SOLID_CPU_C_ELEMTYPE(TYPE)         SOLID_CPU_C_TYPE(SDTYPE_ELEMTYPE(TYPE))
#define SOLID_CPU_C_ELEMWORKTYPE(TYPE)     SOLID_CPU_C_TYPE(SOLID_CPU_ELEMWORKTYPE(TYPE))


/* -------------------------------------------------------------------- */
/* Conversion (CPU - for use in device interfaces)                      */
/* -------------------------------------------------------------------- */

#define SOLID_CPU_TO_WORKTYPE_bool(X)         X
#define SOLID_CPU_TO_WORKTYPE_half(X)         solid_float_to_half((solid_float)(X))
#define SOLID_CPU_TO_WORKTYPE_chalf(X)        { solid_float_to_half((X).real), solid_float_to_half((X).imag) }
#define SOLID_CPU_FROM_WORKTYPE_bool(X)       ((X) ? 1 : 0)
#define SOLID_CPU_FROM_WORKTYPE_half(X)       solid_half_to_float(X)
#define SOLID_CPU_FROM_WORKTYPE_chalf(X)      { solid_half_to_float((X).real), solid_half_to_float((X).imag) }

#define SOLID_CPU_M_TYPE_D0(ETYPE,X,M)        SOLID_CPU_##M##_WORKTYPE_##ETYPE(X)
#define SOLID_CPU_M_TYPE_D1(ETYPE,X,M)        X
#define SOLID_CPU_M_TYPE_C(ETYPE,BASIC,X,M)   SOLID_CPU_M_TYPE_D##BASIC(ETYPE, X, M)
#define SOLID_CPU_M_TYPE_B(ETYPE,BASIC,X,M)   SOLID_CPU_M_TYPE_C(ETYPE, BASIC, X, M)
#define SOLID_CPU_M_TYPE(ETYPE,X,MODE)        SOLID_CPU_M_TYPE_B(ETYPE, SOLID_CPU_BASIC(ETYPE), X, MODE)

#define SOLID_CPU_TO_WORKTYPE(TYPE,X)         SOLID_CPU_M_TYPE(SOLID_CPU_WORKTYPE(TYPE), X, TO)
#define SOLID_CPU_FROM_WORKTYPE(TYPE,X)       SOLID_CPU_M_TYPE(SOLID_CPU_WORKTYPE(TYPE), X, FROM)
#define SOLID_CPU_TO_ELEMWORKTYPE(TYPE,X)     SOLID_CPU_M_TYPE(SOLID_CPU_ELEMWORKTYPE(TYPE), X, TO)
#define SOLID_CPU_FROM_ELEMWORKTYPE(TYPE,X)   SOLID_CPU_M_TYPE(SOLID_CPU_ELEMWORKTYPE(TYPE), X, FROM)


/* -------------------------------------------------------------------- */
/* Assignment (CPU - for use in device interfaces)                      */
/* -------------------------------------------------------------------- */

#define SOLID_CPU_ASSIGN_FROM_WORKTYPE_D0(TYPE,SRC,DST) \
        *((SOLID_CPU_C_TYPE(TYPE) *)(DST)) = SOLID_CPU_FROM_WORKTYPE(TYPE, (*((SOLID_CPU_C_WORKTYPE(TYPE) *)(SRC))))

#define SOLID_CPU_ASSIGN_FROM_WORKTYPE_D1(TYPE,SRC,DST) \
        {  ((SOLID_CPU_C_TYPE(TYPE) *)(DST)) -> real = SOLID_CPU_FROM_ELEMWORKTYPE(TYPE, (((SOLID_CPU_C_WORKTYPE(TYPE) *)(SRC)) -> real)); \
           ((SOLID_CPU_C_TYPE(TYPE) *)(DST)) -> imag = SOLID_CPU_FROM_ELEMWORKTYPE(TYPE, (((SOLID_CPU_C_WORKTYPE(TYPE) *)(SRC)) -> imag)); \
        }

#define SOLID_CPU_ASSIGN_FROM_WORKTYPE_C(TYPE,COMPLEX,SRC,DST)   SOLID_CPU_ASSIGN_FROM_WORKTYPE_D##COMPLEX(TYPE,SRC,DST)
#define SOLID_CPU_ASSIGN_FROM_WORKTYPE_B(TYPE,COMPLEX,SRC,DST)   SOLID_CPU_ASSIGN_FROM_WORKTYPE_C(TYPE,COMPLEX,SRC,DST)
#define SOLID_CPU_ASSIGN_FROM_WORKTYPE(TYPE,SRC,DST)             SOLID_CPU_ASSIGN_FROM_WORKTYPE_B(TYPE, SDTYPE_IS_COMPLEX(TYPE), SRC, DST)

#endif
