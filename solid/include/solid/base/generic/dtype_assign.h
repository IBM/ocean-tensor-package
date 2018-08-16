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

#ifndef __SOLID_DTYPE_ASSIGN_H__
#define __SOLID_DTYPE_ASSIGN_H__

#include "solid/base/generic/dtype_macros.h"


/* -------------------------------------------------------------------- */
/* Macros                                                               */
/* -------------------------------------------------------------------- */
/* SOLID_CONVERT(TYPE1, TYPE2, X)                                       */
/*    Converts scalar X of TYPE1 to a scalar of TYPE2                   */
/*                                                                      */
/* SOLID_BYTESWAP(TYPE, SRC, DST)                                       */
/*    Byteswaps TYPE scalar *SRC to *DST                                */
/*                                                                      */
/* SOLID_ASSIGN(TYPE1, TYPE2, SRC, DST)                                 */
/*    Copies *SRC of TYPE1 to *DST of TYPE2                             */
/*                                                                      */
/* SOLID_BYTESWAP_ASSIGN(TYPE1, TYPE2, SRC, DST)                        */
/*    Copies byte-swapped *SRC of TYPE1 to *DST of TYPE2                */
/*                                                                      */
/* SOLID_TO_WORKTYPE(X)                                                 */
/*    Converts X to the worktype corresponding to SDXTYPE               */
/*                                                                      */
/* SOLID_FROM_WORKTYPE(X)                                               */
/*    Converts X from the worktype back to SDXTYPE                      */
/*                                                                      */
/* SOLID_TO_ELEMTYPE(X)                                                 */
/*    Converts an element of X (.real / .imag) to the element work type */
/*                                                                      */
/* SOLID_FROM_ELEMTYPE(X)                                               */
/*    Converts an element of X back to the element type for SDXTYPE     */
/* -------------------------------------------------------------------- */


/* ==================================================================== */
/* Byteswap helper routines                                             */
/* ==================================================================== */
#define SOLID_BYTESWAP_8(A,B) \
   {  B = A; \
   }

/* Byte-swap variables A and B of type uint16_t */
#define SOLID_BYTESWAP_16(A,B) \
   {  B = (((A & 0x00FF) << 8) | \
           ((A & 0xFF00) >> 8));  \
   }

/* Byte-swap variables A and B of type uint32_t */
#define SOLID_BYTESWAP_32(A,B) \
   {  B = ((((A) & 0x000000FF) << 24) | \
           (((A) & 0x0000FF00) <<  8) | \
           (((A) & 0x00FF0000) >>  8) | \
           (((A) & 0xFF000000) >> 24)); \
   }

/* Byte-swap variables A and B of type uint64_t */
#define SOLID_BYTESWAP_64(A,B) \
   {  B = ((((A) & 0x00000000000000FF) << 56) | \
           (((A) & 0x000000000000FF00) << 40) | \
           (((A) & 0x0000000000FF0000) << 24) | \
           (((A) & 0x00000000FF000000) <<  8) | \
           (((A) & 0x000000FF00000000) >>  8) | \
           (((A) & 0x0000FF0000000000) >> 24) | \
           (((A) & 0x00FF000000000000) >> 40) | \
           (((A) & 0xFF00000000000000) >> 56)); \
   }


/* ==================================================================== */
/* Semicolon helper routines                                            */
/* ==================================================================== */
#define SOLID_SEMICOLON_C0 
#define SOLID_SEMICOLON_C1 ;
#define SOLID_SEMICOLON_B(FLAG) SOLID_SEMICOLON_C##FLAG
#define SOLID_SEMICOLON(FLAG) SOLID_SEMICOLON_B(FLAG)


/* ==================================================================== */
/* SOLID_CONVERT(TYPE1, TYPE2, X)                                       */
/* ==================================================================== */

/* -------------------------------------------------------------------- */
/* Conversion between real types                                        */
/* -------------------------------------------------------------------- */

/* Device specific functions */
/* SOLID_BASIC_TO_SPECIAL_<type2>(TYPE1, X) */
/* SOLID_SPECIAL_TO_BASIC_<type1>(TYPE2, X) */
/* SOLID_CONVERT_<type1>_TO_<type2>         */

/* Conversion between basic and special types */
#define SOLID_CONVERT_BASIC_TO_BASIC(TYPE1, TYPE2, X)      (SOLID_C_TYPE(TYPE2))(X)
#define SOLID_CONVERT_BASIC_TO_SPECIAL(TYPE1, TYPE2, X)    SOLID_BASIC_TO_SPECIAL_##TYPE2(TYPE1, X)
#define SOLID_CONVERT_SPECIAL_TO_BASIC(TYPE1, TYPE2, X)    SOLID_SPECIAL_TO_BASIC_##TYPE1(TYPE2, X) 
#define SOLID_CONVERT_SPECIAL_TO_SPECIAL(TYPE1, TYPE2, X)  SOLID_CONVERT_##TYPE1##_TO_##TYPE2(X)

/* Conversion from real to real */
#define SOLID_CONVERT_REAL_B(TYPE1, TYPE2, BASIC1, BASIC2, X) \
   SOLID_CONVERT_##BASIC1##_TO_##BASIC2(TYPE1, TYPE2, X)
#define SOLID_CONVERT_REAL(TYPE1, TYPE2, BASIC1, BASIC2, X) \
   SOLID_CONVERT_REAL_B(TYPE1, TYPE2, BASIC1, BASIC2, X)
#define SOLID_CONVERT_REAL_TO_REAL_B(TYPE1, TYPE2, X) \
   SOLID_CONVERT_REAL(TYPE1, TYPE2, SDTYPE_BASIC_STR(TYPE1), SDTYPE_BASIC_STR(TYPE2), X)
#define SOLID_CONVERT_REAL_TO_REAL(TYPE1, TYPE2, X) \
   SOLID_CONVERT_REAL_TO_REAL_B(TYPE1, TYPE2, X)

/* -------------------------------------------------------------------- */
/* Conversion with complex types - C99                                  */
/* -------------------------------------------------------------------- */

/* Conversion from real to complex */
#define SOLID_CONVERT_REAL_TO_COMPLEX_D(TYPE, ETYPE,X) \
   (SOLID_C_TYPE(TYPE)) { X, SDTYPE_ZERO(ETYPE) }
#define SOLID_CONVERT_REAL_TO_COMPLEX_C(TYPE1, TYPE2, ETYPE, X) \
   SOLID_CONVERT_REAL_TO_COMPLEX_D(TYPE2, ETYPE, SOLID_CONVERT_REAL_TO_REAL(TYPE1, ETYPE, X))
#define SOLID_CONVERT_REAL_TO_COMPLEX_B(TYPE1, TYPE2, X) \
   SOLID_CONVERT_REAL_TO_COMPLEX_C(TYPE1, TYPE2, SDTYPE_ELEMTYPE(TYPE2), X)
#define SOLID_CONVERT_REAL_TO_COMPLEX(TYPE1, TYPE2, X) \
   SOLID_CONVERT_REAL_TO_COMPLEX_B(TYPE1, TYPE2, X)

/* Conversion from complex to real */
#define SOLID_CONVERT_COMPLEX_TO_REAL_B(TYPE1, TYPE2, X) \
   SOLID_CONVERT_REAL_TO_REAL(SDTYPE_ELEMTYPE(TYPE1), TYPE2, ((X).real))
#define SOLID_CONVERT_COMPLEX_TO_REAL(TYPE1, TYPE2, X) \
   SOLID_CONVERT_COMPLEX_TO_REAL_B(TYPE1, TYPE2, X)

/* Conversion from complex to complex */
#define SOLID_CONVERT_COMPLEX_TO_COMPLEX_C(TYPE1, TYPE2, ETYPE1, ETYPE2, X) \
   ((SOLID_C_TYPE(TYPE2)) {  SOLID_CONVERT_REAL_TO_REAL(ETYPE1, ETYPE2, (X).real), \
                             SOLID_CONVERT_REAL_TO_REAL(ETYPE1, ETYPE2, (X).imag) })
#define SOLID_CONVERT_COMPLEX_TO_COMPLEX_B(TYPE1, TYPE2, X) \
   SOLID_CONVERT_COMPLEX_TO_COMPLEX_C(TYPE1, TYPE2, SDTYPE_ELEMTYPE(TYPE1), SDTYPE_ELEMTYPE(TYPE2), X)
#define SOLID_CONVERT_COMPLEX_TO_COMPLEX(TYPE1, TYPE2, X) \
   SOLID_CONVERT_COMPLEX_TO_COMPLEX_B(TYPE1, TYPE2, X)

/* General conversion */
#define SOLID_CONVERT_0C(TYPE1, TYPE2, CLASS1, CLASS2, X) SOLID_CONVERT_##CLASS1##_TO_##CLASS2(TYPE1, TYPE2, X)
#define SOLID_CONVERT_0B(TYPE1, TYPE2, CLASS1, CLASS2, X) SOLID_CONVERT_0C(TYPE1, TYPE2, CLASS1, CLASS2, X)
#define SOLID_CONVERT_0(TYPE1, TYPE2, X) \
   SOLID_CONVERT_0B(TYPE1, TYPE2, SDTYPE_REAL_STR(TYPE1), SDTYPE_REAL_STR(TYPE2), X)
#define SOLID_CONVERT_1(TYPE1, TYPE2, X) X

#define SOLID_CONVERT_D(TYPE1, TYPE2, EQUAL, X) SOLID_CONVERT_##EQUAL(TYPE1, TYPE2, X)
#define SOLID_CONVERT_C(TYPE1, TYPE2, EQUAL, X) SOLID_CONVERT_D(TYPE1, TYPE2, EQUAL, X)
#define SOLID_CONVERT_B(TYPE1, TYPE2, X) SOLID_CONVERT_C(TYPE1, TYPE2, SDTYPE_EQUAL(TYPE1, TYPE2), X)
#define SOLID_CONVERT(TYPE1, TYPE2, X) SOLID_CONVERT_B(TYPE1, TYPE2, X)



/* ==================================================================== */
/* SOLID_BYTESWAP(TYPE, SRC, DST)                                       */
/* ==================================================================== */

#define SOLID_BYTESWAP_REAL_F(BITS, SRC, DST) \
   {  uint##BITS##_t __raw1; \
      uint##BITS##_t __raw2; \
      __raw1 = *((uint##BITS##_t *)(SRC)); \
      SOLID_BYTESWAP_##BITS(__raw1, __raw2); \
      *((uint##BITS##_t *)(DST)) = __raw2; \
   }
#define SOLID_BYTESWAP_REAL_E1(SRC, DST) \
   {  *((uint8_t *)(DST)) = *((uint8_t *)(SRC)); \
   }
#define SOLID_BYTESWAP_REAL_E2(SRC, DST) SOLID_BYTESWAP_REAL_F(16, SRC, DST)
#define SOLID_BYTESWAP_REAL_E4(SRC, DST) SOLID_BYTESWAP_REAL_F(32, SRC, DST)
#define SOLID_BYTESWAP_REAL_E8(SRC, DST) SOLID_BYTESWAP_REAL_F(64, SRC, DST)
#define SOLID_BYTESWAP_REAL_D(BYTES, SRC, DST) SOLID_BYTESWAP_REAL_E##BYTES(SRC, DST)
#define SOLID_BYTESWAP_REAL_C(BYTES, SRC, DST) SOLID_BYTESWAP_REAL_D(BYTES, SRC, DST)
#define SOLID_BYTESWAP_REAL_B(TYPE, SRC, DST) SOLID_BYTESWAP_REAL_C(SDTYPE_ELEMSIZE(TYPE), SRC, DST)
#define SOLID_BYTESWAP_REAL(TYPE, SRC, DST) SOLID_BYTESWAP_REAL_B(TYPE, SRC, DST)

#define SOLID_BYTESWAP_COMPLEX_F(TYPE, BITS, SRC, DST) \
   {  uint##BITS##_t __raw1; \
      uint##BITS##_t __raw2; \
      __raw1 = *(uint##BITS##_t *)(&(((SOLID_C_TYPE(TYPE) *)(SRC)) -> real)); \
      SOLID_BYTESWAP_##BITS(__raw1, __raw2); \
      *(uint##BITS##_t *)(&(((SOLID_C_TYPE(TYPE) *)(DST)) -> real)) = __raw2; \
      __raw1 = *(uint##BITS##_t *)(&(((SOLID_C_TYPE(TYPE) *)(SRC)) -> imag)); \
      SOLID_BYTESWAP_##BITS(__raw1, __raw2); \
      *(uint##BITS##_t *)(&(((SOLID_C_TYPE(TYPE) *)(DST)) -> imag)) = __raw2; \
   }

#define SOLID_BYTESWAP_COMPLEX_E1(TYPE, SRC, DST) \
   {  *(SOLID_C_TYPE(TYPE) *)(DST) = *(SOLID_C_TYPE(TYPE) *)(SRC); \
   }
#define SOLID_BYTESWAP_COMPLEX_E2(TYPE, SRC, DST) SOLID_BYTESWAP_COMPLEX_F(TYPE, 16, SRC, DST)
#define SOLID_BYTESWAP_COMPLEX_E4(TYPE, SRC, DST) SOLID_BYTESWAP_COMPLEX_F(TYPE, 32, SRC, DST)
#define SOLID_BYTESWAP_COMPLEX_E8(TYPE, SRC, DST) SOLID_BYTESWAP_COMPLEX_F(TYPE, 64, SRC, DST)
#define SOLID_BYTESWAP_COMPLEX_D(TYPE, BYTES, SRC, DST) SOLID_BYTESWAP_COMPLEX_E##BYTES(TYPE, SRC, DST)
#define SOLID_BYTESWAP_COMPLEX_C(TYPE, BYTES, SRC, DST) SOLID_BYTESWAP_COMPLEX_D(TYPE, BYTES, SRC, DST)
#define SOLID_BYTESWAP_COMPLEX_B(TYPE, SRC, DST) SOLID_BYTESWAP_COMPLEX_C(TYPE, SDTYPE_ELEMSIZE(TYPE), SRC, DST)
#define SOLID_BYTESWAP_COMPLEX(TYPE, SRC, DST) SOLID_BYTESWAP_COMPLEX_B(TYPE, SRC, DST)

#define SOLID_BYTESWAP_D(TYPE, CLASS, SRC, DST) SOLID_BYTESWAP_##CLASS(TYPE, SRC, DST)
#define SOLID_BYTESWAP_C(TYPE, CLASS, SRC, DST) SOLID_BYTESWAP_D(TYPE, CLASS, SRC, DST)
#define SOLID_BYTESWAP_B(TYPE, SRC, DST) SOLID_BYTESWAP_C(TYPE, SDTYPE_REAL_STR(TYPE), SRC, DST)
#define SOLID_BYTESWAP(TYPE, SRC, DST) SOLID_BYTESWAP_B(TYPE, SRC, DST)



/* ==================================================================== */
/* SOLID_ASSIGN(TYPE1, TYPE2, SRC, DST)                                 */
/* ==================================================================== */

/* Assign real to real */
#define SOLID_ASSIGN_REAL_TO_REAL_B(TYPE1, TYPE2, SRC, DST, FLAG) \
   *(DST) = SOLID_CONVERT(TYPE1, TYPE2, *(SRC))SOLID_SEMICOLON(FLAG)
#define SOLID_ASSIGN_REAL_TO_REAL(TYPE1, TYPE2, SRC, DST, FLAG) \
   SOLID_ASSIGN_REAL_TO_REAL_B(TYPE1, TYPE2, SRC, DST, FLAG)

/* Assign real to complex */
#define SOLID_ASSIGN_REAL_TO_COMPLEX_C(TYPE1, ETYPE2, SRC, DST, FLAG) \
   {  (DST) -> real = SOLID_CONVERT(TYPE1, ETYPE2, *(SRC)); \
      (DST) -> imag = SDTYPE_ZERO(ETYPE2); \
   }
#define SOLID_ASSIGN_REAL_TO_COMPLEX_B(TYPE1, ETYPE2, SRC, DST, FLAG) \
   SOLID_ASSIGN_REAL_TO_COMPLEX_C(TYPE1, ETYPE2, SRC, DST, FLAG)
#define SOLID_ASSIGN_REAL_TO_COMPLEX(TYPE1, TYPE2, SRC, DST, FLAG) \
   SOLID_ASSIGN_REAL_TO_COMPLEX_B(TYPE1, SDTYPE_ELEMTYPE(TYPE2), SRC, DST, FLAG)

/* Assign complex to real */
#define SOLID_ASSIGN_COMPLEX_TO_REAL_B(TYPE1, TYPE2, SRC, DST, FLAG) \
   *(DST) = SOLID_CONVERT(SDTYPE_ELEMTYPE(TYPE1),TYPE2,(SRC)->real)SOLID_SEMICOLON(FLAG)
#define SOLID_ASSIGN_COMPLEX_TO_REAL(TYPE1, TYPE2, SRC, DST, FLAG) \
   SOLID_ASSIGN_COMPLEX_TO_REAL_B(TYPE1, TYPE2, SRC, DST, FLAG)

/* Assign complex to complex - this may not be optimal when the types match */
#define SOLID_ASSIGN_COMPLEX_TO_COMPLEX_C(ETYPE1, ETYPE2, SRC, DST, FLAG) \
   {  (DST) -> real = SOLID_CONVERT(ETYPE1, ETYPE2, (SRC) -> real); \
      (DST) -> imag = SOLID_CONVERT(ETYPE1, ETYPE2, (SRC) -> imag); \
   }
#define SOLID_ASSIGN_COMPLEX_TO_COMPLEX_B(TYPE1, TYPE2, SRC, DST, FLAG) \
   SOLID_ASSIGN_COMPLEX_TO_COMPLEX_C(SDTYPE_ELEMTYPE(TYPE1), SDTYPE_ELEMTYPE(TYPE2), SRC, DST, FLAG)
#define SOLID_ASSIGN_COMPLEX_TO_COMPLEX(TYPE1, TYPE2, SRC, DST, FLAG) \
   SOLID_ASSIGN_COMPLEX_TO_COMPLEX_B(TYPE1, TYPE2, SRC, DST, FLAG)

/* Assign to Boolean */
#define SOLID_ASSIGN_REAL_TO_BOOL(TYPE1, SRC, DST, FLAG) \
   SOLID_ASSIGN_REAL_TO_REAL(TYPE1, bool, SRC, DST, FLAG)
#define SOLID_ASSIGN_COMPLEX_TO_BOOL_B(TYPE1, SRC, DST, FLAG) \
   *(DST) = ((SOLID_CONVERT(SDTYPE_ELEMTYPE(TYPE1), bool, (SRC)->real)) || \
             (SOLID_CONVERT(SDTYPE_ELEMTYPE(TYPE1), bool, (SRC)->imag)))SOLID_SEMICOLON(FLAG)
#define SOLID_ASSIGN_COMPLEX_TO_BOOL(TYPE1, SRC, DST, FLAG) \
   SOLID_ASSIGN_COMPLEX_TO_BOOL_B(TYPE1, SRC, DST, FLAG)

/* Assignment between different types */
#define SOLID_ASSIGN_0D_0(TYPE1, TYPE2, CLASS1, CLASS2, SRC, DST, FLAG) \
   SOLID_ASSIGN_##CLASS1##_TO_##CLASS2(TYPE1, TYPE2, SRC, DST, FLAG)
#define SOLID_ASSIGN_0D_1(TYPE1, TYPE2, CLASS1, CLASS2, SRC, DST, FLAG) \
   SOLID_ASSIGN_##CLASS1##_TO_BOOL(TYPE1, SRC, DST, FLAG)
#define SOLID_ASSIGN_0C(TYPE1, TYPE2, CLASS1, CLASS2, ISBOOL, SRC, DST, FLAG) \
   SOLID_ASSIGN_0D_##ISBOOL(TYPE1, TYPE2, CLASS1, CLASS2, SRC, DST, FLAG)
#define SOLID_ASSIGN_0B(TYPE1, TYPE2, CLASS1, CLASS2, ISBOOL, SRC, DST, FLAG) \
   SOLID_ASSIGN_0C(TYPE1, TYPE2, CLASS1, CLASS2, ISBOOL, SRC, DST, FLAG)
#define SOLID_ASSIGN_0(TYPE1, TYPE2, SRC, DST, FLAG) \
   SOLID_ASSIGN_0B(TYPE1, TYPE2, SDTYPE_REAL_STR(TYPE1), SDTYPE_REAL_STR(TYPE2), \
                   SDTYPE_EQUAL(TYPE2,bool), (SOLID_C_TYPE(TYPE1) *)(SRC), (SOLID_C_TYPE(TYPE2) *)(DST), FLAG)

/* Assignment to the same type */
#define SOLID_ASSIGN_1(TYPE1, TYPE2, SRC, DST, FLAG) \
   *((SOLID_C_TYPE(TYPE1) *)(DST)) = *((SOLID_C_TYPE(TYPE2) *)(SRC))SOLID_SEMICOLON(FLAG)

/* Assign macro */
#define SOLID_ASSIGN_C(TYPE1, TYPE2, EQUAL, SRC, DST, FLAG) SOLID_ASSIGN_##EQUAL(TYPE1, TYPE2, SRC, DST, FLAG)
#define SOLID_ASSIGN_B(TYPE1, TYPE2, EQUAL, SRC, DST, FLAG) SOLID_ASSIGN_C(TYPE1, TYPE2, EQUAL, SRC, DST, FLAG)

#define SOLID_ASSIGN_NO_SEMICOLON(TYPE1, TYPE2, SRC, DST) \
   SOLID_ASSIGN_B(TYPE1, TYPE2, SDTYPE_EQUAL(TYPE1, TYPE2), SRC, DST, 0)

#define SOLID_ASSIGN(TYPE1, TYPE2, SRC, DST) \
   SOLID_ASSIGN_B(TYPE1, TYPE2, SDTYPE_EQUAL(TYPE1, TYPE2), SRC, DST, 1)



/* ==================================================================== */
/* SOLID_BYTESWAP_ASSIGN(TYPE1, TYPE2, SRC, DST)                        */
/* ==================================================================== */

/* Helper macro */
#define SOLID_BYTESWAP_ASSIGN_HELPER(TYPE1, TYPE2, SRC, DST) \
   {  SOLID_C_TYPE(TYPE1) __byteswap_assign_buffer; \
      SOLID_BYTESWAP(TYPE1, SRC, &(__byteswap_assign_buffer)) \
      SOLID_ASSIGN(TYPE1, TYPE2, &(__byteswap_assign_buffer), DST); \
   }

/* Real to real */
#define SOLID_BYTESWAP_ASSIGN_REAL_TO_REAL(TYPE1, TYPE2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_HELPER(TYPE1, TYPE2, SRC, DST)

/* Real to complex */
#define SOLID_BYTESWAP_ASSIGN_REAL_TO_COMPLEX(TYPE1, TYPE2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_HELPER(TYPE1, TYPE2, SRC, DST)

/* Complex to complex */
#define SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_COMPLEX(TYPE1, TYPE2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_HELPER(TYPE1, TYPE2, SRC, DST)

/* Complex to real - avoid swapping the imaginary part */
#define SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_REAL_D(TYPE1, TYPE2, ETYPE1, SRC, DST) \
  {  SOLID_C_TYPE(ETYPE1) __byteswap_assign_buffer; \
     __byteswap_assign_buffer = ((SOLID_C_TYPE(TYPE1) *)(SRC)) -> real; \
     SOLID_BYTESWAP(ETYPE1, &(__byteswap_assign_buffer), &(__byteswap_assign_buffer)) \
     SOLID_ASSIGN(ETYPE1, TYPE2, &(__byteswap_assign_buffer), DST); \
  }
#define SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_REAL_C(TYPE1, TYPE2, ETYPE1, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_REAL_D(TYPE1, TYPE2, ETYPE1, SRC, DST)
#define SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_REAL_B(TYPE1, TYPE2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_REAL_C(TYPE1, TYPE2, SDTYPE_ELEMTYPE(TYPE1), SRC, DST)
#define SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_REAL(TYPE1, TYPE2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_REAL_B(TYPE1, TYPE2, SRC, DST)

/* Byteswap and assign to Boolean */
#define SOLID_BYTESWAP_ASSIGN_REAL_TO_BOOL(TYPE1, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_REAL_TO_REAL(TYPE1, bool, SRC, DST)

#define SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_BOOL_C(TYPE1, ETYPE1, SRC, DST) \
   {  SOLID_C_TYPE(ETYPE1) __byteswap_assign_buffer; \
      __byteswap_assign_buffer = ((SOLID_C_TYPE(TYPE1) *)(SRC)) -> real; \
      SOLID_BYTESWAP(ETYPE1, &(__byteswap_assign_buffer), &(__byteswap_assign_buffer)) \
      if ((SOLID_ASSIGN_NO_SEMICOLON(ETYPE1, bool, &(__byteswap_assign_buffer), DST)) == 0) \
      {  __byteswap_assign_buffer = ((SOLID_C_TYPE(TYPE1) *)(SRC)) -> imag; \
         SOLID_BYTESWAP(ETYPE1, &(__byteswap_assign_buffer), &(__byteswap_assign_buffer)) \
         SOLID_ASSIGN(ETYPE1, bool, &(__byteswap_assign_buffer), DST) \
      } \
   }
#define SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_BOOL_B(TYPE1, ETYPE, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_BOOL_C(TYPE1, ETYPE, SRC, DST)
#define SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_BOOL(TYPE1, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_COMPLEX_TO_BOOL_B(TYPE1, SDTYPE_ELEMTYPE(TYPE1), SRC, DST)

/* Byteswap and assign - different types */
#define SOLID_BYTESWAP_ASSIGN_0D_0(TYPE1, TYPE2, CLASS1, CLASS2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_##CLASS1##_TO_##CLASS2(TYPE1, TYPE2, SRC, DST)
#define SOLID_BYTESWAP_ASSIGN_0D_1(TYPE1, TYPE2, CLASS1, CLASS2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_##CLASS1##_TO_BOOL(TYPE1, SRC, DST)
#define SOLID_BYTESWAP_ASSIGN_0C(TYPE1, TYPE2, CLASS1, CLASS2, ISBOOL, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_0D_##ISBOOL(TYPE1, TYPE2, CLASS1, CLASS2, SRC, DST)
#define SOLID_BYTESWAP_ASSIGN_0B(TYPE1, TYPE2, CLASS1, CLASS2, ISBOOL, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_0C(TYPE1, TYPE2, CLASS1, CLASS2, ISBOOL, SRC, DST)
#define SOLID_BYTESWAP_ASSIGN_0(TYPE1, TYPE2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_0B(TYPE1, TYPE2, SDTYPE_REAL_STR(TYPE1), SDTYPE_REAL_STR(TYPE2), \
                            SDTYPE_EQUAL(TYPE2, bool), SRC, DST)

/* Byteswap and assign - matching types */
#define SOLID_BYTESWAP_ASSIGN_1(TYPE1, TYPE2, SRC, DST) \
   SOLID_BYTESWAP(TYPE1, SRC, DST)

/* Generic byteswap followed by assignment */
#define SOLID_BYTESWAP_ASSIGN_C(TYPE1, TYPE2, EQUAL, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_##EQUAL(TYPE1, TYPE2, SRC, DST)
#define SOLID_BYTESWAP_ASSIGN_B(TYPE1, TYPE2, EQUAL, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_C(TYPE1, TYPE2, EQUAL, SRC, DST)
#define SOLID_BYTESWAP_ASSIGN(TYPE1, TYPE2, SRC, DST) \
   SOLID_BYTESWAP_ASSIGN_B(TYPE1, TYPE2, SDTYPE_EQUAL(TYPE1, TYPE2), SRC, DST)


/* -------------------------------------------------------------------- */
/* Work type conversion                                                 */
/* -------------------------------------------------------------------- */

/* Convert original type to work type */
#define SOLID_TO_WORKTYPE_TYPE_B(TYPE, X) SOLID_CONVERT(TYPE, SOLID_WORKTYPE(TYPE), X)
#define SOLID_TO_WORKTYPE_TYPE(TYPE, X) SOLID_TO_WORKTYPE_TYPE_B(TYPE, X)
#define SOLID_TO_WORKTYPE(X) SOLID_TO_WORKTYPE_TYPE(SDXTYPE, X)

/* Convert work type to original type */
#define SOLID_FROM_WORKTYPE_TYPE_B(TYPE, X) SOLID_CONVERT(SOLID_WORKTYPE(TYPE), TYPE, X)
#define SOLID_FROM_WORKTYPE_TYPE(TYPE, X) SOLID_FROM_WORKTYPE_TYPE_B(TYPE, X)
#define SOLID_FROM_WORKTYPE(X) SOLID_FROM_WORKTYPE_TYPE(SDXTYPE, X)

/* Convert type to element work type */
#define SOLID_TO_ELEMWORKTYPE_TYPE(TYPE, X) SOLID_TO_WORKTYPE_TYPE(SDTYPE_ELEMTYPE(TYPE), X)
#define SOLID_TO_ELEMWORKTYPE(X) SOLID_TO_ELEMWORKTYPE_TYPE(SDXTYPE, X)

/* Convert element work type to element type */
#define SOLID_FROM_ELEMWORKTYPE_TYPE(TYPE, X) SOLID_FROM_WORKTYPE_TYPE(SDTYPE_ELEMTYPE(TYPE), X)
#define SOLID_FROM_ELEMWORKTYPE(X) SOLID_FROM_ELEMWORKTYPE_TYPE(SDXTYPE, X)

#endif
