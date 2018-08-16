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

#include "solid/base/generic/half.h"
#include "solid/base/generic/fpe.h"


/* ===================================================================== */
/* Function implementation                                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
solid_float solid_half_to_float(solid_half h)
/* -------------------------------------------------------------------- */
{  uint16_t  exp;
   uint32_t  s, d;
   uint32_t  shift;

   /* Extract the sign */
   s  = (uint32_t)(h & 0x8000) << 16;
   h &= 0x7FFF;

   if (h == 0)
   {  return 0.;
   }

   /* Get the exponent */
   exp = h & SD_HALF_EXPONENT_MASK; 

   /* Check for subnormal numbers */
   if (exp == 0)
   {  
      /* Shift h until there is a one bit in the first bit of the */
      /* exponent, that is, the location of the implicit one. The */
      /* fact h is not zero while the top 6 bits are zero ensures */
      /* that the loop will terminate.                            */
      h <<= 1; shift = 1;
      while (!(h & 0x0400)) { shift ++; h <<= 1; }

      /* Shift the exponent and mantissa */
      d  = ((uint32_t)(h)) << 13;

      /* Add 112 = (127-15) to the exponent bias and subtract the */
      /* shift count.                                             */
      d += (uint32_t)(112 - shift) << 23;

      /* Restore the sign */
      d |= s;

      /* Return the floating-point value */
      return *((float *)&d);
   }

   /* Check for NaN and Inf */
   if (exp == SD_HALF_EXPONENT_MASK)
   {  
      /* Shift the exponent and mantissa */
      d  = ((uint32_t)(h)) << 13;

      /* Set all bits in the exponent */
      d |= SD_FLOAT_EXPONENT_MASK;

      /* Restore the sign */
      d |= s;

      /* Return the floating-point value */
      return *((float *)&d);
   }

   /* Regular nonzero numbers */
   {  
      /* Shift the exponent and mantissa */
      d  = ((uint32_t)(h)) << 13;

      /* Add 112 = (127-15) to the exponent bias */
      d += 0x38000000;

      /* Restore the sign */
      d |= s;

      /* Return the floating-point value */
      return *((float *)&d);
   }
}


/* -------------------------------------------------------------------- */
solid_half solid_float_to_half(solid_float f)
/* -------------------------------------------------------------------- */
{  uint32_t    raw, d;
   uint16_t    s, exponent;
   solid_half  h;

   /* Get the raw representation of f */
   raw = *((uint32_t *)&f);
   d   = raw & 0x007FFFFF;

   /* Extract the sign and exponent */
   exponent = (uint16_t)(raw >> 16);
   s = (exponent & 0x8000);
   exponent &= 0x7FFF;
   exponent >>= 7;

   /* Check for zero and underflow */
   if (exponent < (127 - 15))
   {  if ((exponent > 0) || (d != 0)) solid_fpe_raise(SD_FPE_UNDERFLOW);
      return SD_HALF_ZERO;
   }

   /* Check for overflow, infinity, and NaN */
   if (exponent > (127 + 16))
   {  if (exponent == 255)
      {  if (d == 0)
         { h = (s | SD_HALF_EXPONENT_MASK);
         }
         else
         { h = SD_HALF_QUIET_NAN;
         }
      }
      else
      {  /* Plus or minus inifity */
         solid_fpe_raise(SD_FPE_OVERFLOW);
         h = (s | SD_HALF_EXPONENT_MASK);
      }
      return h;
   }

   /* Regular number */
   exponent -= (127 - 15);
   exponent <<= 10;
   exponent |= (d >> 13);
   exponent |= s;
   if ((d & 0x1FFF) >= 0x1000) exponent ++;

   return (solid_half)exponent;
}
