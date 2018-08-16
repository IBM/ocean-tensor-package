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

#include "ocean/base/format.h"
#include "ocean/base/platform.h"
#include "ocean/base/types.h"
#include "ocean/base/half.h"
#include "ocean/base/byteswap.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ocmax(a,b) (((a) > (b)) ? (a) : (b))
#define ocmin(a,b) (((a) < (b)) ? (a) : (b))


/* ===================================================================== */
/* Global variables                                                      */
/* ===================================================================== */

static const char oc_newline[] = "\n";
int oc_format_newline_width = sizeof(oc_newline)-1;

int oc_format_linewidth     = 120;
int oc_format_scientific    =   0;
int oc_format_decimals      =  -1;
int oc_format_min_decimals  =   0;
int oc_format_max_decimals  =   5;


/* ===================================================================== */
/* Analysis and output function declarations                             */
/* ===================================================================== */

/* Analyze the format of a scalar and output to a string. The output   */
/* functions are guaranteed to output at most the number of characters */
/* given by the format width, followed by the terminating null symbol. */
#define OC_TEMPLATE(TYPE) \
static void OcFormatAnalyze_##TYPE(const char *data, OcFormat *format, int index); \
static void OcFormatOutput_##TYPE (const char *data, OcFormat *format, int index, char *str);
OC_GENERATE_ALL_TYPES
#undef OC_TEMPLATE


/* ===================================================================== */
/* Function implementation                                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcFormat *OcFormatAllocate(OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcFormat          *format;
   int                parts, i;
   int                result = 0;

   /* Allocate memory for the format structure */
   format = (OcFormat *)OcMalloc(sizeof(OcFormat));
   if (format == NULL) goto error;

   /* Determine the number of parts in the data type */
   parts = OcDType_parts(dtype);
   format -> parts = parts;

   /* Allocate the format information pointers */
   format -> info  = (OcFormattingInfo **)OcMalloc(sizeof(OcFormattingInfo *) * parts);
   if (format -> info == NULL) goto error;

   /* Allocate the format information structures */
   for (i = 0; i < parts; i++)
   {  format -> info[i] = (OcFormattingInfo *)OcMalloc(sizeof(OcFormattingInfo));
      if (format -> info[i] == NULL) result = -1;
   }
   if (result == -1) goto error;

   /* Success */
   return format;

error :
   OcFormatFree(format);

   OcError(NULL,"Insufficient memory to allocate the format structure");
}


/* -------------------------------------------------------------------- */
void OcFormatFree(OcFormat *format)
/* -------------------------------------------------------------------- */
{  int i;

   if (format != NULL)
   {  
      /* Free the format information structures */
      if (format -> info != NULL)
      {  
         for (i = 0; i < format -> parts; i++)
         {  if (format -> info[i]) OcFree(format -> info[i]);
         }

         /* Free the pointer list */
         OcFree(format -> info);
      }

      /* Free the format structure */
      OcFree(format);
   }
}


/* -------------------------------------------------------------------- */
OcFormat *OcFormatCreate(OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcFormat         *format;
   OcFormattingInfo *info;
   int               i;

   /* Allocate the format structure */
   format = OcFormatAllocate(dtype);
   if (format == NULL) return NULL;

   /* Set default values */
   format -> newlineWidth         = oc_format_newline_width;
   format -> log10                = log(10.0);
   format -> byteswapped          = 0;
   format -> hex                  = 0;
   format -> exponentCharacter    = 'e';
   format -> minimumExponentWidth = 2; /* Including the sign */
   format -> flagScientific       = 0;
   format -> width                = 0;

   /* Set the maximum number of decimals */
   if (OcDType_isFloat(dtype))
   {  format -> flagScientific = oc_format_scientific;
      if (oc_format_decimals >= 0)
      {  OcFormatSetMaxDecimals(format, oc_format_decimals);
      }
      else
      {  OcFormatSetMaxDecimals(format, oc_format_max_decimals);
      }
   }
   else
   {  OcFormatSetMaxDecimals(format, 0);
   }

   for (i = 0; i < format -> parts; i++)
   {  /* Initialize the part-specific information */
      info = format -> info[i];
      info -> flagSeparateSign        = (i == 0) ? 0 : 1;
      info -> signedWidth             = 0;
      info -> unsignedWidth           = 0;
      info -> integerWidth            = 0;
      info -> signedLiteral           = 0;
      info -> unsignedLiteral         = 0;
      info -> literalWidth            = 0;
      info -> flagNegative            = 0;
      info -> flagLeadingSign         = 0;
      info -> width                   = 0;

      /* Initialize the fraction width */
      info -> fractionWidth           = 0;
      info -> fractionWidthScientific = 0;

      if (OcDType_isFloat(dtype))
      {  if (oc_format_decimals >= 0)
         {  info -> fractionWidth           = format -> maximumDecimals;
            info -> fractionWidthScientific = format -> maximumDecimals;
         }
         else if (oc_format_min_decimals > 0)
         {  info -> fractionWidth           = ocmin(oc_format_min_decimals, format -> maximumDecimals);
            info -> fractionWidthScientific = ocmin(oc_format_min_decimals, format -> maximumDecimals);
         }
      }
   }

   /* Return the format structure */
   return format;
}


/* -------------------------------------------------------------------- */
void OcFormatFinalize(OcFormat *format)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   int               fractionWidth = 0;
   int               numberWidth;
   int               i;

   /* Initialize the overall format width to zero */
   format -> width = 0;

   /* Determine the maximum fractionWidth */
   for (i = 0; i < format -> parts; i++)
   {
      /* Get the part-specific information */
      info = format -> info[i];

      /* Update the fraction width in scientific mode */
      if (format -> flagScientific)
      {  info -> fractionWidth = info -> fractionWidthScientific;
      }

      /* Keep track of the maximum fraction width */
      fractionWidth = ocmax(fractionWidth, info -> fractionWidth);
   }

   /* Set the maximum number of decimals */
   OcFormatSetMaxDecimals(format, fractionWidth);

   /* Process each of the parts */
   for (i = 0; i < format -> parts; i++)
   {
      /* Get the part-specific information */
      info = format -> info[i];

      /* Set the fraction width to the maximum */
      info -> fractionWidth = fractionWidth;

      /* ---------------------------------- */
      /*     Determine the number width     */
      /* ---------------------------------- */
      if (format -> flagScientific)
      {
         /* Determine the integer width */
         if (info -> flagSeparateSign)
         {  info -> integerWidth = 1;
         }
         else
         {  info -> integerWidth    = (info -> flagNegative) ? 2 : 1;
            info -> flagLeadingSign = info -> flagNegative;
         }

         /* Initialize the width */
         numberWidth = info -> integerWidth;

         /* Set the default width */
         numberWidth += (info -> fractionWidth > 0) ? (info -> fractionWidth + 1) : 0;            
         numberWidth += 1; /* Exponent sign */
         numberWidth += format -> minimumExponentWidth;
      }
      else
      {  
         /* Determine the integer width */
         if (info -> flagSeparateSign)
         {  info -> integerWidth = ocmax(info -> unsignedWidth, info -> signedWidth - 1);
         }
         else
         {  if (info -> unsignedWidth < info -> signedWidth)
            {  info -> integerWidth    = info -> signedWidth;
               info -> flagLeadingSign = 1;
            }
            else
            {  info -> integerWidth    = info -> unsignedWidth;
            }
         }

         /* Initialize the width */
         numberWidth = info -> integerWidth;

         /* The fractional parts */
         if (info -> fractionWidth > 0)
         {  numberWidth += info -> fractionWidth + 1;
         }
      }

      /* ---------------------------------- */
      /*    Determine the literal width     */
      /* ---------------------------------- */
      if (info -> flagSeparateSign)
      {  info -> literalWidth = ocmax(info -> unsignedLiteral,  info -> signedLiteral - 1);
      }
      else
      {  if ((info -> unsignedLiteral) < (info -> signedLiteral))
         {  info -> literalWidth = info -> signedLiteral;
            if (info -> literalWidth > numberWidth)
            {  info -> flagLeadingSign = 1;
            }
            else
            {  /* Value of leading sign flag does not change */
            }
         }
         else
         {  info -> literalWidth = info -> unsignedLiteral;
            if (info -> literalWidth >= numberWidth)
            {  info -> flagLeadingSign = 0;
            }
         }
      }

      /* Determine the total width; we adjust the integer width if   */
      /* necessary to match the total width.                         */
      if (numberWidth < info -> literalWidth)
      {  info -> width = info -> literalWidth;
         info -> integerWidth += (info -> literalWidth - numberWidth);
      }
      else
      {  info -> width = numberWidth;
      }
      info -> width = ocmax(numberWidth, info -> literalWidth);

      /* Add [+,-] and a space, when using separate sign mode */
      if (info -> flagSeparateSign) info -> width +=2; 
   
      /* Compute the total width, taking into account multiple parts. */
      /* Any additional part has a preceding space, a separate sign   */
      /* format, and a trailing character (such as the complex 'i').  */
      /* To account for the space and character, we add two to the    */
      /* width for all parts other than the first.                    */
      format -> width += info -> width;
      if (i > 0) format -> width += 2;
   }

   /* Flag leading sign for the format */
   format -> flagLeadingSign = format -> info[0] -> flagLeadingSign;
}


/* -------------------------------------------------------------------- */
void OcFormatSetMaxDecimals(OcFormat *format, int maxDecimals)
/* -------------------------------------------------------------------- */
{  
   maxDecimals = ocmax(1,maxDecimals);
   format -> maximumDecimals = (maxDecimals < 16) ? maxDecimals : 16 ;
   format -> scaleDecimals = exp((format -> log10) * (format -> maximumDecimals));
   format -> roundDecimals = 0.5 / format -> scaleDecimals;
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_Bool(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Determine the width */
   if (*data == 0)
        info -> unsignedLiteral = ocmax(info -> unsignedLiteral, 5); /* False */
   else info -> unsignedLiteral = ocmax(info -> unsignedLiteral, 4); /* True */
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_Int8(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   int8_t            v;
   int               width;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Hexadecimal mode */
   if (format -> hex) { info -> unsignedWidth = 2; return ; }

   /* Get the data */
   v = *((int8_t *)data);

   /* Determine the width */
   if (v < 0)
   {  width = (v > -10) ? 2 : (int)(log(0.5 - (double)(v)) / format -> log10) + 2;
      info -> signedWidth = ocmax(info -> signedWidth, width);
      info -> flagNegative = 1;
   }
   else
   {  width = (v < 10) ? 1 : (int)(log((double)v + 0.5) / format -> log10) + 1;
      info -> unsignedWidth = ocmax(info -> unsignedWidth, width);
   }
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_Int16(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   int16_t           raw, v;
   int               width;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Hexadecimal mode */
   if (format -> hex) { info -> unsignedWidth = 4; return ; }

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((int16_t *)data);
      OC_BYTESWAP_16(raw,v);
   }
   else
   {
      v = *((int16_t *)data);
   }

   /* Determine the width */
   if (v < 0)
   {  width = (v > -10) ? 2 : (int)(log(0.5 - (double)(v)) / format -> log10) + 2;
      info -> signedWidth = ocmax(info -> signedWidth, width);
      info -> flagNegative = 1;
   }
   else
   {  width = (v < 10) ? 1 : (int)(log((double)v + 0.5) / format -> log10) + 1;
      info -> unsignedWidth = ocmax(info -> unsignedWidth, width);
   }
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_Int32(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   int32_t           raw, v;
   int               width;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Hexadecimal mode */
   if (format -> hex) { info -> unsignedWidth = 8; return ; }

   /* Byte swap the data if needed */
      if (format -> byteswapped)
   {  raw = *((int32_t *)data);
      OC_BYTESWAP_32(raw,v);
   }
   else
   {
      v = *((int32_t *)data);
   }

   /* Determine the width */
   if (v < 0)
   {  width = (v > -10) ? 2 : (int)(log(0.5 - (double)(v)) / format -> log10) + 2;
      info -> signedWidth = ocmax(info -> signedWidth, width);
      info -> flagNegative = 1;
   }
   else
   {  width = (v < 10) ? 1 : (int)(log((double)v + 0.5) / format -> log10) + 1;
      info -> unsignedWidth = ocmax(info -> unsignedWidth, width);
   }
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_Int64(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   int64_t           raw, v;
   int               width;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Hexadecimal mode */
   if (format -> hex) { info -> unsignedWidth = 16; return ; }

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((int64_t *)data);
      OC_BYTESWAP_64(raw,v);
   }
   else
   {
      v = *((int64_t *)data);
   }

   /* Determine the width */
   if (v < 0)
   {  width = (v > -10) ? 2 : (int)(log(0.5 - (double)(v)) / format -> log10) + 2;
      info -> signedWidth = ocmax(info -> signedWidth, width);
      info -> flagNegative = 1;
   }
   else
   {  width = (v < 10) ? 1 : (int)(log((double)v + 0.5) / format -> log10) + 1;
      info -> unsignedWidth = ocmax(info -> unsignedWidth, width);
   }
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_UInt8(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   uint8_t           v;
   int               width;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Hexadecimal mode */
   if (format -> hex) { info -> unsignedWidth = 2; return ; }

   /* Get the data */
   v = *((uint8_t *)data);

   /* Determine the width */
   width = (v < 10) ? 1 : (int)(log((double)v + 0.5) / format -> log10) + 1;
   info -> unsignedWidth = ocmax(info -> unsignedWidth, width);
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_UInt16(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   uint16_t          raw, v;
   int               width;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Hexadecimal mode */
   if (format -> hex) { info -> unsignedWidth = 4; return ; }

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((uint16_t *)data);
      OC_BYTESWAP_16(raw,v);
   }
   else
   {
      v = *((uint16_t *)data);
   }

   /* Determine the width */
   width = (v < 10) ? 1 : (int)(log((double)v + 0.5) / format -> log10) + 1;
   info -> unsignedWidth = ocmax(info -> unsignedWidth, width);
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_UInt32(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   uint32_t          raw, v;
   int               width;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Hexadecimal mode */
   if (format -> hex) { info -> unsignedWidth = 8; return ; }

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((uint32_t *)data);
      OC_BYTESWAP_32(raw,v);
   }
   else
   {
      v = *((uint32_t *)data);
   }

   /* Determine the width */
   width = (v < 10) ? 1 : (int)(log((double)v + 0.5) / format -> log10) + 1;
   info -> unsignedWidth = ocmax(info -> unsignedWidth, width);
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_UInt64(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   uint64_t          raw, v;
   int               width;

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Hexadecimal mode */
   if (format -> hex) { info -> unsignedWidth = 16; return ; }

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((uint64_t *)data);
      OC_BYTESWAP_64(raw,v);
   }
   else
   {
      v = *((uint64_t *)data);
   }

   /* Determine the width */
   width = (v < 10) ? 1 : (int)(log((double)v + 0.5) / format -> log10) + 1;
   info -> unsignedWidth = ocmax(info -> unsignedWidth, width);
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_Half(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  int16_t   raw1, raw2;
   double    d;
   int       byteswapped;

   /* Get and clear byte-swapped flag */
   byteswapped = format -> byteswapped;
   format -> byteswapped = 0;

   /* Byte swap the data if needed */
   raw1 = *((int16_t *)data);
   if (byteswapped)
   {  OC_BYTESWAP_16(raw1,raw2);
      raw1 = raw2;
   }

   /* Convert half precision to double precision floating-point format */
   d = (double)OcHalfToFloat(raw1);

   /* Analyze the format of the double scalar */
   OcFormatAnalyze_Double((const char *)(&d), format, index);

   /* Restore the byte-swapped flag */
   format -> byteswapped = byteswapped;
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_Float(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  int32_t   raw1,raw2;
   float     v;
   double    d;
   int       byteswapped;

   /* Get and clear byte-swapped flag */
   byteswapped = format -> byteswapped;
   format -> byteswapped = 0;

   /* Convert input to double */
   if (byteswapped)
   {  raw1 = *((int32_t *)data);
      OC_BYTESWAP_32(raw1,raw2);
      v = *((float *)(&raw2));
   }
   else
   {  v = *((float *)data);
   }
   d = (double)v;

   /* Analyze the format of the double scalar */
   OcFormatAnalyze_Double((const char *)(&d), format, index);

   /* Restore the byte-swapped flag */
   format -> byteswapped = byteswapped;
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_Double(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo *info;
   double            value, v, e;
   int               negativeFlag = 0;
   int               exponent;       /* Integer exponent */
   int               exponentWidth;  /* Width of the exponent, minus the sign */
   int               decimalWidth;   /* Number of decimals */
   int               integer;        /* Leading digits */
   int               integerWidth;   /* Width of integer part */
   uint64_t          raw1, raw2;
   unsigned long int d;

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw1 = *((int64_t *)data);
      OC_BYTESWAP_64(raw1,raw2);
      value = *((double *)(&raw2));
   }
   else
   {  value = *((double *)data);
   }

   /* Get the formatting info structure */
   info = format -> info[index];

   /* Inf, -Inf, NaN, -NaN */
   if (isnan(value))
   {  info -> unsignedLiteral = ocmax(info -> unsignedLiteral, 3);
      return ;
   }
   else if (!isfinite(value))
   {  /* Maximum width without the sign */
      if (value < 0)
      {  info -> flagNegative = 1;
         info -> signedLiteral = ocmax(info -> signedLiteral, 4);
      }
      else
      {  info -> unsignedLiteral = ocmax(info -> unsignedLiteral, 3);
      }
      return ;
   }

   /* Get the sign and take absolute value */
   if (value < 0)
    {  if (info -> flagSeparateSign == 0)
       {  info -> flagNegative = 1;
          negativeFlag = 1;
       }
       value *= -1;
    }

   /* Check for zeros if scientific format is needed */
   if (value == 0)
   {  info -> unsignedWidth = ocmax(info -> unsignedWidth, 1);
      return ;
   }
   else if ((value < 1e-4) || (value >= 1e5))
   {  /* Switch to scientific mode */
      format -> flagScientific = 1;
   }

   /* -------------------------------------------------- */
   /*  Determine the parameters for scientific notation  */
   /* -------------------------------------------------- */
   v = value;

   /* Extract the exponent */
   e = log(v) / format -> log10;
   exponent = (int)floor(e);
   v = exp(log(10) * (e - exponent));

   /* Take into account the maximum number of decimals */
   v += format -> roundDecimals;
   integer = (int)v;
   v -= integer;

   /* Take special care of the situation where rounding increases */
   /* the exponent; for example 9.9999 with three decimals.       */
   if (integer == 10) { integer = 1; exponent ++; v = 0; }

   /* Determine the decimals as an integer part of scale decimals.*/
   /* For example, with three decimals 0.052 would be 52 / 1000.  */
   d = (long unsigned int)(v * format -> scaleDecimals);
   if (d == 0)
   {  decimalWidth = 0;
   }
   else
   {  decimalWidth = format -> maximumDecimals;
      while (d % 10 == 0) { decimalWidth --; d /= 10; }
   }

   /* Determine the exponent width */
   exponentWidth = 2;
   if (exponent < 0) exponent *= -1;
   while (exponent >= 10) { exponentWidth ++; exponent /= 10; }

   /* Set the fields */
   info -> fractionWidthScientific = ocmax(info -> fractionWidthScientific, decimalWidth);
   format -> minimumExponentWidth  = ocmax(format -> minimumExponentWidth, exponentWidth);

   /* Return if we are in scientific mode */
   if (format -> flagScientific) return ;


   /* -------------------------------------------------- */
   /*  Determine the parameters for plain notation       */
   /* -------------------------------------------------- */
   v = value;

   /* Round the number and make sure that we can still use plain */
   /* notation. The switch to scientific mode would happen for   */
   /* instance when rounding 9999.9996 to three decimal places.  */
   v += format -> roundDecimals;
   d  = (long unsigned int)v;
   if (v >= 1e5) { format -> flagScientific = 1; return; }
   v -= d;

   /* Determine the width of the integer part */
   integerWidth = 1;
   while (d >= 10) { integerWidth ++; d /= 10; }

   /* Determine the width of the fraction */
   d = (long unsigned int)(v * format -> scaleDecimals);
   if (d == 0)
   {  decimalWidth = 0;
   }
   else
   {  decimalWidth = format -> maximumDecimals;
      while (d % 10 == 0) { decimalWidth --; d /= 10; }
   }

   /* Determine the integer and fraction widths */
   if (negativeFlag)
   {  integerWidth ++;
      info -> signedWidth   = ocmax(info -> signedWidth, integerWidth);
   }
   else
   {  info -> unsignedWidth = ocmax(info -> unsignedWidth, integerWidth);
   }
   info -> fractionWidth = ocmax(info -> fractionWidth, decimalWidth);
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_CHalf(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{
   OcFormatAnalyze_Half((const char *)&(((const OcCHalf *)data) -> real), format, 0);
   OcFormatAnalyze_Half((const char *)&(((const OcCHalf *)data) -> imag), format, 1);

   (void)index; /* Avoid compiler warnings */
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_CFloat(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{
   OcFormatAnalyze_Float((const char *)&(((const OcCFloat *)data) -> real), format, 0);
   OcFormatAnalyze_Float((const char *)&(((const OcCFloat *)data) -> imag), format, 1);

   (void)index; /* Avoid compiler warnings */
}


/* -------------------------------------------------------------------- */
static void OcFormatAnalyze_CDouble(const char *data, OcFormat *format, int index)
/* -------------------------------------------------------------------- */
{
   OcFormatAnalyze_Double((const char *)&(((const OcCDouble *)data) -> real), format, 0);
   OcFormatAnalyze_Double((const char *)&(((const OcCDouble *)data) -> imag), format, 1);

   (void)index; /* Avoid compiler warnings */
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_Bool(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Format the Boolean scalar */
   snprintf(str, info -> width + 1, "%*s", info -> width, (*data) ? "True" : "False");
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_Int8(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   int8_t             v;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Get the data */
   v = *((int8_t *)data);

   /* Output the number */
   if (format -> hex)
        OcFormatHex(str, info -> width, (unsigned long int)v);
   else OcFormatLong(str, info -> width, (long int)v);
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_Int16(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   int16_t            raw, v;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((int16_t *)data);
      OC_BYTESWAP_16(raw,v);
   }
   else
   {
      v = *((int16_t *)data);
   }

   /* Output the number */
   if (format -> hex)
        OcFormatHex(str, info -> width, (unsigned long int)v);
   else OcFormatLong(str, info -> width, (long int)v);
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_Int32(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   int32_t            raw, v;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((int32_t *)data);
      OC_BYTESWAP_32(raw,v);
   }
   else
   {
      v = *((int32_t *)data);
   }

   /* Output the number */
   if (format -> hex)
        OcFormatHex(str, info -> width, (unsigned long int)v);
   else OcFormatLong(str, info -> width, (long int)v);
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_Int64(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   int64_t            raw, v;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((int64_t *)data);
      OC_BYTESWAP_64(raw,v);
   }
   else
   {
      v = *((int64_t *)data);
   }

   /* Output the number */
   if (format -> hex)
        OcFormatHex(str, info -> width, (unsigned long int)v);
   else OcFormatLong(str, info -> width, (long int)v);
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_UInt8(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   uint8_t            v;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Get the data */
   v = *((uint8_t *)data);

   /* Output the number */
   if (format -> hex)
        OcFormatHex(str, info -> width, (unsigned long int)v);
   else OcFormatULong(str, info -> width, (unsigned long int)v);
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_UInt16(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   uint16_t           raw, v;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((uint16_t *)data);
      OC_BYTESWAP_16(raw,v);
   }
   else
   {
      v = *((uint16_t *)data);
   }

   /* Output the number */
   if (format -> hex)
        OcFormatHex(str, info -> width, (unsigned long int)v);
   else OcFormatULong(str, info -> width, (unsigned long int)v);
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_UInt32(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   uint32_t           raw, v;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((uint32_t *)data);
      OC_BYTESWAP_32(raw,v);
   }
   else
   {
      v = *((uint32_t *)data);
   }

   /* Output the number */
   if (format -> hex)
        OcFormatHex(str, info -> width, (unsigned long int)v);
   else OcFormatULong(str, info -> width, (unsigned long int)v);
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_UInt64(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   uint64_t           raw, v;
   
   /* Get the formatting info structure */
   info = format -> info[index];

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw = *((uint64_t *)data);
      OC_BYTESWAP_64(raw,v);
   }
   else
   {
      v = *((uint64_t *)data);
   }

   /* Output the number */
   if (format -> hex)
        OcFormatHex(str, info -> width, (unsigned long int)v);
   else OcFormatULong(str, info -> width, (unsigned long int)v);
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_Half(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  int16_t   raw1, raw2;
   double    d;
   int       byteswapped;

   /* Get and clear byte-swapped flag */
   byteswapped = format -> byteswapped;
   format -> byteswapped = 0;

   /* Byte swap the data if needed */
   raw1 = *((int16_t *)data);
   if (byteswapped)
   {  OC_BYTESWAP_16(raw1,raw2);
      raw1 = raw2;
   }

   /* Convert half precision to double precision floating-point format */
   d = (double)OcHalfToFloat(raw1);

   /* Output the double scalar */
   OcFormatOutput_Double((const char *)(&d), format, index, str);

   /* Restore the byte-swapped flag */
   format -> byteswapped = byteswapped;
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_Float(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  int32_t   raw1, raw2;
   float     v;
   double    d;
   int       byteswapped;

   /* Get and clear byte-swapped flag */
   byteswapped = format -> byteswapped;
   format -> byteswapped = 0;

   /* Convert input to double */
   if (byteswapped)
   {  raw1 = *((int32_t *)data);
      OC_BYTESWAP_32(raw1,raw2);
      v = *((float *)(&raw2));
   }
   else
   {  v = *((float *)data);
   }
   d = (double)v;

   /* Output the double scalar */
   OcFormatOutput_Double((const char *)(&d), format, index, str);

   /* Restore the byte-swapped flag */
   format -> byteswapped = byteswapped;
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_Double(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  OcFormattingInfo  *info;
   double             value, e;
   int                negativeFlag = 0;
   int                exponent;       /* Integer exponent */
   int                integer;        /* Leading digits */
   int                width;
   uint64_t           raw1, raw2;
   unsigned long int  d;

   /* Byte swap the data if needed */
   if (format -> byteswapped)
   {  raw1 = *((int64_t *)data);
      OC_BYTESWAP_64(raw1,raw2);
      value = *((double *)(&raw2));
   }
   else
   {  value = *((double *)data);
   }

   /* Get the formatting info structure */
   info  = format -> info[index];
   width = info -> width;

   /* Separate the sign from the number if needed */
   if (info -> flagSeparateSign)
   {
      /* Output the [+,-] and work with absolute value */
      if ((!isnan(value)) && (value < 0))
      {  *str = '-'; str ++;
         value *= -1;
      }
      else
      {  *str = '+'; str ++;
      }

      /* Output the space */
      *str = ' '; str ++;

      /* Decrease the width */
      width -= 2;
   }

   /* Deal with infinity and NaN */
   if (isnan(value))
   {  snprintf(str, width+1, "%*s", width, "nan");
      return ;
   }
   else if (!isfinite(value))
   {  snprintf(str, width+1, "%*s", width, (value < 0) ? "-inf" : "inf");
      return ;
   }

   /* Work with the absolute value */
   if (value < 0) { negativeFlag = 1; value *= -1; }

   /* --------------------------------------------------- */
   /*  Format floating-point number in scientific format  */
   /* --------------------------------------------------- */
   if (format -> flagScientific)
   {
      /* Extract the exponent */
      if (value > 0)
      {  e = log(value) / format -> log10;
         exponent = (int)floor(e);
         value = exp(log(10) * (e - exponent));
      }
      else
      {  exponent = 0.0;
         value    = 0.0;
      }

      /* Take into account the maximum number of decimals */
      value += format -> roundDecimals;
      integer = (int)value;
      value -= integer;

      /* Take special care of the situation where rounding increases */
      /* the exponent; for example 9.9999 with three decimals.       */
      if (integer == 10) { integer = 1; exponent ++; value = 0; }

      /* Determine the decimals as an integer part of scale decimals.*/
      /* For example, with three decimals 0.052 would be 52 / 1000.  */
      d = (long unsigned int)(value * format -> scaleDecimals);
    
      /* Update the sign of the integer if needed */
      if (negativeFlag)
      {  integer *= -1;
      }

      /* Output the integer part */
      snprintf(str, info -> integerWidth + 1, "%*d", info -> integerWidth, integer);
      str += info -> integerWidth;

      /* Output the fractional part */
      if (info -> fractionWidth > 0)
      {  snprintf(str, info -> fractionWidth + 2,".%0*" OC_FORMAT_LU, info -> fractionWidth, d);
         str += info -> fractionWidth + 1;
      }

      /* Output the exponent part */
      snprintf(str, format -> minimumExponentWidth + 3, "e%+0*d", format -> minimumExponentWidth, exponent);

      return;
   }


   /* --------------------------------------------------- */
   /*    Format floating-point number in plain format     */
   /* --------------------------------------------------- */

   if (value == 0)
   {  /* Make sure we do not output -0 */
      integer = 0; negativeFlag = 0;
      d = 0;      
   }
   else
   {  
      /* Round the number and make sure that we can still use plain */
      /* notation. The switch to scientific mode would happen for   */
      /* instance when rounding 9999.9996 to three decimal places.  */
      value += format -> roundDecimals;
      integer = (long unsigned int)value; /* Integer part */
      value -= integer;

      /* Determine the decimal part */
      d = (long unsigned int)(value * format -> scaleDecimals);

      /* Update the sign of the integer if needed */
      if (negativeFlag)
      {  integer *= -1;
      }
   }

   /* Output the integer part */
   if ((negativeFlag) && (integer == 0))
        snprintf(str, info -> integerWidth + 1, "%*s", info -> integerWidth, "-0");
   else snprintf(str, info -> integerWidth + 1, "%*d", info -> integerWidth, integer);
   
   str += info -> integerWidth;

   /* Output the fractional part */
   if (info -> fractionWidth > 0)
   {  snprintf(str, info -> fractionWidth + 2,".%0*" OC_FORMAT_LU, info -> fractionWidth, d);
   }
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_CHalf(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  int i;

   for (i = 0; i < format -> parts; i++)
   {
      /* Output the half scalar */
      if (i == 0)
           OcFormatOutput_Half((const char *)&(((OcCHalf *)data) -> real), format, i, str);
      else OcFormatOutput_Half((const char *)&(((OcCHalf *)data) -> imag), format, i, str);

      /* Update the string pointer */
      str += format -> info[i] -> width;

      /* Add spacing or imaginary marker */
      if (i == 0)
      {  *str = ' '; str ++; *str = '\0';
      }
      else
      {  *str = 'j'; str ++; *str = '\0';
      }
   }

   (void)index; /* Avoid compiler warnings */
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_CFloat(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  int i;

   for (i = 0; i < format -> parts; i++)
   {
      /* Output the float scalar */
      if (i == 0)
           OcFormatOutput_Float((const char *)&(((OcCFloat *)data) -> real), format, i, str);
      else OcFormatOutput_Float((const char *)&(((OcCFloat *)data) -> imag), format, i, str);

      /* Update the string pointer */
      str += format -> info[i] -> width;

      /* Add spacing or imaginary marker */
      if (i == 0)
      {  *str = ' '; str ++; *str = '\0';
      }
      else
      {  *str = 'j'; str ++; *str = '\0';
      }
   }

   (void)index; /* Avoid compiler warnings */
}


/* -------------------------------------------------------------------- */
static void OcFormatOutput_CDouble(const char *data, OcFormat *format, int index, char *str)
/* -------------------------------------------------------------------- */
{  int i;

   for (i = 0; i < format -> parts; i++)
   {
      /* Output the double scalar */
      if (i == 0)
           OcFormatOutput_Double((const char *)&(((OcCDouble *)data) -> real), format, i, str);
      else OcFormatOutput_Double((const char *)&(((OcCDouble *)data) -> imag), format, i, str);

      /* Update the string pointer */
      str += format -> info[i] -> width;

      /* Add spacing or imaginary marker */
      if (i == 0)
      {  *str = ' '; str ++; *str = '\0';
      }
      else
      {  *str = 'j'; str ++; *str = '\0';
      }
   }

   (void)index; /* Avoid compiler warnings */
}


/* -------------------------------------------------------------------- */
int OcFormatLongWidth(long int v)
/* -------------------------------------------------------------------- */
{
   if (v < 0)
        return (int)(log((double)(-v) + 0.5) / log(10)) + 2;
   else return (int)(log((double)( v) + 0.5) / log(10)) + 1;
}


/* -------------------------------------------------------------------- */
int OcFormatLong(char *str, int width, long int v)
/* -------------------------------------------------------------------- */
{
   return snprintf(str, width+1, "%*" OC_FORMAT_LD, width, v);
}


/* -------------------------------------------------------------------- */
int OcFormatULongWidth(unsigned long int v)
/* -------------------------------------------------------------------- */
{
  return (int)(log((double)(v) + 0.5) / log(10)) + 1;
}


/* -------------------------------------------------------------------- */
int OcFormatULong(char *str, int width, unsigned long int v)
/* -------------------------------------------------------------------- */
{
   return snprintf(str, width+1, "%*" OC_FORMAT_LU, width, v);
}


/* -------------------------------------------------------------------- */
int OcFormatHex(char *str, int width, unsigned long int v)
/* -------------------------------------------------------------------- */
{
   return snprintf(str, width+1, "%0*lX", width, v);
}



/* -------------------------------------------------------------------- */
OcFormatAnalyze_funptr OcFormatAnalyze_function(OcDType dtype)
/* -------------------------------------------------------------------- */
{
   switch(dtype)
   {
      #define OC_TEMPLATE(TYPE) \
      case OcDType##TYPE : return OcFormatAnalyze_##TYPE;
      OC_GENERATE_ALL_TYPES
      #undef OC_TEMPLATE
      default : return 0;
   }
}


/* -------------------------------------------------------------------- */
OcFormatOutput_funptr OcFormatOutput_function(OcDType dtype)
/* -------------------------------------------------------------------- */
{
   switch(dtype)
   {
      #define OC_TEMPLATE(TYPE) \
      case OcDType##TYPE : return OcFormatOutput_##TYPE;
      OC_GENERATE_ALL_TYPES
      #undef OC_TEMPLATE
      default : return 0;
   }
}

