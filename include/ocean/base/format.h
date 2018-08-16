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

#ifndef __OC_FORMAT_H__
#define __OC_FORMAT_H__

/* Size of the workspace buffer, must at least be 4 */
#define OC_FORMAT_BUFFER_SIZE 256

#include "ocean/base/dtype.h"
#include "ocean/base/generate_macros.h"
#include "ocean/base/api.h"


/* ===================================================================== */
/* Global variables                                                      */
/* ===================================================================== */

/* Maximum line-width */
extern int oc_format_linewidth;

/* Characters in the new-line symbol */
extern int oc_format_newline_width;

/* Floating-point formatting settings */
extern int oc_format_scientific;   /* Format all floats as scientific    */
extern int oc_format_decimals;     /* Exact number of decimals or -1     */
extern int oc_format_min_decimals; /* Minimum number of decimals         */
extern int oc_format_max_decimals; /* Maximum number of decimals         */


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{
   /* The user can set the following flag to indicate that the   */
   /* + or - sign should be output separately from the number.   */
   /* This is useful when outputting sums of numbers or the      */
   /* imaginary part of complex numbers. For example, when set,  */
   /* we output -3 as - 3 and 2 as + 2.                          */
   int flagSeparateSign;

   /* The signed and unsigned width fields record the formatting */
   /* width of integers or the integer part of floating-point    */
   /* numbers. For example -1 has a signed width of two and an   */
   /* unsigned width of one. Likewise, -12.45 has a signed width */
   /* of three and an unsigned width of two. For positive numbers*/
   /* the two width are the same. The reason why we do not simply*/
   /* maintain the unsigned with plus the negative flag is that  */
   /* -1 and 12 would have a total output width of 2, whereas    */
   /* combining the unsigned width with the sign would give a    */
   /* width of 3.                                                */
   int signedWidth;
   int unsignedWidth;
   int integerWidth;

   /* String literals are used for 'inf', 'nan' as well as for   */
   /* Boolean values such as 'True' and 'False'.                 */
   int signedLiteral;
   int unsignedLiteral;
   int literalWidth;

   /* The fraction width is the width of the fractional part of  */
   /* floating-point numbers. We maintain a separate variable    */
   /* for the fraction width in scientific mode, because we may  */
   /* switch to this mode at any point and need to know what the */
   /* value would be for all numbers processed up to that point. */
   int fractionWidth;
   int fractionWidthScientific;

   /* Flag indicating that negative numbers were processed.      */
   int flagNegative;

   /* Flag indicating that the leading sign causes the width to  */
   /* be larger. Only applicable when the separate-sign mode is  */
   /* switch off.                                                */
   int flagLeadingSign;

   /* The guaranteed width of the formatted output, for example: */
   /* Standard formatting: "-1.23" or "1.23" have width 4 and 5  */
   /* Separate sign: "- 1.23" or "+ 1.23" have width 6.          */
   int width;          

} OcFormattingInfo;


typedef struct
{  
   OcFormattingInfo **info;
   int                parts;

   /* The byteswapped field can be set by the user to indicate   */
   /* that data must be byte swapped before analysis or          */
   /* formatting.                                                */
   int byteswapped;

   /* The hex field can be set by the user to indicate that the  */
   /* output of integers must be in hexadecimal format.          */
   int hex;

   /* The user can set the following field gives the maximum     */
   /* number of decimals displayed in floating point numbers     */
   int     maximumDecimals;  /* 5 */
   double  roundDecimals;    /* 0.5e-5 */
   double  scaleDecimals;    /* 1.0e+5 */

   /* We pre-compute the log(10) to evaluate log10 and exp10 to  */
   /* make sure that the compiler does not generate warnings on  */
   /* incompatible types in the function declaration.            */
   double  log10;

   /* The user can set the character used in scientific notation */
   char exponentCharacter;

   /* The user can set the following field to specify the minimum*/
   /* width of exponents when strings are formatted in scientific*/
   /* mode.                                                      */
   int minimumExponentWidth;

   /* The scientific flag is used for floating-point numbers,    */
   /* and is set when the exponent is either very small or very  */
   /* large. Once the flag is set the formatting will adhere to  */
   /* the formatting conversions set for scientific notation.    */
   int flagScientific;

   /* Flag indicating that the leading sign causes the width to  */
   /* be larger. Only applicable when the separate-sign mode is  */
   /* switch off.                                                */
   int flagLeadingSign;

   /* The width of the newline symbol.                           */
   int newlineWidth;

   /* The total width of the formatted output                    */
   int width;

} OcFormat;


/* Function types */
typedef void (* OcFormatAnalyze_funptr)(const char *data, OcFormat *format, int index);
typedef void (* OcFormatOutput_funptr )(const char *data, OcFormat *format, int index, char *str);


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Allocation and deallocation of the format structure.       */
OC_API OcFormat *OcFormatCreate(OcDType dtype);
OC_API void      OcFormatFree(OcFormat *format);

/* The finalize function determines the overall output width. */
/* The function returns -1 if the format buffer size is too   */
/* small and may cause buffer overflows!                      */
OC_API void OcFormatFinalize(OcFormat *format);

/* Function to set the maximum number of decimals.            */
OC_API void OcFormatSetMaxDecimals(OcFormat *format, int maxDecimals);

/* Functions to output long integers */
OC_API int OcFormatLongWidth(long int v);
OC_API int OcFormatLong(char *str, int width, long int v);
OC_API int OcFormatULongWidth(unsigned long int v);
OC_API int OcFormatULong(char *str, int width, unsigned long int v);
OC_API int OcFormatHex(char *str, int width, unsigned long int v);

/* Analysis and output functions */
OC_API OcFormatAnalyze_funptr OcFormatAnalyze_function(OcDType dtype);
OC_API OcFormatOutput_funptr  OcFormatOutput_function(OcDType dtype);

#endif
