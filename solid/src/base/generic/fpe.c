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

#include "solid/base/generic/fpe.h"


/* -------------------------------------------------------------------- */
int solid_fpe_get_status(void)
/* -------------------------------------------------------------------- */
{  int status;
   int result = 0;

   /* Get the exception status */
   status = fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT | FE_INVALID);

   if (status & FE_DIVBYZERO) result |= SD_FPE_DIVIDE_BY_ZERO;
   if (status & FE_OVERFLOW ) result |= SD_FPE_OVERFLOW;
   if (status & FE_UNDERFLOW) result |= SD_FPE_UNDERFLOW;
   if (status & FE_INEXACT  ) result |= SD_FPE_INEXACT;
   if (status & FE_INVALID  ) result |= SD_FPE_INVALID;

   return result;
}


/* -------------------------------------------------------------------- */
int solid_fpe_test_status(int exception)
/* -------------------------------------------------------------------- */
{  int status;

   /* Get the status */
   status = solid_fpe_get_status();

   return ((status & exception) == exception) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
void solid_fpe_clear(void)
/* -------------------------------------------------------------------- */
{
   if (solid_fpe_get_status() != 0)
   {  feclearexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT | FE_INVALID);
   }
}


/* -------------------------------------------------------------------- */
void solid_fpe_raise(int exception)
/* -------------------------------------------------------------------- */
{  int flags = 0;

   if (exception & SD_FPE_DIVIDE_BY_ZERO) flags |= FE_DIVBYZERO;
   if (exception & SD_FPE_OVERFLOW      ) flags |= FE_OVERFLOW;
   if (exception & SD_FPE_UNDERFLOW     ) flags |= FE_UNDERFLOW;
   if (exception & SD_FPE_INEXACT       ) flags |= FE_INEXACT;
   if (exception & SD_FPE_INVALID       ) flags |= FE_INVALID;

   if (flags != 0) feraiseexcept(flags);
}
