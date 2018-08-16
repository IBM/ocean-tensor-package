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

#include "ocean/base/fpe.h"


/* -------------------------------------------------------------------- */
int OcFPE_getStatus(void)
/* -------------------------------------------------------------------- */
{  int status;
   int result = 0;

   /* Get the exception status */
   status = fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT | FE_INVALID);

   if (status & FE_DIVBYZERO) result |= OC_FPE_DIVIDE_BY_ZERO;
   if (status & FE_OVERFLOW ) result |= OC_FPE_OVERFLOW;
   if (status & FE_UNDERFLOW) result |= OC_FPE_UNDERFLOW;
   if (status & FE_INEXACT  ) result |= OC_FPE_INEXACT;
   if (status & FE_INVALID  ) result |= OC_FPE_INVALID;

   return result;
}


/* -------------------------------------------------------------------- */
int OcFPE_testStatus(int exception)
/* -------------------------------------------------------------------- */
{  int status;

   /* Get the status */
   status = OcFPE_getStatus();

   return ((status & exception) == exception) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
void OcFPE_clear(void)
/* -------------------------------------------------------------------- */
{
   if (OcFPE_getStatus() != 0)
   {  feclearexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT | FE_INVALID);
   }
}


/* -------------------------------------------------------------------- */
void OcFPE_raise(int exception)
/* -------------------------------------------------------------------- */
{  int flags = 0;

   if (exception & OC_FPE_DIVIDE_BY_ZERO) flags |= FE_DIVBYZERO;
   if (exception & OC_FPE_OVERFLOW      ) flags |= FE_OVERFLOW;
   if (exception & OC_FPE_UNDERFLOW     ) flags |= FE_UNDERFLOW;
   if (exception & OC_FPE_INEXACT       ) flags |= FE_INEXACT;
   if (exception & OC_FPE_INVALID       ) flags |= FE_INVALID;

   if (flags != 0) feraiseexcept(flags);
}
