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

#include "ocean/base/error.h"

#include <stdio.h>

#define OC_ERROR_BUFFER_SIZE 1024


/* ===================================================================== */
/* Local variables                                                       */
/* ===================================================================== */

static char        oc_error_message[OC_ERROR_BUFFER_SIZE] = {"No error has been set yet"};
static const char *oc_error_filename = NULL;
static long int    oc_error_line     = 0;

OcErrorHandler     oc_error_handler   = NULL;
static void       *oc_error_user_data = NULL;


/* ===================================================================== */
/* Function implementations - Query functions                            */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
const char *OcError_lastError(void)
/* --------------------------------------------------------------------- */
{
   return oc_error_message;
}


/* --------------------------------------------------------------------- */
const char *OcError_lastFile(void)
/* --------------------------------------------------------------------- */
{  return oc_error_filename;
}


/* --------------------------------------------------------------------- */
long int OcError_lastLine(void)
/* --------------------------------------------------------------------- */
{  return oc_error_line;
}


/* ===================================================================== */
/* Function implementations - Error handler functions                    */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
void OcError_defaultHandler(const char *error, void *data)
/* --------------------------------------------------------------------- */
{
   printf("Ocean error: %s\n", error);
}


/* --------------------------------------------------------------------- */
OcErrorHandler OcError_getHandler(void)
/* --------------------------------------------------------------------- */
{
   return oc_error_handler;
}


/* --------------------------------------------------------------------- */
void *OcError_getHandlerData(void)
/* --------------------------------------------------------------------- */
{
   return oc_error_user_data;
}


/* --------------------------------------------------------------------- */
void OcError_setDefaultHandler(void)
/* --------------------------------------------------------------------- */
{
   oc_error_handler   = OcError_defaultHandler;
   oc_error_user_data = NULL;
}


/* --------------------------------------------------------------------- */
void OcError_setHandler(OcErrorHandler handler, void *data)
/* --------------------------------------------------------------------- */
{
   oc_error_handler   = handler;
   oc_error_user_data = data;
}


/* ===================================================================== */
/* Function implementations - Internal functions                         */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
char *OcError_getBuffer(void)
/* --------------------------------------------------------------------- */
{
   return oc_error_message;
}


/* --------------------------------------------------------------------- */
size_t OcError_getBufferSize(void)
/* --------------------------------------------------------------------- */
{
   return OC_ERROR_BUFFER_SIZE;
}


/* --------------------------------------------------------------------- */
void OcError_setSource(const char *filename, long int line)
/* --------------------------------------------------------------------- */
{
   oc_error_filename = filename;
   oc_error_line     = line;
}


/* --------------------------------------------------------------------- */
void OcError_callHandler(void)
/* --------------------------------------------------------------------- */
{
   if (oc_error_handler != 0)
   {  oc_error_handler(oc_error_message, oc_error_user_data);
   }
}
