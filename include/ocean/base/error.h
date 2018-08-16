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

#ifndef __OC_ERROR_H__
#define __OC_ERROR_H__

#include "ocean/base/api.h"

#include <stddef.h>
#include <stdio.h>


/* ===================================================================== */
/* Macros                                                                */
/* ===================================================================== */

/* OcErrorMessage sets the error message and location */
#define OcErrorMessage(...) \
   do {  OcError_setSource(__FILE__,__LINE__); \
      snprintf(OcError_getBuffer(), OcError_getBufferSize(), __VA_ARGS__); \
      OcError_callHandler(); \
   } while(0)

/* OcError updates the error message and returns the error code */
#define OcError(CODE, ...) \
   do \
   {  OcErrorMessage(__VA_ARGS__); \
      return (CODE); \
   } while(0)

/* OcFunctionErrorMessage sets the default error message when*/
/* a requested function was not found.                       */
#define OcFunctionErrorMessage(FUNCTION,DTYPE,DEVICE) \
   OcErrorMessage("%s is not supported for type %s on %s", FUNCTION, OcDType_name(DTYPE), (DEVICE)->type->name)

/* OcFunctionError updates the error message and returns the */
/* error code when a requested function was not found.       */
#define OcFunctionError(CODE,FUNCTION,DTYPE,DEVICE) \
   do \
   {  OcFunctionErrorMessage(FUNCTION,DTYPE,DEVICE); \
      return CODE; \
   } while(0)


/* ===================================================================== */
/* Type definitions                                                      */
/* ===================================================================== */

typedef void (*OcErrorHandler)(const char *message, void *data);


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* Query functions */
OC_API const char    *OcError_lastError(void);
OC_API const char    *OcError_lastFile(void);
OC_API long int       OcError_lastLine(void);

/* Error handler functions */
OC_API OcErrorHandler OcError_getHandler(void);
OC_API void          *OcError_getHandlerData(void);
OC_API void           OcError_setDefaultHandler(void);
OC_API void           OcError_setHandler(OcErrorHandler handler, void *data);
OC_API void           OcError_callHandler(void);

/* For use within the OcError function only */
OC_API char          *OcError_getBuffer(void);
OC_API size_t         OcError_getBufferSize(void);
OC_API void           OcError_setSource(const char *filename, long int line);

#endif
