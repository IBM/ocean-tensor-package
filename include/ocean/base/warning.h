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

#ifndef __OC_WARNING_H__
#define __OC_WARNING_H__

#include "ocean/base/api.h"

#include <stddef.h>
#include <stdio.h>


/* ===================================================================== */
/* Macros                                                                */
/* ===================================================================== */

/* OcWarning generates a custom warning message */
#define OcWarning(...) \
   (OcWarning_setSource(__FILE__,__LINE__), \
    snprintf(OcWarning_getBuffer(), OcWarning_getBufferSize(), __VA_ARGS__), \
    OcWarning_callHandler(-1)) \

/* OcWarning_raise generates a typed warning */
#define OcWarning_raise(INDEX) \
   (OcWarning_setSource(__FILE__,__LINE__), \
    OcWarning_callHandler(INDEX)) \

/* ===================================================================== */
/* Type definitions                                                      */
/* ===================================================================== */

typedef int (*OcWarningHandler)(const char *message, void *data);

typedef enum { OC_WARNING_OFF, OC_WARNING_ONCE, OC_WARNING_ON } OcWarningMode;


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* Query functions */
OC_API const char       *OcWarning_lastMessage(void);
OC_API const char       *OcWarning_lastFile(void);
OC_API long int          OcWarning_lastLine(void);

/* Warning handler functions */
OC_API OcWarningHandler  OcWarning_getHandler(void);
OC_API void             *OcWarning_getHandlerData(void);
OC_API void              OcWarning_setDefaultHandler(void);
OC_API void              OcWarning_setHandler(OcWarningHandler handler, void *data);

/* Warning configuration */
OC_API int               OcWarning_enabled(int warningIdx);
OC_API const char       *OcWarning_message(int warningIdx);
OC_API OcWarningMode     OcWarning_getMode(void);
OC_API void              OcWarning_setMode(OcWarningMode mode);

/* Warning registration, and look-up table finalization */
OC_API int               OcWarning_register(int *warningIdx, int flagRaiseOnce, const char *message);
OC_API void              OcWarning_finalize(void);

/* For use within the OcError function only */
OC_API int               OcWarning_callHandler(int warningIdx);
OC_API char             *OcWarning_getBuffer(void);
OC_API size_t            OcWarning_getBufferSize(void);
OC_API void              OcWarning_setSource(const char *filename, long int line);

#endif
