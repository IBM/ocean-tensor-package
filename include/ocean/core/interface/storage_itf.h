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

#ifndef __OC_MODULE_CORE_ITF_STORAGE_H__
#define __OC_MODULE_CORE_ITF_STORAGE_H__

#include "ocean/base/dtype.h"
#include "ocean/base/device.h"
#include "ocean/base/storage.h"
#include "ocean/base/api.h"


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Storage creation */
OC_API OcStorage *OcStorage_create          (OcSize nElements, OcDType dtype, OcDevice *device);
OC_API OcStorage *OcStorage_createTemporary (OcSize nElements, OcDType dtype, OcDevice *device);
OC_API OcStorage *OcStorage_createWithStream(OcSize nElements, OcDType dtype, OcStream *stream);
OC_API OcStorage *OcStorage_createFromObject(OcSize nElements, OcDType dtype, OcDevice *device,
                                             void *object, void *data, int byteswapped,
                                             void (*free)(void *, void *), OcStream *stream);

/* Generic storage operations */
OC_API int        OcStorage_detach          (OcStorage **storage);
OC_API void       OcStorage_setDType        (OcStorage *storage, OcDType dtype);
OC_API void       OcStorage_setDTypeRaw     (OcStorage *storage);
OC_API int        OcStorage_copy            (OcStorage *src, OcStorage *dst);
OC_API OcStorage *OcStorage_clone           (OcStorage *storage);
OC_API OcStorage *OcStorage_cloneTo         (OcStorage *storage, OcDevice *device);
OC_API int        OcStorage_byteswap        (OcStorage *storage);
OC_API int        OcStorage_byteswapNoFlag  (OcStorage *storage);
OC_API int        OcStorage_zero            (OcStorage *storage);
OC_API int        OcStorage_hasHostByteOrder(OcStorage *storage);

/* Cast the data type or device - new or incref */
OC_API OcStorage *OcStorage_castDType       (OcStorage *storage, OcDType dtype);
OC_API OcStorage *OcStorage_castDevice      (OcStorage *storage, OcDevice *device);
OC_API OcStorage *OcStorage_cast            (OcStorage *storage, OcDType dtype, OcDevice *device);

/* Copy raw data between host and storage */
OC_API int        OcStorage_copyToHost      (OcStorage *storage, OcSize offset, OcSize nbytes, void *ptr);
OC_API int        OcStorage_copyFromHost    (OcStorage *storage, OcSize offset, OcSize nbytes, void *ptr);

/* Ensure the data type or device - in-place (decref on failure) */
OC_API int        OcStorage_ensureDType     (OcStorage **storagePtr, OcDType dtype, OcStorage **result);
OC_API int        OcStorage_ensureDevice    (OcStorage **storagePtr, OcDevice *device, OcStorage **result);
OC_API int        OcStorage_ensure          (OcStorage **storagePtr, OcDType dtype, OcDevice *device, OcStorage **result);

/* Storage formatting */
OC_API int        OcStorage_format (OcStorage *storage, char **str, const char *header, const char *footer);
OC_API int        OcStorage_formatFooter(OcStorage *storage, char **str, const char *pre, const char *post);
OC_API int        OcStorage_display(OcStorage *storage);

#endif
