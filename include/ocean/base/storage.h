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

#ifndef __OC_STORAGE_H__
#define __OC_STORAGE_H__

#include "ocean/base/api.h"
#include "ocean/base/dtype.h"
#include "ocean/base/types.h"
#include "ocean/base/device.h"

#include <stddef.h>

/* ===================================================================== */
/* Storage flags                                                         */
/* ===================================================================== */

#define OC_STORAGE_OWNER               0x0001  /* Ownership flag         */
#define OC_STORAGE_RAW                 0x0002  /* Raw storage flag       */
#define OC_STORAGE_BYTESWAPPED         0x0004  /* Byte-order information */
#define OC_STORAGE_READONLY            0x0008  /* Read-only status       */


/* ===================================================================== */
/* Auxilliary data types                                                 */
/* ===================================================================== */

typedef uint32_t OcStorageFlags;


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct __OcStorage
{  OcStream      *stream;      /* Stream                                 */
   OcEvent       *event;       /* Event for synchronization              */

   char          *data;        /* Pointer to the data                    */
   OcDType        dtype;       /* Data type of the elements              */
   OcSize         size;        /* Storage size in bytes                  */
   OcSize         capacity;    /* Storage capacity in bytes              */
   OcSize         nelem;       /* Number of elements in the storage      */
   int            elemsize;    /* Element size for the given data type   */
   OcStorageFlags flags;       /* Flags to specialize the storage        */

   long int       refcount;/* Number of references to the storage object */

   /* Device-specific finalize function */
   void     (*finalize)(struct __OcStorage *storage);
} OcStorage;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Functions to manipulate storage reference counts */
OC_API OcStorage *OcIncrefStorage(OcStorage *storage);
OC_API void       OcDecrefStorage(OcStorage *storage);

/* Function for storage structure allocation; the size field can be used */
/* when instantiated subclasses of the standard storage structure.       */
OC_API OcStorage *OcAllocateStorage(OcDevice *device, OcDType dtype, OcStream *stream, OcSize size);

/* Query functions */
OC_API int OcStorage_isAligned      (OcStorage *storage);
OC_API int OcStorage_isByteswapped  (OcStorage *storage);
OC_API int OcStorage_isReadOnly     (OcStorage *storage);
OC_API int OcStorage_isOwner        (OcStorage *storage);
OC_API int OcStorage_isRaw          (OcStorage *storage);
OC_API int OcStorage_isDetached     (OcStorage *storage);

/* Update functions */
OC_API void OcStorage_setByteswapped(OcStorage *storage, int flag);
OC_API void OcStorage_setReadOnly   (OcStorage *storage, int flag);
OC_API void OcStorage_setOwner      (OcStorage *storage, int flag);

/* Generic functions */
OC_API int OcStorage_overlap        (OcStorage *storage1, OcStorage *storage2);

/* Synchronization */
OC_API int OcStorage_synchronize    (OcStorage *storage);
OC_API int OcStorage_update         (OcStorage *storage);
OC_API int OcStorage_startRead      (OcStorage *storage, OcStorage *readStorage);
OC_API int OcStorage_finishRead     (OcStorage *storage, OcStorage *readStorage);
OC_API int OcStorage_startWrite     (OcStorage *storage, OcStorage *writeStorage);
OC_API int OcStorage_finishWrite    (OcStorage *storage, OcStorage *writeStorage);


/* ===================================================================== */
/* Macros                                                                */
/* ===================================================================== */

/* Reference count macros allowing NULL storage; */
/* these may be replaced by functions later.     */
#define OcXIncrefStorage(storage)      OcIncrefStorage(storage)
#define OcXDecrefStorage(storage)      OcDecrefStorage(storage)


#define OcStorage_data(storage)        ((storage) -> data)
#define OcStorage_device(storage)      ((storage) -> stream -> device)
#define OcStorage_deviceType(storage)  ((storage) -> device -> type)
#define OcStorage_stream(storage)      ((storage) -> stream)
#define OcStorage_scheduler(storage)   ((storage) -> stream -> scheduler)

#endif
