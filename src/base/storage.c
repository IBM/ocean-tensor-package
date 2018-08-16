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

#include "ocean/base/storage.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <stdlib.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

static void OcDeallocateStorage(OcStorage *storage);


/* ===================================================================== */
/* Reference count operations                                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcStorage *OcIncrefStorage(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  /* Returns storage to allow: storage = ocIncrefStorage(storage) */

   if (storage != NULL) storage -> refcount ++;
   return storage;
}


/* -------------------------------------------------------------------- */
void OcDecrefStorage(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   if (storage == NULL) return;

   storage -> refcount --;
   if (storage -> refcount == 0)
   {  /* Synchronize the storage */
      if (storage -> stream) OcStorage_synchronize(storage);

      /* Call the finalize function */
      if (storage -> finalize) storage -> finalize(storage);

      /* Deallocate the storage structure */
      OcDeallocateStorage(storage);
   }
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcStorage *OcAllocateStorage(OcDevice *device, OcDType dtype, OcStream *stream, OcSize size)
/* -------------------------------------------------------------------- */
{  OcStorage *storage;

   /* Check the data type */
   if (dtype == OcDTypeNone) OcError(NULL, "Invalid data type in storage allocation");

   /* Allocate memory for the storage structure */
   storage = (OcStorage *)OcMalloc(sizeof(char) * size);
   if (storage == NULL) OcError(NULL, "Insufficient memory to allocate the storage structure");

   /* Initialize basic fields */
   storage -> stream      = NULL;
   storage -> event       = NULL;
   storage -> data        = NULL;
   storage -> dtype       = dtype;
   storage -> size        = 0;
   storage -> capacity    = 0;
   storage -> nelem       = 0;
   storage -> elemsize    = OcDType_size(dtype);
   storage -> flags       = 0;
   storage -> refcount    = 1;
   storage -> finalize    = 0;

   /* Create a new stream object */
   if (stream != NULL)
   {  storage -> stream = OcIncrefStream(stream);
   }
   else
   {  if ((storage -> stream = OcDevice_createStream(device)) == NULL)
      {  OcDeallocateStorage(storage);
         return NULL;
      }
   }

   /* Create a new event */
   if (OcDevice_createEvent(device, &(storage -> event)) != 0)
   {  OcDeallocateStorage(storage);
      return NULL;
   }

   return storage;
}


/* -------------------------------------------------------------------- */
static void OcDeallocateStorage(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  OcEvent *eventDoneRead;
   OcStream *stream;

   if (storage == NULL) return ;

   if ((stream = storage -> stream) != NULL)
   {  /* Synchronize the event */
      eventDoneRead = storage -> event;
      OcEvent_synchronize(stream -> device, eventDoneRead);

      /* Free the event */
      OcDevice_freeEvent(stream -> device, eventDoneRead);

      /* Decrement the stream */
      OcDecrefStream(storage -> stream);
   }

   /* Free the storage object */
   OcFree(storage);
}


/* -------------------------------------------------------------------- */
int OcStorage_synchronize(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  OcDevice *device = storage -> stream -> device;
   OcEvent  *eventDoneWrite = storage -> event;

   return OcEvent_synchronize(device, eventDoneWrite);
}


/* -------------------------------------------------------------------- */
int OcStorage_update(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  OcEvent *eventDoneWrite = storage -> event;

   /* This function should be called on the storage whose stream was*/
   /* used for the operation, after the operation is scheduled and  */
   /* before any of the finish functions. In case of separate read  */
   /* and write events this needs two statements:                   */
   /* OcEvent_record(storage -> eventDoneWrite, storage -> stream); */
   /* OcEvent_record(storage -> eventDoneRead,  storage -> stream); */

   return OcEvent_record(eventDoneWrite, storage -> stream);
}


/* -------------------------------------------------------------------- */
int OcStorage_startRead(OcStorage *storage, OcStorage *readStorage)
/* -------------------------------------------------------------------- */
{  OcEvent *eventDoneWrite = readStorage -> event;

   /* Wait for all writes to the storage to be completed */
   return OcStream_waitFor(storage -> stream, eventDoneWrite);
}


/* -------------------------------------------------------------------- */
int OcStorage_finishRead(OcStorage *storage, OcStorage *readStorage)
/* -------------------------------------------------------------------- */
{  OcEvent *eventDoneRead  = readStorage -> event;
   OcEvent *eventDoneWrite = storage -> event;
   int      result;

   /* Wait for the reference storage to complete its writes  */
   /* and then record an event in the read storage stream to */
   /* indicate that all read operations have completed.      */
   result = OcStream_waitFor(readStorage -> stream, eventDoneWrite);
   if (result != 0)
        return -1;
   else return OcEvent_record(eventDoneRead, readStorage -> stream);
}


/* -------------------------------------------------------------------- */
int OcStorage_startWrite(OcStorage *storage, OcStorage *writeStorage)
/* -------------------------------------------------------------------- */
{  OcEvent *eventDoneRead = writeStorage -> event;

   /* The write storage will be written to in the stream    */
   /* corresponding to the reference storage. The reference */
   /* stream must therefore wait until all read operations  */
   /* have completed (since eventDoneRead always follows    */
   /* eventDoneWrite updates, this also ensures there there */
   /* are no pending or active write updates).              */
   return OcStream_waitFor(storage -> stream, eventDoneRead);
}


/* -------------------------------------------------------------------- */
int OcStorage_finishWrite(OcStorage *storage, OcStorage *writeStorage)
/* -------------------------------------------------------------------- */
{  OcEvent *eventDoneWrite = storage -> event;
   int      result;

   /* Make sure that the updates to the reference storage  */
   /* have completed and then register an update to the    */
   /* write storage.                                       */
   result = OcStream_waitFor(writeStorage -> stream, eventDoneWrite);
   if (result != 0)
        return -1;
   else return OcStorage_update(writeStorage);

}


/* -------------------------------------------------------------------- */
int OcStorage_isAligned(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   if (((uintptr_t)(storage -> data)) % (storage -> elemsize) == 0)
        return 1;
   else return 0;
}


/* -------------------------------------------------------------------- */
int OcStorage_isByteswapped(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   return (storage -> flags & OC_STORAGE_BYTESWAPPED) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
int OcStorage_isReadOnly(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   return (storage -> flags & OC_STORAGE_READONLY) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
int OcStorage_isOwner(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   return (storage -> flags & OC_STORAGE_OWNER) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
int OcStorage_isRaw(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   return (storage -> flags & OC_STORAGE_RAW) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
int OcStorage_isDetached(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   if ((storage -> refcount != 1) || (!OcStorage_isOwner(storage)))
        return 0;
   else return 1;
}


/* -------------------------------------------------------------------- */
void OcStorage_setByteswapped(OcStorage *storage, int flag)
/* -------------------------------------------------------------------- */
{
   if (flag)
        storage -> flags |= OC_STORAGE_BYTESWAPPED;
   else storage -> flags &=~OC_STORAGE_BYTESWAPPED;
}


/* -------------------------------------------------------------------- */
void OcStorage_setReadOnly(OcStorage *storage, int flag)
/* -------------------------------------------------------------------- */
{
   if (flag)
        storage -> flags |= OC_STORAGE_READONLY;
   else storage -> flags &=~OC_STORAGE_READONLY;
}


/* -------------------------------------------------------------------- */
void OcStorage_setOwner(OcStorage *storage, int flag)
/* -------------------------------------------------------------------- */
{
   if (flag)
        storage -> flags |= OC_STORAGE_OWNER;
   else storage -> flags &=~OC_STORAGE_OWNER;
}


/* -------------------------------------------------------------------- */
int OcStorage_overlap(OcStorage *storage1, OcStorage *storage2)
/* -------------------------------------------------------------------- */
{  uintptr_t ptr1, ptr2;
   uintptr_t size;

   if (storage1 -> stream -> device != storage2 -> stream -> device) return 0;

   ptr1  = (uintptr_t)(storage1 -> data);
   ptr2  = (uintptr_t)(storage2 -> data);

   if (ptr1 <= ptr2)
   {  size = (storage1 -> elemsize) * (storage1 -> nelem);
      return (ptr1 + size > ptr2) ? 1 : 0;
   }
   else
   {  size = (storage2 -> elemsize) * (storage2 -> nelem);
      return (ptr2 + size > ptr1) ? 1 : 0;
   }
}
