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

#include "ocean/core/cpu/storage_cpu.h"
#include "ocean/base/platform.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"
#include "solid/base/cpu/solid_omp.h"

#include <stdlib.h>
#include <string.h>


/* ===================================================================== */
/* Internal type definitions                                             */
/* ===================================================================== */

typedef struct
{  OcSize  nbytes;
   int     njobs;
   char   *srcData;
   char   *dstData;
} OcBufferCPU_threadData;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Storage create and free */
static OcStorage *OcStorageCPU_create(OcSize nelem, OcDType dtype, OcDevice *device, OcStream *stream);
static void       OcStorageCPU_finalize(OcStorage *storage);
static OcStorage *OcStorageCPU_fromObject(OcSize nelem, OcDType dtype, OcDevice *device,
                                          void *object, void *data, int byteswapped,
                                          void (*free)(void *object, void *data), OcStream *stream);


/* ===================================================================== */
/* Function registration                                                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcRegisterStorageCPU(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set the function pointers */
   module -> Storage_create         = OcStorageCPU_create;
   module -> Storage_fromObject     = OcStorageCPU_fromObject;
   
   module -> Buffer_copy            = OcBufferCPU_copy;
   module -> Buffer_copyFromCPU     = OcBufferCPU_copy;
   module -> Buffer_copyToCPU       = OcBufferCPU_copy;
   module -> Buffer_copyHostStorage = OcBufferCPU_copyHostToStorage;
   module -> Buffer_copyStorageHost = OcBufferCPU_copyStorageToHost;

   module -> Buffer_zero        = OcBufferCPU_zero;
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcStorage *OcStorageCPU_create(OcSize nelem, OcDType dtype, OcDevice *device, OcStream *stream)
/* -------------------------------------------------------------------- */
{  OcStorage *storage;
   OcSize     size;

   /* Allocate storage and initialize basic fields */
   if (stream == NULL) stream = device -> defaultStream;
   storage = OcAllocateStorage(device, dtype, stream, sizeof(OcStorageCPU));
   if (storage == NULL) return NULL;

   /* Initialize fields */
   storage -> flags   |= OC_STORAGE_OWNER;
   storage -> finalize = OcStorageCPU_finalize;
   storage -> data     = NULL;

   /* Initialize CPU specific fields */
   ((OcStorageCPU *)storage) -> referenceFree = 0;
   ((OcStorageCPU *)storage) -> reference     = NULL;

   /* Allocate storage memory */
   size = nelem * (storage -> elemsize);
   storage -> size     = size;
   storage -> capacity = size;
   storage -> nelem    = nelem;

   if (nelem > 0)
   {  
      /* Allocate memory */
      storage -> data = (char *)OcMalloc(size);
      if (storage -> data == NULL)
      {  OcDecrefStorage(storage);
         OcError(NULL, "Error allocating %s storage of size %" OC_FORMAT_LU " on %s",
                 OcDType_name(dtype), (unsigned long)nelem, device -> name);
      }
   }

   return storage;
}


/* -------------------------------------------------------------------- */
void OcStorageCPU_finalize(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   /* Free storage memory */
   if (storage -> data)
   {  if (OcStorage_isOwner(storage))
      {  OcFree(storage -> data);
      }
      else if (((OcStorageCPU *)storage) -> referenceFree)
      {  ((OcStorageCPU *)storage) -> referenceFree(((OcStorageCPU *)storage) -> reference,
                                                    storage -> data);
      }
   }
}


/* -------------------------------------------------------------------- */
OcStorage *OcStorageCPU_fromObject(OcSize nelem, OcDType dtype, OcDevice *device,
                                   void *object, void *data, int byteswapped,
                                   void (*free)(void *object, void *data), OcStream *stream)
/* -------------------------------------------------------------------- */
{  OcStorage *storage;

   /* Allocate storage and initialize basic fields */
   if (stream == NULL) stream = device -> defaultStream;
   storage = OcAllocateStorage(device, dtype, stream, sizeof(OcStorageCPU));
   if (storage == NULL) return NULL;

   /* Initialize fields */
   storage -> data        = data;
   storage -> size        = nelem * (storage -> elemsize);
   storage -> capacity    = nelem * (storage -> elemsize);
   storage -> nelem       = nelem;
   storage -> finalize    = OcStorageCPU_finalize;

   /* Update the flags */
   if (byteswapped)
   {  storage -> flags |= OC_STORAGE_BYTESWAPPED;
   }
   
   /* Initialize CPU specific fields */
   ((OcStorageCPU *)storage) -> referenceFree = free;
   ((OcStorageCPU *)storage) -> reference     = object;
  
   return storage;
}


/* -------------------------------------------------------------------- */
void OcBufferCPU_intrnlCopy(int jobID, void *data)
/* -------------------------------------------------------------------- */
{  OcSize offset, n;
   int    njobs;

   /* Compute the offset */
   n      = ((OcBufferCPU_threadData *)data) -> nbytes;
   njobs  = ((OcBufferCPU_threadData *)data) -> njobs;
   offset = n % njobs;
   n     /= njobs;

   if (jobID < offset)
   {  n ++;
      offset = jobID * n;
   }
   else
   {  offset+= jobID * n;
   }
         
   /* Copy the data */
   memcpy((void *)((((OcBufferCPU_threadData *)data) -> dstData) + offset),
          (void *)((((OcBufferCPU_threadData *)data) -> srcData) + offset), n);
}


/* -------------------------------------------------------------------- */
int OcBufferCPU_copy(OcStorage *srcStorage, void *srcData,
                     OcStorage *dstStorage, void *dstData,
                     OcDType dtype, OcSize n)
/* -------------------------------------------------------------------- */
{  OcBufferCPU_threadData data;
   OcSize  nbytes;
   int     nthreads;
   
   /* Determine the number of bytes to copy */
   nbytes = OcDType_size(dtype) * n;

   /* Determine how many threads to use */
   nthreads = solid_omp_get_max_threads();
   if (nthreads > nbytes / 1024) nthreads = nbytes / 1024;

   /* Copy the data */
   if (nthreads > 1)
   {  data.nbytes  = nbytes;
      data.njobs   = nthreads;
      data.srcData = (char *)srcData;
      data.dstData = (char *)dstData;

      solid_omp_run_parallel(OcBufferCPU_intrnlCopy, nthreads, nthreads, (void *)(&data));
   }
   else
   {  /* Copy the data */
      memcpy(dstData, srcData, nbytes);
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcBufferCPU_copyHostToStorage(OcStorage *storage, OcSize offset,
                                  OcSize nbytes, void *ptr)
/* -------------------------------------------------------------------- */
{  OcBufferCPU_threadData data;
   void *srcData, *dstData;
   int   nthreads;

   /* Get the data pointers */
   srcData = ptr;
   dstData = (void *)((char *)(storage -> data) + offset);

   /* Determine how many threads to use */
   nthreads = solid_omp_get_max_threads();
   if (nthreads > nbytes / 1024) nthreads = nbytes / 1024;

   /* Copy the data */
   if (nthreads > 1)
   {  data.nbytes  = nbytes;
      data.njobs   = nthreads;
      data.srcData = (char *)srcData;
      data.dstData = (char *)dstData;

      solid_omp_run_parallel(OcBufferCPU_intrnlCopy, nthreads, nthreads, (void *)(&data));
   }
   else
   {  /* Copy the data */
      memcpy(dstData, srcData, nbytes);
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcBufferCPU_copyStorageToHost(OcStorage *storage, OcSize offset,
                                  OcSize nbytes, void *ptr)
/* -------------------------------------------------------------------- */
{  OcBufferCPU_threadData data;
   void *srcData, *dstData;
   int   nthreads;

   /* Get the data pointers */
   srcData = (void *)((char *)(storage -> data) + offset);
   dstData = ptr;

   /* Determine how many threads to use */
   nthreads = solid_omp_get_max_threads();
   if (nthreads > nbytes / 1024) nthreads = nbytes / 1024;

   /* Copy the data */
   if (nthreads > 1)
   {  data.nbytes  = nbytes;
      data.njobs   = nthreads;
      data.srcData = (char *)srcData;
      data.dstData = (char *)dstData;

      solid_omp_run_parallel(OcBufferCPU_intrnlCopy, nthreads, nthreads, (void *)(&data));
   }
   else
   {  /* Copy the data */
      memcpy(dstData, srcData, nbytes);
   }

   return 0;
}


/* -------------------------------------------------------------------- */
void OcBufferCPU_intrnlZero(int jobID, void *data)
/* -------------------------------------------------------------------- */
{  OcSize offset, n;
   int    njobs;

   /* Compute the offset */
   n      = ((OcBufferCPU_threadData *)data) -> nbytes;
   njobs  = ((OcBufferCPU_threadData *)data) -> njobs;
   offset = n % njobs;
   n     /= njobs;

   if (jobID < offset)
   {  n ++;
      offset = jobID * n;
   }
   else
   {  offset+= jobID * n;
   }
         
   /* Zero the data */
   memset((void *)((((OcBufferCPU_threadData *)data) -> dstData) + offset), 0, n);
}


/* -------------------------------------------------------------------- */
int OcBufferCPU_zero(OcStorage *storage, void *data, OcDType dtype, OcSize n)
/* -------------------------------------------------------------------- */
{  OcBufferCPU_threadData threadData;
   OcSize nbytes;
   int    nthreads;

   /* Determine the number of bytes to zero */
   nbytes = OcDType_size(dtype) * n;

   /* Determine how many threads to use */
   nthreads = solid_omp_get_max_threads();
   if (nthreads > nbytes / 1024) nthreads = nbytes / 1024;

   /* Copy the data */
   if (nthreads > 1)
   {  threadData.nbytes  = nbytes;
      threadData.njobs   = nthreads;
      threadData.dstData = (char *)data;

      solid_omp_run_parallel(OcBufferCPU_intrnlZero, nthreads, nthreads, (void *)(&threadData));
   }
   else
   {  /* Set the memory to zero */
      memset(data, 0, nbytes);
   }

   return 0;
}
