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

#include "ocean/core/cpu/device_cpu.h"
#include "ocean/core/interface/module_core.h"
#include "ocean/core/interface/storage_itf.h"
#include "ocean/core/interface/tensor_itf.h"

#include "ocean/base/format.h"
#include "ocean/base/scalar.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>



/* ===================================================================== */
/* Function implementation - Storage creation                            */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
OcStorage *OcStorage_create(OcSize nElements, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcStorage *(*funptr)(OcSize, OcDType, OcDevice *, OcStream *);

   /* Basic checks */
   if (nElements < 0) OcError(NULL, "Storage size cannot be negative");

   /* Apply default data type and device */
   if ((dtype == OcDTypeNone) && ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone)) return NULL;
   if ((device == NULL) && ((device = OcDevice_applyDefault(device)) == NULL)) return NULL;

   /* Look up the StorageCreate function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Storage_create)) == 0)
   {  OcError(NULL, "Storage creation is not supported on device %s", device -> type -> name);
   }

   /* Call the function */
   return funptr(nElements, dtype, device, device -> defaultStream);
}


/* --------------------------------------------------------------------- */
OcStorage *OcStorage_createTemporary(OcSize nElements, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcModuleCore_Context *ctx;
   OcStorage *buffer;
   OcSize requested;
   int elemsize = OcDType_size(dtype);
   int idx;

   /* Basic checks */
   if (nElements < 0) OcError(NULL, "Storage size cannot be negative");

   /* Determine the requested buffer size */
   requested = nElements * elemsize;

   /* Get the core module device context */
   ctx = (OcModuleCore_Context *)(OC_GET_DEVICE_CONTEXT(device, oc_module_core));
   if ((ctx == NULL) || (ctx -> bufferList == NULL) || (ctx -> bufferCount == 0) ||
       ((requested > ctx -> bufferMaxSize) && (ctx -> bufferMaxSize > 0)))
   {  return OcStorage_create(nElements, dtype, device);
   }

   /* Get and update the current buffer index */
   idx = ctx -> bufferIndex;
   ctx -> bufferIndex = (idx + 1) % (ctx -> bufferCount);

   /* Check the buffer status */
   buffer = ctx -> bufferList[idx];
   if ((buffer != NULL) && (buffer -> refcount == 1))
   {  if (buffer -> capacity >= requested)
      {  /* Reset the buffer fields */
         buffer -> dtype    = dtype;
         buffer -> size     = nElements * elemsize;
         buffer -> nelem    = nElements;
         buffer -> elemsize = elemsize;
         buffer -> flags   &= ~(OC_STORAGE_BYTESWAPPED | OC_STORAGE_READONLY | OC_STORAGE_RAW);
         return OcIncrefStorage(buffer);
      }
      else
      {  OcDecrefStorage(buffer);
         ctx -> bufferList[idx] = NULL;
      }
   }

   /* Create a new buffer */
   buffer = OcStorage_create(nElements, dtype, device);
   if (buffer == NULL) return NULL;

   /* Check if we should update the buffer list */
   if (ctx -> bufferList[idx] == NULL)
   {  ctx -> bufferList[idx] = OcIncrefStorage(buffer);
   }
   else /* Buffer is already in use (refcount > 1) */
   {  if (ctx -> bufferList[idx] -> capacity < buffer -> capacity)
      {  OcDecrefStorage(ctx -> bufferList[idx]);
         ctx -> bufferList[idx] = OcIncrefStorage(buffer);
      }
   }

   /* Return the buffer */
   return buffer;
}


/* --------------------------------------------------------------------- */
OcStorage *OcStorage_createWithStream(OcSize nElements, OcDType dtype, OcStream *stream)
/* --------------------------------------------------------------------- */
{  OcStorage *(*funptr)(OcSize, OcDType, OcDevice *, OcStream *);
   OcDevice  *device;

   /* Basic checks */
   if (nElements < 0) OcError(NULL, "Storage size cannot be negative");

   /* Make sure the stream is valid */
   if (stream == NULL)
      OcError(NULL, "Stream cannot be empty");
   if ((device = stream -> device) == NULL)
      OcError(NULL, "Stream does not have a valid device");

   /* Apply default data type */
   if ((dtype == OcDTypeNone) && ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone)) return NULL;

   /* Look up the StorageCreate function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Storage_create)) == 0)
   {  OcError(NULL, "Storage creation is not supported on device %s", device -> type -> name);
   }

   /* Call the function */
   return funptr(nElements, dtype, device, stream);
}


/* --------------------------------------------------------------------- */
OcStorage *OcStorage_createFromObject(OcSize nElements, OcDType dtype, OcDevice *device, 
                                      void *object, void *data, int byteswapped,
                                      void (*free)(void *, void *), OcStream *stream)
/* --------------------------------------------------------------------- */
{  OcStorage *(*funptr)(OcSize, OcDType, OcDevice *, void *, void *,
                        int, void (*free)(void *, void*), OcStream *);

   /* Basic checks */
   if (nElements < 0) OcError(NULL, "Storage size cannot be negative");

   /* Apply default data type */
   if ((dtype == OcDTypeNone) && ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone)) return NULL;
   
   /* Make sure the stream is valid */
   if (stream != NULL)
   {  if ((device != NULL) && (stream -> device != device))
      {  OcError(NULL, "Device does not match the stream device");
      }
      else
      {  device = stream -> device;
      }
   }
   else if (device == NULL)
   {  /* Apply default device */
      if ((device = OcDevice_applyDefault(device)) == NULL) return NULL;
   }

   /* Look up the StorageFromObject function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Storage_fromObject)) == 0)
   {  OcError(NULL, "Storage creation from object is not supported on device %s", device -> type -> name);
   }

   /* Call the function */
   return funptr(nElements, dtype, device, object, data, byteswapped, free, stream);
}


/* ===================================================================== */
/* Function implementations - Generic operations                         */
/* ===================================================================== */


/* -------------------------------------------------------------------- */
int OcStorage_detach(OcStorage **storage)
/* -------------------------------------------------------------------- */
{  OcStorage *s;

   /* Replace the storage if the storage is shared */
   if ((*storage != NULL) && (((*storage) -> refcount > 1) || (!OcStorage_isOwner(*storage))))
   {  s = OcStorage_clone(*storage);
      if (s == NULL) return -1;

      /* Replace the storage */
      OcDecrefStorage(*storage);
      *storage = s;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
OcTensor *OcStorage_asTensor(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  return OcTensor_createFromStorage(storage, -1, NULL, NULL, 0, storage -> dtype);
}


/* -------------------------------------------------------------------- */
void OcStorage_setDType(OcStorage *storage, OcDType dtype)
/* -------------------------------------------------------------------- */
{
   storage -> dtype    = dtype;
   storage -> elemsize = OcDType_size(dtype);
   storage -> nelem    = storage -> size / storage -> elemsize;
   storage -> flags   &=~OC_STORAGE_RAW;
}


/* -------------------------------------------------------------------- */
void OcStorage_setDTypeRaw(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   storage -> dtype    = OcDTypeUInt8;
   storage -> elemsize = OcDType_size(OcDTypeUInt8);
   storage -> nelem    = storage -> size / storage -> elemsize;
   storage -> flags   |= OC_STORAGE_RAW;
}


/* -------------------------------------------------------------------- */
int OcStorage_copy(OcStorage *src, OcStorage *dst)
/* -------------------------------------------------------------------- */
{  OcTensor  *tensorSrc, *tensorDst;
   int        result = -1;

   /* Check read-only flag */
   if (OcStorage_isReadOnly(dst))
      OcError(-1, "Cannot copy to read-only storage");
   
   /* Ensure compatibility of the source and destination storage */
   if (src == NULL) OcError(-1, "Invalid source storage pointer");
   if (dst == NULL) OcError(-1, "Invalid destination storage pointer");
   if (src -> nelem != dst -> nelem ) OcError(-1, "Mismatch in number of storage elements");


   /* ------------------------------------------------- */
   /* Generic copy through tensors                      */
   /* ------------------------------------------------- */
   
   /* Cast the storage objects as tensors and copy them */
   tensorSrc = OcStorage_asTensor(src);
   tensorDst = OcStorage_asTensor(dst);

   /* Copy the data */
   if ((tensorSrc != NULL) && (tensorDst != NULL))
   {  result = OcTensor_copy(tensorSrc, tensorDst);
   }
   else
   {  result = -1;
   }
   
   /* Free the temporary tensors */
   if (tensorSrc) OcDecrefTensor(tensorSrc);
   if (tensorDst) OcDecrefTensor(tensorDst);

   return result;
}


/* -------------------------------------------------------------------- */
OcStorage *OcStorage_clone(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  return OcStorage_cloneTo(storage, storage -> stream -> device);  
}


/* -------------------------------------------------------------------- */
OcStorage *OcStorage_cloneTo(OcStorage *storage, OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcStorage *result;

   /* Create the result storage */
   result = OcStorage_create(storage -> nelem, storage -> dtype, device);
   if (result == NULL) return NULL;

   /* Set the raw flag if needed */
   if (OcStorage_isRaw(storage)) OcStorage_setDTypeRaw(result);

   /* Set the byte-swap flag, if needed */
   if ((OcStorage_isByteswapped(storage)) &&
       (storage -> stream -> device -> endianness == device -> endianness) &&
       (OC_GET_CORE_FUNCTION(device, Tensor_byteswapNoFlag) != 0))
   {  result -> flags |= OC_STORAGE_BYTESWAPPED;
   }

   /* Copy the data */
   if (OcStorage_copy(storage, result) != 0)
   {  OcDecrefStorage(result);
      return NULL;
   }

   return result;
}


/* -------------------------------------------------------------------- */
int OcStorage_byteswap(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  int result;
   
   /* Check read-only flag */
   if (OcStorage_isReadOnly(storage))
      OcError(-1, "Cannot byte-swap read-only storage");

   /* Return if byteswap has no effect */
   if (storage -> elemsize == 1) return 0;

   /* Byte swap the data */
   result = OcStorage_byteswapNoFlag(storage);

   /* Flip the byte-swapped flag */
   if (result == 0)
   {  storage -> flags ^= OC_STORAGE_BYTESWAPPED;
   }

   return result;
}


/* -------------------------------------------------------------------- */
int OcStorage_byteswapNoFlag(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor;
   OcDevice *device;
   int result;

   /* Check read-only flag */
   if (OcStorage_isReadOnly(storage))
      OcError(-1, "Cannot byte-swap read-only storage");

   /* Return if byteswap has no effect */
   if (storage -> elemsize == 1) return 0;

   /* Make sure tensor byteswap is supported */
   device = storage -> stream -> device;
   if (OC_GET_CORE_FUNCTION(device, Tensor_byteswapNoFlag) == 0)
   {  OcError(-1, "Storage byteswap is not supported on device %s",  device -> type -> name);
   }
   
   /* Wrap storage as a tensor and byteswap the data */
   if ((tensor = OcStorage_asTensor(storage)) == NULL) return -1;
   result = OcTensor_byteswap(tensor);
   OcDecrefTensor(tensor);

   return result;
}


/* -------------------------------------------------------------------- */
int OcStorage_zero(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  int (*funptr)(OcStorage *, void *, OcDType, OcSize);
   OcDevice *device;

   /* Check read-only flag */
   if (OcStorage_isReadOnly(storage))
      OcError(-1, "Cannot zero read-only storage");

   /* Look up the BufferZero function */
   device = storage -> stream -> device;
   if ((funptr = OC_GET_CORE_FUNCTION(device, Buffer_zero)) == 0)
   {  OcError(-1, "Storage zero is not supported on device %s", device -> type -> name);
   }

   /* Call the function */
   return funptr(storage, storage -> data, storage -> dtype, storage -> nelem);
}


/* -------------------------------------------------------------------- */
int OcStorage_hasHostByteOrder(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   if (storage -> elemsize == 1) return 1;

   if (storage -> stream -> device -> endianness == OcCPU -> endianness)
        return (OcStorage_isByteswapped(storage) ? 0 : 1);
   else return (OcStorage_isByteswapped(storage) ? 1 : 0);
}


/* -------------------------------------------------------------------- */
OcStorage *OcStorage_castDType(OcStorage *storage, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcStorage *result;

   OcStorage_ensure(&storage, dtype, storage -> stream -> device, &result);
   if (result != NULL)
   {  if (OcStorage_detach(&result) != 0)
      {  OcDecrefStorage(result);
         result = NULL;
      }
   }
   return result;
}


/* -------------------------------------------------------------------- */
OcStorage *OcStorage_castDevice(OcStorage *storage, OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcStorage *result;

   OcStorage_ensureDevice(&storage, device, &result);
   if (result != NULL)
   {  if (OcStorage_detach(&result) != 0)
      {  OcDecrefStorage(result);
         result = NULL;
      }
   }
   return result;
}


/* -------------------------------------------------------------------- */
OcStorage *OcStorage_cast(OcStorage *storage, OcDType dtype, OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcStorage *result;

   OcStorage_ensure(&storage, dtype, device, &result);
   if (result != NULL)
   {  if (OcStorage_detach(&result) != 0)
      {  OcDecrefStorage(result);
         result = NULL;
      }
   }
   return result;
}

/* -------------------------------------------------------------------- */
int OcStorage_copyToHost(OcStorage *storage, OcSize offset,
                         OcSize nbytes, void *ptr)
/* -------------------------------------------------------------------- */
{  int (*funptr)(OcStorage *, OcIndex, OcSize, void *);
   OcDevice *device = OcStorage_device(storage);
 
   /* Range check */
   if ((offset < 0) || ((offset + nbytes) > (storage -> size)))
      OcError(-1, "Requested index range exceeds storage bounds");

   /* Look up the copy function function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Buffer_copyStorageHost)) == 0)
   {  OcError(-1, "Copying of storage data to host memory is not supported on device %s", device -> type -> name);
   }

   return funptr(storage, offset, nbytes, ptr);
}


/* -------------------------------------------------------------------- */
int OcStorage_copyFromHost(OcStorage *storage, OcSize offset,
                           OcSize nbytes, void *ptr)
/* -------------------------------------------------------------------- */
{  int (*funptr)(OcStorage *, OcIndex, OcSize, void *);
   OcDevice *device = OcStorage_device(storage);
 
   /* Range check */
   if ((offset < 0) || ((offset + nbytes) > (storage -> size)))
      OcError(-1, "Requested index range exceeds storage bounds");

   /* Look up the copy function function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Buffer_copyHostStorage)) == 0)
   {  OcError(-1, "Copying of host memory to storage data is not supported on device %s", device -> type -> name);
   }

   return funptr(storage, offset, nbytes, ptr);
}


/* -------------------------------------------------------------------- */
int OcStorage_ensureDType(OcStorage **storagePtr, OcDType dtype, OcStorage **result)
/* -------------------------------------------------------------------- */
{
   return OcStorage_ensure(storagePtr, dtype, (*storagePtr) -> stream -> device, result);
}


/* -------------------------------------------------------------------- */
int OcStorage_ensureDevice(OcStorage **storagePtr, OcDevice *device, OcStorage **result)
/* -------------------------------------------------------------------- */
{  OcDType dtype = (OcStorage_isRaw(*storagePtr)) ? OcDTypeNone : (*storagePtr) -> dtype;

   return OcStorage_ensure(storagePtr, dtype, device, result);
}


/* -------------------------------------------------------------------- */
int OcStorage_ensure(OcStorage **storagePtr, OcDType dtype, OcDevice *device, OcStorage **result)
/* -------------------------------------------------------------------- */
{  OcStorage *storage = *storagePtr;
   int flagRaw;
   int status = -1;

   /* Raw tensors */
   flagRaw = (dtype == OcDTypeNone) ? 1 : 0;
   if (flagRaw) dtype = OcDTypeUInt8;
   if (device == NULL) device = storage -> stream -> device;

   /* Return when the data type and device matches */
   if ((((!OcStorage_isRaw(storage)) && (storage -> dtype == dtype)) ||
        (( OcStorage_isRaw(storage)) && (flagRaw))) &&
       (storage -> stream -> device == device))
   {   /* Set result if needed */
       if (result != NULL) *result = OcIncrefStorage(storage);
       return 0;
   }
       
   /* Create a new storage */
   storage = OcStorage_create((*storagePtr) -> nelem, dtype, device);
   if (storage == NULL) goto final;

   /* Copy the storage */
   if (OcStorage_copy(*storagePtr, storage) != 0)
   {  OcDecrefStorage(storage); storage = NULL;
      goto final;
   }

   /* Set raw mode if needed */
   if (flagRaw) OcStorage_setDTypeRaw(storage);

   /* Success */
   status = 0;
   
final :
   if (result != NULL)
   {  *result = storage;
   }
   else
   {  OcDecrefStorage(*storagePtr);
      *storagePtr = storage;
   }
   return status;
}


/* ===================================================================== */
/* Function implementations - Formatting routines                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcStorage_format(OcStorage *storage, char **str, const char *header, const char *footer)
/* -------------------------------------------------------------------- */
{  OcFormatAnalyze_funptr funptrAnalyze;
   OcFormatOutput_funptr  funptrOutput;
   OcFormat  *format = NULL;
   char      *s = NULL, *data, *buffer = NULL;
   size_t     i, j, n, slen;
   int        k, mode;
   int        result = -1;

   /* Formatting related variables */
   int        colIndex;
   int        rowIndent = 3;
   int        rowWidth;           /* Maximum width */
   int        rowsPerBlock = 10;  /* Maximum numbers of rows per block */
   int        itemSpacing = 3;
   int        itemsPerRow;
   int        itemsPerBlock;

   /* Make sure the storage is on CPU */
   if (OcStorage_ensureDevice(&storage, OcCPU, &storage) != 0) goto final;

   /* Synchronize the storage */
   OcStorage_synchronize(storage);

   /* Create and initialize a new format structure */
   if ((format = OcFormatCreate(storage -> dtype)) == NULL)
   {  OcErrorMessage("Could not allocate the formatting information");
      goto final;
   }

   /* Determine the byteswap flag */
   if (OcStorage_isByteswapped(storage))
      format -> byteswapped = 1;

   /* Determine the hexadecimal flag */
   if (OcStorage_isRaw(storage))
      format -> hex = 1;

   /* Determine the function handle to use */
   funptrAnalyze = OcFormatAnalyze_function(storage -> dtype);
   funptrOutput  = OcFormatOutput_function(storage -> dtype);

   if ((funptrAnalyze == 0) || (funptrOutput == 0))
   {  OcErrorMessage("Could not find formatting analysis and output functions");
      goto final;
   }

   /* Determine upper bounds on the number of items per block */
   rowWidth = oc_format_linewidth;
   itemsPerRow = (int)((rowWidth - rowIndent - 1) / (1 + itemSpacing)) + 1;
   if (itemsPerRow <= 0) itemsPerRow = 1;
   itemsPerBlock = itemsPerRow * rowsPerBlock; 

   /* Get a pointer to the data */
   data = storage -> data;

   /* Determine the format to use */
   n = (storage -> nelem > 2*itemsPerBlock) ? itemsPerBlock : storage -> nelem;
   for (i = 0; i < n; i++)
   {  funptrAnalyze(data + i * (storage -> elemsize), format, 0);
   }
   n = (storage -> nelem > 2*itemsPerBlock) ? storage -> nelem - itemsPerBlock : n;
   for (i = n; i < storage -> nelem; i++)
   {  funptrAnalyze(data + i * (storage -> elemsize), format, 0);
   }

   /* Finalize formatting */
   OcFormatFinalize(format);

   /* Adjust spacing based on the number of parts */
   if (format -> parts > 1) itemSpacing ++;

   /* Reduce the item spacing by one if only signs   */
   /* appear as the first character of each element. */
   if (format -> flagLeadingSign) itemSpacing --;

   /* Determine the actual number of items per block */
   itemsPerRow = (int)((rowWidth - rowIndent - (format -> width)) / (format -> width + itemSpacing)) + 1;
   if (itemsPerRow <= 0) itemsPerRow = 1;
   itemsPerBlock = itemsPerRow * rowsPerBlock; 


   /* --------------------------- */
   /* Format the tensor contents  */
   /* --------------------------- */
   for (mode = 0; mode < 2; mode ++)
   {  slen = 0;

      /* Output the header */
      if (header != NULL)
      {  k = strlen(header); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", header);
      }

      /* Output the body */
      n = storage -> nelem; colIndex = 0;
      for (i = 0; i < n; i++)
      {
         /* Indentation or separation */
         k = (colIndex == 0) ? rowIndent : itemSpacing; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%*s", k, "");        

         /* Format the element */
         k = format -> width; slen += k;
         if (mode == 1)
         {  funptrOutput(data + i * (storage -> elemsize), format, 0, s);
            s += format -> width;
         }

         /* Add a new line */
         if (++colIndex == itemsPerRow)
         {  colIndex = 0;
            k = format -> newlineWidth; slen += k;
            if (mode == 1) s += snprintf(s, k+1, "\n");
          }

         if ((n > 2*itemsPerBlock) && (i == itemsPerBlock-1))
         {
            for (j = 0; j < itemsPerRow; j++)
            {
               k = (j == 0) ? rowIndent : itemSpacing; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%*s", k, "");

               k = (format -> width + 1) / 2; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%*s", k, ":");

               k = (format -> width) / 2; slen += k;
               if ((mode == 1) && (k > 0)) s += snprintf(s, k+1, "%*s", k, "");
            }

            k = format -> newlineWidth; slen += k;
            if (mode == 1) s += snprintf(s, k+1, "\n");

            i = n - itemsPerBlock - 1;
         }
      }
      if (colIndex != 0)
      {  k = format -> newlineWidth; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "\n");
      }

      /* Output the footer */
      if (footer != NULL)
      {  k = strlen(footer); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", footer);
      }

      /* Allocate memory for the string */
      if (mode == 0)
      {
         /* ------------------------------------------------------------- */
         /* Allocate the memory for the string. We use a regular malloc   */
         /* here instead of OcMalloc to ensure that the library can be    */
         /* recompiled with new memory allocation routines without having */
         /* to recompile any language bindings.                           */
         /* ------------------------------------------------------------- */
         buffer = (char *)malloc(sizeof(char) * (slen + 1));
         s = buffer;
         if (buffer == NULL)
         {  OcErrorMessage("Insufficient memory for output string");
            goto final;
         }
      }

   } /* Mode */

   /* Success */
   *str = buffer;
   result = 0;

final :
   /* -------------------------------------------------------- */
   /* Clean up. Note that freeing of the buffer has to be done */
   /* using the regular free function to match its allocation  */
   /* above using malloc.                                      */
   /* -------------------------------------------------------- */
   if (format != NULL) OcFormatFree(format);
   if ((result != 0) && (buffer != NULL)) { free(buffer); }
   OcDecrefStorage(storage);

   return result;
}


/* -------------------------------------------------------------------- */
int OcStorage_formatFooter(OcStorage *storage, char **str, const char *pre, const char *post)
/* -------------------------------------------------------------------- */
{  char   *s = NULL, *buffer = NULL;
   size_t  slen;
   int     k, mode;

   for (mode = 0; mode < 2; mode ++)
   {  slen = 0;

      /* Pre-string */
      if (pre != NULL)
      {  k = strlen(pre); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", pre);
      }

      /* Empty */
      if (storage -> nelem == 0)
      {  k = 6; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "empty ");
      }

      /* Storage */
      if (OcStorage_isRaw(storage))
      {  k = 11; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "raw storage");
      }
      else
      {  k = 8 + strlen(OcDType_name(storage -> dtype)); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "storage.%s", OcDType_name(storage -> dtype));
      }

      /* Size */
      if (storage -> nelem > 0)
      {  k = 9; slen += k;
         if (mode == 1) s+= snprintf(s, k+1, " of size ");

         k = OcFormatULongWidth((unsigned long)(storage -> nelem)); slen += k;
         if (mode == 1) s += OcFormatULong(s, k, (unsigned long int)(storage -> nelem));
      }

      /* Device */
      k = 4 + strlen(storage -> stream -> device -> name); slen += k;
      if (mode == 1) s += snprintf(s, k+1, " on %s", storage -> stream -> device -> name);

      /* Special properties */
      if (OcStorage_isByteswapped(storage) ||
          OcStorage_isReadOnly(storage))
      {  int flag = 0;

         /* Opening bracket */
         k = 2; slen += k;
         if (mode == 1) s += snprintf(s, k+1, " (");
      
         /* Byteswapped */
         if (OcStorage_isByteswapped(storage))
         {  k = 11; slen += k; flag = 1;
            if (mode == 1) s += snprintf(s, k+1, "byteswapped");
         }

         /* Read-only */
         if (OcStorage_isReadOnly(storage))
         {  if (flag)
            {  k = 2; slen += k;
               if (mode == 1) s += snprintf(s, k+1, ", ");
            }
            k = 9; slen += k; flag = 1;
            if (mode == 1) s += snprintf(s, k+1, "read-only");
         }

         /* Closing bracket */
         k = 1; slen += k;
         if (mode == 1) s += snprintf(s, k+1, ")");
      }

      /* Post-string */
      if (post != NULL)
      {  k = strlen(post); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", post);
      }

      /* Allocate memory for the string */
      if (mode == 0)
      {
         /* ------------------------------------------------------------- */
         /* Allocate the memory for the string. We use a regular malloc   */
         /* here instead of OcMalloc to ensure that the library can be    */
         /* recompiled with new memory allocation routines without having */
         /* to recompile any language bindings.                           */
         /* ------------------------------------------------------------- */
         buffer = (char *)malloc(sizeof(char) * (slen + 1));
         s = buffer; *str = buffer;
         if (buffer == NULL) OcError(-1, "Insufficient memory for output string");
      }
   }

   /* Ensure that the string is terminated properly */
   *s = '\0';

   return 0;
}


/* --------------------------------------------------------------------- */
int OcStorage_display(OcStorage *storage)
/* --------------------------------------------------------------------- */
{  char *str = NULL, *footer = NULL;
   int   result;

   /* Format the footer */
   if (OcStorage_formatFooter(storage, &footer, "<", ">\n") != 0) return -1;

   /* Format and display the storage */
   result = OcStorage_format(storage, &str, NULL, footer);
   if (result == 0)
   {  printf("%s", str);
   }

   /* Deallocate memory */
   if (str) free(str);
   if (footer) free(footer);

   return result;
}
