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

#include "ocean/base/malloc.h"
#include "ocean/base/platform.h"

#include <stdio.h>
#include <stdint.h>


/* ===================================================================== */
/* Internal structure definitions                                        */
/* ===================================================================== */

typedef struct __OcMallocEntry
{  void       *ptr;
   const char *file;
   size_t      line;
   struct __OcMallocEntry *next;
} OcMallocEntry;


/* ===================================================================== */
/* Function implementation                                               */
/* ===================================================================== */

static OcMallocEntry *oc_malloc_entries = NULL;


/* --------------------------------------------------------------------- */
void OcMallocDebugInit(void)
/* --------------------------------------------------------------------- */
{
   oc_malloc_entries = NULL;
}


/* --------------------------------------------------------------------- */
void OcMallocDebugFinalize(void)
/* --------------------------------------------------------------------- */
{  OcMallocEntry *entry;

   if ((entry = oc_malloc_entries) != NULL)
   {  printf("---------------------------------------------------------------------\n");
      printf("Allocated memory                                                     \n");
      printf("---------------------------------------------------------------------\n");

      while(entry)
      {  printf("Line %-5d %s\n", (int)(entry -> line), entry -> file);
         entry = entry -> next;
      }

      printf("---------------------------------------------------------------------\n");
   }
}


/* --------------------------------------------------------------------- */
void *OcMallocDebug(const char *file, size_t line, size_t size)
/* --------------------------------------------------------------------- */
{  OcMallocEntry *entry;
   char          *ptr;

   /* Allocate the memory chunk */
   ptr = (char *)malloc(sizeof(char) * size);
   entry = (OcMallocEntry *)malloc(sizeof(OcMallocEntry));
   if ((ptr == NULL) || (entry == NULL))
   {  if (ptr) free(ptr);
      if (entry) free(entry);
      return NULL;
   }
   
   /* Add the malloc entry */
   entry -> ptr  = ptr;
   entry -> file = file;
   entry -> line = line;
   entry -> next = oc_malloc_entries;
   oc_malloc_entries = entry;

   /* Return the result */
   return (void *)ptr;
}


/* --------------------------------------------------------------------- */
void OcFreeDebug(const char *file, size_t line, void *ptr)
/* --------------------------------------------------------------------- */
{  OcMallocEntry *entry, *prev;

   /* Find the entry */
   entry = oc_malloc_entries; prev = NULL;
   while (entry != NULL)
   {  if (entry -> ptr == ptr) break;
      prev = entry; entry = entry -> next;
   }

   /* Unlink the entry or generate a warning */
   if (entry != NULL)
   {  if (prev == NULL)
           oc_malloc_entries = entry -> next;
      else prev -> next = entry -> next;
      free(entry);
   }
   else
   {  printf("OcFree: Memory free from %s(%d) was not allocated using OcMalloc", file, (int)line);
   }

   /* Free the data */
   free(ptr);
}


/* --------------------------------------------------------------------- */
void *OcMallocAligned(size_t size)
/* --------------------------------------------------------------------- */
{  char   *buffer, *ptr;
   size_t  shift;

   /* Allocate the memory chunk */
   buffer = (char *)OcMalloc(size + sizeof(void *) + (OC_MEMORY_BYTE_ALIGNMENT - 1));
   
   /* Determine the result pointer */
   ptr = buffer + sizeof(void *);
   shift = ((uintptr_t)ptr % OC_MEMORY_BYTE_ALIGNMENT);
   if (shift)
   {  ptr += (OC_MEMORY_BYTE_ALIGNMENT - shift);
   }

   /* Include the original buffer */
   ((void **)ptr)[-1] = buffer;

   return (void *)ptr;
}


/* --------------------------------------------------------------------- */
void OcFreeAligned(void *ptr)
/* --------------------------------------------------------------------- */
{  void *buffer = ((void **)ptr)[-1];

   OcFree(buffer);
}
