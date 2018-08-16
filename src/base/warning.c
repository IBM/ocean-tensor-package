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
#include "ocean/base/warning.h"
#include "ocean/base/error.h"

#include <stdio.h>
#include <string.h>


#define OC_WARNING_BUFFER_SIZE 1024


/* ===================================================================== */
/* Local function declarations                                           */
/* ===================================================================== */

int OcWarning_defaultHandler(const char *message, void *data);


/* ===================================================================== */
/* Local variables                                                       */
/* ===================================================================== */

/* Warning data */
static char              oc_warning_buffer[OC_WARNING_BUFFER_SIZE] = {"No warning has been set yet"};
static const char       *oc_warning_filename  = NULL;
static long int          oc_warning_line      = 0;
static const char       *oc_warning_message   = oc_warning_buffer;

/* Warning configuration */
static OcWarningMode     oc_warning_mode      = OC_WARNING_ONCE;

/* Warning handler */
static OcWarningHandler  oc_warning_handler   = OcWarning_defaultHandler;
static void             *oc_warning_user_data = NULL;

/* Warning look-up table */
typedef struct
{  int    flagRaised;
   int    flagRaiseOnce;
   char  *message;
} OcWarningEntry;

static OcWarningEntry *oc_warning_table          = NULL;
static int             oc_warning_table_size     = 0;
static int             oc_warning_table_capacity = 0;


/* ===================================================================== */
/* Function implementations - Query functions                            */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
const char *OcWarning_lastMessage(void)
/* --------------------------------------------------------------------- */
{
   return oc_warning_message;
}


/* --------------------------------------------------------------------- */
const char *OcWarning_lastFile(void)
/* --------------------------------------------------------------------- */
{  return oc_warning_filename;
}


/* --------------------------------------------------------------------- */
long int OcWarning_lastLine(void)
/* --------------------------------------------------------------------- */
{  return oc_warning_line;
}



/* ===================================================================== */
/* Function implementations - Warning configuration                      */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcWarning_enabled(int warningIdx)
/* --------------------------------------------------------------------- */
{
   /* Make sure the handler is set and the index is valid */
   if (oc_warning_handler == NULL) return 0;
   if (warningIdx > oc_warning_table_size) return 0;

   switch (oc_warning_mode)
   {
      case OC_WARNING_OFF :
         return 0;

      case OC_WARNING_ONCE :
         /* Fall throught when warning index is negative */
         if ((warningIdx >= 0) &&
             (oc_warning_table[warningIdx].flagRaiseOnce) &&
             (oc_warning_table[warningIdx].flagRaised)) return 0;

      case OC_WARNING_ON :
         return 1;
   }

   return 0;
}


/* --------------------------------------------------------------------- */
OC_API const char *OcWarning_message(int warningIdx)
/* --------------------------------------------------------------------- */
{
   /* Make sure the handler is set and the index is valid */
   if (oc_warning_handler == NULL) return NULL;
   if ((warningIdx < 0) || (warningIdx > oc_warning_table_size)) return NULL;

   return oc_warning_table[warningIdx].message;
}


/* --------------------------------------------------------------------- */
OcWarningMode OcWarning_getMode(void)
/* --------------------------------------------------------------------- */
{  return oc_warning_mode;
}


/* --------------------------------------------------------------------- */
void OcWarning_setMode(OcWarningMode mode)
/* --------------------------------------------------------------------- */
{
   oc_warning_mode = mode;
}



/* ===================================================================== */
/* Function implementations - Warning handler functions                  */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcWarning_defaultHandler(const char *message, void *data)
/* --------------------------------------------------------------------- */
{
   printf("Ocean warning: %s\n", message);
   return 0;
}


/* --------------------------------------------------------------------- */
OcWarningHandler OcWarning_getHandler(void)
/* --------------------------------------------------------------------- */
{
   return oc_warning_handler;
}


/* --------------------------------------------------------------------- */
void *OcWarning_getHandlerData(void)
/* --------------------------------------------------------------------- */
{
   return oc_warning_user_data;
}


/* --------------------------------------------------------------------- */
void OcWarning_setDefaultHandler(void)
/* --------------------------------------------------------------------- */
{
   oc_warning_handler   = OcWarning_defaultHandler;
   oc_warning_user_data = NULL;
}


/* --------------------------------------------------------------------- */
void OcWarning_setHandler(OcWarningHandler handler, void *data)
/* --------------------------------------------------------------------- */
{
   oc_warning_handler   = handler;
   oc_warning_user_data = data;
}



/* ===================================================================== */
/* Function implementations - Look-up table                              */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcWarning_register(int *warningIdx, int flagRaiseOnce, const char *message)
/* --------------------------------------------------------------------- */
{  OcWarningEntry *table;
   char           *messageStr;
   int             newSize;
   int             i;

   /* Warning index in case of failure */
   *warningIdx = -1;

   /* Resize the warning table if needed */
   if (oc_warning_table_capacity <= oc_warning_table_size)
   {  newSize = (oc_warning_table_size == 0) ? 32 : 2 * oc_warning_table_size;
      table = (OcWarningEntry *)OcMalloc(sizeof(OcWarningEntry) * newSize);
      if (table == NULL) OcError(-1, "Insufficient memory for warning table");

      /* Initialize the table */
      oc_warning_table_capacity = newSize;
      for (i = 1; i < oc_warning_table_size; i++)
      {  table[i].flagRaised    = oc_warning_table[i].flagRaised;
         table[i].flagRaiseOnce = oc_warning_table[i].flagRaiseOnce;
         table[i].message       = oc_warning_table[i].message;
      }
      for ( ;  i < oc_warning_table_capacity; i++)
      {  table[i].message = NULL;
      }

      /* Set the new table */
      if (oc_warning_table) OcFree(oc_warning_table);
      oc_warning_table = table;
   }

   /* Copy the message */
   messageStr = (char *)OcMalloc(sizeof(char) * (strlen(message) + 1));
   if (messageStr == NULL) OcError(-1, "Insufficient memory for the warning entry");
   strcpy(messageStr, message);

   /* Add the entry */
   oc_warning_table[oc_warning_table_size].flagRaised    = 0;
   oc_warning_table[oc_warning_table_size].flagRaiseOnce = flagRaiseOnce;
   oc_warning_table[oc_warning_table_size].message       = messageStr;

   /* Update the size */
   *warningIdx = oc_warning_table_size;
   oc_warning_table_size ++;

   /* Success */
   return 0;
}


/* --------------------------------------------------------------------- */
void OcWarning_finalize(void)
/* --------------------------------------------------------------------- */
{  int i;

   if (oc_warning_table != NULL)
   {
      /* Delete the table entries */
      for (i = 0; i < oc_warning_table_size; i++)
      {  if (oc_warning_table[i].message)
           OcFree(oc_warning_table[i].message);
      }

      /* Finalize the table */
      OcFree(oc_warning_table);
      oc_warning_table          = NULL;
      oc_warning_table_size     = 0;
      oc_warning_table_capacity = 0;
   }
}



/* ===================================================================== */
/* Function implementations - Internal functions for use by macros       */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcWarning_callHandler(int warningIdx)
/* --------------------------------------------------------------------- */
{  int result;

   if (!OcWarning_enabled(warningIdx)) return 0;

   if (warningIdx < 0)
   {  oc_warning_message = oc_warning_buffer;
   }
   else
   {  oc_warning_message = oc_warning_table[warningIdx].message;
      oc_warning_table[warningIdx].flagRaised = 1;
   }

   /* Call the warning handler */
   result = oc_warning_handler(oc_warning_message, oc_warning_user_data);
   if (result != 0) OcErrorMessage("Warning error: %s", oc_warning_message); 

   return result;
}


/* --------------------------------------------------------------------- */
char *OcWarning_getBuffer(void)
/* --------------------------------------------------------------------- */
{
   return oc_warning_buffer;
}


/* --------------------------------------------------------------------- */
size_t OcWarning_getBufferSize(void)
/* --------------------------------------------------------------------- */
{
   return OC_WARNING_BUFFER_SIZE;
}


/* --------------------------------------------------------------------- */
void OcWarning_setSource(const char *filename, long int line)
/* --------------------------------------------------------------------- */
{
   oc_warning_filename = filename;
   oc_warning_line     = line;
}
