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

#include "ocean/base/pthread.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */


/* -------------------------------------------------------------------- */
int OcPthread_createMutex(OcPthreadMutex **mutex, const OcPthreadMutexAttr *attr)
/* -------------------------------------------------------------------- */
{
   /* Allocate the mutex */
   if ((*mutex = (OcPthreadMutex *)OcMalloc(sizeof(OcPthreadMutex))) == NULL)
      OcError(-1, "Insufficient memory to allocate mutex");

   /* Initialize the mutex */
   if (pthread_mutex_init((pthread_mutex_t *)(*mutex), attr) != 0)
   {  OcFree(*mutex); *mutex = NULL;
      OcError(-1, "Error initializing mutex");
   }

   return 0;
}

/* -------------------------------------------------------------------- */
int OcPthread_createCondition(OcPthreadCond **condition, const OcPthreadCondAttr *attr)
/* -------------------------------------------------------------------- */
{
   /* Allocate the condition */
   if ((*condition = (OcPthreadCond *)OcMalloc(sizeof(OcPthreadCond))) == NULL)
      OcError(-1, "Insufficient memory to allocate condition");

   /* Initialize the condition */
   if (pthread_cond_init((pthread_cond_t *)(*condition), attr) != 0)
   {  OcFree(*condition); *condition = NULL;
      OcError(-1, "Error initializing condition");
   }

   return 0;
}


/* -------------------------------------------------------------------- */
void OcPthread_destroyMutex(OcPthreadMutex *mutex)
/* -------------------------------------------------------------------- */
{
   if (mutex != NULL)
   {  pthread_mutex_destroy((pthread_mutex_t *)mutex);
      OcFree(mutex);
   }
}


/* -------------------------------------------------------------------- */
void OcPthread_destroyCondition(OcPthreadCond *condition)
/* -------------------------------------------------------------------- */
{
   if (condition != NULL)
   {  pthread_cond_destroy((pthread_cond_t *)condition);
      OcFree(condition);
   }
}
