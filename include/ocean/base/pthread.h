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

#ifndef __OC_PTHREAD_H__
#define __OC_PTHREAD_H__

#include "ocean/base/api.h"

#include <pthread.h>


/* ===================================================================== */
/* Defines                                                               */
/* ===================================================================== */

#define OC_PTHREAD_CREATE_JOINABLE   PTHREAD_CREATE_JOINABLE

#define OcPthread_create(THREAD, ATTR, ENTRY, ARG) \
          pthread_create(THREAD, ATTR, ENTRY, ARG)
#define OcPthread_join(THREAD, STATUS) \
          pthread_join(THREAD, STATUS)
#define OcPthread_exit(PTR) \
          pthread_exit(PTR)

#define OcPthreadMutex_lock(MUTEX) \
          pthread_mutex_lock((pthread_mutex_t *)(MUTEX))
#define OcPthreadMutex_unlock(MUTEX) \
          pthread_mutex_unlock((pthread_mutex_t *)(MUTEX))

#define OcPthreadCond_wait(COND,MUTEX)\
          pthread_cond_wait((pthread_cond_t *)(COND), (pthread_mutex_t *)(MUTEX))
#define OcPthreadCond_signal(COND) \
          pthread_cond_signal((pthread_cond_t *)(COND))
#define OcPthreadCond_broadcast(COND) \
          pthread_cond_broadcast((pthread_cond_t *)(COND))

#define OcPthreadAttr_init(ATTR) \
          pthread_attr_init((pthread_attr_t *)(ATTR))
#define OcPthreadAttr_destroy(ATTR) \
          pthread_attr_destroy((pthread_attr_t *)(ATTR))
#define OcPthreadAttr_setStacksize(ATTR, SIZE) \
          pthread_attr_setstacksize((pthread_attr_t *)(ATTR), (size_t)(SIZE))
#define OcPthreadAttr_setDetachState(ATTR, DETACHSTATE) \
          pthread_attr_setdetachstate((pthread_attr_t *)(ATTR), DETACHSTATE)


/* ===================================================================== */
/* Type definitions                                                      */
/* ===================================================================== */
typedef pthread_t             OcPthread;
typedef pthread_attr_t        OcPthreadAttr;
typedef pthread_mutex_t       OcPthreadMutex;
typedef pthread_mutexattr_t   OcPthreadMutexAttr;
typedef pthread_cond_t        OcPthreadCond;
typedef pthread_condattr_t    OcPthreadCondAttr;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

OC_API int  OcPthread_createMutex(OcPthreadMutex **mutex, const OcPthreadMutexAttr *attr);
OC_API int  OcPthread_createCondition(OcPthreadCond **condition, const OcPthreadCondAttr *attr);
OC_API void OcPthread_destroyMutex(OcPthreadMutex *mutex);
OC_API void OcPthread_destroyCondition(OcPthreadCond *condition);

#endif
