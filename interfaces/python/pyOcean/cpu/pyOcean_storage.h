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

#ifndef __PYOCEAN_STORAGE_H__
#define __PYOCEAN_STORAGE_H__

#include <Python.h>

#include "ocean.h"

typedef struct {
    PyObject_HEAD
    OcStorage *storage;
} pyOcStorage;

extern PyTypeObject *PyOceanStorage;

#define PYOC_GET_STORAGE(object) (((pyOcStorage *)(object)) -> storage)

/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

PyObject *PyOceanStorage_New(OcStorage *storage);
PyObject *PyOceanStorage_Wrap(OcStorage *storage);
int       PyOceanStorage_Check(PyObject *obj);


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Internal use only - Routines for module setup */
int  pyOcStorage_Initialize(void);
int  pyOcStorage_InitializeModule(PyObject *module);
void pyOcStorage_Finalize(void);

#endif
