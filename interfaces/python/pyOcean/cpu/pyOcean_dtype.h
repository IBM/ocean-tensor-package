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

#ifndef __PYOCEAN_DTYPE_H__
#define __PYOCEAN_DTYPE_H__

#include <Python.h>

#include "ocean.h"

typedef struct {
    PyObject_HEAD
    OcDType dtype;
} pyOcDType;

extern PyTypeObject *PyOceanDType;

#define PYOC_GET_DTYPE(object) (((pyOcDType *)(object)) -> dtype)

/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

PyObject *PyOceanDType_New(OcDType dtype);
int       PyOceanDType_Check(PyObject *obj);


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Internal use only - Routines for module setup */
int  pyOcDType_Initialize(void);
int  pyOcDType_InitializeModule(PyObject *module);
void pyOcDType_Finalize(void);


#endif
