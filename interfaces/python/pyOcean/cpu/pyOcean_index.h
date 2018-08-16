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

#ifndef __PYOCEAN_INDEX_H__
#define __PYOCEAN_INDEX_H__

#include <Python.h>
#include "ocean.h"


/* ===================================================================== */
/* Index type                                                            */
/* ===================================================================== */

typedef struct {
    PyObject_HEAD
    OcTensorIndex *index;
} pyOcTensorIndex;

typedef struct {
    PyObject_HEAD
} pyOcIndexCreate;

#define PYOC_GET_TENSOR_INDEX(object) (((pyOcTensorIndex *)(object)) -> index)

extern PyTypeObject *PyOceanTensorIndex;
extern PyTypeObject *PyOceanIndexCreate;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

PyObject *PyOceanTensorIndex_New(OcTensorIndex *index);
PyObject *PyOceanTensorIndex_Wrap(OcTensorIndex *index);
int       PyOceanTensorIndex_Check(PyObject *obj);


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Internal use only - Routines for module setup */
int  pyOcIndex_Initialize(void);
int  pyOcIndex_InitializeModule(PyObject *module);
void pyOcIndex_Finalize(void);

#endif
