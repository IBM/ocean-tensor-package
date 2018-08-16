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

#ifndef __PYOCEAN_TENSOR_H__
#define __PYOCEAN_TENSOR_H__

#include <Python.h>
#include "ocean.h"


/* ===================================================================== */
/* Tensor type                                                           */
/* ===================================================================== */

typedef struct {
    PyObject_HEAD
    OcTensor *tensor;
} pyOcTensor;

#define PYOC_GET_TENSOR(object) (((pyOcTensor *)(object)) -> tensor)

extern PyTypeObject *PyOceanTensor;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

PyObject *PyOceanTensor_New(OcTensor *tensor);
PyObject *PyOceanTensor_Wrap(OcTensor *tensor);
int       PyOceanTensor_Check(PyObject *obj);


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Internal use only - Routines for module setup */
int  pyOcTensor_Initialize(void);
int  pyOcTensor_InitializeModule(PyObject *module);
void pyOcTensor_Finalize(void);

#endif
