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

#ifndef __PYOCEAN_STREAM_H__
#define __PYOCEAN_STREAM_H__

#include <Python.h>

#include "ocean.h"

typedef struct {
    PyObject_HEAD
    OcStream *stream;
} pyOcStream;

extern PyTypeObject *PyOceanStream;

#define PYOC_GET_STREAM(object) (((pyOcStream *)(object)) -> stream)


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

PyObject *PyOceanStream_New(OcStream *stream);
PyObject *PyOceanStream_Wrap(OcStream *stream);
int       PyOceanStream_Check(PyObject *obj);


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Internal use only - Routines for module setup */
int  pyOcStream_Initialize(void);
int  pyOcStream_InitializeModule(PyObject *module);
void pyOcStream_Finalize(void);


#endif
