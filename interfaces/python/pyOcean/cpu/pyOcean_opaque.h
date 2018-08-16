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

#ifndef __PYOCEAN_OPAQUE_H__
#define __PYOCEAN_OPAQUE_H__

#include <Python.h>


/* ===================================================================== */
/* Opaque type                                                           */
/* ===================================================================== */

typedef struct {
    PyObject_HEAD
    void  *data;
    int    magic;
    void (*free)(void *);
} pyOcOpaque;

extern PyTypeObject *PyOceanOpaque;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

PyObject *PyOceanOpaque_New(void *data, int magic);
PyObject *PyOceanOpaque_NewWithFree(void *data, int magic, void (*fptrFree)(void *));
int       PyOceanOpaque_Check(PyObject *obj);
int       PyOceanOpaque_CheckMagic(PyObject *obj, int magic);

                                       
/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Create a unique magic number (limited to int range) */
int  pyOcOpaque_CreateMagic(void);

/* Internal use only - Routines for module setup */
int  pyOcOpaque_Initialize(void);
int  pyOcOpaque_InitializeModule(PyObject *module);
void pyOcOpaque_Finalize(void);


#endif
