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

#ifndef __PYOCEAN_DEVICE_H__
#define __PYOCEAN_DEVICE_H__

#include <Python.h>

#include "ocean.h"

typedef struct {
    PyObject_HEAD
    OcDevice *device;
} pyOcDevice;

extern PyTypeObject *PyOceanDevice;

#define PYOC_GET_DEVICE(object) (((pyOcDevice *)(object)) -> device)


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

PyObject *PyOceanDevice_New(OcDevice *device);
PyObject *PyOceanDevice_Wrap(OcDevice *device);
int       PyOceanDevice_Check(PyObject *obj);


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Internal use only - Routines for module setup */
int  pyOcDevice_Initialize(void);
int  pyOcDevice_InitializeModule(PyObject *module);
void pyOcDevice_Finalize(void);

/* Internal use only - Registration of device types */
int pyOcDevice_RegisterType(const char *deviceType, PyTypeObject *type);

#endif
