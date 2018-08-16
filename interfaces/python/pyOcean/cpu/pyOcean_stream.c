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

#include "pyOcean_stream.h"
#include "pyOcean_device.h"
#include "pyOcean_args.h"

#include <stdio.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Standard functions */
static void      pyOcStream_dealloc     (pyOcStream *self);
static PyObject *pyOcStream_new         (PyTypeObject *subtype, PyObject *args, PyObject *kwargs);
static PyObject *pyOcStream_str         (pyOcStream *self);
static PyObject *pyOcStream_richcompare (PyObject *o1, PyObject *o2, int opid);

/* Get and set functions */
static PyObject *pyOcStream_getdevice   (pyOcStream *self, void *closure);
static PyObject *pyOcStream_getrefcount (pyOcStream *self, void *closure);

/* Generic function */
static PyObject *pyOcStream_sync        (pyOcStream *self, PyObject *args);

/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

struct PyGetSetDef py_oc_stream_getseters[] = {
   {"device",    (getter)pyOcStream_getdevice,    NULL, "associated device",    NULL},
   {"refcount", (getter)pyOcStream_getrefcount,   NULL, "reference count",      NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef py_oc_stream_methods[] = {
   {"sync", (PyCFunction)pyOcStream_sync, METH_NOARGS, "Synchronize the stream"},
   {NULL}  /* Sentinel */
};

PyTypeObject py_oc_stream_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.stream",             /* tp_name      */
   sizeof(pyOcStream),         /* tp_basicsize */
};

PyTypeObject *PyOceanStream;


/* -------------------------------------------------------------------- */
int pyOcStream_Initialize(void)
/* -------------------------------------------------------------------- */
{
    /* Construct the stream type object */
    PyOceanStream = &py_oc_stream_type;

    PyOceanStream -> tp_flags       = Py_TPFLAGS_DEFAULT;
    PyOceanStream -> tp_alloc       = PyType_GenericAlloc;
    PyOceanStream -> tp_dealloc     = (destructor)pyOcStream_dealloc;
    PyOceanStream -> tp_new         = (newfunc)pyOcStream_new;
    PyOceanStream -> tp_str         = (reprfunc)pyOcStream_str;
    PyOceanStream -> tp_repr        = (reprfunc)pyOcStream_str; /* [sic] */
    PyOceanStream -> tp_richcompare = pyOcStream_richcompare;
    PyOceanStream -> tp_getset      = py_oc_stream_getseters;
    PyOceanStream -> tp_methods     = py_oc_stream_methods;
    PyOceanStream -> tp_doc         = "Ocean stream";

    if (PyType_Ready(PyOceanStream) < 0) return -1;

    return 0;
}


/* -------------------------------------------------------------------- */
int pyOcStream_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{
    Py_INCREF(PyOceanStream); /* Static object - do not delete */
    PyModule_AddObject(module, "stream", (PyObject *)PyOceanStream);

    return 0;
}


/* -------------------------------------------------------------------- */
void pyOcStream_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Empty */
}



/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *PyOceanStream_New(OcStream *stream)
/* -------------------------------------------------------------------- */
{  pyOcStream *obj;

   if (stream == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcStream *)PyOceanStream -> tp_alloc(PyOceanStream,0);
   if (obj == NULL) return NULL;

   /* Set the stream  */
   obj -> stream = OcIncrefStream(stream);

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
PyObject *PyOceanStream_Wrap(OcStream *stream)
/* -------------------------------------------------------------------- */
{  pyOcStream *obj;

   if (stream == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcStream *)PyOceanStream -> tp_alloc(PyOceanStream,0);
   if (obj == NULL)
   {  /* Decrement the reference count */
      OcDecrefStream(stream);
      return NULL;
   }

   /* Set the stream (do not increment the reference count) */
   obj -> stream = stream;

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
int PyOceanStream_Check(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   return PyObject_IsInstance(obj, (PyObject *)PyOceanStream);
}



/* ===================================================================== */
/* Internal function definitions                                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static void pyOcStream_dealloc(pyOcStream *self)
/* -------------------------------------------------------------------- */
{
   if (self -> stream) OcDecrefStream(self -> stream);

   Py_TYPE(self)->tp_free((PyObject *)self);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStream_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcDevice   *device;

   /* ========================= */
   /* Syntax: stream([device])  */
   /* ========================= */

   /* Make sure there are no key-word arguments */
   if (kwargs != NULL) OcError(NULL, "The scalar constructor does not take keyword arguments");

   /* Parse the parameters */
   PyOceanArgs_Init(&param, args, "ocean.stream");
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Apply the default device */
   if ((device = OcDevice_applyDefault(device)) == NULL) return NULL;

   /* Create the new stream */
   return PyOceanStream_Wrap(OcDevice_createStream(device));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStream_str(pyOcStream *self)
/* -------------------------------------------------------------------- */
{  OcDevice *device;
   char buffer[128];

   device = self -> stream -> device;
   snprintf(buffer, 128, "<stream %p on %s>", (void *)(self -> stream), device -> name);

   return PyString_FromString(buffer);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStream_richcompare(PyObject *self, PyObject *obj, int opid)
/* -------------------------------------------------------------------- */
{  OcStream *stream1, *stream2;

   /* Make sure the object is an Ocean stream */
   if (!PyOceanStream_Check(obj))
   {  if (opid == Py_EQ) Py_RETURN_FALSE;
      if (opid == Py_NE) Py_RETURN_TRUE;
   }
   else
   {  /* Get the streams */
      stream1 = ((pyOcStream *)self) -> stream;
      stream2 = ((pyOcStream *)obj) -> stream;

      /* Evaluate supported comparison operations */
      if (opid == Py_EQ) return PyBool_FromLong(stream1 == stream2);
      if (opid == Py_NE) return PyBool_FromLong(stream1 != stream2);
   }

   OcError(NULL, "The given comparison operation is not implemented");
}


/* ===================================================================== */
/* Class functions - get and set functions                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcStream_getdevice(pyOcStream *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanDevice_New(self -> stream -> device);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStream_getrefcount(pyOcStream *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> stream -> refcount);
}


/* ===================================================================== */
/* Class functions - generic functions                                   */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcStream_sync(pyOcStream *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   OcStream_synchronize(self -> stream);

   Py_RETURN_NONE;
}
