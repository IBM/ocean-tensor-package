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

#include "pyOcean_opaque.h"
#include "pyOcean_compatibility.h"
#include "ocean.h"


int py_oc_opaque_magic = 0;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Standard functions */
static void      pyOcOpaque_dealloc    (pyOcOpaque *self);
static PyObject *pyOcOpaque_richcompare(PyObject *o1, PyObject *o2, int opid);
static PyObject *pyOcOpaque_str        (pyOcOpaque *self);


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

static PyMethodDef py_oc_opaque_methods[] = {
   {NULL}  /* Sentinel */
};

PyTypeObject py_oc_opaque_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.opaque",            /* tp_name      */
   sizeof(pyOcOpaque),        /* tp_basicsize */
};

PyTypeObject *PyOceanOpaque;


/* -------------------------------------------------------------------- */
int pyOcOpaque_Initialize(void)
/* -------------------------------------------------------------------- */
{
    /* Construct the opaque type object */
    PyOceanOpaque = &py_oc_opaque_type;

    PyOceanOpaque -> tp_flags       = Py_TPFLAGS_DEFAULT;
    PyOceanOpaque -> tp_alloc       = PyType_GenericAlloc;
    PyOceanOpaque -> tp_dealloc     = (destructor)pyOcOpaque_dealloc;
    PyOceanOpaque -> tp_str         = (reprfunc)pyOcOpaque_str;
    PyOceanOpaque -> tp_repr        = (reprfunc)pyOcOpaque_str;
    PyOceanOpaque -> tp_richcompare = pyOcOpaque_richcompare;
    PyOceanOpaque -> tp_methods     = py_oc_opaque_methods;
    PyOceanOpaque -> tp_doc         = "Ocean opaque";

    if (PyType_Ready(PyOceanOpaque) < 0) return -1;

    return 0;
}


/* -------------------------------------------------------------------- */
int pyOcOpaque_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{
    Py_INCREF(PyOceanOpaque); /* Static object - do not delete */
    PyModule_AddObject(module, "opaque", (PyObject *)PyOceanOpaque);

    return 0;
}


/* -------------------------------------------------------------------- */
void pyOcOpaque_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Empty */
}


/* -------------------------------------------------------------------- */
int pyOcOpaque_CreateMagic(void)
/* -------------------------------------------------------------------- */
{  int result = py_oc_opaque_magic;

   py_oc_opaque_magic ++;

   return result;
}


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *PyOceanOpaque_New(void *data, int magic)
/* -------------------------------------------------------------------- */
{
   return PyOceanOpaque_NewWithFree(data, magic, 0);
}


/* -------------------------------------------------------------------- */
PyObject *PyOceanOpaque_NewWithFree(void *data, int magic, void (*fptrFree)(void *))
/* -------------------------------------------------------------------- */
{  pyOcOpaque *obj;

   /* Construct the object */
   obj = (pyOcOpaque *)PyOceanOpaque -> tp_alloc(PyOceanOpaque,0);
   if (obj == NULL)
   {  if ((fptrFree) && (data)) fptrFree(data);
      return NULL;
   }

   /* Set the data */
   obj -> data    = data;
   obj -> magic   = magic;
   obj -> free    = fptrFree;

   return (PyObject *)obj;
}

/* -------------------------------------------------------------------- */
int PyOceanOpaque_Check(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   return PyObject_IsInstance(obj, (PyObject *)PyOceanOpaque);
}


/* -------------------------------------------------------------------- */
int PyOceanOpaque_CheckMagic(PyObject *obj, int magic)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   if (!PyObject_IsInstance(obj, (PyObject *)PyOceanOpaque)) return 0;

   return (((pyOcOpaque *)(obj)) -> magic == magic) ? 1 : 0;
}


/* ===================================================================== */
/* Class functions - standard methods                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static void pyOcOpaque_dealloc(pyOcOpaque *self)
/* -------------------------------------------------------------------- */
{
   if ((self -> free) && (self -> data)) self -> free(self -> data);

   Py_TYPE(self)->tp_free((PyObject *)self);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcOpaque_richcompare(PyObject *self, PyObject *obj, int opid)
/* -------------------------------------------------------------------- */
{  pyOcOpaque *opaque1, *opaque2;
   int flagEqual = 0;

   /* Make sure the object is an Ocean device */
   if (PyOceanOpaque_Check(obj))
   {  
      opaque1 = (pyOcOpaque *)self;
      opaque2 = (pyOcOpaque *)obj;

      /* Compare the magic numbers */
      if (opaque1 -> magic != opaque2 -> magic)
      {  flagEqual = 0;
      }
      else
      {  flagEqual = (opaque1 -> data == opaque2 -> data);
      }

   }

   /* Return the result */
   if (opid == Py_EQ)
   {  if (flagEqual)
           Py_RETURN_TRUE;
      else Py_RETURN_FALSE;
   }
   else if (opid == Py_NE)
   {  if (flagEqual)
           Py_RETURN_FALSE;
      else Py_RETURN_TRUE;
   }
   else
   {  OcError(NULL, "The given comparison operation is not implemented");
   }
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcOpaque_str(pyOcOpaque *self)
/* -------------------------------------------------------------------- */
{
   return PyString_FromString("<opaque>");
}
