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

#include "pyOcean.h"

#include "ocean.h"
#include "ocean/module_dummy/module_dummy_gpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

#if PY_MAJOR_VERSION < 3
static void   pyOceanDummy_gpu_dealloc(PyObject *obj);
static void (*pyOceanDummy_gpu_original_dealloc)(PyObject *) = NULL;
#endif

static void   pyOceanDummy_gpu_free(void *data);


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

static PyObject *py_oc_module_ocean = NULL;
static PyObject* module = NULL;


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef py_ocean_dummy_gpu_module = {
        PyModuleDef_HEAD_INIT,
        PyMODULE_NAME_STR("pyOceanDummy_gpu"),  /* Module name               (m_base) */
        NULL,                                   /* Module docstring           (m_doc) */
        -1,                                     /* Module state size         (m_size) */
        NULL,                                   /* Module methods         (m_methods) */
        NULL,                                   /* Reload func      (m_reload, < 3.5) */
                                                /* Slot definitions (m_slots, >= 3.5) */
        NULL,                                   /* Traverse func         (m_traverse) */
        NULL,                                   /* Clear func               (m_clear) */
        pyOceanDummy_gpu_free                   /* Free function             (m_free) */
};
#endif

/* -------------------------------------------------------------------- */
PyMODULE_INIT(pyOceanDummy_gpu)              /* Module name must match! */
/* -------------------------------------------------------------------- */
{  int status = 0;

   /* Load the pyOcean module to ensure proper */
   /* Ocean initialization and finalization.   */
   py_oc_module_ocean = PyImport_ImportModule("pyOcean_cpu");
   if (py_oc_module_ocean == NULL)
   {  PyErr_SetString(PyExc_RuntimeError, "Unable to load Ocean core gpu module (pyOcean_cpu)");
      status = -1; goto final;
   }

   /* Initialize the Dummy GPU module */
   if (OcInitModule_Dummy_GPU() != 0) { status = -1; goto final; }

   /* Create the module */
   #if PY_MAJOR_VERSION >= 3
      module = PyModule_Create(&py_ocean_dummy_gpu_module);
      if (module == NULL) { status = -1; goto final; }
   #else
      module = Py_InitModule3(PyMODULE_NAME_STR("pyOceanDummy_gpu"), NULL,
                              "pyOceanDummy is an example module for Ocean");
      if (module == NULL) { status = -1; goto final; }
   #endif

   /* Finalization */
   #if PY_MAJOR_VERSION < 3
   pyOceanDummy_gpu_original_dealloc = Py_TYPE(module) -> tp_dealloc;
   Py_TYPE(module) -> tp_dealloc = pyOceanDummy_gpu_dealloc;
   #endif

final : ;
   /* Check for errors */
   if (status != 0)
   {  Py_XDECREF(py_oc_module_ocean); py_oc_module_ocean = NULL;
      Py_XDECREF(module);
      PyMODULE_INIT_ERROR;
   }

   /* Return the module */
   #if PY_MAJOR_VERSION >= 3
   return module;
   #endif
}


/* ===================================================================== */
/* Finalization                                                          */
/* ===================================================================== */

#if PY_MAJOR_VERSION < 3
/* -------------------------------------------------------------------- */
static void pyOceanDummy_gpu_dealloc(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (pyOceanDummy_gpu_original_dealloc) pyOceanDummy_gpu_original_dealloc(obj);

   if (obj == module) pyOceanDummy_gpu_free((void *)obj);
}
#endif


/* -------------------------------------------------------------------- */
static void pyOceanDummy_gpu_free(void *data)
/* -------------------------------------------------------------------- */
{
   /* Finalize */
  if (py_oc_module_ocean != NULL)
  {  Py_DECREF(py_oc_module_ocean); py_oc_module_ocean = NULL;
  }
}
