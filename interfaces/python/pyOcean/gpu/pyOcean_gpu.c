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
#include "pyOcean/gpu/pyOcean_gpu.h"
#include "pyOcean/gpu/pyOcean_device_gpu.h"

#include "ocean.h"
#include "ocean_gpu.h"

#include <stdio.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

#if PY_MAJOR_VERSION < 3
static void   pyOcean_gpu_dealloc(PyObject *obj);
static void (*pyOcean_gpu_original_dealloc)(PyObject *) = NULL;
#endif

static void   pyOcean_gpu_free(void *data);
static void   pyOcean_gpu_finalize(void);


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

static PyObject *module = NULL;
   
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef py_ocean_gpu_module = {
        PyModuleDef_HEAD_INIT,
        PyMODULE_NAME_STR("pyOcean_gpu"),  /* Module name               (m_base) */
        NULL,                              /* Module docstring           (m_doc) */
        -1,                                /* Module state size         (m_size) */
        NULL,                              /* Module methods         (m_methods) */
        NULL,                              /* Reload func      (m_reload, < 3.5) */
                                           /* Slot definitions (m_slots, >= 3.5) */
        NULL,                              /* Traverse func         (m_traverse) */
        NULL,                              /* Clear func               (m_clear) */
        pyOcean_gpu_free                   /* Free function             (m_free) */
};
#endif

/* -------------------------------------------------------------------- */
PyMODULE_INIT(pyOcean_gpu)
/* -------------------------------------------------------------------- */
{  PyObject *py_oc_module_ocean = NULL;
   int status = 0;

   /* Load the pyOcean module */
   py_oc_module_ocean = PyImport_ImportModule("pyOcean_cpu");
   if (py_oc_module_ocean == NULL)
   {  PyErr_SetString(PyExc_RuntimeError, "Unable to load Ocean core module (pyOcean_cpu)");
      status = -1; goto final;
   }

   /* Initialize the Ocean GPU package */
   if (OcInitGPU() != 0) { status = -1; goto final; }

   /* Register the finalization function */
   pyOcean_RegisterFinalizer(pyOcean_gpu_finalize);

   /* Initialize the different components */
   if (pyOcDeviceGPU_Initialize() != 0) { status = -1; goto final; }

   /* Create the module */
   #if PY_MAJOR_VERSION >= 3
      module = PyModule_Create(&py_ocean_gpu_module);
      if (module == NULL) { status = -1; goto final; }
   #else
      module = Py_InitModule3(PyMODULE_NAME_STR("pyOcean_gpu"), NULL,
                              "pyOcean_gpu implements the core module on GPU");
      if (module == NULL) { status = -1; goto final; }
   #endif
   
   /* Finalization */
   #if PY_MAJOR_VERSION < 3
   pyOcean_gpu_original_dealloc = Py_TYPE(module) -> tp_dealloc;
   Py_TYPE(module) -> tp_dealloc = pyOcean_gpu_dealloc;
   #endif

   /* Complete module initialization */
   if (pyOcDeviceGPU_InitializeModule(module) != 0) { status = -1; goto final; }

final : ;
   /* Free the local reference to pyOcean */
   Py_XDECREF(py_oc_module_ocean); py_oc_module_ocean = NULL;

   /* Check for errors */
   if (status != 0)
   {  OcErrorMessage("Error initializing module pyOcean_gpu");
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
static void pyOcean_gpu_dealloc(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (pyOcean_gpu_original_dealloc) pyOcean_gpu_original_dealloc(obj);

   if (obj == module) pyOcean_gpu_free((void *)obj);
}
#endif


/* -------------------------------------------------------------------- */
static void pyOcean_gpu_free(void *data)
/* -------------------------------------------------------------------- */
{
   /* ------------------------------------------------------ */
   /* Note: the original code kept a global variable for the */
   /*       pyOcean module, but this is not needed as Python */
   /*       maintains an internal reference to modules that  */
   /*       have been loaded.                                */
   /* ------------------------------------------------------ */
   /* Decref the Ocean module */
   /*
   if (py_oc_module_ocean != NULL)
   {  Py_DECREF(py_oc_module_ocean); py_oc_module_ocean = NULL;
   }
   */
}

/* -------------------------------------------------------------------- */
static void pyOcean_gpu_finalize(void)
/* -------------------------------------------------------------------- */
{  
   /* Finalize the different components */
   pyOcDeviceGPU_Finalize();
}
