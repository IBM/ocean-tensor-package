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

#include <stdio.h>


/* ===================================================================== */
/* Type definitions                                                      */
/* ===================================================================== */

typedef struct __pyOcean_finalizer
{  void (*func)(void);
   struct __pyOcean_finalizer *next;
} pyOcean_finalizer;

pyOcean_finalizer *py_ocean_finalizers = NULL;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

#if PY_MAJOR_VERSION < 3
static void   pyOcean_cpu_dealloc(PyObject *obj);
static void (*pyOcean_cpu_original_dealloc)(PyObject *) = NULL;
#endif

static void   pyOcean_cpu_free(void *data);
static void   pyOcean_cpu_finalize(void);


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

static PyObject *module = NULL;

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef py_ocean_cpu_module = {
        PyModuleDef_HEAD_INIT,
        PyMODULE_NAME_STR("pyOcean_cpu"),  /* Module name               (m_base) */
        NULL,                              /* Module docstring           (m_doc) */
        -1,                                /* Module state size         (m_size) */
        py_ocean_core_methods,             /* Module methods         (m_methods) */
        NULL,                              /* Reload func      (m_reload, < 3.5) */
                                           /* Slot definitions (m_slots, >= 3.5) */
        NULL,                              /* Traverse func         (m_traverse) */
        NULL,                              /* Clear func               (m_clear) */
        pyOcean_cpu_free                   /* Free function             (m_free) */
};
#endif


/* -------------------------------------------------------------------- */
static void pyOcean_ErrorHandler(const char *error, void *data)
/* -------------------------------------------------------------------- */
{
   PyErr_SetString(PyExc_RuntimeError, error);
}


/* -------------------------------------------------------------------- */
static int pyOcean_WarningHandler(const char *message, void *data)
/* -------------------------------------------------------------------- */
{
   PyErr_WarnEx(PyExc_RuntimeWarning, message, 0);
   return 0;
}


/* -------------------------------------------------------------------- */
PyMODULE_INIT(pyOcean_cpu)
/* -------------------------------------------------------------------- */
{  int       status = 0;

   /* Initialize the Ocean package */
   if (OcInit() != 0)
   {  pyOcean_ErrorHandler(OcError_lastError(), NULL);
      PyMODULE_INIT_ERROR;
   }

   /* Set the error and warning handlers */
   OcError_setHandler(pyOcean_ErrorHandler, NULL);
   OcWarning_setHandler(pyOcean_WarningHandler, NULL);

   /* Register the finalization function */
   pyOcean_RegisterFinalizer(pyOcean_cpu_finalize);

   /* Initialize the different components */
   if (pyOcDType_Initialize()     != 0) PyMODULE_INIT_ERROR;
   if (pyOcDevice_Initialize()    != 0) PyMODULE_INIT_ERROR;
   if (pyOcDeviceCPU_Initialize() != 0) PyMODULE_INIT_ERROR;
   if (pyOcStream_Initialize()    != 0) PyMODULE_INIT_ERROR;
   if (pyOcScalar_Initialize()    != 0) PyMODULE_INIT_ERROR;
   if (pyOcStorage_Initialize()   != 0) PyMODULE_INIT_ERROR;
   if (pyOcTensor_Initialize()    != 0) PyMODULE_INIT_ERROR;
   if (pyOcIndex_Initialize()     != 0) PyMODULE_INIT_ERROR;
   if (pyOcOpaque_Initialize()    != 0) PyMODULE_INIT_ERROR;

   /* Create the module */
   #if PY_MAJOR_VERSION >= 3
      module = PyModule_Create(&py_ocean_cpu_module);
      if (module == NULL) PyMODULE_INIT_ERROR;
   #else
      module = Py_InitModule3(PyMODULE_NAME_STR("pyOcean_cpu"), py_ocean_core_methods,
                              "pyOcean_cpu is the core module for Ocean");
      if (module == NULL) PyMODULE_INIT_ERROR;
   #endif

   /* Finalization */
   #if PY_MAJOR_VERSION < 3
   pyOcean_cpu_original_dealloc = Py_TYPE(module) -> tp_dealloc;
   Py_TYPE(module) -> tp_dealloc = pyOcean_cpu_dealloc;
   #endif

   /* Complete module initialization */
   if (pyOcDType_InitializeModule(module)      != 0) status = -1;
   if (pyOcDevice_InitializeModule(module)     != 0) status = -1;
   if (pyOcDeviceCPU_InitializeModule(module)  != 0) status = -1;
   if (pyOcStream_InitializeModule(module)     != 0) status = -1;
   if (pyOcScalar_InitializeModule(module)     != 0) status = -1;
   if (pyOcStorage_InitializeModule(module)    != 0) status = -1;
   if (pyOcTensor_InitializeModule(module)     != 0) status = -1;
   if (pyOcIndex_InitializeModule(module)      != 0) status = -1;
   if (pyOcOpaque_InitializeModule(module)     != 0) status = -1;

   /* Set the default device and data type */
   OcDevice_setDefault(OcCPU);
   OcDType_setDefault(OcDTypeFloat);

   /* Check for errors */
   if (status != 0)
   {  OcErrorMessage("Error initializing module pyOcean_cpu");
      Py_DECREF(module); PyMODULE_INIT_ERROR;
   }

   /* Return the module */
   #if PY_MAJOR_VERSION >= 3
   return module;
   #endif
}


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void pyOcean_RegisterFinalizer(void (*func)(void))
/* -------------------------------------------------------------------- */
{  pyOcean_finalizer *finalizer;

   /* Allocate memory for the finalizer */
   finalizer = (pyOcean_finalizer *)malloc(sizeof(pyOcean_finalizer));
   if (finalizer == NULL) return ;

   /* Initialize the finalizer and add it to the list */
   finalizer -> func = func;
   finalizer -> next = py_ocean_finalizers;
   py_ocean_finalizers = finalizer;   
}


#if PY_MAJOR_VERSION < 3
/* -------------------------------------------------------------------- */
static void pyOcean_cpu_dealloc(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (pyOcean_cpu_original_dealloc) pyOcean_cpu_original_dealloc(obj);

   if (obj == module) pyOcean_cpu_free((void *)obj);
}
#endif


/* -------------------------------------------------------------------- */
static void pyOcean_cpu_free(void *data)
/* -------------------------------------------------------------------- */
{  pyOcean_finalizer *finalizer;

   /* Finalize all components */
   if (py_ocean_finalizers != NULL)
   {  
      while ((finalizer = py_ocean_finalizers) != NULL)
      {
         /* Set the next finalizer */
         py_ocean_finalizers = finalizer -> next;

         /* Call the finalizer function */
         finalizer -> func();

         /* Free the finalizer */
         free(finalizer);
      }
   }
}


/* -------------------------------------------------------------------- */
static void pyOcean_cpu_finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Finalize the different components */
   pyOcOpaque_Finalize();
   pyOcIndex_Finalize();
   pyOcTensor_Finalize();
   pyOcStorage_Finalize();
   pyOcScalar_Finalize();
   pyOcStream_Finalize();
   pyOcDeviceCPU_Finalize();
   pyOcDevice_Finalize();
   pyOcDType_Finalize();

   /* Finalize Ocean */
   OcFinalize();
}
