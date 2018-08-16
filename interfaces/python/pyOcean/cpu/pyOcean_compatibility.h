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

#ifndef __PYOCEAN_COMPATIBILITY_H__
#define __PYOCEAN_COMPATIBILITY_H__

/* PYTHON VERSIONS */
#if (PY_MAJOR_VERSION >= 2)
#define PY_VERSION_2_0
#endif
#if (PY_MAJOR_VERSION >=3) || ((PY_MAJOR_VERSION >= 2) && (PY_MINOR_VERSION >= 1))
#define PY_VERSION_2_1
#endif
#if (PY_MAJOR_VERSION >=3) || ((PY_MAJOR_VERSION >= 2) && (PY_MINOR_VERSION >= 2))
#define PY_VERSION_2_2
#endif
#if (PY_MAJOR_VERSION >=3) || ((PY_MAJOR_VERSION >= 2) && (PY_MINOR_VERSION >= 3))
#define PY_VERSION_2_3
#endif
#if (PY_MAJOR_VERSION >=3) || ((PY_MAJOR_VERSION >= 2) && (PY_MINOR_VERSION >= 4))
#define PY_VERSION_2_4
#endif
#if (PY_MAJOR_VERSION >=3) || ((PY_MAJOR_VERSION >= 2) && (PY_MINOR_VERSION >= 5))
#define PY_VERSION_2_5
#endif

/* Module functions */
#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

/* Macros */
#ifndef PyVarObject_HEAD_INIT
#define PyVarObject_HEAD_INIT(TYPE, SIZE) PyObject_HEAD_INIT(TYPE) SIZE,
#endif

#ifndef Py_TYPE
#define Py_TYPE(OBJ) (((PyObject *)(OBJ)) -> ob_type)
#endif


/* Version string */
#define PyMODULE_NAME_C(NAME,MAJOR,MINOR)     NAME##_v##MAJOR##_##MINOR
#define PyMODULE_NAME_B(NAME,MAJOR,MINOR)     PyMODULE_NAME_C(NAME,MAJOR,MINOR)
#define PyMODULE_NAME(NAME)                   PyMODULE_NAME_B(NAME,PY_MAJOR_VERSION,PY_MINOR_VERSION)

#define PyMODULE_NAME_STR_C(NAME,MAJOR,MINOR) NAME "_v" #MAJOR "_" #MINOR
#define PyMODULE_NAME_STR_B(NAME,MAJOR,MINOR) PyMODULE_NAME_STR_C(NAME,MAJOR,MINOR)
#define PyMODULE_NAME_STR(NAME)               PyMODULE_NAME_STR_B(NAME,PY_MAJOR_VERSION,PY_MINOR_VERSION)


#if PY_MAJOR_VERSION >= 3
    #define PyMODULE_INIT(name) PyMODINIT_FUNC PyMODULE_NAME(PyInit_ ## name)(void)
#else
    #define PyMODULE_INIT(name) PyMODINIT_FUNC PyMODULE_NAME(init ## name)(void)
#endif

#if PY_MAJOR_VERSION >= 3
   #define PyMODULE_INIT_ERROR return NULL
#else
   #define PyMODULE_INIT_ERROR return
#endif


/* Size types */
#ifdef PY_VERSION_2_5
typedef Py_ssize_t  Py_int_ssize_t;
#else
typedef int Py_int_ssize_t;
#endif

/* Integer and string objects */
#if PY_MAJOR_VERSION >= 3
#define PyInt_Check              PyLong_Check
#define PyInt_AsLong             PyLong_AsLong
#define PyInt_AS_LONG            PyLong_AS_LONG
#define PyInt_FromLong           PyLong_FromLong
#endif

/* String objects */
#if PY_MAJOR_VERSION >= 3
#define PyString_FromString(S)   PyUnicode_FromString(S)
#define PyString_AsString(S)     PyBytes_AsString(S)
#endif

#endif
