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

#ifndef __PYOCEAN_ARGS_H__
#define __PYOCEAN_ARGS_H__

#include <Python.h>

#include "ocean.h"
#include "pyOcean_compatibility.h"


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct __PyOceanArgObject
{  PyObject *obj;
   struct __PyOceanArgObject *next;
} PyOceanArgObject;

typedef struct __PyOceanArgTensor
{  OcTensor *obj;
   struct __PyOceanArgTensor *next;
} PyOceanArgTensor;

typedef struct __PyOceanArgScalar
{  OcScalar *obj;
   struct __PyOceanArgScalar *next;
} PyOceanArgScalar;

typedef struct __PyOceanArgData
{  void *data;
   struct __PyOceanArgData *next;
} PyOceanArgData;


typedef struct
{  PyObject         *arguments;  /* Argument tuple                   */
   PyObject         *obj;        /* Pointer to the current parameter */
   Py_int_ssize_t    size;       /* Length of the argument list      */
   Py_int_ssize_t    index;      /* Parameter index                  */
   const char       *function;   /* Name of the function             */
   PyOceanArgObject *objects;    /* Dynamically allocated objects    */
   PyOceanArgTensor *tensors;    /* Dynamically allocated tensors    */
   PyOceanArgScalar *scalars;    /* Dynamically allocated scalars    */
   PyOceanArgData   *data;       /* Dynamically allocated data       */
   int               error;      /* Error status (0 = success)       */
} PyOceanArgs;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* NOTE: The argument functions return the number of successfully    */
/* parsed arguments (either 0 or 1) and return a negative error code */
/* in case an error occurred. Once an error occurred all subsequent  */
/* calls to parse an argument will immediately return -1 (possibly   */
/* after initializing the result value or  pointer).                 */

void PyOceanArgs_Init            (PyOceanArgs *args, PyObject *arguments, const char *function);
void PyOceanArgs_Next            (PyOceanArgs *args);
void PyOceanArgs_Prev            (PyOceanArgs *args);
int  PyOceanArgs_Error           (PyOceanArgs *args, const char *message);
void PyOceanArgs_ErrorIf         (PyOceanArgs *args, int condition, const char *message);
int  PyOceanArgs_Validate        (PyOceanArgs *args, int condition, const char *message, int mandatory);
int  PyOceanArgs_Success         (PyOceanArgs *args);
int  PyOceanArgs_Finalize        (PyOceanArgs *args);
int  PyOceanArgs_Length          (PyOceanArgs *args);

/* The following function parses a generic Python object.  */
int PyOceanArgs_GetPyObject      (PyOceanArgs *args, PyObject **obj, int mandatory); /* [obj = NULL] */

/* The following functions extract a python object of the given type */
/* from the current input argument. When the types match, the object */
/* reference is set, the input argument is advanced, and the function*/
/* returns 0. Otherwise the object reference is set to NULL and the  */
/* function returns -1. Note that a non-zero return value here does  */
/* not indicate an error.                                            */
int PyOceanArgs_GetObject        (PyOceanArgs *args, PyTypeObject *type,  PyObject **obj, int mandatory); /* [obj = NULL] */
int PyOceanArgs_GetOcDType       (PyOceanArgs *args, OcDType    *dtype,   int mandatory); /* [dtype = OcDTypeNone] */
int PyOceanArgs_GetOcDevice      (PyOceanArgs *args, OcDevice  **device,  int mandatory); /* [device = NULL] */
int PyOceanArgs_GetOcStream      (PyOceanArgs *args, OcStream  **stream,  int mandatory); /* [stream = NULL] */
int PyOceanArgs_GetOcScalar      (PyOceanArgs *args, OcScalar  **scalar,  int mandatory); /* [scalar = NULL] */
int PyOceanArgs_GetOcStorage     (PyOceanArgs *args, OcStorage **storage, int mandatory); /* [storage = NULL] */
int PyOceanArgs_GetOcTensor      (PyOceanArgs *args, OcTensor  **tensor,  int mandatory); /* [tensor = NULL] */
int PyOceanArgs_GetOcDeviceList  (PyOceanArgs *args, PyObject  **devices, int mandatory); /* [devices = NULL] */
int PyOceanArgs_GetOcTensorList  (PyOceanArgs *args, PyObject  **tensors, int mandatory); /* [tensors = NULL] */
int PyOceanArgs_GetOcScalarTensor(PyOceanArgs *args, OcTensor  **tensor,  int mandatory); /* [tensor = NULL] */

/* The following function parses scalar integer objects. The function*/
/* returns the number of successfully parsed arguments (0 or 1), or  */
/* returns a negative number when the object could not be converted. */
/* On success the parsed value is assigned through the value pointer.*/
int PyOceanArgs_GetScalarInt     (PyOceanArgs *args, long int *value, int mandatory); /* No default */
int PyOceanArgs_GetScalarUInt    (PyOceanArgs *args, long unsigned int *value, int mandatory); /* No default */
int PyOceanArgs_GetScalarDouble  (PyOceanArgs *args, double *value, int mandatory); /* No default */
int PyOceanArgs_GetAsciiString   (PyOceanArgs *args, char **string, int mandatory); /* [string = NULL] - DYNAMIC */
int PyOceanArgs_GetNone          (PyOceanArgs *args, int mandatory);
int PyOceanArgs_GetBool          (PyOceanArgs *args, int *value, int mandatory); /* No default */
int PyOceanArgs_GetChar          (PyOceanArgs *args, char *value, int mandatory); /* No default */
int PyOceanArgs_GetAxis          (PyOceanArgs *args, int *value, int mandatory); /* No default */
int PyOceanArgs_GetAxes          (PyOceanArgs *args, int *values, int *n, int allowScalar, int allowTensor, int mandatory); /* No default */

/* The following functions parse lists of integers. Because there is */
/* no distinction in Python between signed and unsigned numbers all  */
/* functions accept a list of int and long objects. Boolean values   */
/* are explicitly excluded. When successful a Python object is added */
/* to the list of dynamically allocated objects.                     */
/* The tensor stride and size functions first parse an index or size */
/* list and then check to make sure that the number of elements does */
/* not exceed the maximum number of tensor dimensions. When done, the*/
/* list is copied to the given pointer for size or strides, to allow */
/* easier processing. In effect this makes the tensors size and      */
/* stride functions static (when all other parsed arguments are also */
/* static this allows the PyOceanArgs_Finalize functions to be called*/
/* directly, before processing, rather than after.                   */
int PyOceanArgs_GetSizeList     (PyOceanArgs *args, OcSize  **list, Py_int_ssize_t *n, int mandatory); /* [list = NULL] - DYNAMIC */
int PyOceanArgs_GetIndexList    (PyOceanArgs *args, OcIndex **list, Py_int_ssize_t *n, int mandatory); /* [list = NULL] - DYNAMIC */
int PyOceanArgs_GetTensorSize   (PyOceanArgs *args, OcSize **size, int *nSize, int mandatory);         /* [size = NULL] - DYNAMIC */
int PyOceanArgs_GetTensorStrides(PyOceanArgs *args, OcIndex **strides, int *nStrides, int mandatory);  /* [strides = NULL] - DYNAMIC */
int PyOceanArgs_GetTensorStrideType(PyOceanArgs *args, char *type, int mandatory); /* No default */

/* Generic conversion to tensors and scalars */
int PyOceanArgs_GetScalar          (PyOceanArgs *args, OcScalar **scalar, int mandatory); /* [scalar = NULL] - DYNAMIC */
int PyOceanArgs_GetScalarOrNone    (PyOceanArgs *args, OcScalar **scalar, int mandatory); /* [scalar = NULL] - DYNAMIC */
int PyOceanArgs_GetScalarLike      (PyOceanArgs *args, OcScalar **scalar, int mandatory); /* [scalar = NULL] - DYNAMIC */
int PyOceanArgs_GetScalarTensorLike(PyOceanArgs *args, OcScalar **scalar, int mandatory); /* [scalar = NULL] - DYNAMIC */
int PyOceanArgs_GetScalarTensor    (PyOceanArgs *args, OcTensor **tensor, int mandatory); /* [tensor = NULL] - DYNAMIC */
int PyOceanArgs_GetTensor          (PyOceanArgs *args, OcTensor **tensor, int mandatory); /* [tensor = NULL] - DYNAMIC */
int PyOceanArgs_GetTensorNone      (PyOceanArgs *args, OcTensor **tensor, int mandatory); /* [tensor = NULL] - DYNAMIC */
int PyOceanArgs_GetTensorLike      (PyOceanArgs *args, OcTensor **tensor, int mandatory); /* [tensor = NULL] - DYNAMIC */
int PyOceanArgs_GetTensorLikeOnly  (PyOceanArgs *args, OcTensor **tensor, int mandatory); /* [tensor = NULL] - DYNAMIC */
int PyOceanArgs_GetTensorList      (PyOceanArgs *args, PyObject **tensors, int mandatory); /* [tensors = NULL] - DYNAMIC */

#endif
