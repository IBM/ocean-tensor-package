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

#include "pyOcean_args.h"
#include "pyOcean_dtype.h"
#include "pyOcean_device.h"
#include "pyOcean_stream.h"
#include "pyOcean_scalar.h"
#include "pyOcean_storage.h"
#include "pyOcean_tensor.h"
#include "pyOcean_opaque.h"
#include "pyOcean_convert.h"

#include <stdlib.h>
#include <string.h>


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void PyOceanArgs_Init(PyOceanArgs *args, PyObject *arguments, const char *function)
/* -------------------------------------------------------------------- */
{
   args -> arguments = arguments;
   args -> size      = PyTuple_Size(arguments);
   args -> index     = 0;
   args -> error     = 0;
   args -> function  = (function != NULL) ? function : "<function>";
   args -> objects   = NULL;
   args -> tensors   = NULL;
   args -> scalars   = NULL;
   args -> data      = NULL;

   if (args -> index < args -> size)
   {  args -> obj = PyTuple_GetItem(arguments, 0);
   }
   else
   {  args -> obj = NULL;
   }
}


/* -------------------------------------------------------------------- */
void PyOceanArgs_Next(PyOceanArgs *args)
/* -------------------------------------------------------------------- */
{
   if (args -> index + 1 < args -> size)
   {  args -> index ++;
      args -> obj = PyTuple_GetItem(args -> arguments, args -> index);
   }
   else
   {  args -> index = args -> size;
      args -> obj = NULL;
   }
}


/* -------------------------------------------------------------------- */
void PyOceanArgs_Prev(PyOceanArgs *args)
/* -------------------------------------------------------------------- */
{
   if (args -> index > 0)
     args -> index --;
}


/* -------------------------------------------------------------------- */
void PyOceanArgs_ErrorIf(PyOceanArgs *args, int condition, const char *message)
/* -------------------------------------------------------------------- */
{
   if (!condition) return ;

   if (args -> error == 0)
   {  args -> error = 1;
      OcErrorMessage("%s",message);
   }
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_AddObject(PyOceanArgs *args, PyObject *object)
/* -------------------------------------------------------------------- */
{  PyOceanArgObject *element;

   element = (PyOceanArgObject *)malloc(sizeof(PyOceanArgObject));
   if (element == NULL)
   {  args -> error = 1;
      Py_XDECREF(object);
      OcError(-1, "Error allocating memory for parameter object");
   }

   /* Add the element */
   element -> next = args -> objects;
   element -> obj  = object;
   args -> objects = element;

   return 0;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_AddTensor(PyOceanArgs *args, OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  PyOceanArgTensor *element;

   element = (PyOceanArgTensor *)malloc(sizeof(PyOceanArgTensor));
   if (element == NULL)
   {  args -> error = 1;
      OcDecrefTensor(tensor);
      OcError(-1, "Error allocating memory for tensor parameter object");
   }

   /* Add the element */
   element -> next = args -> tensors;
   element -> obj  = tensor;
   args -> tensors = element;

   return 0;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_AddScalar(PyOceanArgs *args, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  PyOceanArgScalar *element;

   element = (PyOceanArgScalar *)malloc(sizeof(PyOceanArgScalar));
   if (element == NULL)
   {  args -> error = 1;
      OcScalar_free(scalar);
      OcError(-1, "Error allocating memory for scalar parameter object");
   }

   /* Add the element */
   element -> next = args -> scalars;
   element -> obj  = scalar;
   args -> scalars = element;

   return 0;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_AddData(PyOceanArgs *args, void *data)
/* -------------------------------------------------------------------- */
{  PyOceanArgData *element;

   element = (PyOceanArgData *)malloc(sizeof(PyOceanArgData));
   if (element == NULL)
   {  args -> error = 1;
      OcFree(data);
      OcError(-1, "Error allocating memory for scalar parameter object");
   }

   /* Add the element */
   element -> next = args -> data;
   element -> data = data;
   args -> data = element;

   return 0;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_Success(PyOceanArgs *args)
/* -------------------------------------------------------------------- */
{
   /* Check for errors and unused parameters */
   if ((args -> obj != NULL) && (args -> error == 0))
   {  OcErrorMessage("Invalid parameters to function %s", args -> function);
      args -> error = 1;
   }

   /* Finalize in case of error */
   if (args -> error)
   {  /* Reset the fields and finalize */
      args -> obj = NULL;
      args -> error = 0;
      PyOceanArgs_Finalize(args);
      return 0;
   }

   /* Return 1 on success and 0 on failure */
   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_Finalize(PyOceanArgs *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgObject *object;
   PyOceanArgTensor *tensor;
   PyOceanArgScalar *scalar;
   PyOceanArgData   *data;

   /* Delete all intermediate Python objects */
   while ((object = args -> objects) != NULL)
   {  args -> objects = object -> next;
      Py_XDECREF(object -> obj);
      free(object);
   }

   /* Delete all intermediate Ocean tensors */
   while ((tensor = args -> tensors) != NULL)
   {  args -> tensors = tensor -> next;
      OcDecrefTensor(tensor -> obj);
      free(tensor);
   }

   /* Delete all intermediate Ocean scalars */
   while ((scalar = args -> scalars) != NULL)
   {  args -> scalars = scalar -> next;
      OcScalar_free(scalar -> obj);
      free(tensor);
   }

   /* Delete all intermediate data elements */
   while ((data = args -> data) != NULL)
   {  args -> data = data -> next;
      if (data -> data) OcFree(data -> data);
      free(data);
   }

   /* Check for errors and unused parameters */
   if ((args -> obj != NULL) && (args -> error == 0))
   {  args -> obj = NULL;
      args -> error = 0;
      OcError(0, "Invalid parameters to function %s", args -> function);
   }

   /* Return 1 on success and 0 on failure */
   return (args -> error) ? 0 : 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_Length(PyOceanArgs *args)
/* -------------------------------------------------------------------- */
{
   return (int)(args -> size);
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_EnsureParam(PyOceanArgs *args, int mandatory)
/* -------------------------------------------------------------------- */
{
   if (args -> obj == NULL)
   {  if (mandatory)
      {  args -> error = 1;
         OcError(-1, "Missing parameter %d in %s", (int)(args -> index + 1), args -> function);
      }
      return 0;
   }

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_Error(PyOceanArgs *args, const char *message)
/* -------------------------------------------------------------------- */
{
   args -> error = 1;
   OcErrorMessage("%s at parameter %d of %s", message, (int)(args -> index + 1), args -> function);

   return -1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_Validate(PyOceanArgs *args, int condition, const char *message, int mandatory)
/* -------------------------------------------------------------------- */
{
   if (!condition)
   {  if (mandatory)
           return PyOceanArgs_Error(args, message);
      else return 0;
   }

   return 1;
}


/* -------------------------------------------------------------------- */
void *PyOceanArgs_Malloc(PyOceanArgs *args, size_t size)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   void     *buffer;

   /* Allocate memory */
   buffer = (OcSize *)malloc(size);
   if (buffer == NULL)
   {  args -> error = 1;
      OcError(NULL, "Error allocating memory");
   }

   /* Create an opaque object to store the pointer */
   obj = PyOceanOpaque_NewWithFree(buffer, 0, &free);
   if (obj == NULL) return NULL;

   /* Add the opaque object to the list of dynamic parameters */
   if (PyOceanArgs_AddObject(args, obj) != 0) return NULL;

   return buffer;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetPyObject(PyOceanArgs *args, PyObject **obj, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the result */
   *obj = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Set the object */
   *obj = args -> obj;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetObject(PyOceanArgs *args, PyTypeObject *type, PyObject **obj, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the result */
   *obj = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   result = PyOceanArgs_EnsureParam(args, mandatory);
   if (result != 1) return result;
   result = PyOceanArgs_Validate(args, PyObject_IsInstance(args -> obj, (PyObject *)type), "Type mismatch", mandatory);
   if (result != 1) return result;

   /* Set the object */
   *obj = args -> obj;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetOcDType(PyOceanArgs *args, OcDType *dtype, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   int result;

   result = PyOceanArgs_GetObject(args, PyOceanDType, &obj, mandatory);
   *dtype = (result == 1) ? PYOC_GET_DTYPE(obj) : OcDTypeNone;

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetOcScalar(PyOceanArgs *args, OcScalar **scalar, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   OcScalar *ptr = NULL;
   int result;
   
   result = PyOceanArgs_GetObject(args, PyOceanScalar, &obj, mandatory);
   if (result == 1)
   {  ptr = OcScalar_clone(PYOC_GET_SCALAR(obj));
      if ((ptr == NULL) || (PyOceanArgs_AddScalar(args, ptr) != 0))
      {  ptr = NULL; result = -1;
      }
   }

   /* Set the scalar */
   *scalar = ptr;

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetOcDevice(PyOceanArgs *args, OcDevice **device, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   int result;

   result = PyOceanArgs_GetObject(args, PyOceanDevice, &obj, mandatory);
   *device = (result == 1) ? PYOC_GET_DEVICE(obj) : NULL;

   return result;
}


/* -------------------------------------------------------------------- */
int  PyOceanArgs_GetOcStream(PyOceanArgs *args, OcStream **stream, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   int result;
   
   result = PyOceanArgs_GetObject(args, PyOceanStream, &obj, mandatory);
   *stream = (result == 1) ? PYOC_GET_STREAM(obj) : NULL;

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetOcStorage(PyOceanArgs *args, OcStorage **storage, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   int result;
   
   result = PyOceanArgs_GetObject(args, PyOceanStorage, &obj, mandatory);
   *storage = (result == 1) ? PYOC_GET_STORAGE(obj) : NULL;

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetOcTensor(PyOceanArgs *args, OcTensor **tensor, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   int result;
   
   result = PyOceanArgs_GetObject(args, PyOceanTensor, &obj, mandatory);
   *tensor = (result == 1) ? PYOC_GET_TENSOR(obj) : NULL;

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetOcDeviceList(PyOceanArgs *args, PyObject  **devices, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject       *obj, *arg;
   Py_int_ssize_t  i, n;
   int             result;

   /* Initialize the result */
   *devices = NULL;

   /* Return if the argument is not a list */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Check for lists or tuples */
   if (!PySequence_Check(args -> obj))
   {  return PyOceanArgs_Validate(args, 0, "Device list expected", mandatory);
   }
   
   /* Get the list size */
   arg = args -> obj;
   n = PySequence_Size(arg);
   result = PyOceanArgs_Validate(args, (n > 0), "Device list cannot be empty", mandatory);
   if (result != 1) return result;

   /* Ensure that all entries are devices */
   for (i = 0; i < n; i++)
   {  obj = PySequence_ITEM(arg, i);
      if (!PyOceanDevice_Check(obj))
      {  if ((mandatory) || (i > 0))
              return PyOceanArgs_Error(args, "Invalid element type in device list");
         else return 0;
      }
   }

   /* Set the result */
   *devices = args -> obj;
   
   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetOcTensorList(PyOceanArgs *args, PyObject  **tensors, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject       *obj, *arg;
   Py_int_ssize_t  i, n;
   int             result;

   /* Initialize the result */
   *tensors = NULL;

   /* Return if the argument is not a list */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Check for lists or tuples */
   if (!PySequence_Check(args -> obj))
   {  return PyOceanArgs_Validate(args, 0, "Tensor list expected", mandatory);
   }
   
   /* Get the list size */
   arg = args -> obj;
   n = PySequence_Size(arg);
   result = PyOceanArgs_Validate(args, (n > 0), "Tensor list cannot be empty", mandatory);
   if (result != 1) return result;

   /* Ensure that all entries are tensors */
   for (i = 0; i < n; i++)
   {  obj = PySequence_GetItem(arg, i);
      if (!PyOceanTensor_Check(obj))
      {  if ((mandatory) || (i > 0))
              return PyOceanArgs_Error(args, "Invalid element type in tensor list");
         else return 0;
      }
   }

   /* Set the result */
   *tensors = args -> obj;
   
   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetOcScalarTensor(PyOceanArgs *args, OcTensor **tensor, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the result */
   *tensor = NULL;

   /* Return if the argument is not a list */
   if (args -> error) return -1;
   result = PyOceanArgs_EnsureParam(args, mandatory);
   if (result != 1) return result;
   result = PyOceanArgs_Validate(args, PyOceanTensor_Check(args -> obj), "Tensor type expected", mandatory);
   if (result != 1) return result;

   /* Check the number of elements */
   if (PYOC_GET_TENSOR(args -> obj) -> nelem != 1)
   {  return PyOceanArgs_Validate(args, 0, "Scalar Ocean tensor expected", mandatory);
   }

   *tensor = PYOC_GET_TENSOR(args -> obj);
   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetScalarInt(PyOceanArgs *args, long int *value, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *arg;
   int result;

   /* Get the input argument */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Clear the error status */
   PyErr_Clear();

   /* Scalar from Python numeric type */
   arg = args -> obj;
   if (PyBool_Check(arg))
   {  return PyOceanArgs_Validate(args, 0, "Expected integer type instead of Boolean", mandatory);
   }
   if (PyInt_Check(arg))
   {  *value = PyInt_AsLong(arg);
   }
   else if (PyLong_Check(arg))
   {  *value = PyLong_AsLong(arg);
   }
   else
   {  return PyOceanArgs_Validate(args, 0, "Expected integer type", mandatory);
   }

   /* Check for conversion errors */
   if (PyErr_Occurred())
   {  return PyOceanArgs_Error(args, "Error converting to scalar long");
   }

   /* Determine the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetScalarUInt(PyOceanArgs *args, long unsigned int *value, int mandatory)
/* -------------------------------------------------------------------- */
{  long int v;
   int result;
   
   result = PyOceanArgs_GetScalarInt(args, &v, mandatory);
   if (result != 1) return result;

   if (v < 0)
   {  return PyOceanArgs_Error(args, "Expected nonnegative integer");
   }
   else
   {  *value = (long unsigned int)v;
   }

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetScalarDouble(PyOceanArgs *args, double *value, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *arg;
   int result;

   /* Get the input argument */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Clear the error status */
   PyErr_Clear();

   /* Scalar from Python numeric type */
   arg = args -> obj;
   if (PyBool_Check(arg))
   {  return PyOceanArgs_Validate(args, 0, "Expected double type instead of Boolean", mandatory);
   }
   if (PyInt_Check(arg))
   {  *value = (double)PyInt_AsLong(arg);
   }
   else if (PyLong_Check(arg))
   {  *value = (double)PyLong_AsLong(arg);
   }
   else if (PyFloat_Check(arg))
   {  *value = PyFloat_AsDouble(arg);
   }
   else
   {  return PyOceanArgs_Validate(args, 0, "Expected double type", mandatory);
   }

   /* Check for conversion errors */
   if (PyErr_Occurred())
   {  return PyOceanArgs_Error(args, "Error converting to scalar double");
   }

   /* Determine the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetAsciiString(PyOceanArgs *args, char **string, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   Py_ssize_t size;
   int result;

   /* Initialize the output */
   *string = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Parse an ASCII string */
   if (PyUnicode_Check(args -> obj))
   {  if ((obj = PyUnicode_AsASCIIString(args -> obj)) == NULL)
         return PyOceanArgs_Validate(args, 0, "Unicode string is not ASCII-compatible", mandatory);
   }
#if PY_MAJOR_VERSION < 3
   else if (PyString_Check(args -> obj))
   {  obj = args -> obj;
      Py_INCREF(obj);
   }
#endif
   else
   {  return PyOceanArgs_Validate(args, 0, "ASCII-compatible string expected", mandatory);
   }

   /* Get the string length */
   #if PY_MAJOR_VERSION < 3
      size = PyString_Size(obj);
   #else
      size = PyBytes_Size(obj);
   #endif

   /* Allocate memory for the string */
   *string = (char *)OcMalloc(sizeof(char) * (size + 1));
   if (*string != NULL)
   {  /* Copy the string */
      #if PY_MAJOR_VERSION < 3
         strcpy(*string, PyString_AsString(obj));
      #else
         strcpy(*string, PyBytes_AsString(obj));
      #endif

      /* Register the data */
      if (PyOceanArgs_AddData(args, (void *)(*string)) != 0)
      {  *string = NULL;
      }
   }

   /* Decref the String or Bytes object */
   Py_DECREF(obj);

   if (*string != NULL)
   {  /* Proceed to the next argument */
      PyOceanArgs_Next(args);
      return 1;
   }
   else
   {  /* Return an error */
      return PyOceanArgs_Error(args, "Could not allocate memory for string");
   }
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetNone(PyOceanArgs *args, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Check the object */
   result = PyOceanArgs_Validate(args, (args -> obj == Py_None), "None object expected", mandatory);
   if (result != 1) return result;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetBool(PyOceanArgs *args, int *value, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   int result;

   result = PyOceanArgs_GetObject(args, &PyBool_Type, &obj, mandatory);
   if (result == 1) *value = ((obj == Py_True) ? 1 : 0);

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetChar(PyOceanArgs *args, char *value, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   char     *s;
   int       result;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Make sure the entry is a string of length one */
   if (PyUnicode_Check(args -> obj))
   {  if ((obj = PyUnicode_AsASCIIString(args -> obj)) == NULL)
         return PyOceanArgs_Validate(args, 0, "Error converting unicode string", mandatory);

      #if PY_MAJOR_VERSION >= 3
      s = PyBytes_AS_STRING(obj);
      #else
      s = PyString_AS_STRING(obj);
      #endif

      if (s == NULL)
      {  result = PyOceanArgs_Validate(args, 0, "Error converting unicode string", mandatory);
      }
      else if (strlen(s) != 1)
      {  result = PyOceanArgs_Validate(args, 0, "Single character expected", mandatory);
      }
      else
      {  *value = s[0];
         result = 1;
      }

      /* Decrement the intermediate string/buffer object */
      Py_DECREF(obj);
   }
#if PY_MAJOR_VERSION < 3
   else if (PyString_Check(args -> obj))
   {  
      s = PyString_AsString(args -> obj);
      if ((PyString_Size(args -> obj) != 1) || (strlen(s) != 1))
         return PyOceanArgs_Validate(args, 0, "Single character expected", mandatory);

      *value = s[0];
      result = 1;
   }
#endif
   else
   {  return PyOceanArgs_Validate(args, 0, "Character type expected", mandatory);
   }

   /* Proceed to the next argument */
   if (result == 1) PyOceanArgs_Next(args);

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetAxis(PyOceanArgs *args, int *value, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject *arg;
   long int v;
   int result;

   /* Get the input argument */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Clear the error status */
   PyErr_Clear();

   /* Scalar from Python numeric type */
   arg = args -> obj;
   if (PyBool_Check(arg))
   {  return PyOceanArgs_Validate(args, 0, "Expected integer type instead of Boolean", mandatory);
   }
   if (PyInt_Check(arg))
   {  v = PyInt_AsLong(arg);
   }
   else if (PyLong_Check(arg))
   {  v = PyLong_AsLong(arg);
   }
   else
   {  return PyOceanArgs_Validate(args, 0, "Expected integer type", mandatory);
   }

   /* Check for conversion errors */
   if (PyErr_Occurred())
   {  return PyOceanArgs_Error(args, "Error converting to scalar long");
   }

   /* Basic range check on dimensions */
   if ((v < -32768L) || (v >= 32768L))
      return PyOceanArgs_Validate(args, 0, "Axis index out of range", mandatory);
   else *value = (int)v;

   /* Determine the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_intrnlGetDataList(PyOceanArgs *args, char **list, Py_int_ssize_t *n, OcDType dtype,
                                  int mandatory, int maxlen, int allowTensor, const char *str)
/* -------------------------------------------------------------------- */
{  PyObject       *obj;
   Py_int_ssize_t  i, k;
   OcTensor       *tensor;
   OcIndex         stride, elemsize;
   OcScalar        scalar;
   char           *buffer = NULL, *data;
   int             result;

   /* Initialize */ 
   *list = NULL;
   elemsize = OcDType_size(dtype);

   /* Get the input argument */
   if (args -> error) return -1;
   result = PyOceanArgs_EnsureParam(args, mandatory);
   if (result == 1) obj = args -> obj; else return result;

   /* Check for string types */
#if PY_MAJOR_VERSION >= 3
   if (PyUnicode_Check(obj))
#else
   if (PyUnicode_Check(obj) || PyString_Check(obj))
#endif
   {  return PyOceanArgs_Validate(args, 0, "Strings are not a valid size type ", mandatory);
   }

   /* Check for sequence types - this could have been included as */
   /* part of the generic tensor parsing. For now leave it as a   */
   /* specialized part for size parsing.                          */
   if ((PySequence_Check(obj)) && ((k = PySequence_Length(obj)) != -1))
   {
      /* Return if the sequence is empty */
      if (k == 0) goto success;

      /* Make sure the number of elements is valid */
      if ((maxlen > 0) && (k > maxlen))
      {  args -> error = 1;
         OcError(-1, "Maximum number of %s elements exceeded (%d)", str, maxlen);
      }

      /* Allocate registered memory */
      buffer = (char *)PyOceanArgs_Malloc(args, elemsize * k);
      if (buffer == NULL) return -1;

      /* Convert the elements */
      for (i = 0; i < k; i++)
      {
         /* Check for scalars */
         result = pyOcean_getScalarLike(PySequence_GetItem(obj, i), &scalar);
         if (result != 1)
         {  args -> error = 1;
            OcError(-1, "Invalid %s type", str);
         }
         else
         {  if ((!OcDType_isSigned(dtype)) && (OcScalar_isLTZero(&scalar)))
            {  args -> error = 1;
               OcError(-1, "The %s values must be nonnegative", str);
            }
            else if (!OcScalar_inRange(&scalar, dtype))
            {  args -> error = 1;
               OcError(-1, "The %s value at index %"OC_FORMAT_LU" of out of range", str, (long unsigned)(i+1));
            }
            else
            {  OcScalar_castTo(&scalar, dtype, NULL);
               OcScalar_exportData(&scalar, (void *)(buffer + i * elemsize), 0);
            }
         }
      }

      /* Success */
      goto success;
   }

   /* Check for tensor-like types */
   if (allowTensor)
        result = pyOcean_getTensorLike(obj, &tensor, dtype, OcCPU);
   else result = 0;
   if (result < 0) return result;
   if (result == 1)
   {  
      /* Add the tensor */
      if (PyOceanArgs_AddTensor(args, tensor) != 0) return -1;
      
      /* Check the number of dimensions */
      if (OcTensor_ndims(tensor) > 1)
      {  args -> error = 1;
         OcError(-1, "The %s vector cannot have more than one dimension", str);
      }

      /* Return if the tensor is empty */
      if ((k = OcTensor_nelem(tensor)) == 0) goto success;

      /* Check the number of elements */
      if ((maxlen > 0) && (k > maxlen))
      {  args -> error = 1;
         OcError(-1, "Maximum number of %s elements exceeded (%d)", str, maxlen);
      }

      /* Allocate registered memory */
      buffer = (char *)PyOceanArgs_Malloc(args, elemsize * k);
      if (buffer == NULL) return -1;

      /* Copy the information */
      data = OcTensor_data(tensor);
      stride = (OcTensor_ndims(tensor) == 0) ? 0 : tensor -> strides[0];
      for (i = 0; i < k; i++)
      {  OcScalar_copyRaw(((void *)(data + i * stride)), dtype,
                           (void *)(buffer + i * elemsize), dtype);
      }

      /* Success */
      goto success;
   }

   /* Invalid data type */
   return PyOceanArgs_Validate(args, 0, "Sequence or vector type expected", mandatory);

success :
   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   /* Set the result */
   *n = k; *list = buffer;

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetAxes(PyOceanArgs *args, int *axes, int *n, int allowScalar, int allowTensor, int mandatory)
/* -------------------------------------------------------------------- */
{  Py_int_ssize_t nAxes;
   OcInt64 v;
   char *buffer = NULL;
   int   result, i;

   /* Parse a scalar axis */
   if (allowScalar)
   {  result = PyOceanArgs_GetAxis(args, axes, 0);
      if (result == 1) { *n = 1; return result; }
      if (result != 0) return result;
   }

   /* Parse a list */
   result = PyOceanArgs_intrnlGetDataList(args, &buffer, &nAxes, OcDTypeInt64, mandatory,
                                          OC_TENSOR_MAX_DIMS, allowTensor, "axis indices");
   if (result == 1)
   {  /* Copy the dimensions */
      for (i = 0; i < nAxes; i++)
      {  /* Basic range check on dimensions */
         v = ((OcInt64 *)buffer)[i];
         if ((v < -32768L) || (v >= 32768L))
         {  PyOceanArgs_Prev(args);
            return PyOceanArgs_Validate(args, 0, "Axis index out of range", 1);
         }
         axes[i] = (int)v;
      }
      *n = (int) nAxes;
   }

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetSizeList(PyOceanArgs *args, OcSize **list, Py_int_ssize_t *n, int mandatory)
/* -------------------------------------------------------------------- */
{  char *buffer = NULL;
   int result;

   result = PyOceanArgs_intrnlGetDataList(args, &buffer, n, OcDTypeUInt64, mandatory, -1, 1, "size");
   *list = (OcSize *)buffer;

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetIndexList(PyOceanArgs *args, OcIndex **list, Py_int_ssize_t *n, int mandatory)
/* -------------------------------------------------------------------- */
{  char *buffer = NULL;
   int result;

   result = PyOceanArgs_intrnlGetDataList(args, &buffer, n, OcDTypeInt64, mandatory, -1, 1, "index");
   *list = (OcIndex *)buffer;

   return result;
}


/* -------------------------------------------------------------------- */
int  PyOceanArgs_GetTensorSize(PyOceanArgs *args, OcSize **size, int *nSize, int mandatory)
/* -------------------------------------------------------------------- */
{  Py_int_ssize_t n;
   char *buffer;
   int result;

   /* Get the size list */
   result = PyOceanArgs_intrnlGetDataList(args, &buffer, &n, OcDTypeUInt64, mandatory, OC_TENSOR_MAX_DIMS, 1, "tensor dimension");
   *size = (OcSize *)buffer;

   /* Output the number of dimensions */
   *nSize = (result == 1) ? (int)n : -1;

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetTensorStrides(PyOceanArgs *args, OcIndex **strides, int *nStrides, int mandatory)
/* -------------------------------------------------------------------- */
{  Py_int_ssize_t n;
   char *buffer;
   int result;

   /* Get the size list */
   result = PyOceanArgs_intrnlGetDataList(args, &buffer, &n, OcDTypeInt64, mandatory, OC_TENSOR_MAX_DIMS, 1, "tensor stride");
   *strides = (OcIndex *)buffer;

   /* Output the number of dimensions */
   *nStrides = (result == 1) ? n : -1;

   return result;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetTensorStrideType(PyOceanArgs *args, char *type, int mandatory)
/* -------------------------------------------------------------------- */
{  char value;
   int result;

   result = PyOceanArgs_GetChar(args, &value, mandatory);
   if (result != 1) return result;

   /* Check validity of the type */
   if ((value == 'F') || (value == 'f')) { *type = 'F'; return 1; }
   if ((value == 'C') || (value == 'c')) { *type = 'C'; return 1; }
   if ((value == 'R') || (value == 'r')) { *type = 'R'; return 1; }

   args -> index -= 1;
   return PyOceanArgs_Validate(args, 0, "Invalid stride type", 1);
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetScalar(PyOceanArgs *args, OcScalar **scalar, int mandatory)
/* -------------------------------------------------------------------- */
{  OcScalar value, *ptr;
   int result;

   /* Initialize the scalar */
   *scalar = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Parse the scalar */
   result = pyOcean_getScalar(args -> obj, &value);
   if (result  < 0) { args -> error = 1; return result; }
   if (result != 1)
      return PyOceanArgs_Validate(args, 0, "Scalar type expected", mandatory);

   /* Copy the scalar and add to the arguments */
   ptr = OcScalar_clone(&value);
   if ((ptr == NULL) || (PyOceanArgs_AddScalar(args, ptr) != 0))
        return -1;
   else *scalar = ptr;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetScalarOrNone(PyOceanArgs *args, OcScalar **scalar, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the scalar */
   *scalar = NULL;

   /* Try to parse a scalar */
   result = PyOceanArgs_GetScalar(args, scalar, 0);
   if (result != 0) return result;

   /* Try to parse None */
   result = PyOceanArgs_GetNone(args, 0);
   if (result != 0) return result;

   return PyOceanArgs_Validate(args, 0, "Scalar type or None expected", mandatory);
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetScalarLike(PyOceanArgs *args, OcScalar **scalar, int mandatory)
/* -------------------------------------------------------------------- */
{  OcScalar value, *ptr;
   int result;

   /* Initialize the scalar */
   *scalar = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Parse the scalar */
   result = pyOcean_getScalarLike(args -> obj, &value);
   if (result  < 0) { args -> error = 1; return result; }
   if (result != 1)
      return PyOceanArgs_Validate(args, 0, "Scalar-like type expected", mandatory);

   /* Copy the scalar and add to the arguments */
   ptr = OcScalar_clone(&value);
   if ((ptr == NULL) || (PyOceanArgs_AddScalar(args, ptr) != 0))
        return -1;
   else *scalar = ptr;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetScalarTensorLike(PyOceanArgs *args, OcScalar **scalar, int mandatory)
/* -------------------------------------------------------------------- */
{  OcScalar value, *ptr;
   int result;

   /* Initialize the scalar */
   *scalar = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Parse the scalar */
   result = pyOcean_getScalarTensorLike(args -> obj, &value);
   if (result  < 0) { args -> error = 1; return result; }
   if (result != 1)
      return PyOceanArgs_Validate(args, 0, "Tensor-like type expected", mandatory);

   /* Copy the scalar and add to the arguments */
   ptr = OcScalar_clone(&value);
   if ((ptr == NULL) || (PyOceanArgs_AddScalar(args, ptr) != 0))
        return -1;
   else *scalar = ptr;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetScalarTensor(PyOceanArgs *args, OcTensor **tensor, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the tensor */
   *tensor = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Parse the tensor */
   result = pyOcean_getScalarTensor(args -> obj, tensor);
   if (result  < 0) { args -> error = 1; return result; }
   if (result != 1)
      return PyOceanArgs_Validate(args, 0, "Scalar tensor expected", mandatory);

   /* Add the tensor */
   if (PyOceanArgs_AddTensor(args, *tensor) != 0) return -1;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetTensor(PyOceanArgs *args, OcTensor **tensor, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the tensor */
   *tensor = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Parse the tensor */
   result = pyOcean_getTensor(args -> obj, tensor);
   if (result  < 0) { args -> error = 1; return result; }
   if (result != 1)
      return PyOceanArgs_Validate(args, 0, "Tensor expected", mandatory);

   /* Add the tensor */
   if (PyOceanArgs_AddTensor(args, *tensor) != 0) return -1;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetTensorNone(PyOceanArgs *args, OcTensor **tensor, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the tensor */
   *tensor = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Accept None or parse a tensor */
   if (args -> obj != Py_None)
   {
      /* Parse the tensor */
      result = pyOcean_getTensor(args -> obj, tensor);
      if (result  < 0) { args -> error = 1; return result; }
      if (result != 1)
         return PyOceanArgs_Validate(args, 0, "Tensor expected", mandatory);

      /* Add the tensor */
      if (PyOceanArgs_AddTensor(args, *tensor) != 0) return -1;
   }

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetTensorLike(PyOceanArgs *args, OcTensor **tensor, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the tensor */
   *tensor = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Parse the tensor */
   result = pyOcean_getTensorLike(args -> obj, tensor, OcDTypeNone, NULL);
   if (result  < 0) { args -> error = 1; return result; }
   if (result != 1)
      return PyOceanArgs_Validate(args, 0, "Tensor-like type expected", mandatory);

   /* Add the tensor */
   if (PyOceanArgs_AddTensor(args, *tensor) != 0) return -1;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetTensorLikeOnly(PyOceanArgs *args, OcTensor **tensor, int mandatory)
/* -------------------------------------------------------------------- */
{  int result;

   /* Initialize the tensor */
   *tensor = NULL;

   /* Basic checks */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Parse the tensor */
   result = pyOcean_getTensorLikeOnly(args -> obj, tensor, NULL, OcDTypeNone, NULL, 'F');
   if (result  < 0) { args -> error = 1; return result; }
   if (result != 1)
      return PyOceanArgs_Validate(args, 0, "Tensor-like expected", mandatory);

   /* Add the tensor */
   if (PyOceanArgs_AddTensor(args, *tensor) != 0) return -1;

   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}


/* -------------------------------------------------------------------- */
int PyOceanArgs_GetTensorList(PyOceanArgs *args, PyObject **tensors, int mandatory)
/* -------------------------------------------------------------------- */
{  PyObject       *obj, *arg, *list;
   Py_int_ssize_t  i, n;
   OcTensor       *tensor;
   int             result;

   /* Initialize the result */
   *tensors = NULL;

   /* Return if the argument is not a list */
   if (args -> error) return -1;
   if ((result = PyOceanArgs_EnsureParam(args, mandatory)) != 1) return result;

   /* Check for lists or tuples */
   if (!PySequence_Check(args -> obj))
   {  return PyOceanArgs_Validate(args, 0, "Tensor list expected", mandatory);
   }
   
   /* Get the list size */
   arg = args -> obj;
   n = PySequence_Size(arg);
   result = PyOceanArgs_Validate(args, (n > 0), "Tensor list cannot be empty", mandatory);
   if (result != 1) return result;

   /* Create the tensor list and register the object to ensure */
   /* proper clean up even when populating it fails.           */
   list = PyList_New(n);
   if (list == NULL) PyOceanArgs_Error(args, "Error creating tensor list");
   if (PyOceanArgs_AddObject(args, list) != 0) return -1;

   /* Parse tensor elements */
   for (i = 0; i < n; i++)
   {  obj = PySequence_GetItem(arg, i);
      if (pyOcean_getTensorLike(obj, &tensor, OcDTypeNone, NULL) != 1)
      {  if ((mandatory) || (i > 0))
              return PyOceanArgs_Error(args, "Invalid element type in tensor list");
         else return 0;
      }

      /* Create a new tensor object */
      obj = PyOceanTensor_Wrap(tensor);
      if (obj == NULL) return PyOceanArgs_Error(args, "Error creating tensor object");

      /* Add the tensor object to the list */
      if (PyList_SetItem(list, i, obj) != 0)
         return PyOceanArgs_Error(args, "Error adding tensor object to list");
   }

   /* Set the result */
   *tensors = list;
   
   /* Proceed to the next argument */
   PyOceanArgs_Next(args);

   return 1;
}
