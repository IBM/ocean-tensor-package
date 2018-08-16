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

#ifndef __PYOCEAN_H__
#define __PYOCEAN_H__

#include <Python.h>

#include "pyOcean_args.h"
#include "pyOcean_dtype.h"
#include "pyOcean_device.h"
#include "pyOcean_device_cpu.h"
#include "pyOcean_stream.h"
#include "pyOcean_scalar.h"
#include "pyOcean_storage.h"
#include "pyOcean_tensor.h"
#include "pyOcean_opaque.h"
#include "pyOcean_convert.h"
#include "pyOcean_index.h"
#include "pyOcean_core.h"
#include "pyOcean_module_core.h"
#include "pyOcean_compatibility.h"

void pyOcean_RegisterFinalizer(void (*func)(void));

#endif
