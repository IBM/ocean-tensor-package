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

#ifndef __OCEAN_CPU_H__
#define __OCEAN_CPU_H__

/* Ocean device-independent header files */
#include "ocean/base/api.h"
#include "ocean/base/config.h"
#include "ocean/base/ocean.h"
#include "ocean/base/dtype.h"
#include "ocean/base/module.h"
#include "ocean/base/device.h"
#include "ocean/base/device_module.h"
#include "ocean/base/scalar.h"
#include "ocean/base/storage.h"
#include "ocean/base/tensor.h"
#include "ocean/base/index.h"
#include "ocean/base/shape.h"
#include "ocean/base/malloc.h"
#include "ocean/base/format.h"
#include "ocean/base/platform.h"
#include "ocean/base/warning.h"
#include "ocean/base/error.h"

/* Ocean main module */
#include "ocean/core/ocean.h"

/* Core module */
#include "ocean/core/interface/module_core.h"
#include "ocean/core/interface/device_itf.h"
#include "ocean/core/interface/scalar_itf.h"
#include "ocean/core/interface/storage_itf.h"
#include "ocean/core/interface/tensor_itf.h"
#include "ocean/core/interface/index_itf.h"
#include "ocean/core/cpu/device_cpu.h"
#include "ocean/core/cpu/storage_cpu.h"
#include "ocean/core/cpu/tensor_cpu.h"

#endif
