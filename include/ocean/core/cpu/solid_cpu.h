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

#ifndef __OC_SOLID_CPU_H__
#define __OC_SOLID_CPU_H__

#define OC_SOLID_DEVICE          cpu
#define OC_DEVICE                CPU


#define OC_SOLID_FUNPTR(FUN)     solid_funptr_ # OC_SOLID_DEVICE # _ # FUN
#define OC_SOLID_FUNCTION(FUN)   solid_ # OC_SOLID_DEVICE # _ # FUN
#define OC_SOLID_FUN_PARAM(FUN)  OcTensor # OC_DEVICE # _ # FUN # _param
#define OC_SOLID_TASK(FUN)       OcSolidTask_cpu_ # FUN
#define OC_SOLID_TASK_ARGS(FUN)  solid_param_ # OC_SOLID_DEVICE # _ # FUN
#define OC_SOLID_MACRO(FUN)      solid_macro_ # OC_SOLID_DEVICE # _ # FUN

#define OC_APPLY_CONFIG(TYPE)    Oc # TYPE # _config



#define OC_SOLID_FUN_DECL(FUN, TYPE) \
   static int OC_KERNEL(FUN # _operation)(void *ptr_param) \
   {  OC_SOLID_FUN_PARAM(FUN) *param = (OC_PARAM(fill) *)(ptr_param); \
      OC_SOLID_TASK_ARGS(FUN) *args = &(param -> args); \
      \
      OC_SOLID_MACRO(FUN)(param -> funptr, args); \
   } \
   \
   typedef struct \
   {  OC_APPLY_CONFIG(TYPE)    desc; \
      OC_SOLID_TASK_ARGS(FUN)  args; \
   } OC_SOLID_FUN_PARAM(FUN);


#define OC_SOLID_FUN_PREAMBLE(FUN) \
   OC_SOLID_FUN_PARAM(FUN) __p, *__param=&__p; \
   OC_SOLID_TASK(FUN) *__task = NULL; \
   OC_SOLID_FUNPTR(FUN) __funptr = 0; \

#define OC_SOLID_FUN_PREPARE(FUN, TYPE, DTYPE, NAME_STR) \
   {  int __index; \
      \
      __index = OcSolid_getType(DTYPE); \
      if (__index >= 0) __funptr = OC_SOLID_FUNCTION(fill)[__index]; \
      if (__funptr == 0) OcError(-1,"Function " ## FUN ## " is not supported on CPU"); \

           if (scheduler)
           {  task = create task;
              task -> function = generic_function;
              param = &(task -> param);

              task -> data -> size[i] = size[i];
              task -> data -> strides[i] = ...;
           }

           /* Analyze */
           apply_elemwise1_cpu_analyze(tensor, &(param -> config));

           /* Set the parameters */
           param -> args -> ndims = ...
           param -> args -> size = (task ? param -> size : size);
           param -> function = OC_SOLID_FUNCTION(cpu_fill, tensor -> dtype) --> use lut, conversion OC_SDTYPE[TYPE]


#endif
