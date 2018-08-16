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

#ifndef __SOLID_ERROR_H__
#define __SOLID_ERROR_H__

#include <stdio.h>

#define SOLID_ERRMSG_SIZE 512

extern char *solid_errmsg;


/* SOLID_ERROR_MESSAGE sets the error message */
#define SOLID_ERROR_MESSAGE(...) snprintf(solid_errmsg, SOLID_ERRMSG_SIZE, __VA_ARGS__)

/* SOLID_ERROR sets the error message and returns the error code */
#define SOLID_ERROR(CODE, ...) \
   do \
   {  SOLID_ERROR_MESSAGE(__VA_ARGS__); \
      return (CODE); \
   } while(0)

#endif
