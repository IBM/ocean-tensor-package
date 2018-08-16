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

#include "ocean/core/cpu/device_cpu.h"
#include "ocean/base/device.h"
#include "ocean/base/malloc.h"
#include "ocean/base/types.h"
#include "ocean/base/error.h"
#include "ocean.h"

#include <string.h>
#include <stdlib.h>


/* ===================================================================== */
/* Global CPU device                                                     */
/* ===================================================================== */

OcDevice *OcCPU = NULL;



/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

static OcStream *OcDeviceCPU_createStream(OcDevice *device);
static void      OcDeviceCPU_deleteStream(OcStream *stream);


/* ===================================================================== */
/* Functions for device creation and deletion                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcFreeDeviceCPU(OcDevice *self)
/* -------------------------------------------------------------------- */
{  OcDeviceCPU *device = (OcDeviceCPU *)self;
   OcStream    *stream;

   /* Free all buffered streams */
   while ((stream = device -> streamBuffer) != NULL)
   {  device -> streamBuffer = stream -> next;
      OcFree(stream);
   }
}


/* -------------------------------------------------------------------- */
OcDevice *OcCreateDeviceCPU(OcDeviceType *type, int index, const char *name)
/* -------------------------------------------------------------------- */
{  OcDeviceCPU *deviceCPU;
   OcDevice    *device;
   OcStream    *stream;
   uint16_t     v = 0x0001;

   /* Create the device structure */
   device    = OcCreateDevice(type, index, name, sizeof(OcDeviceCPU));
   deviceCPU = (OcDeviceCPU *)device;

   /* Initialize the device */
   if (device != NULL)
   {
      /* CPU device-specific initialization */
      device -> finalize = OcFreeDeviceCPU;

      /* Endianness */
      device -> endianness = (*((uint8_t *)&v) == 0x01) ? 0 : 1;
      device -> requiresAlignedData = 0;

      /* Stream and event related functions */
      device -> create_stream = OcDeviceCPU_createStream;
      device -> create_event  = 0;
      device -> delete_stream = OcDeviceCPU_deleteStream;
      device -> delete_event  = 0;
      device -> sync_stream   = 0;
      device -> sync_event    = 0;
      device -> record_event  = 0;
      device -> wait_event    = 0;

      /* Buffering of streams */
      deviceCPU -> streamBuffer = NULL;
      deviceCPU -> streamCount  = 0;

      /* Create the default stream */
      stream = OcDeviceCPU_createStream(device);
      OcDevice_setDefaultStream(device, stream);
      OcDecrefStream(stream);
   }

   return (OcDevice *)device;
}


/* -------------------------------------------------------------------- */
void OcFinalizeDeviceCPU(void)
/* -------------------------------------------------------------------- */
{
   if (OcCPU != NULL)
   {  OcDecrefDevice(OcCPU);
      OcCPU = NULL;
   }
}


/* -------------------------------------------------------------------- */
int OcInitDevicesCPU(void)
/* -------------------------------------------------------------------- */
{  OcDeviceType *type;
   OcDevice *device;

   /* Create the CPU device type */
   type = OcCreateDeviceType("CPU");
   if (type == NULL) return -1;

   /* Create and register the CPU device */
   device = OcCreateDeviceCPU(type, 0, "cpu");
   if ((device == NULL) || (OcRegisterDevice(device) != 0))
   {  return -1;
   }

   /* Set the global CPU device */
   OcCPU = OcIncrefDevice(device);

   return OcFinalizeAddHandler(OcFinalizeDeviceCPU, "Finalize device CPU");
}



/* ===================================================================== */
/* Stream functions                                                      */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcStream *OcDeviceCPU_createStream(OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcDeviceCPU *cpu = (OcDeviceCPU *)device;
   OcStream *stream;

   if (cpu -> streamCount > 0)
   {  /* Recycle a stream */
      stream = cpu -> streamBuffer;
      cpu -> streamBuffer = stream -> next;
      cpu -> streamCount --;
   }
   else
   {  /* Create a new stream */
      stream = (OcStream *)OcMalloc(sizeof(OcStream));
      if (stream == NULL) OcError(NULL, "Error allocating CPU stream");
   }

   /* Initialize the stream */
   stream -> device   = OcIncrefDevice(device);
   stream -> refcount = 1;

   return stream;
}


/* -------------------------------------------------------------------- */
void OcDeviceCPU_deleteStream(OcStream *stream)
/* -------------------------------------------------------------------- */
{  OcDeviceCPU *device;

   if (stream)
   {
      /* Get the stream device */
      device = (OcDeviceCPU *)(stream -> device);

      /* Check if we want to recycle the stream (maximum of 64) */
      if (device -> streamCount < 64)
      {  stream -> next = device -> streamBuffer;
         device -> streamBuffer = stream;
         device -> streamCount ++;

         /* Decrement the device reference count to make sure that  */
         /* devices can be freed while containing buffered streams. */
         OcDecrefDevice(stream -> device);
      }
      else
      {  /* Decrement the device reference count */
         OcDecrefDevice(stream -> device);

         /* Delete the stream object */
         OcFree(stream);
      }
   }
}
