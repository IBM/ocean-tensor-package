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

#include "ocean/core/gpu/device_gpu.h"
#include "ocean/core/gpu/device_info.h"
#include "ocean.h"

#include <string.h>
#include <stdlib.h>


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

static int OcDeviceGPU_formatInfo(OcDevice *device, char **str, const char *header, const char *footer);


/* ===================================================================== */
/* Function implemenations                                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcRegisterDevicesGPU(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set the core module function pointers */
   module -> Device_formatInfo = OcDeviceGPU_formatInfo;

   return 0;
}


/* -------------------------------------------------------------------- */
static const char *OcDeviceGPU_formatSupported(int value)
/* -------------------------------------------------------------------- */
{  if (value < 0) return "---";
   return ((value) ? "Supported" : "Not supported");
}


/* -------------------------------------------------------------------- */
static const char *OcDeviceGPU_formatEnabled(int value)
/* -------------------------------------------------------------------- */
{  if (value < 0) return "---";
   return ((value) ? "Enabled" : "Disabled");
}


/* -------------------------------------------------------------------- */
static const char *OcDeviceGPU_formatYes(int value)
/* -------------------------------------------------------------------- */
{  if (value < 0) return "---";
   return ((value) ? "Yes" : "No");
}


/* -------------------------------------------------------------------- */
static int OcDeviceGPU_formatInfo(OcDevice *device, char **str,
                                  const char *header, const char *footer)
/* -------------------------------------------------------------------- */
{  OcDevicePropGPU *properties;
   int     newlineWidth;
   int     devicePeerCount = 0;
   char   *s = NULL, *p, *buffer = NULL;
   size_t  slen;
   int     k, c, mode;
   int     i, j, jMax;

   /* Update the device properties */
   if (OcDeviceGPU_updateProperties((OcDeviceGPU *)device) != 0) return -1;
   
   /* Get the device properties */
   properties = &(((OcDeviceGPU *)device) -> properties);

   /* Column width */
   newlineWidth = strlen("\n");
   c = 46;

   for (mode = 0; mode < 2; mode ++)
   {  slen = 0;

      /* Header */
      if (header != NULL)
      {  k = strlen(header); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", header);
      }

      /* Device */
      k  = 10 + strlen(device -> name) + newlineWidth;
      k += strlen(properties -> name); slen += k;
      if (mode == 1) s += snprintf(s, k+1, "Device %s - %s\n",
                                   device -> name,
                                   properties -> name);

      /* Add a separator */
      k = 2*c + newlineWidth; slen += k;
      if (mode == 1)
      {  for (i = 0; i < 2*c; i++, s++) *s = '-';
         s += snprintf(s, newlineWidth+1, "\n");
      }


      /* Device properties */
      jMax = 40;
      for (j = 0; j < jMax; j++)
      {
         p = s;
         i = (j % 2 == 0) ? j / 2 : ((jMax+1)/2) + j / 2;

         switch(i)
         {
            case 0: /* Clock rate */
               k = 22 + 9; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %.1fMHz", "Clockrate",
                                            properties -> clockRate / 1000.0);
               break;

            case 1: /* Memory clock rate */
               k = 22 + 9; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %.1fMHz", "Memory clockrate",
                                            properties -> memoryClockRate / 1000.0);
               break;

            case 2: /* Total global memory */
               k = 22 + 9 + 3 + 9; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %.0fMb (%.0fMb)", "Total global memory",
                                            (properties -> totalGlobalMem) / (1024. * 1024.),
                                            (properties -> freeGlobalMem) / (1024. * 1024.));
               break;

            case 3: /* Total constant memory */
               k = 22 + 9; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %.0fkb", "Total const memory",
                                            (properties -> totalConstMem) / (1024.));
               break;

            case 4: /* L2 Cache size */
               k = 22 + 8; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %.1fMb", "L2 Cache size",
                                            properties -> l2CacheSize / (1024. * 1024.));
               break;


            case 5: /* Global L1 cache supported */
               k = 22 + 8; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Global caching L1",
                                            OcDeviceGPU_formatEnabled(properties -> globalL1CacheSupported));
               break;

            case 6: /* Local L1 cache supported */
               k = 22 + 8; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Local caching L1",
                                            OcDeviceGPU_formatEnabled(properties -> localL1CacheSupported));
               break;


            case 7: /* Can map host memory */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Can map host memory",
                                            properties -> canMapHostMemory ? "Yes" : "No");
               break;

            case 8: /* Maximum copy mempitch */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %"OC_FORMAT_LU"", "Max. copy mempitch",
                                            (unsigned long)(properties -> memPitch));
               break;

            case 9: /* Memory bus width */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Memory bus width",
                                            properties -> memoryBusWidth);
               break;


            case 10: /* Unified memory addressing */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Unified addressing",
                                            properties -> unifiedAddressing ? "Yes" : "No");
               break;

            case 11: /* Managed memory */
               k = 22 + 13; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Managed memory",
                                            OcDeviceGPU_formatSupported(properties -> managedMemSupported));
               break;

            case 12: /* Pageable memory access */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Pageable memory",
                                            OcDeviceGPU_formatYes(properties -> pageableMemoryAccess));
               break;

            case 13 : /* ECC enabled */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "ECC enabled",
                                            properties -> ECCEnabled ? "Yes" : "No");
               break;

            case 14: /* TCC driver */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "TCC driver",
                                            properties -> tccDriver ? "Yes" : "No");
               break;

            case 15: /* Integrated */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Integrated",
                                            properties -> integrated ? "Yes" : "No");
               break;

            case 16: /* Multi-GPU board */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Multi-GPU board",
                                            OcDeviceGPU_formatYes(properties -> isMultiGpuBoard));
               break;

            case 17: /* Multi-GPU board ID */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Multi-GPU group ID",
                                            properties -> multiGpuBoardGroupID);
               break;

            case 18 : /* Concurrent managed access */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Concurrent managed",
                                            OcDeviceGPU_formatYes(properties -> concurrentManagedAccess));
               break;

            case 19: /* Host native atomic supported */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Host native atomic",
                                            OcDeviceGPU_formatYes(properties -> hostNativeAtomicSupported));
               break;


            /* ------------------------------------------------------------ */

            case 20: /* Multi-processor count */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Multi-processors",
                                            properties -> multiProcessorCount);
               break;


            case 21: /* Major and minor compute capability */
               k = 22 + 2*6; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d.%d", "Compute capability",
                                            properties -> major, properties -> minor);
               break;

            case 22: /* Maximum grid size */
               k = 22 + 3*12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d, %d, %d", "Maximum grid size",
                                            properties -> maxGridSize[0],
                                            properties -> maxGridSize[1],
                                            properties -> maxGridSize[2]);
               break;

            case 23: /* Maximum thread dimensions - block size */
               k = 22 + 3*12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d, %d, %d", "Maximum block size",
                                            properties -> maxThreadsDim[0],
                                            properties -> maxThreadsDim[1],
                                            properties -> maxThreadsDim[2]);
               break;

            case 24: /* Maximum threads per block */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Threads per block",
                                            properties -> maxThreadsPerBlock);
               break;

            case 25: /* Warp size in threads */
               k = 22 + 6; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Warp size",
                                            properties -> warpSize);
               break;

            case 26: /* Maximum resident threads per multiprocessor */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Max resident threads",
                                            properties -> maxThreadsPerMultiProcessor);
               break;

            case 27: /* Maximum resident blocks per multiprocessor */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Max resident blocks",
                                            properties -> maxBlocksPerMultiProcessor);
               break;

            case 28: /* Maximum resident warps per multiprocessor */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Max resident warps",
                                            properties -> maxWarpsPerMultiProcessor);
               break;

            case 29: /* Shared memory per multiprocessor */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %.0fkb", "Shared mem per mproc",
                                            properties -> sharedMemPerMultiprocessor / 1024.);
               break;

            case 30: /* Shared memory per block */
               k = 22 + 12; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %.0fkb", "Shared mem per block",
                                            properties -> sharedMemPerBlock / 1024.);
               break;

            case 31: /* Registers per block */
               k = 22 + 6; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Registers per block",
                                            properties -> regsPerBlock);
               break;

            case 32: /* Registers per multi-processor */
               k = 22 + 6; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %d", "Registers per mproc",
                                            properties -> regsPerMultiprocessor);
               break;

            case 33: /* Compute mode */
               k = 22 + 2; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %-2d", "Compute mode",
                                            properties -> computeMode);
               break;

            case 34: /* Asynchronous engine count */
               k = 22 + 6; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %-6d", "Async. engine count",
                                            properties -> asyncEngineCount);
               break;

            case 35: /* Concurrent kernels */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Concurrent kernels",
                                            properties -> concurrentKernels ? "Yes" : "No");
               break;

            case 36: /* Kernel execution time-out enabled */
               k = 22 + 8; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Kernel time-out",
                                            properties -> kernelExecTimeoutEnabled ? "Enabled" : "Disabled");
               break;

            case 37: /* Stream priorities */
               k = 22 + 3; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %s", "Stream priorities",
                                            properties -> streamPrioritiesSupported ? "Yes" : "No");
               break;

            case 38: /* PCI */
               k = 22 + 3*8; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %04X:%04X:%02X", "PCI",
                                            properties -> pciDomainID,
                                            properties -> pciBusID,
                                            properties -> pciDeviceID);
               break;

            case 39: /* Single-double performance ratio */
               k = 22 + 6; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%-20s: %-6d", "Single-double perf.",
                                            properties -> singleToDoublePrecisionPerfRatio);
               break;


            default :
               k = 0;
         }

         if ((j % 2) == 0)
         {  if (k < c) slen += (c-k);
            if (mode == 1)
            {  k = (long int)(s) - (long int)(p);
               if (k < c) s += snprintf(s, (k-c)+1, "%*s", k-c, "");
            }
         }

         if (((j % 2) == 1) || (j+1 == jMax))
         {  slen += newlineWidth;
            if (mode == 1) s += snprintf(s, newlineWidth+1, "\n");
         }
      }

      /* List peer devices */
      k = 14 + newlineWidth; slen += k;
      if (mode == 1) s += snprintf(s, k+1, "\nPeer devices: ");

      if (mode == 0)
      {  for (i = 0; i < OcDeviceGPUCount(); i++)
         {  if (i == device -> index) continue;
            if (OcDeviceGPU_peerAccess(device -> index, i)) devicePeerCount ++;
         }
      }

      if (devicePeerCount == 0)
      {  k = 4; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "None");
      }
      else
      {  k = 9 * devicePeerCount + 2 * (devicePeerCount - 1); slen += k;

         if (mode == 1)
         {  for (i = 0; i < OcDeviceGPUCount(); i++)
            {  if (i == device -> index) continue;
               if (OcDeviceGPU_peerAccess(device -> index, i))
               {  s += snprintf(s, 9+1, "gpu%d", i);
                  if ((--devicePeerCount) > 0) s += snprintf(s, 3, ", ");
               }
            }
         }
      }

      /* Footer */
      if (footer != NULL)
      {  k = strlen(footer); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", footer);
      }

      /* Allocate memory for the string */
      if (mode == 0)
      {
         /* ------------------------------------------------------------- */
         /* Allocate the memory for the string. We use a regular malloc   */
         /* here instead of OcMalloc to ensure that the library can be    */
         /* recompiled with new memory allocation routines without having */
         /* to recompile any language bindings.                           */
         /* ------------------------------------------------------------- */
         buffer = (char *)malloc(sizeof(char) * (slen + 1));
         s = buffer; *str = buffer;
         if (buffer == NULL) OcError(-1, "Insufficient memory for output string");
      }
   }  
   
   /* Ensure that the string is terminated properly */
   *s = '\0';

   return 0;
}
