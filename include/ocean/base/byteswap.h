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

#ifndef __OC_BYTESWAP_H__
#define __OC_BYTESWAP_H__

#define OC_BYTESWAP_8(A,B) \
   {  B = A; \
   }

/* Byte-swap variables A and B of type uint16_t */
#define OC_BYTESWAP_16(A,B) \
   {  B = (((A & 0x00FF) << 8) | \
           ((A & 0xFF00) >> 8));  \
   }

/* Byte-swap variables A and B of type uint32_t */
#define OC_BYTESWAP_32(A,B) \
   {  B = ((((A) & 0x000000FF) << 24) | \
           (((A) & 0x0000FF00) <<  8) | \
           (((A) & 0x00FF0000) >>  8) | \
           (((A) & 0xFF000000) >> 24)); \
   }

/* Byte-swap variables A and B of type uint64_t */
#define OC_BYTESWAP_64(A,B) \
   {  B = ((((A) & 0x00000000000000FF) << 56) | \
           (((A) & 0x000000000000FF00) << 40) | \
           (((A) & 0x0000000000FF0000) << 24) | \
           (((A) & 0x00000000FF000000) <<  8) | \
           (((A) & 0x000000FF00000000) >>  8) | \
           (((A) & 0x0000FF0000000000) >> 24) | \
           (((A) & 0x00FF000000000000) >> 40) | \
           (((A) & 0xFF00000000000000) >> 56)); \
   }

#endif
