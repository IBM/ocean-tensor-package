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

/*          Ocean name      Solid name      Check  Description              */
/*          --------------  --------------  -----  ------------------------ */
OC_TEMPLATE(add,            add,            0,     "add"                     )
OC_TEMPLATE(subtract,       subtract,       0,     "subtract"                )
OC_TEMPLATE(scale,          multiply,       0,     "elementwise multiply"    )
OC_TEMPLATE(divide,         divide,         0,     "elementwise divide"      )
OC_TEMPLATE(trueDivide,     true_divide,    0,     "elementwise true-divide" )
OC_TEMPLATE(floorDivide,    floor_divide,   0,     "elementwise floor-divide")
OC_TEMPLATE(power,          power,          1,     "power"                   )
OC_TEMPLATE(mod,            mod,            0,     "mod"                     )
OC_TEMPLATE(fmod,           fmod,           0,     "fmod"                    )

OC_TEMPLATE(elemwiseMin,    min,            0,     "pairwise minimum"        )
OC_TEMPLATE(elemwiseMax,    max,            0,     "pairwise maximum"        )
OC_TEMPLATE(elemwiseFMin,   fmin,           0,     "pairwise minimum"        )
OC_TEMPLATE(elemwiseFMax,   fmax,           0,     "pairwise maximum"        )
OC_TEMPLATE(elemwiseLT,     lt,             0,     "less-than"               )
OC_TEMPLATE(elemwiseLE,     le,             0,     "less-equal"              )
OC_TEMPLATE(elemwiseEQ,     eq,             0,     "equal"                   )
OC_TEMPLATE(elemwiseNE,     ne,             0,     "not-equal"               )
OC_TEMPLATE(elemwiseGE,     ge,             0,     "greater-equal"           )
OC_TEMPLATE(elemwiseGT,     gt,             0,     "greater-than"            )

OC_TEMPLATE(bitwiseAnd,     bitwise_and,    0,     "bitwise AND"             )
OC_TEMPLATE(bitwiseOr,      bitwise_or,     0,     "bitwise OR"              )
OC_TEMPLATE(bitwiseXor,     bitwise_xor,    0,     "bitwise XOR"             )
OC_TEMPLATE(logicalAnd,     logical_and,    0,     "logical AND"             )
OC_TEMPLATE(logicalOr,      logical_or,     0,     "logical OR"              )
OC_TEMPLATE(logicalXor,     logical_xor,    0,     "logical XOR"             )
OC_TEMPLATE(bitshiftLeft,   bitshift_left,  0,     "bitshift left"           )
OC_TEMPLATE(bitshiftRight,  bitshift_right, 0,     "bitshift right"          )
