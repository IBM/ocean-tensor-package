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

/*          Ocean name   Solid name    Check  Description                */
/*          -----------  ------------  ------ -------------------------- */
OC_TEMPLATE(negative,    negative,     0,     "unary negative"            )
OC_TEMPLATE(bitwiseNot,  bitwise_not,  0,     "bitwise NOT"               )
OC_TEMPLATE(logicalNot,  logical_not,  0,     "logical NOT"               )
OC_TEMPLATE(conj,        conj,         0,     "conjugate"                 )
OC_TEMPLATE(reciprocal,  reciprocal,   0,     "reciprocal"                )

OC_TEMPLATE(sin,         sin,          0,     "sine"                      )
OC_TEMPLATE(cos,         cos,          0,     "cosine"                    )
OC_TEMPLATE(tan,         tan,          0,     "tangent"                   )
OC_TEMPLATE(sinh,        sinh,         0,     "hyperbolic sine"           )
OC_TEMPLATE(cosh,        cosh,         0,     "hyperbolic cosine"         )
OC_TEMPLATE(tanh,        tanh,         0,     "hyperbolic tangent"        )
OC_TEMPLATE(arcsin,      arcsin,       1,     "inverse sine"              )
OC_TEMPLATE(arccos,      arccos,       1,     "inverse cosine"            )
OC_TEMPLATE(arctan,      arctan,       0,     "inverse tangent"           )
OC_TEMPLATE(arcsinh,     arcsinh,      0,     "inverse hyperbolic sine"   )
OC_TEMPLATE(arccosh,     arccosh,      1,     "inverse hyperbolic cosine" )
OC_TEMPLATE(arctanh,     arctanh,      1,     "inverse hyperbolic tangent")

OC_TEMPLATE(sqrt,        sqrt,         1,     "square root"               )
OC_TEMPLATE(cbrt,        cbrt,         0,     "cube root"                 )
OC_TEMPLATE(square,      square,       0,     "square"                    )
OC_TEMPLATE(exp,         exp,          0,     "exp"                       )
OC_TEMPLATE(exp2,        exp2,         0,     "exp2"                      )
OC_TEMPLATE(exp10,       exp10,        0,     "exp10"                     )
OC_TEMPLATE(expm1,       expm1,        0,     "expm1"                     )
OC_TEMPLATE(log,         log,          1,     "log"                       )
OC_TEMPLATE(log2,        log2,         1,     "log2"                      )
OC_TEMPLATE(log10,       log10,        1,     "log10"                     )
OC_TEMPLATE(log1p,       log1p,        1,     "log1p"                     )

OC_TEMPLATE(absolute,    absolute,     0,     "absolute"                  )
OC_TEMPLATE(fabs,        fabs,         0,     "fabs"                      )
OC_TEMPLATE(sign,        sign,         0,     "sign"                      )
OC_TEMPLATE(ceil,        ceil,         0,     "ceil"                      )
OC_TEMPLATE(floor,       floor,        0,     "floor"                     )
OC_TEMPLATE(trunc,       trunc,        0,     "truncate"                  )
OC_TEMPLATE(round,       round,        0,     "round"                     )

OC_TEMPLATE(isinf,       isinf,        0,     "isinf"                     )
OC_TEMPLATE(isnan,       isnan,        0,     "isnan"                     )
OC_TEMPLATE(isfinite,    isfinite,     0,     "isfinite"                  )
OC_TEMPLATE(isposinf,    isposinf,     0,     "isposinf"                  )
OC_TEMPLATE(isneginf,    isneginf,     0,     "isneginf"                  )
