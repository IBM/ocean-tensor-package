# -------------------------------------------------------------------------
# Copyright 2018, IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import sys
import os.path


# Syntax: generate <-c|-h|-a>  <base.src>  [<output-dir>]

# -----------------------------------------------------------------
# Open all files
# -----------------------------------------------------------------

# Filename of the form <base>.src
filetype     = sys.argv[1]
filenameSrc  = sys.argv[2]

# Check file type
if (filetype == '-h') :
   flagHeader = True
   flagSource = False
elif (filetype == '-c') :
   flagHeader = False
   flagSource = True
elif (filetype == '-a') :
   flagHeader = True
   flagSource = True
else :
   raise Exception("Invalid file type '%s'", filetype)

# Get the base form
path, filenameBase = os.path.split(filenameSrc)
filenameBase, ext  = os.path.splitext(filenameBase)

# Base directory
if (len(sys.argv) > 3) :
   basedir = sys.argv[3]
else :
   basedir = path
   
# Destination files
filenameDst1 = os.path.join(basedir,"%s.h" % filenameBase)
filenameDst2 = os.path.join(basedir,"%s.lut.c" % filenameBase)

# Open the files
fpSrc  = open(filenameSrc,'r')
if flagHeader :
   fpDst1 = open(filenameDst1,'w')
if flagSource :
   fpDst2 = open(filenameDst2,'w')


# -----------------------------------------------------------------
# Setup
# -----------------------------------------------------------------

# Define the FUNCTIONS and FUNCTIONS2 keywords
sFunction1 = 'FUNCTIONS('
nFunction1 = len(sFunction1)
sFunction2 = 'FUNCTIONS2('
nFunction2 = len(sFunction2)

# Available types
types = ['bool','uint8','uint16','uint32','uint64',
         'int8','int16','int32','int64','half',
         'float','double','chalf','cfloat','cdouble']

# Group definitions
groups = {'bool'    : [0],
          'uint8'   : [1],
          'uint16'  : [2],
          'uint32'  : [3],
          'uint64'  : [4],
          'int8'    : [5],
          'int16'   : [6],
          'int32'   : [7],
          'int64'   : [8],
          'float16' : [9],
          'float32' : [10],
          'float64' : [11],
          'cfloat16': [12],
          'cfloat32': [13],
          'cfloat64': [14],
          'uint'    : [1,2,3,4],
          'int'     : [5,6,7,8],
          'integer' : [1,2,3,4,5,6,7,8],
          'real'    : [9,10,11],
          'complex' : [12,13,14],
          'float'   : [9,10,11,12,13,14],
          'all'     : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]}

nGroups = max([max(groups[key]) for key in groups]) + 1
nTypes = len(types)
lTypes = max([len(t) for t in types])


# -----------------------------------------------------------------
# Normalized and continued line processing
# -----------------------------------------------------------------
def normalized_lines(f) :
   for line in f :
      line = line.rstrip('\n')
      while (line.endswith('\\')) :
         line = line[:-1] + next(f).rstrip('\n')
      yield line


# -----------------------------------------------------------------
# Process the input
# -----------------------------------------------------------------

# Helper functions
def parseGroupString(str) :
   mask = [0] * nTypes
   param1 = str.split('+')
   param2 = []
   for p in param1 :
      p = p.strip()
      allNegative = (p[0] == '-')
      p = p.split('-')
      for i in range(len(p)) :
         if ((i > 0) or allNegative) :
            value = 0
         else :
            value = 1
         for j in groups[p[i].strip()] :
            mask[j] = value
   return mask

def parseFunction(line,flagMask) :
   # FUNCTION(device, result, name, [, "param"]*, groups)
   # FUNCTION2(device, result, name, [, "param"]*)
   idx1 = line.find("(")
   idx2 = line.rfind(")")
   args = line[idx1+1:idx2].split('"')

   # Parameter and device
   param = args[0].split(',')
   device = param[0].strip()
   result = param[1].strip()
   name   = param[2].strip()

   # Group specification
   if (flagMask) :
      # This also works when the function has no parameters
      param = args[-1].split(',')
      mask = parseGroupString(param[-1]);
   else :
      mask = None

   # C arguments
   param = args[1:-1:2]
   args = []
   for i in range(len(param)) :
      str = param[i].rstrip()
      if (str[-1] == ',') :
         str = str[:-1]
      if (i < len(param)-1) :
         str = str + ","
      else :
         str = str + ");"
      args.append(str)

   # Return
   return (device, name, result, args, mask)


# Output the header declaration
if flagHeader :
   fpDst1.write("/* This file was automatically generated from %s */\n" % (filenameSrc))
   fpDst1.write("#ifndef __%s_H__\n" % filenameBase.upper())
   fpDst1.write("#define __%s_H__\n" % filenameBase.upper())
   fpDst1.write("\n");

def typedecl(fp, functions) :
   resultStr = []
   nameStr   = []
   argsList  = []
   for (ndims, device, name, result, args, mask) in functions :
      resultStr.append(result)
      nameStr.append("solid_funptr_%s_%s" % (device, name))
      argsList.append(args)

   resultLength = max([len(str) for str in resultStr])
   nameLength   = max([len(str) for str in nameStr])

   for i in range(len(resultStr)) :
      prefix = "typedef %*s (*%*s)(" % (-resultLength, resultStr[i], -nameLength, nameStr[i])
      prefixLength = len(prefix)
      args = argsList[i]
      if (len(args) == 0) :
         args = ["void);"]
      for k in range(len(args)) :
         if (k == 0) :
            fp.write("%s%s\n" % (prefix, args[0]))
         else :
            fp.write("%*s%s\n" % (prefixLength,"",args[k]))

# Process the source file
newFunctions = 0
functions = []
for line in normalized_lines(fpSrc):
    nLine = len(line)
    if ((nLine > nFunction1) and (line[:nFunction1] == sFunction1)) :
       (device, name, result, args, mask) = parseFunction(line, True)
       functions.append((1, device, name, result, args, mask))
       newFunctions += 1
    elif ((nLine > nFunction2) and (line[:nFunction2] == sFunction2)) :
       (device, name, result, args, mask) = parseFunction(line, False)
       functions.append((2, device, name, result, args, mask))
       newFunctions += 1

    if flagHeader :
       if newFunctions :
          # Output all new type declarations at once
          typedecl(fpDst1, functions[len(functions)-newFunctions:])
          newFunctions = 0
       else :
          # Output original line
          fpDst1.write("%s\n" % line)
       

if flagHeader :
   # Add the look-up table declarations
   fpDst1.write("/* -------------------------------------------------------------------- */\n")
   fpDst1.write("/* Look-up-table declarations                                           */\n")
   fpDst1.write("/* -------------------------------------------------------------------- */\n")

   # Add the function declarations
   width = 15 + max([len(device) + len(name) for (ndims,device,name,result,args,mask) in functions])
   for (ndims,device,name,result,args,mask) in functions :
       if ((ndims == 1) and (not any(mask))) :
           continue
       funptr = "solid_funptr_%s_%s" % (device,name);
       fpDst1.write("extern %*s solid_%s_%s%s;\n" % (-width, funptr, device, name, ("[%d]"%nTypes)*ndims))

   fpDst1.write("\n")
   fpDst1.write("\n")
   fpDst1.write("/* -------------------------------------------------------------------- */\n")
   fpDst1.write("/* Function declarations                                                */\n")
   fpDst1.write("/* -------------------------------------------------------------------- */\n")
   fpDst1.write("\n");
   fpDst1.write("#if __cplusplus\n");
   fpDst1.write("extern \"C\" {\n");
   fpDst1.write("#endif\n");

   def fundecl(fp, prefix, prefixLength, args) :
      if (len(args) == 0) :
         args = ["void);"]
      for k in range(len(args)) :
         if (k == 0) :
            fp.write("%*s(%s\n" % (-prefixLength+1, prefix, args[0]))
         else :
            fp.write("%*s%s\n" % (-prefixLength, "", args[k]))

   for (ndims,device,name,result,args,mask) in functions :
      if ((ndims == 1) and (not any(mask))) :
         continue

      fpDst1.write("\n")
      fpDst1.write("/* Function declarations for solid_%s_%s */\n" % (device,name))
      if (ndims == 1) :
         typeLength = max([len(types[i]) for i in range(nTypes) if (mask[i]==1)])
         prefixLength = 10 + len(result) + len(device) + len(name) + typeLength
         for i in range(nTypes) :
            if (mask[i] == 0) :
                continue
            prefix = "%s solid_%s_%s_%s" % (result, device, name, types[i])
            fundecl(fpDst1, prefix, prefixLength, args)
      else :
         prefixLength = 11 + len(result) + len(device) + len(name) + lTypes * 2
         for i in range(nTypes) :
            for j in range(nTypes) :
               prefix = "%s solid_%s_%s_%s_%s" % (result, device, name, types[i], types[j])
               fundecl(fpDst1, prefix, prefixLength, args)

   fpDst1.write("\n")
   fpDst1.write("#if __cplusplus\n");
   fpDst1.write("}\n");
   fpDst1.write("#endif\n");
   fpDst1.write("\n")
   fpDst1.write("\n")
   fpDst1.write("/* -------------------------------------------------------------------- */\n")
   fpDst1.write("/* Parameter structures                                                 */\n")
   fpDst1.write("/* -------------------------------------------------------------------- */\n")

   for (ndims,device,name,result,args,mask) in functions :
      if (len(args) == 0) :
         continue

      fpDst1.write("\n")
      fpDst1.write("/* Parameter structure for solid_%s_%s */\n" % (device,name))

      args = "".join(args)
      args = args[:-2]
      args = args.split(",")

      fields    = []
      ptrwidth  = 0
      typewidth = 0
      for arg in args :
         arg = arg.strip()
         if ((len(arg) > 6) and (arg[:6] == "const ")) :
            arg = arg[6:].strip()
         idx  = arg.rfind(" ")
         idx2 = arg.find("*")
         if (idx2 > -1) :
            idx = idx2-1;
      
         argtype = arg[:idx+1].strip()
         argname = arg[idx+1:]
         argptr  = argname.count('*')
         argname = argname.replace('*','')
         argname = argname.strip()

         typewidth = max(typewidth, len(argtype))
         ptrwidth  = max(ptrwidth,argptr)
      
         fields.append((argtype, argptr, argname))

      # Output the structure definition
      fpDst1.write("typedef struct\n")
      prefix = "{  "
      for (argtype, argptr, argname) in fields :
         fpDst1.write("%s%*s %*s%s;\n" % (prefix, -typewidth, argtype, ptrwidth, '*'*argptr, argname))
         prefix = "   "
      fpDst1.write("} solid_param_%s_%s;\n" % (device, name));


   fpDst1.write("\n")
   fpDst1.write("\n")
   fpDst1.write("/* -------------------------------------------------------------------- */\n")
   fpDst1.write("/* Function macros                                                      */\n")
   fpDst1.write("/* -------------------------------------------------------------------- */\n")
   fpDst1.write("\n")

   macroWidth = 0
   for (ndims,device,name,result,args,mask) in functions :
      macroWidth = max(macroWidth,len(device)+len(name))
   macroWidth += 35 + 8

   for (ndims,device,name,result,args,mask) in functions :
      if (len(args) == 0) :
         continue

      nargs = len(args)
      for i in range(nargs) :
         if (i == 0) :
            fpDst1.write("%*s FUNPTR(" % (-macroWidth+8, "#define solid_macro_%s_%s(FUNPTR,PARAM)" % (device,name)))
         else :
            fpDst1.write("%*s" % (macroWidth,""))

         # Get the argument names
         largs = args[i]
         if (i == nargs -1) :
            largs = largs[:-2] # Trailing ');'
         elif (largs[-1] == ',') :
            largs = largs[:-1] # Trailing ','
         largs = largs.split(",")

         for j in range(len(largs)) :
            s   = largs[j].strip()
            idx = max(s.rfind(' '), s.rfind('*'))
            fpDst1.write("(PARAM)->%s" % s[idx+1:])
            if (j < len(largs)-1) :
               fpDst1.write(", ")

         if (i == nargs - 1) :
            fpDst1.write(")\n");
         else :
            fpDst1.write(",\\\n");

   fpDst1.write("\n");
   fpDst1.write("#endif\n");


if flagSource :
   # Output the look-up tables
   fpDst2.write("#include \"%s.h\"\n" % (filenameBase))
   for (ndims,device,name,result,args,mask) in functions :
       if ((ndims == 1) and (not any(mask))) :
           continue

       postfix = ','
       fpDst2.write("\n")
       fpDst2.write("/* Lookup table for solid_%s_%s */\n" % (device,name))
       fpDst2.write("solid_funptr_%s_%s solid_%s_%s%s = {\n" % (device,name,device,name,("[%d]" % nTypes)*ndims))
       if (ndims == 1) :
          for i in range(nTypes) :
             if (i == nTypes - 1) :
                postfix = ""
        
             if (mask[i] == 0) :
                fpDst2.write("   0%s\n" % postfix)
             else :
                fpDst2.write("   solid_%s_%s_%s%s\n" % (device,name,types[i],postfix))
       else :
          for i in range(nTypes) :
             for j in range(nTypes) :
                if (j == 0) :
                   prefix = "  {"
                else :
                   prefix = "   "
                if (j == nTypes-1) :
                   postfix = ""
                else :
                   postfix = ","
                fpDst2.write("%ssolid_%s_%s_%s_%s%s\n" % (prefix,device,name,types[i],types[j],postfix))
             if (i == nTypes-1) :
                fpDst2.write("  }\n")
             else :
                fpDst2.write("  },\n")
       fpDst2.write("};\n")


# -----------------------------------------------------------------
# Close all files
# -----------------------------------------------------------------

fpSrc.close()
if flagHeader :
   fpDst1.close()
if flagSource :
   fpDst2.close()
