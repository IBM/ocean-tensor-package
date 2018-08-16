import pyOcean_cpu as ocean

def checkDType(dtype) :
   nError = 0
   for transA in ['N','T','C'] :
      if (transA == 'N') :
         sizeA = [2,3]
      else :
         sizeA = [3,2]

      for transB in ['N','T','C'] :
         if (transB == 'N') :
            sizeB = [3,4]
         else :
            sizeB = [4,3]

         for orderA in ['F','C'] :
            A = ocean.tensor(sizeA, orderA, dtype)
            A.copy(ocean.arange(sizeA[0]*sizeA[1], dtype).reshape(sizeA))
            An = ocean.tensor(sizeA, dtype)
            An.copy(A)

            for orderB in ['F','C'] :
               B = ocean.tensor(sizeB, orderB, dtype)
               B.copy(ocean.arange(sizeB[0]*sizeB[1], dtype).reshape(sizeB))
               Bn = ocean.tensor(sizeB, dtype)
               Bn.copy(B)

               for orderC in ['F','C'] :
                  C = ocean.tensor([2,4], orderC, dtype)
                  print("Setting: transA='%s', transB='%s', orderA='%s', orderB='%s', orderC='%s'\n" %
                        (transA, transB, orderA, orderB, orderC))

                  ocean.gemm(1, A, transA, B, transB, 0, C)
                  Cn = ocean.gemm(1, An, transA, Bn, transB)
                  if (not ocean.all(C == Cn)) :
                     expectedStr = str(C).replace('\n','\n         ');
                     obtainedStr = str(Cn).replace('\n','\n         ');
                     print("Expected : %s" % expectedStr)
                     print("Obtained : %s" % obtainedStr)
                     nError += 1
                  else :
                     expectedStr = str(C).replace('\n','\n         ');
                     print("Success  : %s" % expectedStr)

   if (nError == 0) :
      print("All checks passed for data type %s" % dtype.name)


checkDType(ocean.double)
