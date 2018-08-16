import pyOcean_cpu as ocean

def check(reference, size, strides, elemsize=1) :
   overlap = ocean.checkSelfOverlap(size, strides, elemsize)
   if (overlap == reference) :
      s = ''
   else :
      if (reference and not overlap) :
         s = '*** Incorrect -- false negative ***'
      else :
         s = '*** Incorrect ***'
   print("%-5s %-5s %s" % (reference, overlap, s))

check(True,  [3,2,2,2], [1,2,6,12])
check(True,  [3,2], [1,2])
check(False, [4,3], [4,5])
check(True,  [4,3], [4,5], 2)


