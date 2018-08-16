import pyOcean_cpu as ocean

def fun(idx) :
   print("Index: %s" % str(idx))
   print("Scalar = %s" % str(idx.isScalar()))
   print("View   = %s" % str(idx.isView()))
   print("Bound  = %s" % str(idx.isBound()))
   print("")


fun(ocean.index[3,4])
fun(ocean.index[:,2])
fun(ocean.index[:,...,3])


idx = ocean.index[1,...,3]
fun(idx)
idx.bind([5,6],True)
fun(idx)
idx = idx.bind([5,6],[0,8])
fun(idx)


idx = ocean.index[[1,2,3]]
fun(idx)
