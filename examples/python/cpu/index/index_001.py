import pyOcean_cpu as ocean

def fun(idx, size) :
   idx2 = idx.bind(size)
   print("Bind(%s, %s) = %s" % (idx,size,idx2))
   print("    input,output = (%s, %s) --> (%s, %s)" %
         (idx.inputSize, idx.outputSize, idx2.inputSize, idx2.outputSize))
   print("")

fun(ocean.index[1], [5])
fun(ocean.index[[1,2,3]], [6])
fun(ocean.index[[[True,False,True],[True,False,False]]],[3,2])
fun(ocean.index[None],[2])
fun(ocean.index[:],[8])
fun(ocean.index[...],[4,5])
fun(ocean.index[1:5],[7])
fun(ocean.index[-1::-1],[7])
fun(ocean.index[2,3:6,[True,False],[1,2]],[8,8,2,8])
