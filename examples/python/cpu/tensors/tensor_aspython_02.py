import pyOcean_cpu as ocean

for mode in ['F','C','R'] :
   print("\n========= Mode '%s' =========" % mode)
   t = ocean.asTensor([[[1,2,3],[4,5,6]]], mode, ocean.float)
   print(t)
   print(t.asPython(mode))

