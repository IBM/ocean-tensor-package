import ocean

for dtype in [ocean.int16, ocean.float, ocean.chalf] :
   a = ocean.tensor([],dtype,ocean.gpu[0])
   a.fill(24)
   print(a)

   a.fill(36)
   print(a)


