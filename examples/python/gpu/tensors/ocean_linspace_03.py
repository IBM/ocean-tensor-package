import ocean

print(ocean.linspace(0,10,5))
print(ocean.linspace(0,10,5,False))
print(ocean.linspace(1+2j,2+1j,5,ocean.chalf,ocean.gpu[0]))

(a,step) = ocean.linspace(0,2,5,True,True)

print(a)
print(step)

