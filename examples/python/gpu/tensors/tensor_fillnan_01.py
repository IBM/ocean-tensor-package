import ocean
a = ocean.asTensor([ocean.nan,3,4,ocean.nan], ocean.float, ocean.gpu[0])
a.fillNaN(ocean.inf)
print(a)

