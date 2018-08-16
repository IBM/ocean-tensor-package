import pyOcean_cpu as ocean
a = ocean.asTensor([ocean.nan,3,4,ocean.nan], ocean.float)
a.fillNaN(ocean.inf)
print(a)

