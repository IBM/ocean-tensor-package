import pyOcean_cpu as ocean

a = [-20,20,-20,20]
b = [3,3,-3,-3]
c = [1,2,-2,-1]
print(ocean.mod(ocean.int64(a),ocean.int64(b)))
print(ocean.mod(ocean.float(a),ocean.float(b)))
print(ocean.mod(ocean.double(a),ocean.double(b)))
print(ocean.int64(c))
print("")

c = [-2,2,-2,2]
print(ocean.fmod(ocean.int64(a), ocean.int64(b)))
print(ocean.fmod(ocean.float(a),ocean.float(b)))
print(ocean.fmod(ocean.double(a),ocean.double(b)))
print(ocean.int64(c))


