## Copy an empty tensor from gpu to cpu
import ocean

a = ocean.tensor([0],ocean.gpu[0])
print(a)

