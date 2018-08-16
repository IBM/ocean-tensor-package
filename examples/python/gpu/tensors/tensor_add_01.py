import ocean

device = ocean.gpu[0]

A = ocean.zeros([100],ocean.float,device)
B = ocean.tensor(A.storage,0,[15])
B.copy(range(15))

C = ocean.tensor(A.storage,0,[5,3],[3,1]);

print(A)
print(B)
print(C)

print("========= Evaluating C += 100 =========")
C += 100

print(C)
print(A)

