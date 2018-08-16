import ocean

def run(cmd) :
   print("====== %s ======" % cmd)
   print(eval(cmd))

a = ocean.arange(25,ocean.gpu[0]).reshape(5,5)
print(a)

run("a[[1,2,1]]")
run("a[:,[0,2,1]]")

run("a[[2,1],:]")
run("a[:,[1,2]]")
run("a[[2,1],[1,2,2,2],None]")
run("a[[[1,2],[2,3],[3,4]]]")

b = (a <= a.T);
run("b")

f = ocean.find(b);
run("f")

run("a[None,b]")
run("a[None,f]")
run("a[[True,False,True,False,True],[True,True,False,False,True]]")
