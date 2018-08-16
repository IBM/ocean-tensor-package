import pyOcean_cpu as ocean

def run(cmd) :
   result = eval(cmd);
   print("%s --> %s" % (cmd, result))


run("ocean.false and ocean.false")
run("ocean.false and ocean.true")
run("ocean.true and ocean.false")
run("ocean.true and ocean.true")

run("ocean.true and False")
run("ocean.true and True")
run("True and ocean.false")
run("True and ocean.true")

run("ocean.false or ocean.false")
run("ocean.false or ocean.true")
run("ocean.true or ocean.false")
run("ocean.true or ocean.true")

run("ocean.false or False")
run("ocean.false or True")
run("False or ocean.false")
run("False or ocean.true")

run("not ocean.false")
run("not ocean.true")




