import pyOcean_cpu as ocean

def check(idx,a,b) :
   if (a==b) :
      print("[%d] Correct" % idx)
   else :
      print("[%d] Failed" % idx)

check(1, True,  ocean.checkOverlap([5],[4],0,1,[4],[3],7,1))
check(2, False, ocean.checkOverlap([5],[4],0,1,[3],[3],7,1))
check(3, True,  ocean.checkOverlap([5,3],[4,1],0,1,[3],[10],-3,1))
check(4, True,  ocean.checkOverlap([5,3],[4,1],0,1,[2,3],[-40,10],37,1))
check(5, True,  ocean.checkOverlap([3],[8],0,1,[4],[5],5,2))
