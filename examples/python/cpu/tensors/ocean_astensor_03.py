import pyOcean_cpu as ocean

# Custom iterator class
class RangeIterator:
    def __init__ (self, r):
        self.n = r.n
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self.idx == self.n) :
           raise StopIteration()
        self.idx += 1
        return self.idx-1

    def next(self) :
       return self.__next__()

class Range :
   def __init__(self, n) :
      self.n = n

   def __iter__(self) :
      return RangeIterator(self)


# Regular lists and tuples
print(ocean.asTensor([[1,2,3],(4,5,6)]))

# Xrange objects
print(ocean.asTensor(range(10)))

# Import iterable object
r = Range(10)
print(ocean.asTensor(r))
print(ocean.asTensor([r,r]))
