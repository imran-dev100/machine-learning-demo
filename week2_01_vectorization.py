import numpy
import time

# NumPy routines which allocate memory and fill arrays with value

a = numpy.zeros(4)
print(f"numpy.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = numpy.zeros((4,))
print(f"numpy.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = numpy.random.random_sample(4)
print(f"numpy.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

print(f"\n")


# NumPy routines which allocate memory and fill arrays with value but do not accept shape as inumpy.t argument
a = numpy.arange(4.);              print(f"numpy.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = numpy.random.rand(4);          print(f"numpy.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

print(f"\n")

# NumPy routines which allocate memory and fill with user specified values
a = numpy.array([5,4,3,2]);  print(f"numpy.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = numpy.array([5.,4,3,2]); print(f"numpy.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

print(f"\nINDEXING:")


#vector indexing operations on 1-D vectors
a = numpy.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
    print(f"c: {c}")
except Exception as e:
    print("The error message you'll see is:")
    print(e)




