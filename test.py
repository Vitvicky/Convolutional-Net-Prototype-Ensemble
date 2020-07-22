import numpy as np
def c():
    return 1

def d():
    return 2

b = True
a = c if b else d
print(a())

arr = np.array([[1,2],[3, 4]])
print(arr)