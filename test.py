def c():
    return 1

def d():
    return 2

b = True
a = c if b else d
print(a())

