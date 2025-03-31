import taichi as ti
ti.init(arch=ti.gpu)
a=ti.Vector([1,2])
b=ti.Vector([2,1])
print(a>b)
@ti.kernel
def f():
    print(ti.zero(a))
f()

