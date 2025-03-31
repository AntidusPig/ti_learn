import taichi as ti
from taichi import math as tm
import numpy as np
from random import random

ti.init(arch=ti.gpu)

h = 2

poly6 = 4./tm.pi/h**8
spiky = 10./tm.pi/h**5
visco = 45/tm.pi/h**6

@ti.func
def dist(pi,pj):
    sub = pi-pj
    return tm.sqrt(sub[0]*sub[0]+sub[1]*sub[1])

@ti.func
def W(x):
    ans=0.
    if x>=h:
        ans=0.
    else:
        temp = (h*h-x*x)
        ans = temp**3*poly6
    return ans

@ti.func
def dW(x):
    ans=0.
    if x>=h:
        ans=0.
    else:
        ans = -3*(h-x)**2*spiky
    return ans

@ti.func
def ddW(x):
    ans=0.
    if x>=h:
        ans=0.
    else:
        ans = (h-x)*visco
    return ans

row = 10
col = 10
num_particles = row*col
x = ti.Vector.field(2,float,shape = (num_particles,))
v = ti.Vector.field(2,float,shape = (num_particles,))

rho = ti.field(float,shape=(num_particles,))
pressure = ti.field(float,shape=(num_particles,))
fp = ti.Vector.field(2,float,shape=(num_particles,))
fv = ti.Vector.field(2,float,shape=(num_particles,))

rho_0 = ti.field(float,shape=())
@ti.kernel
def init():
    rho_0[None] = W(0)        
init()

sep = 5.
offset = ti.Vector([40,40])
random_amount = 1.4

for i in range(row):
    for j in range(col):
        x[i*col+j] = ti.Vector([i*sep+random_amount*random(),
                                j*sep+random_amount*random()])+offset
        # v[i*col+j][1] = -30

eta = 100
grav=ti.Vector([0.,-9.8])
k = 1.e2
dt = 1/60

@ti.kernel
def compute_pressure():
    rho_0=W(0)
    for i in pressure:
        pressure[i] = k*(rho[i]-rho_0)

@ti.kernel
def compute_density():
    rho.fill(0)
    for i,j in ti.ndrange(num_particles,num_particles):
        if i!=j:
            r = dist(x[i],x[j])
            if 0<r<h:
                rho[i] += W(r)

@ti.kernel
def update2():
    fp.fill(0)
    fv.fill(0)
    for i,j in ti.ndrange(num_particles,num_particles):
        if i!=j:
            r = dist(x[i],x[j])
            if 0<r<h:
                r_hat = (x[i]-x[j])/r
                # print(rho_0[None]/rho[j])
                fp[i] += -(pressure[i]+pressure[j])/2*dW(r)*r_hat/rho[j]
                # fv[i] += eta*(v[j]-v[i])*ddW(r)/rho[j]
                # print(dW(r),laplacianW(r))
    # print('----------------')

eps = 1.e-8
coef_restitu = 0.8
size=500
scale = 3
@ti.kernel
def update3():
    for i in x:
        v[i] += dt*(grav+fp[i]+fv[i])
        x[i] += dt*v[i]

        if x[i][1] <= 0:
            x[i][1] = eps
            v[i][1] = -v[i][1] * coef_restitu
        elif x[i][1] >= size/scale:
            x[i][1] = size/scale-eps
            v[i][1] = -v[i][1] * coef_restitu
        if x[i][0] <= 0:
            x[i][0] = eps
            v[i][0] = -v[i][0] * coef_restitu
        elif x[i][0] >= size/scale:
            x[i][0] = size/scale-eps
            v[i][0] = -v[i][0] * coef_restitu
        

def loop():
    compute_density()
    compute_pressure()
    update2()
    update3()


gui = ti.GUI("SPH", res=(size,size))

def clamp(x,a,b):
    if x>b:
        return b
    elif x<a:
        return a
    return x

show_pressure = ti.field(bool,shape=())
def paint():
    for i in range(num_particles):
        position = x[i]*scale/size
        bridgtness = 100
        if show_pressure[None]:
            clr = clamp(int(255*(pressure[i]+bridgtness)/(500+bridgtness)*256**2),0,256**3-1)
        else:
            clr = clamp(int(255*(i+bridgtness)/(num_particles+bridgtness)*256**2),0,256**3-1)
        gui.circle(pos=[position[0],position[1]],radius=5,color=clr)

while gui.running:
    loop()
    paint()

    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'p':
            show_pressure[None] = not show_pressure[None]
    gui.show()