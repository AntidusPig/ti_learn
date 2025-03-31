import taichi as ti
from taichi import math as tm
import numpy as np
from random import random

ti.init(arch=ti.gpu)

# a few example settings
preset_index = 3
export = False
show_pressure = ti.field(bool,shape=())
# show_pressure.fill(1)

h = 10  # kernel radius

poly6 = 4./tm.pi/h**8 # 2d version, different normalizing constant
spiky = 10./tm.pi/h**5
visco = 45/tm.pi/h**6 # 3d version: differ by a constant, not a big deal

row = 30
col = 100
num_particles = row*col

rho_0 = ti.field(float,shape=())

sep = 3
offset = ti.Vector([40,90])
# random_amount = 1.4
random_amount = 0.5

eta = 400
grav=ti.Vector([0.,-1.])
k = 500.
dt = 1./60.
substeps = 4

render_size = .003

if preset_index==0:
    # high viscousity
    pass
elif preset_index==1:
    # free space
    row=30
    col=30
    num_particles = row*col
    sep = 10
    offset = ti.Vector([100,100])
    h = 30
    grav[1]=0.
elif preset_index==2:
    # low viscosity
    eta = 0.1
elif preset_index==3:
    # 10,000 particles
    row = 100
    col = 100
    sep = 3
    num_particles = row*col
    offset[0] = 90
elif preset_index==4:
    # higher k
    row = 100
    col = 100
    sep = 3
    k = 1000
    num_particles = row*col
    offset[0] = 90

@ti.func
def dist(pi,pj):
    sub = pi-pj
    return tm.sqrt(sub[0]*sub[0]+sub[1]*sub[1])

# kernel functions chosen as suggested in "Particle-Based Fluid Simulation for Interactive Applications" Matthias

@ti.func
def W(x): #poly6
    ans=0.
    if x>=h:
        ans=0.
    else:
        temp = (h*h-x*x)
        ans = temp**3*poly6
    return ans

@ti.func
def dW(x): #spiky
    ans=0.
    if x>=h:
        ans=0.
    else:
        ans = -3*(h-x)**2*spiky
    return ans

@ti.func
def ddW(x): #viscosity
    ans=0.
    if x>=h:
        ans=0.
    else:
        ans = (h-x)*visco
    return ans

x = ti.Vector.field(2,float,shape = (num_particles,))
v = ti.Vector.field(2,float,shape = (num_particles,))

rho = ti.field(float,shape=(num_particles,))
pressure = ti.field(float,shape=(num_particles,))
f = ti.Vector.field(2,float,shape=(num_particles,))

@ti.kernel
def init():
    rho_0[None] = W(0)
init()

for i in range(row):
    for j in range(col):
        x[i*col+j] = ti.Vector([i*sep+random_amount*random(),
                                j*sep+random_amount*random()])+offset

@ti.kernel
def compute_pressure():
    rho_0=W(0)
    for i in pressure:
        pressure[i] = k*(rho[i]-rho_0)

@ti.func
def density_task(i,j):
    r = dist(x[i],x[j])
    rho[i] += W(r)

@ti.kernel
def compute_density():
    rho.fill(0)
    for i in x:
        do_for_neighbours(i,density_task)

@ti.func
def force_task(i,j):
    r = dist(x[i],x[j])
    r_hat = (x[i]-x[j])/(r+eps)
    # print(rho_0[None]/rho[j])
    f[i] += -(pressure[i]+pressure[j])/2*dW(r)*r_hat/(rho[j]+eps)
    f[i] += eta*(v[j]-v[i])*ddW(r)/(rho[j]+eps)
    # print(dW(r),laplacianW(r))

@ti.kernel
def update_force():
    f.fill(0)
    for i in range(num_particles):
        do_for_neighbours(i,force_task)
                
    # print('----------------')

eps = 1.e-4
coef_restitution = 0.8
size=500
@ti.kernel
def update_vx():
    for i in x:
        v[i] += dt*(grav+f[i])
        x[i] += dt*v[i]

        if x[i][1] <= 0:
            x[i][1] = eps
            v[i][1] = -v[i][1] * coef_restitution
        elif x[i][1] >= size:
            x[i][1] = size-eps
            v[i][1] = -v[i][1] * coef_restitution
        if x[i][0] <= 0:
            x[i][0] = eps
            v[i][0] = -v[i][0] * coef_restitution
        elif x[i][0] >= size:
            x[i][0] = size-eps
            v[i][0] = -v[i][0] * coef_restitution
        
# optimization
# https://www.youtube.com/watch?v=rSKMYc1CQHE&t=20s

@ti.func
def position_to_cell_coord(x:ti.template()):
    return ti.Vector([x[0]//h, x[1]//h],int)

@ti.func
def get_cell_key(cell:ti.template())->int:
    return ti.cast(cell[0]*15823+cell[1]*9737333,int)%num_particles


# first element particle index, second element grid key
spatial_lookup = ti.Vector.field(2,ti.i32,num_particles)
spatial_lookup_np = np.zeros((num_particles,2),dtype=np.int32)
# the ith element is the starting index of the elements with key i in sorted spatial lookup
start_indicies = ti.field(ti.i32,shape=(num_particles,))

@ti.kernel
def usl1():
    for i in range(num_particles):
        spatial_lookup[i][0] = ti.cast(i,ti.i32)
        spatial_lookup[i][1] = get_cell_key(position_to_cell_coord(x[i]))

@ti.kernel
def usl2():
    start_indicies.fill(-1)
    for i  in spatial_lookup:
        j = spatial_lookup[i][1]
        if i==0 or j != spatial_lookup[i-1][1]:
            start_indicies[j] = i

def update_spatial_lookup():
    usl1()
    spatial_lookup_np = spatial_lookup.to_numpy()
    spatial_lookup_np = spatial_lookup_np[np.lexsort(spatial_lookup_np.T)]  # sort by second number, the hash key
    spatial_lookup.from_numpy(spatial_lookup_np)
    usl2()
    

cell_offsets_np = np.array([[-1,-1],[-1,0],[-1,1],
                            [0,-1],[0,0],[0,1],
                            [1,-1],[1,0],[1,1]])
cell_offsets = ti.Vector.field(2,int,shape=(9,))
cell_offsets.from_numpy(cell_offsets_np)

@ti.func
def do_for_neighbours(pi_index:int, task:ti.template()):
    cell = position_to_cell_coord(x[pi_index])
    for i in range(9):
        key = get_cell_key(cell_offsets[i]+cell)
        for j in range(start_indicies[key], num_particles):
            if spatial_lookup[j][1] != key:
                break
            pj_index = spatial_lookup[j][0]
            if pi_index!=pj_index:
                d = dist(x[pi_index],x[pj_index])
                if d<h:
                    task(pi_index,pj_index)

@ti.func
def lerpf(a,b,t):
    return a+(b-a)*t

@ti.func
def lerp_color(a,b,t):
    return (lerpf(a[0],b[0],t),lerpf(a[1],b[1],t),lerpf(a[2],b[2],t))


window = ti.ui.Window("SPH", res=(size,size))
canvas = window.get_canvas()


red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
white = (1,1,1)
grey = (0.1,0.1,0.1)
canvas.set_background_color(grey)

@ti.func
def gradient(t):
    c = ti.Vector([0,0,0],ti.f32)
    if t<0.5:
        c = lerp_color(blue, green, t-0.5)
    else:
        c = lerp_color(green, red, t*2-1)
    return c

@ti.func
def sigmoid(x,a):
    return 1./(1.+tm.e**(-x/a))

point_color = ti.Vector.field(3, int,num_particles)  # color of the circle
bridgtness = 100
@ti.func
def debug_red_task(i:int,j:int):
    point_color[j] = red
    point_color[i] = blue
@ti.kernel
def set_point_color():
    for i in point_color:
        if show_pressure[None]:
            point_color[i] = gradient(pressure[i]/50)
        else:
            point_color[i] = white
    do_for_neighbours(num_particles//2-row//2,debug_red_task) # paint points around it red

def loop():
    for i in range(substeps):
        update_spatial_lookup()
        compute_density()
        compute_pressure()
        update_force()
        update_vx()
        set_point_color()

x_screenpos = ti.Vector.field(2,float,num_particles)
@ti.kernel
def calc_screenpos():
    for i in range(num_particles):
        x_screenpos[i]=x[i]/size

frame_i = 0
total_frame = 500
frames_per_save = 5
counter2 = 0
while window.running:
    loop()

    # paint
    # gui.set_image(canvas)
    # for i, j in ti.ndrange(size,size):
    #     # pressure = 
    #     canvas[i,j] = lerp(blue,red,0.5)

    # canvas.circles(vertices, radius, color, per_vertex_color)
    calc_screenpos()
    canvas.circles(x_screenpos,render_size,white,point_color)
    # print(x2[0],size,point_color[0])

    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
        elif e.key == 'p':
            show_pressure[None] = not show_pressure[None]

    if export:
        if counter2%frames_per_save==0:
            window.save_image(f'./img/sph{frame_i}.png')
            frame_i+=1
        counter2+=1
        # save
        if frame_i==total_frame:
            break
        
    window.show()