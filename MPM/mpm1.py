import taichi as ti
import numpy as np
import taichi.math as tm
from random import random

# Huge thanks to yuanming for sharing this cool method!
# and the cool language :)
# https://www.bilibili.com/video/BV1ZK411H7Hc/
# Coolest thing I've ever seen

ti.init(arch=ti.gpu)

preset = 3
export = False

n_grid, substeps, dt = 64, 25, 2e-4

num_particles = 3000

dx = 1./n_grid

# particle attributes
rho = 1.0                 # density
vol = (dx*0.5)**2       # volume
mass = vol*rho          # particle mass
GRAVITY = ti.Vector.field(2,float,shape=())
GRAVITY[None] = ti.Vector([0.,-10.]) # grav acc
E = 1000.0                # young's modulus
sigma = 0.2             # poisson's ratio
G_0, lambda_0 = E/(2*(1+sigma)), E*sigma/((1+sigma)*(1-2*sigma)) # shear modulus, lame constant
xi = 1.                # hardening coefficient
snow_def_lower = 2.5e-2
snow_def_upper = 4.5e-3
mud_def_lower = 25.e-2
mud_def_upper = 19.e-3
snow_def_upper = 4.5e-3
bound = 3               # width of grid considered to be the wall, some padding to protect kernel sampling

x = ti.Vector.field(2,float,num_particles,)  # particle position
v = ti.Vector.field(2,float,num_particles)  # particle velocity
C = ti.Matrix.field(2,2,float,num_particles)  # velocity gradient C_p
F = ti.Matrix.field(2,2,float,num_particles)  # deformation gradient
J = ti.field(float, num_particles)          # jacobian of F

colors = ti.Vector.field(3,float,num_particles)# render color
grid_v = ti.Vector.field(2,float,(num_particles,num_particles)) # grid velocity
grid_m = ti.field(float,(num_particles,num_particles)) # grid mass
materials = ti.field(int,num_particles)      # type of substance
used = ti.field(bool, num_particles)
next_p = 0 # pointer/index to next uninitialized particle


# quadratic kernel
neighbour = (3,3)  # loop through to get offset for neighbouring grid
# for cubic kernel
# neighbour = (5,5)
# although one of them is 0, it's annoying to figure out which

WATER = 0
JELLY = 1
SNOW = 2
MUD = 3
WATER2 = 4  # different color
MAX_MATERIAL = 5
mat_color = ti.Vector.field(3,float,MAX_MATERIAL)
mat_color.from_numpy(np.array([[51/255,153/255,1],[1,0.02,0],[1,1,1],[1,153/255,51/255],[0,102/255,204/255]]))

@ti.kernel
def p2g():
    grid_m.fill(0)
    grid_v.fill(0)
    for p in x:
        if not used[p]:continue
        x_scaled = x[p]/dx
        base = int(x_scaled-0.5)  # so that base+1 is the closest grid point
        t = x_scaled-base
        # quadratic interpolation
        w = [0.5 * (t - 1.5) ** 2, 0.75 - (t - 1) ** 2, 0.5 * (t - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 2)+dt*C[p])@F[p]
        
        # hardening
        G, lamb = G_0, lambda_0
        if materials[p]==WATER or materials[p]==WATER2:
            G = 0.0
        else:
            h = 0.3 # jelly
            if materials[p]==SNOW:
                h = ti.exp(xi*(1.0-J[p]))
            G *= h
            lamb *= h

        # yeilding
        U, Sig, V = ti.svd(F[p])
        J_local = 1.0
        for i in ti.static(range(2)):
            new_sig = Sig[i,i]
            if materials[p] == SNOW:
                # ad hoc
                new_sig = tm.clamp(Sig[i,i],1-snow_def_lower,1+snow_def_upper)
            elif materials[p] == MUD:
                new_sig = tm.clamp(Sig[i,i],1-mud_def_lower,1+mud_def_upper)
            # if plastic deformation, store new volume
            J[p] *= Sig[i,i]/new_sig
            Sig[i,i] = new_sig
            J_local*=new_sig
        if materials[p] == WATER or materials[p] == WATER2:
            # Reset deformation gradient to avoid numerical instability
            # TODO: not entirely sure why
            # need to read more about this
            new_F = ti.Matrix.identity(float,2)
            new_F[0,0] = J_local  # what about new_F[1,1]?
            F[p] = new_F
        elif materials[p] == SNOW or materials[p] == MUD:
            # Accidentally wrote it as 
            # F = U@ Sig @V.transpose()
            # but the code runs
            # took me an hour to figure out
            F[p] = U@ Sig @V.transpose()
        # R = UV^T
        # P = 2*mu*(F-R)+lamb*(J-1)JF^{-T}
        # PF^T = 2*mu*(F-R)F^T+lamb*(J-1)J*I
        stress = 2*G*(F[p]-U@V.transpose())@F[p].transpose()
        stress += ti.Matrix.identity(float,2)*lamb*J_local*(J_local-1)
        stress *= -4*dt/dx**2*vol
        affine = stress + mass * C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(3,3))):
            dpos = (offset-t)*dx
            weight=1.
            for i in ti.static(range(2)):
                weight *= w[offset[i]][i]
            grid_v[base+offset]+=weight*(mass*v[p]+affine@dpos)
            grid_m[base+offset]+=weight*mass

@ti.kernel
def grid_ops():
    for i in ti.grouped(grid_m):
        if grid_m[i]>0:
            grid_v[i]/=grid_m[i]
        grid_v[i] += dt*GRAVITY[None]
        # boundary condition, slip (type 2)
        # note that cond is a vector, like i
        cond = (i < bound) & (grid_v[i] < 0) | (i > n_grid - bound) & (grid_v[i] > 0)
        grid_v[i] = ti.select(cond, 0, grid_v[i])

@ti.kernel
def g2p():
    for p in x:
        if not used[p]:
            continue
        x_scaled = x[p]/dx
        base=int(x_scaled-.5)
        t = x_scaled-base
        w = [0.5*(1.5-t)**2, 0.75-(t - 1)**2, 0.5*(t-0.5)**2] # a 3x2 matrix
        new_v = ti.zero(v[p]) # element filled with 0
        new_C = ti.zero(C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(3,3))):
            dpos = (offset - t)*dx
            weight = 1.0
            for i in ti.static(range(2)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base+offset]
            new_v += weight*g_v
            new_C += 4*weight*g_v.outer_product(dpos)/dx**2
            a = base+offset
        v[p] = new_v
        x[p] += dt*v[p]
        C[p] = new_C
                
def loop():
    p2g()
    grid_ops()
    g2p()

class CubeVolume:
    def __init__(self, minimum, size, sep, material):
        self.minimum = minimum
        self.size = size
        self.sep = sep
        self.volume = self.size.x * self.size.y
        self.material = material


def init_cube_vol(x_begin: float, y_begin: float,
                x_size: int, y_size: int,
                sep: float,
                material: int):
    global next_p
    for i in range(x_size):
        for j in range(y_size):
            if used[next_p]:
                return
            x[next_p] = ti.Vector([i,j])*sep + ti.Vector([x_begin, y_begin])
            J[next_p] = 1
            F[next_p] = ti.Matrix([[1, 0], [0, 1]])
            v[next_p] = ti.Vector([0.0, 0.0])
            materials[next_p] = material
            colors[next_p] = mat_color[material]
            used[next_p] = True

            next_p = min(next_p+1, num_particles-1)

def init_vols(vols):
    total_vol = 0
    for v in vols:
        total_vol += v.volume
    for i, v in enumerate(vols):
        v = vols[i]
        if isinstance(v, CubeVolume):
            init_cube_vol(*v.minimum, *v.size, v.sep, v.material)
        else:
            raise Exception(f"Init volume of type{type(v)} not implemented")

cubes = []

def load_preset(preset:int):
    global cubes
    if preset == 0:
        cubes = [
            CubeVolume(ti.Vector([0.3, 0.5]), ti.Vector([20, 20]), 0.02, WATER),
        ]
    elif preset==1:
        cubes = [
            CubeVolume(ti.Vector([0.20, 0.35]), ti.Vector([20, 25]), 0.01, WATER),
            CubeVolume(ti.Vector([0.50, 0.35]), ti.Vector([25, 20]), 0.01, WATER2),
        ]
    elif preset==2:
        cubes = [
            CubeVolume(ti.Vector([0.35, 0.75]), ti.Vector([10, 30]), 0.01, SNOW),
            CubeVolume(ti.Vector([0.55, 0.75]), ti.Vector([15, 25]), 0.01, MUD),
            CubeVolume(ti.Vector([0.45, 0.6]), ti.Vector([30, 10]), 0.01, JELLY),
            CubeVolume(ti.Vector([0.1, 0.05]), ti.Vector([80, 20]), 0.01, WATER),
        ]
    elif preset==3:
        cubes = [
            CubeVolume(ti.Vector([0.2, 0.2]), ti.Vector([60, 5]), 0.01, SNOW),
            CubeVolume(ti.Vector([0.45, 0.3]), ti.Vector([5, 35]), 0.01, JELLY),
            CubeVolume(ti.Vector([0.1, 0.75]), ti.Vector([80, 5]), 0.01, MUD),
            CubeVolume(ti.Vector([0.1, 0.8]), ti.Vector([80, 20]), 0.01, WATER2),
        ]
    elif preset==4:
        cubes = [
            CubeVolume(ti.Vector([0.20, 0.35]), ti.Vector([20, 10]), 0.01, SNOW),
            CubeVolume(ti.Vector([0.5, 0.35]), ti.Vector([20, 10]), 0.01, MUD),
            CubeVolume(ti.Vector([0.35, 0.75]), ti.Vector([20, 10]), 0.01, JELLY),
            CubeVolume(ti.Vector([0.05, 0.55]), ti.Vector([40, 10]), 0.01, WATER),
            CubeVolume(ti.Vector([0.65, 0.75]), ti.Vector([30, 10]), 0.01, WATER2),
        ]
    elif preset==5:
        cubes = [
            CubeVolume(ti.Vector([0.15, 0.1]), ti.Vector([30, 30]), 0.01, MUD),
            CubeVolume(ti.Vector([0.05, 0.6]), ti.Vector([10, 10]), 0.01, JELLY),
            CubeVolume(ti.Vector([0.2, 0.6]), ti.Vector([10, 10]), 0.01, JELLY),
            CubeVolume(ti.Vector([0.45, 0.7]), ti.Vector([15, 15]), 0.01, JELLY),
            CubeVolume(ti.Vector([0.6, 0.6]), ti.Vector([10, 10]), 0.01, JELLY),
        ]
    init_vols(cubes)

def load_next_preset():
    global next_p
    global preset
    next_p = 0
    x.fill(100)
    used.fill(False)
    preset = (preset+1)%6
    load_preset(preset)

load_preset(preset)

res = (1000, 1000)
window = ti.ui.Window("Real MPM 2D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
# scene = window.get_scene()
from time import time
dir_keys = ('w','a','s','d')
dir_key_held = {'w':False,'a':False,'s':False,'d':False}
redo_action_time = 0.1
dir_key_last_time = {'w':0,'a':0,'s':0,'d':0}
dir_key_hold_timer = {'w':0,'a':0,'s':0,'d':0}
dir_vectors = {'w':ti.Vector([0,1.]), 'a':ti.Vector([-1.,0.]), 's':ti.Vector([0.,-1.]), 'd':ti.Vector([1.,0.])}

def dir_key_pressed(key):
    dir_key_held[key] = True
    GRAVITY[None] += dir_vectors[key]
    dir_key_last_time[key] = time()

def dir_key_released(key):
    dir_key_held[key] = False
    dir_key_hold_timer[key] = 0

def check_hold_key():
    for i in dir_keys:
        if dir_key_held[i]:
            t = time()
            dir_key_hold_timer[i] += t-dir_key_last_time[i]
            if dir_key_hold_timer[i]>=redo_action_time:
                GRAVITY[None] += dir_vectors[i]
                dir_key_hold_timer[i] = 0
            dir_key_last_time[i] = t
            
frame_id = 0
paused = False
total_frame = 300
frames_per_save = 2
counter2 = 0

def sign(x):
    if x>0:return 1
    else:return -1

while window.running:
    # simulate
    if not paused:
        for _ in range(substeps):
            loop()

    # render
    canvas.circles(x,0.005, per_vertex_color = colors)
    
    # input
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
        elif e.key == 'p':
            paused = not paused
        elif e.key in dir_keys:
            dir_key_pressed(e.key)
        elif e.key == ti.ui.SPACE:
            load_next_preset()
        elif e.key == ti.ui.RETURN:
            GRAVITY[None] = ti.Vector([random()*-10.*sign(GRAVITY[None][0]),random()*-10.*sign(GRAVITY[None][1])])

    for e in window.get_events(ti.ui.RELEASE):
        if e.key in dir_keys:
            dir_key_released(e.key)
    check_hold_key()

    # text
    with gui.sub_window("Sub Window", x=0, y=0, width=0.6, height=0.1):
        gui.text("gravity: %.2f, %.2f"%(GRAVITY[None][0],GRAVITY[None][1]))

    # export
    if export:
        if counter2%frames_per_save==0:
            window.save_image(f'./img/mpm{frame_id}.png')
            frame_id+=1
        counter2+=1
        # save
        if frame_id==total_frame:
            break
    
    window.show()