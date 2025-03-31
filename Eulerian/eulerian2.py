import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)


@ti.func
def sample(vf:ti.template(), u, v):
    i, j = int(u), int(v)
    # Nearest
    i = tm.clamp(i, 0, vf.shape[0] - 1)
    j = tm.clamp(j, 0, vf.shape[1] - 1)
    return vf[i, j]

@ti.func
def lerp(a, b, t):
    # frac: [0.0, 1.0]
    return (1 - t) * a + t * b


@ti.func
def bilerp(vf:ti.template(), u, v):

    s, t = u - 0.5, v - 0.5
    iu, iv = int(s), int(t)
    a = sample(vf, iu + 0.5, iv + 0.5)
    b = sample(vf, iu + 1.5, iv + 0.5)
    c = sample(vf, iu + 0.5, iv + 1.5)
    d = sample(vf, iu + 1.5, iv + 1.5)

    fu, fv = s - iu, t - iv
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

eulerSimParam = {
    "shape": (516, 516),
    "dt": 1 / 60.0,
    "iteration_step": 20,
    "eta": 0,
    "flow_on": True,
    "initial_ring_on": False,
    "ring_obstacle_on": True,
    "gravity_on": False,
    "render": True,
}

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


v1 = ti.Vector.field(2, float, shape=(eulerSimParam["shape"]))
v2 = ti.Vector.field(2, float, shape=(eulerSimParam["shape"]))
color1 = ti.Vector.field(3, float, shape=(eulerSimParam["shape"]))
color2 = ti.Vector.field(3, float, shape=(eulerSimParam["shape"]))

div = ti.field(float, shape=(eulerSimParam["shape"])) # div
p1 = ti.field(float, shape=(eulerSimParam["shape"]))
p2 = ti.field(float, shape=(eulerSimParam["shape"]))

v = TexPair(v1, v2)
p = TexPair(p1, p2)
color = TexPair(color1, color2)

obstacle = ti.field(bool, shape=eulerSimParam['shape'])

@ti.func
def no_obstacle(i:int, j:int):
    return not ((i<1 or j<1 or i>obstacle.shape[0]-2 or j>obstacle.shape[1]-2) or obstacle[i,j])

@ti.kernel
def init_field():
    # initial condition

    p1.fill(0)
    v1.fill(0)
    for i, j in ti.ndrange(eulerSimParam["shape"][0], eulerSimParam["shape"][1]):
        # checkerboard pattern
        color.cur[i,j] = (i//40)%2 != (j//40)%2

        # rotating fluid
        center = v.cur.shape[0]//2
        r = tm.sqrt((i-center)**2+(j-center)**2)
        if eulerSimParam['initial_ring_on']:
            radius1 = center/4
            radius2 = center/2
            x = i-center
            y = j-center
            if radius1<r<radius2:
                v.cur[i,j]=ti.Vector([-y,x])/center*10.

        # circular obstacle
        if eulerSimParam['ring_obstacle_on']:
            if center/8<r<center/4:
                obstacle[i,j] = True

@ti.kernel
def advection(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        if no_obstacle(i,j):
            coord_cur = ti.Vector([i, j]) + ti.Vector([0.5, 0.5])
            vel_cur = vf[i, j]
            coord_prev = coord_cur - vel_cur * eulerSimParam["dt"]
            q_prev = bilerp(qf, coord_prev[0], coord_prev[1])
            new_qf[i, j] = q_prev

@ti.kernel
def calc_div(vf: ti.template(), divf: ti.template()):
    for i, j in vf:
        if no_obstacle(i,j):
            divf[i, j] = (vf[i + 1, j][0] - vf[i - 1, j][0] + vf[i, j + 1][1] - vf[i, j - 1][1])

@ti.kernel
def update_p(divf: ti.template(), pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        if no_obstacle(i,j):
            new_pf[i, j] = (pf[i + 1, j] + pf[i - 1, j] + pf[i, j - 1] + pf[i, j + 1] - divf[i, j]) / 4

@ti.kernel
def velocity_from_pressure(pf: ti.template(), vf: ti.template(), vf_new: ti.template()):
    for i, j in vf:
        if no_obstacle(i,j):
            vf_new[i, j] = vf[i, j] - eulerSimParam['dt'] * ti.Vector(
                [(p_with_boundary(pf, i + 1, j)- p_with_boundary(pf, i - 1, j))/2.,
                (p_with_boundary(pf, i, j + 1)- p_with_boundary(pf, i, j - 1))/2.])

@ti.kernel
def apply_vel_bc(vf: ti.template()):
    # flow boundary width
    width = 30
    center = v.cur.shape[0]//2

    for i, j in vf:
        # static boundary
        if (i <= 0) or (i >= vf.shape[0] - 1) or (j >= vf.shape[1] - 1) or (j <= 0):
            vf[i, j] = ti.Vector([0.0, 0.0])
        
        # flow boundary
        if eulerSimParam['flow_on'] and -width<(i-center)<width and v.cur.shape[1]-j<5:
            v.cur[i,j] = ti.Vector([0.,-10.])

@ti.func
def p_with_boundary(pf: ti.template(), i: int, j: int) -> ti.f32:
    shape = pf.shape
    if ((i == j == 0)
        or (i == shape[0] - 1 and j == shape[1] - 1)
        or (i == 0 and j == shape[1] - 1)
        or (i == shape[0] - 1 and j == 0)):
        pf[i, j] = 0.0
    elif i == 0:
        pf[0, j] = pf[1, j]
    elif j == 0:
        pf[i, 0] = pf[i, 1]
    elif i == shape[0] - 1:
        pf[shape[0] - 1, j] = pf[shape[0] - 2, j]
    elif j == shape[1] - 1:
        pf[i, shape[1] - 1] = pf[i, shape[1] - 2]
    return pf[i, j]

@ti.kernel
def apply_p_bc(pf: ti.template()):
    for i, j in pf:
        p_with_boundary(pf,i,j)

@ti.kernel
def gravity(vf:ti.template()):  # accelerate downwards
    for i,j in vf:
        if no_obstacle(i,j):
            vf[i,j] += ti.Vector([0.,-1.])*eulerSimParam['dt']

@ti.kernel
def viscous(vf:ti.template(), vf_new:ti.template()):
    for i,j in vf:
        if no_obstacle(i,j):
            vf_new[i,j] = vf[i,j] + eulerSimParam['eta']*eulerSimParam["dt"]*(vf[i-1,j]+vf[i+1,j]+vf[i,j-1]+vf[i,j+1]-4*vf[i,j])

def advection_step():
    advection(v.cur, color.cur, color.nxt)
    advection(v.cur, v.cur, v.nxt)
    color.swap()
    v.swap()
    apply_vel_bc(v.cur)

def pressure_step():
    calc_div(v.cur, div)
    for i in range(eulerSimParam["iteration_step"]):
        update_p(div, p.cur, p.nxt)
        p.swap()
        apply_p_bc(p.cur)
        
    velocity_from_pressure(p.cur, v.cur, v.nxt)
    v.swap()
    apply_vel_bc(v.cur)

def viscous_step():
    viscous(v.cur,v.nxt)
    v.swap()

canvas = ti.Vector.field(3,float,(eulerSimParam["shape"][0]*2, eulerSimParam['shape'][1]))

@ti.kernel
def paint(color:ti.template()):
    width = 30
    center = v.cur.shape[0]//2
    for i,j in v.cur:
        offset = eulerSimParam["shape"][0]
        canvas[i+offset,j] = ti.Vector([p.cur[i,j],0,0])
        canvas[i,j] = color[i,j]

        if obstacle[i,j]:
            canvas[i,j] = ti.Vector([0.8,0.8,0.8])
        
        if eulerSimParam['flow_on'] and -width<(i-center)<width and v.cur.shape[1]-j<5:
            canvas[i,j] *= ti.Vector([0,1,0])

init_field()
apply_vel_bc(v.cur)
window = ti.GUI("Euler 2D Simulation", res=canvas.shape)

frames_per_save = 25
counter = 0
counter2 = 0
        
while window.running:
    if eulerSimParam['gravity_on']:
        gravity(v.cur)
    advection_step()
    pressure_step()
    viscous_step()

    if eulerSimParam['render']:
        if counter % 5==0:
            ti.tools.imwrite(canvas.to_numpy(), './imgs1/euler%d.png'%counter2)
            counter2+=1
        counter += 1

    for e in window.get_events(window.PRESS):
        if e.key == window.ESCAPE:
            window.running = False

    paint(color1)
    window.set_image(canvas)
    window.show()