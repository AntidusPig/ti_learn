##################
### DEPRECATED ###
##################
import taichi as ti
import taichi.math as tm

# ti.init(arch=ti.gpu)
ti.init(arch=ti.gpu, debug=True)

eulerSimParam = {
    "shape": (516, 516),
    "dt": 1 / 1000.0,
    "iteration_step": 20,
    "o": 1.2 # SOR factor
}

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


# if the corner of the grid is at (0,0)
# origin of v_x is at (0.5,0)
# origin of v_y is at (0,0.5)
# origin of q (for any other variables)  is at (0.5,0.5)

v1 = ti.Vector.field(2,float, shape=eulerSimParam['shape'])
v2 = ti.Vector.field(2,float, shape=eulerSimParam['shape'])
v = TexPair(v1,v2)

color1 = ti.Vector.field(3,float,shape=eulerSimParam["shape"])
color2 = ti.Vector.field(3,float,shape=eulerSimParam["shape"])
color = TexPair(color1, color2)

obstacle = ti.field(bool, shape=eulerSimParam['shape'])

p = ti.field(float, shape=eulerSimParam['shape'])

@ti.func
def no_obstacle(i:int, j:int):
    return not ((i<1 or j<1 or i>obstacle.shape[0]-2 or j>obstacle.shape[1]-2) or obstacle[i,j])

@ti.kernel
def vel_projection(vf:ti.template(), vf_new:ti.template()):
    cp = eulerSimParam['density']/eulerSimParam['dt']  # (*h, where h is 1.0)
    p.fill(0)
    for i,j in ti.ndrange(vf.shape[0]-1,vf.shape[1]-1):
        if i==0 or j==0:continue
        if no_obstacle(i,j):
            a1,a2,a3,a4 = no_obstacle(i-1,j),no_obstacle(i+1,j),no_obstacle(i,j-1),no_obstacle(i,j+1)
            s=0.+a1+a2+a3+a4
            if s==0.:continue
            d = vf[i+1,j][0] - vf[i,j][0] + vf[i,j+1][1] - vf[i,j][1]
            c = -d/s
            c *= eulerSimParam['o']
            p[i,j] += -cp*c
            vf_new[i,j][0] -= a1*c
            vf_new[i+1,j][0] += a2*c
            vf_new[i,j][1] -= a3*c
            vf_new[i,j+1][1] += a4*c

@ti.func
def lerp(a,b,t):
    # t in [0,1]
    return (1-t)*a+t*b

@ti.func
def sample(f:ti.template(), x:float, y:float):
    x = tm.clamp(x,0,f.shape[0]-2)
    y = tm.clamp(y,0,f.shape[1]-2)
    x1 = ti.cast(tm.floor(x),ti.i32)
    x2 = x1+1
    y1 = ti.cast(tm.floor(y),ti.i32)
    y2 = y1+1
    a = f[x1,y1]
    b = f[x2,y1]
    c = f[x1,y2]
    d = f[x2,y2]
    u = x-x1
    v = y-y1
    return lerp(lerp(a,b,u),lerp(c,d,u),v)

@ti.kernel
def advect_q(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i,j in ti.ndrange(vf.shape[0]-1,vf.shape[1]-1):
        if i==0 or j==0:continue
        vx = sample(vf,i+0.5,j)[0]
        vy = sample(vf,i,j+0.5)[1]
        coord_prev = ti.Vector([i, j]) - ti.Vector([vx,vy]) * eulerSimParam["dt"]
        q_prev = sample(qf, coord_prev[0], coord_prev[1])
        new_qf[i, j] = q_prev

@ti.kernel
def advect_v(vf: ti.template(), new_vf: ti.template()):
    for i,j in ti.ndrange(vf.shape[0]-1,vf.shape[1]-1):
        if i==0 or j==0:continue
        vx = vf[i,j][0]
        vy = sample(vf,i,j+0.5)[1]
        coord_prev = ti.Vector([i, j]) - ti.Vector([vx,vy]) * eulerSimParam["dt"]
        vx_sample = sample(vf, coord_prev[0], coord_prev[1])[0]
        
        vx = sample(vf,i+0.5,j)[0]
        vy = vf[i,j][1]
        coord_prev = ti.Vector([i, j]) - ti.Vector([vx,vy]) * eulerSimParam["dt"]
        vy_sample = sample(vf, coord_prev[0], coord_prev[1])[1]

        new_vf[i, j][0] = vx_sample
        new_vf[i, j][1] = vy_sample

@ti.kernel
def init():
    v.cur.fill(0)
    for i, j in ti.ndrange(eulerSimParam["shape"][0], eulerSimParam["shape"][1]):
        # checkerboard pattern
        color.cur[i,j] = (i//40)%2 != (j//40)%2

        # rotating fluid
        center = v.cur.shape[0]//2
        # radius1 = center/4
        # radius2 = center/2
        r = tm.sqrt((i-center)**2+(j-center+150)**2)
        # if radius1<r<radius2:
        #     v.cur[i,j]=ti.Vector([-j,i])/center*0.02

        # circular obstruction
        if r<center/4:
            obstacle[i,j] = True
@ti.kernel
def boundary_condition():
    width = 30
    center = v.cur.shape[0]//2
    for i,j in v.cur:
        if -width<(i-center)<width and j<5:
            v.cur[i,j] = ti.Vector([0.,10.])

def simulate():
    for i in range(eulerSimParam['iteration_step']):
        boundary_condition()
        vel_projection(v.cur, v.nxt)
        v.swap()

        advect_v(v.cur, v.nxt)
        v.swap()
        
        advect_q(v.cur, color.cur, color.nxt)
        color.swap()


zoom = 1

canvas = ti.Vector.field(3,float,(eulerSimParam["shape"][0]*zoom, eulerSimParam["shape"][1]*zoom))
@ti.kernel
def paint(color:ti.template()):
    width = 30
    center = v.cur.shape[0]//2
    for i,j in canvas:
        canvas[i,j] = color[i//zoom,j//zoom]
        # canvas[i,j][0] = (v.cur[i,j][0]**2+v.cur[i,j][1]**2)/50
        if obstacle[i,j]:
            canvas[i,j] = ti.Vector([0.5,0.5,0.5])
        
        if -width<(i-center)<width and j<5:
            canvas[i,j] *= 0.9


window = ti.GUI("Euler 2D Simulation", res=(eulerSimParam["shape"][0]*zoom, eulerSimParam["shape"][1]*zoom))
mouseX, mouseY = 0,0
init()
# save_per_frame = 5
# counter = 0
# counter2 = 0
while window.running:
    simulate()
    paint(color.cur)
    # if counter % 5==0:
    #     ti.tools.imwrite(canvas.to_numpy(), './imgs1/euler%d.png'%counter2)
    #     counter2+=1
    # counter += 1
    window.set_image(canvas)
    window.show()