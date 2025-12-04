import math, random
import tkinter as tk
from tkinter import ttk

# ----------------- small vector helpers -----------------
def add(a,b): return (a[0]+b[0], a[1]+b[1])
def sub(a,b): return (a[0]-b[0], a[1]-b[1])
def mul(a,k): return (a[0]*k, a[1]*k)
def norm(a):
    n = math.hypot(a[0], a[1])
    return (a[0]/n, a[1]/n) if n>1e-12 else (0.0,0.0)
def limit(vec, maxlen):
    x,y = vec
    n2 = x*x + y*y
    if n2 == 0: return (0.0,0.0)
    if n2 > maxlen*maxlen:
        n = math.sqrt(n2)
        return (x*maxlen/n, y*maxlen/n)
    return vec

# ----------------- torus helpers -----------------
def wrap(x, L):
    while x < 0: x += L
    while x >= L: x -= L
    return x

def torus_delta(a, b, L):
    d = b - a
    if d > L/2: d -= L
    if d < -L/2: d += L
    return d

# ----------------- Agent & Sim (simplified extended) -----------------
class Agent:
    def __init__(self, x,y,vx,vy, atype='A', energy=1.0, leader=False):
        self.x = x; self.y = y
        self.vx = vx; self.vy = vy
        self.type = atype
        self.energy = energy
        self.is_leader = leader

    def pos(self): return (self.x, self.y)
    def vel(self): return (self.vx, self.vy)

class MultiAgentSim:
    def __init__(self,
                 W=160.0, H=100.0, N=60, seed=1,
                 v_max=2.0, r_neigh=12.0,
                 w_sep=1.2, w_coh=0.8, w_align=0.9,
                 sep_dist=3.5, noise=0.05, dt=1.0,
                 leader=False, leader_dir=(1.0,0.0), leader_influence=0.6,
                 reflective=False, use_cell_list=True,
                 energy_model=False, energy_cost=0.01, recharge_zones=None,
                 types_fraction=(0.4,0.3,0.3)):
        random.seed(seed)
        self.W, self.H = W, H
        self.N = N
        self.v_max = v_max
        self.r_neigh = r_neigh
        self.w_sep = w_sep; self.w_coh = w_coh; self.w_align = w_align
        self.sep_dist = sep_dist
        self.noise = noise
        self.dt = dt
        self.t = 0
        self.leader_enabled = leader
        self.leader_dir = norm(leader_dir)
        self.leader_influence = leader_influence
        self.reflective = reflective
        self.use_cell_list = use_cell_list
        self.energy_model = energy_model
        self.energy_cost = energy_cost
        self.recharge_zones = recharge_zones or []

        # prepare types list
        fa, fb, fc = types_fraction
        types = []
        for k in range(N):
            r = random.random()
            if r < fa: types.append('A')
            elif r < fa+fb: types.append('B')
            else: types.append('C')

        self.agents = []
        leader_assigned = False
        for i in range(N):
            x = random.random()*W
            y = random.random()*H
            ang = random.random()*2*math.pi
            v = (math.cos(ang)*v_max*0.5, math.sin(ang)*v_max*0.5)
            is_leader = False
            if self.leader_enabled and not leader_assigned:
                is_leader=True; leader_assigned=True
                v = mul(self.leader_dir, v_max)
            a = Agent(x,y,v[0],v[1], atype=types[i], energy=1.0, leader=is_leader)
            self.agents.append(a)

        # cell list
        self.cell_size = max(1.0, self.r_neigh)
        self.nx = max(1, int(self.W / self.cell_size))
        self.ny = max(1, int(self.H / self.cell_size))

    # neighbors
    def build_cell_list(self):
        cells = {}
        for idx,a in enumerate(self.agents):
            cx = int(a.x / self.cell_size) % self.nx
            cy = int(a.y / self.cell_size) % self.ny
            cells.setdefault((cx,cy), []).append(idx)
        return cells

    def neighbors_cell(self, i, cells):
        ai = self.agents[i]
        res = []
        cx = int(ai.x / self.cell_size) % self.nx
        cy = int(ai.y / self.cell_size) % self.ny
        for dx_cell in (-1,0,1):
            for dy_cell in (-1,0,1):
                ncx = (cx+dx_cell) % self.nx
                ncy = (cy+dy_cell) % self.ny
                for j in cells.get((ncx,ncy), []):
                    if j == i: continue
                    aj = self.agents[j]
                    if self.reflective:
                        dx = aj.x - ai.x; dy = aj.y - ai.y
                    else:
                        dx = torus_delta(ai.x, aj.x, self.W); dy = torus_delta(ai.y, aj.y, self.H)
                    d2 = dx*dx + dy*dy
                    if d2 <= self.r_neigh*self.r_neigh:
                        res.append((j, dx, dy, math.sqrt(d2)))
        return res

    def neighbors_bruteforce(self, i):
        ai = self.agents[i]
        res=[]
        for j,aj in enumerate(self.agents):
            if i==j: continue
            if self.reflective:
                dx = aj.x - ai.x; dy = aj.y - ai.y
            else:
                dx = torus_delta(ai.x, aj.x, self.W); dy = torus_delta(ai.y, aj.y, self.H)
            d2 = dx*dx + dy*dy
            if d2 <= self.r_neigh*self.r_neigh:
                res.append((j, dx, dy, math.sqrt(d2)))
        return res

    def neighbors(self, i, cells=None):
        if self.use_cell_list:
            return self.neighbors_cell(i, cells)
        else:
            return self.neighbors_bruteforce(i)

    def step(self):
        new_accs = []
        cells = self.build_cell_list() if self.use_cell_list else None

        # compute accelerations
        for i in range(self.N):
            ai = self.agents[i]
            if ai.is_leader:
                # leader keeps direction
                desired_v = mul(self.leader_dir, self.v_max)
                acc = sub(desired_v, ai.vel())
                acc = add(acc, ((random.random()-0.5)*self.noise, (random.random()-0.5)*self.noise))
                new_accs.append(acc); continue

            neigh = self.neighbors(i, cells)
            if not neigh:
                jitter = ((random.random()-0.5)*self.noise, (random.random()-0.5)*self.noise)
                new_accs.append(jitter); continue

            sep=(0.0,0.0); cx=0.0; cy=0.0; avx=0.0; avy=0.0; count=0
            for j, dx, dy, d in neigh:
                count+=1
                cx += dx; cy += dy
                vj = self.agents[j].vel(); avx += vj[0]; avy += vj[1]
                if d < self.sep_dist and d>1e-9:
                    push = mul((dx/d, dy/d), - (self.sep_dist - d) / self.sep_dist)
                    sep = add(sep, push)

            coh = (0.0,0.0); align=(0.0,0.0)
            if count>0:
                coh = mul((-cx/count, -cy/count), 1.0)
                align = sub((avx/count, avy/count), ai.vel())

            # leader influence
            if self.leader_enabled:
                leader_v = None
                for a in self.agents:
                    if a.is_leader:
                        leader_v = a.vel(); break
                if leader_v is not None:
                    align = add(align, mul(sub(leader_v, ai.vel()), self.leader_influence))

            acc = (0.0,0.0)
            acc = add(acc, mul(sep, self.w_sep))
            acc = add(acc, mul(coh, self.w_coh))
            acc = add(acc, mul(align, self.w_align))
            acc = add(acc, ((random.random()-0.5)*self.noise, (random.random()-0.5)*self.noise))
            new_accs.append(acc)

        # integrate
        for i,a in enumerate(new_accs):
            ag = self.agents[i]
            vx, vy = add(ag.vel(), mul(a, self.dt))
            vx, vy = limit((vx,vy), self.v_max)
            nx = ag.x + vx*self.dt
            ny = ag.y + vy*self.dt
            if self.reflective:
                if nx < 0: nx = -nx; vx = -vx
                if nx >= self.W: nx = 2*self.W - nx; vx = -vx
                if ny < 0: ny = -ny; vy = -vy
                if ny >= self.H: ny = 2*self.H - ny; vy = -vy
                nx = max(0.0, min(self.W, nx)); ny = max(0.0, min(self.H, ny))
            else:
                nx = wrap(nx, self.W); ny = wrap(ny, self.H)

            if self.energy_model and not ag.is_leader:
                # harakat tezligiga qarab energiya kamayadi
                speed = math.hypot(vx, vy)
                ag.energy -= self.energy_cost * speed * self.dt * 0.2

                # energiya zaryad zonalari orqali tiklanadi
                for zx, zy, zr, rate in self.recharge_zones:
                    if self.reflective:
                        dx = nx - zx
                        dy = ny - zy
                    else:
                        dx = torus_delta(nx, zx, self.W)
                        dy = torus_delta(ny, zy, self.H)
                    if math.hypot(dx, dy) <= zr:
                        ag.energy += rate * self.dt

                # energiyani 0-1 oralig‘ida ushlab turamiz
                ag.energy = max(0.0, min(1.0, ag.energy))

                # ✴️ kuchliroq energiya-ta'sirli tezlik modeli
                # past energiyada 10% tezlik, yuqorida 200%
                energy_factor = energy_factor = 0.5 + 1.5 * (ag.energy ** 0.6)
                vx *= energy_factor
                vy *= energy_factor

                # rangni energiyaga qarab o‘zgartiramiz (vizual sezilsin)
                ag.color = f"#{int(255*(1-ag.energy)):02x}{int(255*ag.energy):02x}40"

            # pozitsiyani yangilaymiz
            ag.x, ag.y, ag.vx, ag.vy = nx, ny, vx, vy

        self.t += 1

    # metrics
    def polarization(self):
        if self.N==0: return 0.0
        v_sum=(0.0,0.0); spd_sum=0.0
        for a in self.agents:
            v_sum = add(v_sum, a.vel())
            spd_sum += math.hypot(a.vx,a.vy)
        if spd_sum < 1e-12: return 0.0
        return math.hypot(v_sum[0], v_sum[1]) / spd_sum

    def mean_speed(self):
        if self.N==0: return 0.0
        return sum(math.hypot(a.vx,a.vy) for a in self.agents) / self.N

    def mean_nn_distance(self):
        if self.N<=1: return 0.0
        total=0.0
        for i in range(self.N):
            ai = self.agents[i]
            best=1e9
            for j in range(self.N):
                if i==j: continue
                if self.reflective:
                    dx = self.agents[j].x - ai.x; dy = self.agents[j].y - ai.y
                else:
                    dx = torus_delta(ai.x, self.agents[j].x, self.W); dy = torus_delta(ai.y, self.agents[j].y, self.H)
                d = math.hypot(dx,dy)
                if d < best: best=d
            total += best
        return total / self.N

# ----------------- GUI -----------------
class MiniBoidsGUI:
    COLORS = {'A':'#4E79A7', 'B':'#E15759', 'C':'#59A14F', 'L':'#F28E2B'}
    def __init__(self, root):
        self.root = root
        root.title("Mini-Boids — Minimal GUI")
        root.geometry("980x640")
        root.configure(bg="#F6F7F9")

        # left: canvas, right: controls
        self.canvas_frame = tk.Frame(root, bg="#F6F7F9")
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self.ctrl_frame = tk.Frame(root, bg="#FFFFFF", bd=0, relief="flat")
        self.ctrl_frame.pack(side="right", fill="y", padx=12, pady=12)

        # simulation canvas
        self.CANVAS_W, self.CANVAS_H = 760, 580
        self.canvas = tk.Canvas(self.canvas_frame, width=self.CANVAS_W, height=self.CANVAS_H,
                                bg="#FFFFFF", bd=0, highlightthickness=0)
        self.canvas.pack(expand=True, padx=8, pady=8)

        # create initial sim (⚠️ FULL SIZE, not half)
        self.sim = MultiAgentSim(W=self.CANVAS_W, H=self.CANVAS_H, N=80,
                                 r_neigh=12.0, v_max=2.2,
                                 w_sep=1.2, w_coh=0.8, w_align=0.9,
                                 sep_dist=3.5, noise=0.05,
                                 leader=True, leader_dir=(1,0), leader_influence=0.6,
                                 reflective=False, use_cell_list=True,
                                 energy_model=False,
                                 recharge_zones=[(self.CANVAS_W/2, self.CANVAS_H/2, 40, 0.05)])

        self.build_controls()
        self.running = False
        self.after_id = None
        self.speed_scale.set(1.0)
        self.draw_once()


    def build_controls(self):
        pad = dict(padx=8, pady=6)
        header = tk.Label(self.ctrl_frame, text="Simulation Controls", bg="#FFFFFF", fg="#333333",
                          font=("Segoe UI", 12, "bold"))
        header.pack(pady=(8,4))

        # Start / Pause / Reset buttons
        btn_frame = tk.Frame(self.ctrl_frame, bg="#FFFFFF")
        btn_frame.pack(fill="x", **pad)
        self.start_btn = tk.Button(btn_frame, text="Start", command=self.start, bg="#4E79A7", fg="white",
                                   relief="flat", padx=8, pady=6)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0,4))
        self.pause_btn = tk.Button(btn_frame, text="Pause", command=self.pause, bg="#E15759", fg="white",
                                   relief="flat", padx=8, pady=6)
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=(4,0))

        reset_btn = tk.Button(self.ctrl_frame, text="Reset", command=self.reset, bg="#FFFFFF", fg="#333333",
                              relief="ridge", padx=8, pady=6)
        reset_btn.pack(fill="x", **pad)

        # Parameter sliders
        lbl = tk.Label(self.ctrl_frame, text="Parameters", bg="#FFFFFF", fg="#333333", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=(12,4))

        def mk_slider(text, var_from, var_to, init, resolution=0.1):
            f = tk.Frame(self.ctrl_frame, bg="#FFFFFF")
            f.pack(fill="x", padx=8, pady=4)
            tk.Label(f, text=text, bg="#FFFFFF", anchor="w").pack(fill="x")
            s = tk.Scale(f, from_=var_from, to=var_to, resolution=resolution, orient="horizontal",
                         length=220, showvalue=True)
            s.set(init)
            s.pack()
            return s

        self.sep_scale = mk_slider("Separation weight", 0.0, 3.0, self.sim.w_sep, 0.05)
        self.coh_scale = mk_slider("Cohesion weight", 0.0, 3.0, self.sim.w_coh, 0.05)
        self.aln_scale = mk_slider("Alignment weight", 0.0, 3.0, self.sim.w_align, 0.05)
        self.noise_scale = mk_slider("Noise", 0.0, 0.5, self.sim.noise, 0.01)
        self.rnei_scale = mk_slider("Neighbor radius", 2.0, 40.0, self.sim.r_neigh, 1.0)
        self.speed_scale = mk_slider("Play speed (x)", 0.1, 3.0, 1.0, 0.1)

        # toggles
        tframe = tk.Frame(self.ctrl_frame, bg="#FFFFFF")
        tframe.pack(fill="x", padx=8, pady=6)
        self.leader_var = tk.BooleanVar(value=self.sim.leader_enabled)
        self.reflect_var = tk.BooleanVar(value=self.sim.reflective)
        self.energy_var = tk.BooleanVar(value=self.sim.energy_model)
        tk.Checkbutton(tframe, text="Leader", variable=self.leader_var, bg="#FFFFFF", anchor="w").pack(anchor="w")
        tk.Checkbutton(tframe, text="Reflective walls", variable=self.reflect_var, bg="#FFFFFF", anchor="w").pack(anchor="w")
        tk.Checkbutton(tframe, text="Energy model", variable=self.energy_var, bg="#FFFFFF", anchor="w").pack(anchor="w")

        # metrics
        sep = ttk.Separator(self.ctrl_frame, orient="horizontal")
        sep.pack(fill="x", padx=8, pady=10)
        metrics_label = tk.Label(self.ctrl_frame, text="Metrics", bg="#FFFFFF", fg="#333333", font=("Segoe UI", 10, "bold"))
        metrics_label.pack(pady=(0,4))
        self.pol_label = tk.Label(self.ctrl_frame, text="Polarization: —", bg="#FFFFFF")
        self.pol_label.pack(anchor="w", padx=8)
        self.ms_label = tk.Label(self.ctrl_frame, text="Mean speed: —", bg="#FFFFFF")
        self.ms_label.pack(anchor="w", padx=8)
        self.nn_label = tk.Label(self.ctrl_frame, text="Mean NN dist: —", bg="#FFFFFF")
        self.nn_label.pack(anchor="w", padx=8)

        # small footer
        tk.Label(self.ctrl_frame, text="Tip: change params and press Reset", bg="#FFFFFF", fg="#777777",
                 font=("Segoe UI", 8)).pack(side="bottom", pady=8)

    def reset(self):
        # apply sliders/toggles to sim and re-create (⚠️ FULL SIZE)
        W, H = self.CANVAS_W, self.CANVAS_H
        N = 80
        self.sim = MultiAgentSim(W=W, H=H, N=N,
                                 r_neigh=self.rnei_scale.get(),
                                 v_max=2.2,
                                 w_sep=self.sep_scale.get(), w_coh=self.coh_scale.get(), w_align=self.aln_scale.get(),
                                 sep_dist=3.5, noise=self.noise_scale.get(),
                                 leader=self.leader_var.get(), leader_dir=(1,0), leader_influence=0.6,
                                 reflective=self.reflect_var.get(), use_cell_list=True,
                                 energy_model=self.energy_var.get(),
                                 recharge_zones=[(W/2, H/2, 40, 0.05)])
        self.draw_once()

    def start(self):
        if self.running: return
        self.running = True
        self.run_loop()

    def pause(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id); self.after_id = None

    def run_loop(self):
        # step sim several times depending on speed multiplier
        speed = self.speed_scale.get()
        steps = max(1, int(speed))
        for _ in range(steps):
            self.sim.step()
        self.draw_once()
        # update metrics
        pol = self.sim.polarization(); ms = self.sim.mean_speed(); mnn = self.sim.mean_nn_distance()
        self.pol_label.config(text=f"Polarization: {pol:0.3f}")
        self.ms_label.config(text=f"Mean speed: {ms:0.3f}")
        self.nn_label.config(text=f"Mean NN dist: {mnn:0.2f}")
        if self.running:
            self.after_id = self.root.after(60, self.run_loop)  # ~16 FPS

    def draw_once(self):
        self.canvas.delete("all")

        # Energiya zonalarini chizish
        for zx, zy, zr, rate in self.sim.recharge_zones:
            self.canvas.create_oval(zx-zr, zy-zr, zx+zr, zy+zr,
                                    outline="#8FD694", width=2, dash=(4, 3))
            self.canvas.create_oval(zx-6, zy-6, zx+6, zy+6,
                                    fill="#93D977", outline="")

        # Agentlarni chizish
        for a in self.sim.agents:
            sx, sy = a.x, a.y
            r = 4
            color = self.COLORS.get('L' if a.is_leader else a.type, "#444444")

            if self.sim.energy_model:
                fade = 0.4 + 0.6 * a.energy
                def blend(hexc, f):
                    hexc = hexc.lstrip('#')
                    rint, gint, bint = int(hexc[0:2],16), int(hexc[2:4],16), int(hexc[4:6],16)
                    r2 = int(255 + (rint-255)*f)
                    g2 = int(255 + (gint-255)*f)
                    b2 = int(255 + (bint-255)*f)
                    return f"#{r2:02x}{g2:02x}{b2:02x}"
                color = blend(color, fade)

            self.canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill=color, outline="")

            # Harakat yo‘nalishi
            hx = sx + a.vx*3
            hy = sy + a.vy*3
            self.canvas.create_line(sx, sy, hx, hy, fill="#222222", width=1)

            # Lider belgisi
            if a.is_leader:
                self.canvas.create_oval(sx-7, sy-7, sx+7, sy+7, outline="#F28E2B", width=2)

            # Energiya qiymatini agent tepasiga yozamiz
            if self.sim.energy_model:
                energy_text = f"{a.energy:.2f}"
                self.canvas.create_text(sx, sy - 10, text=energy_text,
                                        fill="#333333", font=("Segoe UI", 8, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = MiniBoidsGUI(root)
    root.mainloop()
