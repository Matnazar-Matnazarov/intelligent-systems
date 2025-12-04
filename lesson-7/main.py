import random
import tkinter as tk
from tkinter import ttk
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Agent:
    def __init__(self, kind):
        self.kind = kind

class Grid:
    def __init__(self, size, share_A=0.45, share_B=0.45, seed=42, neighborhood='moore', share_C=0.0):
        random.seed(seed)
        self.n = size
        self.neighborhood = neighborhood
        total = size * size
        num_A = int(total * share_A)
        num_B = int(total * share_B)
        num_C = int(total * share_C)
        items = ['A'] * num_A + ['B'] * num_B + ['C'] * num_C + [None] * (total - num_A - num_B - num_C)
        random.shuffle(items)
        it = iter(items)
        self.cells = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                k = next(it)
                self.cells[i][j] = Agent(k) if k else None

    def neighbors(self, r, c):
        if self.neighborhood == 'moore':
            dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        else:
            dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        for dr, dc in dirs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.n and 0 <= cc < self.n:
                yield rr, cc

    def similar_share(self, r, c):
        me = self.cells[r][c]
        if me is None:
            return None
        same, tot = 0, 0
        for rr, cc in self.neighbors(r, c):
            other = self.cells[rr][cc]
            if other is None:
                continue
            tot += 1
            if other.kind == me.kind:
                same += 1
        return (same / tot) if tot > 0 else 1.0

    def empty_positions(self):
        return [(i, j) for i in range(self.n) for j in range(self.n) if self.cells[i][j] is None]

    def population_stats(self):
        cnt = Counter()
        for i in range(self.n):
            for j in range(self.n):
                a = self.cells[i][j]
                cnt['Empty' if a is None else a.kind] += 1
        return cnt

class Simulation:
    def __init__(self, size=20, share_A=0.45, share_B=0.45, share_C=0.0,
                 th_A=0.5, th_B=0.5, th_C=0.5, max_steps=200,
                 seed=42, neighborhood='moore', move_strategy='best'):
        self.grid = Grid(size, share_A, share_B, seed=seed,
                         neighborhood=neighborhood, share_C=share_C)
        self.th_A, self.th_B, self.th_C = th_A, th_B, th_C
        self.max_steps = max_steps
        self.move_strategy = move_strategy
        self.history = []
        self.step_count = 0

    def dissatisfied_for(self, kind, s):
        if kind == 'A':
            return s < self.th_A
        if kind == 'B':
            return s < self.th_B
        if kind == 'C':
            return s < self.th_C
        return False

    def step(self):
        n = self.grid.n
        dissatisfied, similarities = [], []
        for i in range(n):
            for j in range(n):
                a = self.grid.cells[i][j]
                if a is None:
                    continue
                s = self.grid.similar_share(i, j)
                similarities.append(s)
                if self.dissatisfied_for(a.kind, s):
                    dissatisfied.append((i, j))
        empties = self.grid.empty_positions()
        random.shuffle(dissatisfied)
        random.shuffle(empties)
        moves = 0
        while dissatisfied and empties:
            ri, rj = dissatisfied.pop()
            ei, ej = empties.pop()
            self.grid.cells[ei][ej] = self.grid.cells[ri][rj]
            self.grid.cells[ri][rj] = None
            moves += 1
        self.step_count += 1
        dr = len(dissatisfied) / max(1, (n * n - len(empties)))
        ms = sum(similarities) / len(similarities) if similarities else 1.0
        seg = sum(abs(s - 0.5) for s in similarities) / len(similarities) if similarities else 0.0
        self.history.append((self.step_count, dr, ms, seg))
        return moves, dr, ms, seg

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Schelling Model Simulation")
        self.geometry("900x600")
        self.configure(bg="#222831")
        self.sim = None
        self.running = False
        self._after_id = None
        self._build_ui()

    def _build_ui(self):
        frm_left = ttk.Frame(self, padding=10)
        frm_left.pack(side="left", fill="y")

        ttk.Label(frm_left, text="Grid Size:").pack()
        self.var_N = tk.IntVar(value=20)
        ttk.Entry(frm_left, textvariable=self.var_N, width=10).pack(pady=2)

        ttk.Button(frm_left, text="Init Grid", command=self.init_grid).pack(pady=5)
        ttk.Button(frm_left, text="Step", command=self.step_once).pack(pady=5)
        ttk.Button(frm_left, text="Run", command=self.run_sim).pack(pady=5)
        ttk.Button(frm_left, text="Stop", command=self.stop_sim).pack(pady=5)
        ttk.Button(frm_left, text="Reset", command=self.reset_sim).pack(pady=5)

        self.canvas = tk.Canvas(self, width=400, height=400, bg="white")
        self.canvas.pack(side="left", padx=20, pady=20)

        # Grafik
        self.fig, self.ax = plt.subplots(figsize=(4,3))
        self.ax.set_title("Metrics")
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Values")
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_graph.get_tk_widget().pack(side="right", fill="both", expand=True)

    def init_grid(self):
        n = self.var_N.get()
        self.sim = Simulation(size=n)
        self._draw_grid()
        self.ax.clear()
        self.ax.set_title("Metrics")
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Values")
        self.canvas_graph.draw()
        self.running = False

    def _draw_grid(self):
        self.canvas.delete("all")
        n = self.sim.grid.n
        size = 400 / n
        self.rects = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                color = self._color_for(self.sim.grid.cells[i][j])
                x0, y0 = j*size, i*size
                rect = self.canvas.create_rectangle(x0, y0, x0+size, y0+size, fill=color, outline="gray")
                self.rects[i][j] = rect

    def _color_for(self, cell):
        if cell is None:
            return "white"
        if cell.kind == "A":
            return "#00ADB5"
        if cell.kind == "B":
            return "#FF5722"
        if cell.kind == "C":
            return "#4CAF50"
        return "white"

    def _update_grid(self):
        n = self.sim.grid.n
        for i in range(n):
            for j in range(n):
                color = self._color_for(self.sim.grid.cells[i][j])
                self.canvas.itemconfig(self.rects[i][j], fill=color)

    def step_once(self):
        if not self.sim:
            return
        moves, dr, ms, seg = self.sim.step()
        self._update_grid()
        self._update_graph()

    def _update_graph(self):
        steps = [h[0] for h in self.sim.history]
        drs = [h[1] for h in self.sim.history]
        ms = [h[2] for h in self.sim.history]
        seg = [h[3] for h in self.sim.history]
        self.ax.clear()
        self.ax.plot(steps, drs, label="Dissatisfied", color="red")
        self.ax.plot(steps, ms, label="Mean Similarity", color="blue")
        self.ax.plot(steps, seg, label="Seg. Index", color="green")
        self.ax.legend()
        self.canvas_graph.draw()

    def run_sim(self):
        if not self.sim or self.running:
            return
        self.running = True
        self._loop()

    def _loop(self):
        if not self.running:
            return
        moves, dr, ms, seg = self.sim.step()
        self._update_grid()
        self._update_graph()
        if moves == 0 or self.sim.step_count >= self.sim.max_steps:
            self.running = False
            return
        self._after_id = self.after(100, self._loop)

    def stop_sim(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.running = False

    def reset_sim(self):
        self.stop_sim()
        self.sim = None
        self.canvas.delete("all")
        self.ax.clear()
        self.canvas_graph.draw()

if __name__ == "__main__":
    app = App()
    app.mainloop()
