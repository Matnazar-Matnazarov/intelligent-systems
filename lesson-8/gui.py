import tkinter as tk
from tkinter import ttk
import random
import math

class Agent:
    def __init__(self, kind):
        self.kind = kind
        self.satisfied = False

class Grid:
    def __init__(self, size, empty_ratio=0.2):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        kinds = [1, 2]
        for i in range(size):
            for j in range(size):
                if random.random() > empty_ratio:
                    self.grid[i][j] = Agent(random.choice(kinds))

    def get_neighbors(self, x, y, radius=1):
        neighbors = []
        for i in range(max(0, x-radius), min(self.size, x+radius+1)):
            for j in range(max(0, y-radius), min(self.size, y+radius+1)):
                if (i, j) != (x, y) and self.grid[i][j]:
                    neighbors.append(self.grid[i][j])
        return neighbors

    def update(self, tolerance=0.3):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if not self.grid[i][j]]
        random.shuffle(empty_cells)
        for i in range(self.size):
            for j in range(self.size):
                agent = self.grid[i][j]
                if agent:
                    neighbors = self.get_neighbors(i, j)
                    if not neighbors:
                        continue
                    same_kind = sum(1 for n in neighbors if n.kind == agent.kind)
                    ratio = same_kind / len(neighbors)
                    agent.satisfied = ratio >= tolerance
                    if not agent.satisfied and empty_cells:
                        new_x, new_y = empty_cells.pop()
                        self.grid[new_x][new_y] = agent
                        self.grid[i][j] = None
                        empty_cells.append((i, j))

class SchellingApp(tk.Tk):
    def __init__(self, size=20, cell_size=25):
        super().__init__()
        self.title("Schelling Segregation Model — Vizual Versiya")
        self.geometry("700x750")
        self.resizable(False, False)

        self.size = size
        self.cell_size = cell_size
        self.grid_model = Grid(size)
        self.radius = 2  # qo‘shni radiusi
        self.energy_radius = 5  # energiya radiusi
        self.leader = (random.randint(0, size - 1), random.randint(0, size - 1))

        # --- Frame ---
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Canvas (markazda joylashgan va to‘liq ekranli) ---
        canvas_size = size * cell_size + 100
        self.canvas = tk.Canvas(self.main_frame, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.pack(expand=True)

        # --- Start Button ---
        self.start_btn = ttk.Button(self, text="Start Simulation", command=self.start_simulation)
        self.start_btn.pack(pady=10)

        self.running = False
        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        leader_x, leader_y = self.leader

        # --- Energiya doirasi ---
        cx = leader_y * self.cell_size + self.cell_size / 2 + 50
        cy = leader_x * self.cell_size + self.cell_size / 2 + 50
        energy_radius_px = self.energy_radius * self.cell_size
        self.canvas.create_oval(cx - energy_radius_px, cy - energy_radius_px,
                                cx + energy_radius_px, cy + energy_radius_px,
                                outline="green", width=2, dash=(4, 2))

        # --- Qo‘shni radiusi ---
        radius_px = self.radius * self.cell_size
        self.canvas.create_oval(cx - radius_px, cy - radius_px,
                                cx + radius_px, cy + radius_px,
                                outline="blue", width=2, dash=(4, 2))

        # --- Agentlarni chizish ---
        for i in range(self.size):
            for j in range(self.size):
                agent = self.grid_model.grid[i][j]
                x1 = j * self.cell_size + 50
                y1 = i * self.cell_size + 50
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if agent:
                    color = "skyblue" if agent.kind == 1 else "salmon"
                    if (i, j) == self.leader:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="red", width=3)
                        self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="★", font=("Arial", 12, "bold"))
                    else:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

        # --- Grid chiziqlari ---
        for i in range(self.size + 1):
            self.canvas.create_line(50, 50 + i * self.cell_size,
                                    50 + self.size * self.cell_size, 50 + i * self.cell_size, fill="#ddd")
            self.canvas.create_line(50 + i * self.cell_size, 50,
                                    50 + i * self.cell_size, 50 + self.size * self.cell_size, fill="#ddd")

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.animate()

    def animate(self):
        if not self.running:
            return
        self.grid_model.update()
        self.draw_grid()
        self.after(300, self.animate)

if __name__ == "__main__":
    app = SchellingApp()
    app.mainloop()
