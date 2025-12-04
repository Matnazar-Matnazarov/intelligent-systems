import random, math, time, threading
import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def fitness(x):
    return x * math.sin(10 * math.pi * x) + 1.0

def init_population(size):
    return [random.random() for _ in range(size)]

def selection(pop, k=3):
    best = None
    for _ in range(k):
        ind = random.choice(pop)
        if (best is None) or (fitness(ind) > fitness(best)):
            best = ind
    return best

def crossover(p1, p2, rate=0.8):
    if random.random() < rate:
        alpha = random.random()
        return alpha * p1 + (1 - alpha) * p2
    return p1

def mutate(ind, rate=0.1, scale=0.1):
    if random.random() < rate:
        ind += random.uniform(-scale, scale)
        ind = max(0.0, min(1.0, ind))
    return ind


class GeneticGUI(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("🧬 Genetic Algorithm Optimizer - Modern Edition")
        self.geometry("1000x650")
        self.resizable(False, False)
        self.running = False

        ttk.Label(self, text="GENETIC ALGORITHM OPTIMIZATION", font=("Segoe UI", 16, "bold"), bootstyle="primary").pack(pady=15)

        # ---- Frames ----
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

        left = ttk.Frame(main_frame, bootstyle="secondary")
        left.pack(side=LEFT, fill=Y, padx=(0, 15))

        right = ttk.Frame(main_frame, bootstyle="secondary")
        right.pack(side=LEFT, fill=BOTH, expand=True)

        # ---- Controls ----
        ttk.Label(left, text="Algorithm Settings", font=("Segoe UI", 13, "bold")).pack(pady=10)

        self.gen = ttk.Entry(left, width=10)
        self.pop = ttk.Entry(left, width=10)
        self.cross = ttk.Entry(left, width=10)
        self.mut = ttk.Entry(left, width=10)

        self._add_field(left, "Avlodlar soni:", self.gen, "50")
        self._add_field(left, "Populyatsiya hajmi:", self.pop, "30")
        self._add_field(left, "Krossover ehtimoli:", self.cross, "0.8")
        self._add_field(left, "Mutatsiya ehtimoli:", self.mut, "0.1")

        self.start_btn = ttk.Button(left, text="🚀 Start", bootstyle="success-outline", command=self.start_algorithm)
        self.start_btn.pack(pady=10, fill=X)

        self.stop_btn = ttk.Button(left, text="⛔ Stop", bootstyle="danger-outline", command=self.stop_algorithm, state="disabled")
        self.stop_btn.pack(pady=5, fill=X)

        self.progress = ttk.Progressbar(left, bootstyle="info-striped")
        self.progress.pack(pady=15, fill=X, padx=10)

        ttk.Label(left, text="Log:", font=("Segoe UI", 11, "bold")).pack(pady=(10, 0))
        self.text = tk.Text(left, height=20, width=35, bg="#1e1e1e", fg="#00ff99", font=("Consolas", 10))
        self.text.pack(padx=10, pady=5)

        self.result_lbl = ttk.Label(left, text="", font=("Consolas", 11), bootstyle="light")
        self.result_lbl.pack(pady=10)

        # ---- Right side: Graph ----
        ttk.Label(right, text="Fitness Evolution", font=("Segoe UI", 13, "bold")).pack(pady=10)
        self.figure, self.ax = plt.subplots(figsize=(6.8, 4.5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack()

    def _add_field(self, parent, label, entry, default):
        ttk.Label(parent, text=label).pack(pady=(5, 0))
        entry.insert(0, default)
        entry.pack(pady=3)

    def start_algorithm(self):
        try:
            self.generations = int(self.gen.get())
            self.pop_size = int(self.pop.get())
            self.cross_rate = float(self.cross.get())
            self.mut_rate = float(self.mut.get())
        except ValueError:
            messagebox.showerror("Xato", "Kiritilgan qiymatlar to'g'ri emas!")
            return

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.text.delete("1.0", tk.END)
        self.progress["value"] = 0
        self.result_lbl.config(text="")
        self.ax.clear()

        threading.Thread(target=self._run_algorithm, daemon=True).start()

    def stop_algorithm(self):
        self.running = False
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")

    # ---------- Core Algorithm ----------
    def _run_algorithm(self):
        pop = init_population(self.pop_size)
        best_history = []

        for g in range(self.generations):
            if not self.running:
                self._log(f"\n❌ To‘xtatildi {g}-avlodda.\n")
                return

            new_pop = []
            for _ in range(self.pop_size):
                p1 = selection(pop)
                p2 = selection(pop)
                child = crossover(p1, p2, rate=self.cross_rate)
                child = mutate(child, rate=self.mut_rate)
                new_pop.append(child)

            pop = new_pop
            best = max(pop, key=fitness)
            fit = fitness(best)
            best_history.append(fit)

            self._log(f"Avlod {g+1:02d}: Best x={best:.4f}, Fit={fit:.4f}")
            self._update_progress((g + 1) / self.generations * 100)
            self._update_chart(best_history)
            time.sleep(0.05)

        best_overall = max(pop, key=fitness)
        self.result_lbl.config(text=f"✅ Yakuniy natija:\nEng yaxshi x = {best_overall:.4f}\nFitnes = {fitness(best_overall):.4f}")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    # ---------- Helpers ----------
    def _log(self, msg):
        self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)

    def _update_progress(self, val):
        self.progress["value"] = val
        self.update_idletasks()

    def _update_chart(self, data):
        self.ax.clear()
        self.ax.plot(range(1, len(data) + 1), data, color="#00e6e6", linewidth=2, marker="o")
        self.ax.set_xlabel("Avlodlar")
        self.ax.set_ylabel("Fitnes qiymati")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.canvas.draw()

# ---------- Run ----------
if __name__ == "__main__":
    app = GeneticGUI()
    app.mainloop()
