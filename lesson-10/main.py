import tkinter as tk
from tkinter import ttk, messagebox
import math
import random

# ---------------- Objective functions ----------------
def sphere(x):
    return sum(xi * xi for xi in x)

def rastrigin(x):
    n = len(x)
    return 10 * n + sum(xi * xi - 10 * math.cos(2 * math.pi * xi) for xi in x)

OBJECTIVES = {"sphere": sphere, "rastrigin": rastrigin}

# ---------------- Individual ----------------
class Individual:
    def __init__(self, x, sigmas):
        self.x = x
        self.sigmas = sigmas
        self.f = None

# ---------------- Utility functions ----------------
def clamp_vec(x, low, high):
    return [max(low, min(high, xi)) for xi in x]

def mean_vectors(pop):
    n = len(pop[0].x)
    mx = [0.0] * n
    ms = [0.0] * n
    for ind in pop:
        for i in range(n):
            mx[i] += ind.x[i]
            ms[i] += ind.sigmas[i]
    return [v / len(pop) for v in mx], [v / len(pop) for v in ms]

def recombine(parents, method="arithmetic"):
    if method == "arithmetic":
        return mean_vectors(parents)
    elif method == "uniform":
        n = len(parents[0].x)
        x = [random.choice([p.x[i] for p in parents]) for i in range(n)]
        sigmas = [random.choice([p.sigmas[i] for p in parents]) for i in range(n)]
        return x, sigmas
    elif method == "median":
        n = len(parents[0].x)
        x = [sorted([p.x[i] for p in parents])[len(parents)//2] for i in range(n)]
        sigmas = [sorted([p.sigmas[i] for p in parents])[len(parents)//2] for i in range(n)]
        return x, sigmas
    else:
        return mean_vectors(parents)

def mutate(x, sigmas, tau0, tau, bounds):
    n = len(x)
    N0 = random.gauss(0.0, 1.0)
    new_sigmas = []
    for i in range(n):
        Ni = random.gauss(0.0, 1.0)
        si = sigmas[i] * math.exp(tau0 * N0 + tau * Ni)
        si = max(1e-10, min(si, (bounds[1] - bounds[0])))
        new_sigmas.append(si)
    new_x = [x[i] + new_sigmas[i] * random.gauss(0.0, 1.0) for i in range(n)]
    new_x = clamp_vec(new_x, bounds[0], bounds[1])
    return new_x, new_sigmas

def evaluate(pop, obj):
    for ind in pop:
        ind.f = obj(ind.x)

def select_best(pop, mu):
    return sorted(pop, key=lambda s: s.f)[:mu]

# ---------------- ES main loop ----------------
def es_optimize(n_dim, bounds, obj_name, generations, mu, lambd, seed, init_sigma, plus_selection, recomb_method):
    random.seed(seed)
    obj = OBJECTIVES[obj_name]
    low, high = bounds
    tau0 = 1.0 / math.sqrt(2.0 * n_dim)
    tau  = 1.0 / math.sqrt(2.0 * math.sqrt(n_dim))

    parents = []
    for _ in range(mu):
        x = [random.uniform(low, high) for _ in range(n_dim)]
        sig = [init_sigma] * n_dim
        parents.append(Individual(x, sig))
    evaluate(parents, obj)
    best = min(parents, key=lambda s: s.f)

    log = []
    for g in range(1, generations + 1):
        offspring = []
        for _ in range(lambd):
            k = max(2, min(mu, 5))
            pars = random.sample(parents, k=k)
            rx, rs = recombine(pars, method=recomb_method)
            cx, cs = mutate(rx, rs, tau0, tau, bounds)
            offspring.append(Individual(cx, cs))
        evaluate(offspring, obj)
        pool = parents + offspring if plus_selection else offspring
        parents = select_best(pool, mu)
        cur_best = parents[0]
        if cur_best.f < best.f:
            best = cur_best
        if g % max(1, generations // 10) == 0 or g == generations:
            log.append((g, best.f, sum(sum(p.sigmas) for p in parents)/(mu*n_dim)))
    return best, log

# ---------------- GUI ----------------
class ESGui:
    def __init__(self, root):
        self.root = root
        root.title("Evolution Strategy (μ,λ)-ES GUI")

        tk.Label(root, text="Objective Function").grid(row=0, column=0)
        self.obj_var = tk.StringVar(value="rastrigin")
        ttk.Combobox(root, textvariable=self.obj_var, values=list(OBJECTIVES.keys())).grid(row=0, column=1)

        tk.Label(root, text="Dimensions (n_dim)").grid(row=1, column=0)
        self.dim_entry = tk.Entry(root)
        self.dim_entry.insert(0, "10")
        self.dim_entry.grid(row=1, column=1)

        tk.Label(root, text="Generations").grid(row=2, column=0)
        self.gen_entry = tk.Entry(root)
        self.gen_entry.insert(0, "200")
        self.gen_entry.grid(row=2, column=1)

        tk.Label(root, text="μ (parents)").grid(row=3, column=0)
        self.mu_entry = tk.Entry(root)
        self.mu_entry.insert(0, "15")
        self.mu_entry.grid(row=3, column=1)

        tk.Label(root, text="λ (offspring)").grid(row=4, column=0)
        self.lam_entry = tk.Entry(root)
        self.lam_entry.insert(0, "100")
        self.lam_entry.grid(row=4, column=1)

        tk.Label(root, text="Initial sigma").grid(row=5, column=0)
        self.sig_entry = tk.Entry(root)
        self.sig_entry.insert(0, "0.5")
        self.sig_entry.grid(row=5, column=1)

        tk.Label(root, text="Plus Selection").grid(row=6, column=0)
        self.plus_var = tk.BooleanVar()
        tk.Checkbutton(root, variable=self.plus_var).grid(row=6, column=1)

        tk.Label(root, text="Recombination").grid(row=7, column=0)
        self.rec_var = tk.StringVar(value="arithmetic")
        ttk.Combobox(root, textvariable=self.rec_var, values=["arithmetic","median","uniform"]).grid(row=7, column=1)

        tk.Button(root, text="Run ES", command=self.run_es).grid(row=8, column=0, columnspan=2, pady=10)
        self.output = tk.Text(root, height=15, width=60)
        self.output.grid(row=9, column=0, columnspan=2)

    def run_es(self):
        try:
            n_dim = int(self.dim_entry.get())
            generations = int(self.gen_entry.get())
            mu = int(self.mu_entry.get())
            lambd = int(self.lam_entry.get())
            init_sigma = float(self.sig_entry.get())
            obj_name = self.obj_var.get()
            plus_selection = self.plus_var.get()
            recomb_method = self.rec_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input")
            return

        self.output.delete(1.0, tk.END)
        best, log = es_optimize(n_dim=n_dim,
                                bounds=(-5,5),
                                obj_name=obj_name,
                                generations=generations,
                                mu=mu,
                                lambd=lambd,
                                seed=random.randint(0,1000),
                                init_sigma=init_sigma,
                                plus_selection=plus_selection,
                                recomb_method=recomb_method)
        self.output.insert(tk.END, f"Best f = {best.f:.6f}\n")
        self.output.insert(tk.END, "Best x (first 5 dims): " + str([round(v,4) for v in best.x[:5]]) + "\n\n")
        self.output.insert(tk.END, "Gen | Best f | Mean sigma\n")
        for g,f,s in log:
            self.output.insert(tk.END, f"{g:4d} | {f:.6f} | {s:.4f}\n")

# ---------------- Run GUI ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ESGui(root)
    root.mainloop()
