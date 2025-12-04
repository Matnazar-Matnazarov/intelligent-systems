#!/usr/bin/env python3
# fuzzy_gui.py
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import matplotlib
matplotlib.use("TkAgg")

# ----------------- Fuzzy logic code (original, slightly refactored) -----------------

def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a + 1e-12)
    elif b <= x < c:
        return (c - x) / (c - b + 1e-12)
    else:
        return 0.0

def trap(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a + 1e-12)
    elif b <= x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c + 1e-12)
    else:
        return 0.0

def temperature_mu(x):
    return {
        "cold":    trap(x, -10, 0, 10, 20),
        "warm":    tri(x, 15, 25, 35),
        "hot":     trap(x, 30, 35, 40, 45),
        "very_hot":trap(x, 30, 35, 45, 55),
    }

def humidity_mu(x):
    return {
        "dry":     trap(x, -10, 0, 20, 35),
        "normal":  tri(x, 30, 50, 70),
        "humid":   trap(x, 65, 80, 100, 110),
    }

def air_quality_mu(x):
    return {
        "good":    trap(x, -10, 0, 50, 75),
        "medium":  tri(x, 50, 125, 200),
        "poor":    trap(x, 175, 225, 300, 320),
    }

def control_level_sets():
    return {
        "low":       lambda y: trap(y, -10, 0, 30, 45),
        "medium":    lambda y: tri(y, 40, 55, 70),
        "high":      lambda y: tri(y, 65, 80, 90),
        "turbo":     lambda y: trap(y, 90, 95, 100, 110),
    }

RULES = [
    (("cold", "humid"), "medium"),
    (("warm", "dry"), "medium"),
    (("very_hot", "humid"), "turbo"),
    (("hot|warm", "normal"), "high"),
    (("cold", "dry|normal"), "low"),
]

def get_membership(mu_dict, label):
    if '|' in label:
        parts = label.split('|')
        return max(mu_dict.get(p.strip(), 0.0) for p in parts)
    else:
        return mu_dict.get(label, 0.0)

def mamdani(temp, hum, aqi, y_min=0, y_max=100, y_step=0.5):
    temp_mu = temperature_mu(temp)
    hum_mu = humidity_mu(hum)

    y_values = [y_min + i * y_step for i in range(int((y_max - y_min) / y_step) + 1)]
    aggregated = [0.0 for _ in y_values]
    out_sets = control_level_sets()

    for (temp_lbl, hum_lbl), out_lbl in RULES:
        alpha_temp = get_membership(temp_mu, temp_lbl)
        alpha_hum = get_membership(hum_mu, hum_lbl)
        alpha = min(alpha_temp, alpha_hum)
        if alpha <= 0.0:
            continue
        mu_out = out_sets[out_lbl]
        for i, y in enumerate(y_values):
            aggregated[i] = max(aggregated[i], min(alpha, mu_out(y)))

    return y_values, aggregated

def centroid(y_values, mu_values):
    num = 0.0
    den = 0.0
    for y, m in zip(y_values, mu_values):
        num += y * m
        den += m
    return (num / den) if den > 1e-12 else 0.0

def mom(y_values, mu_values):
    max_mu = max(mu_values)
    max_points = [y for y, m in zip(y_values, mu_values) if abs(m - max_mu) < 1e-9]
    return sum(max_points) / len(max_points) if max_points else 0.0

def bisector(y_values, mu_values):
    total_area = sum(mu_values)
    if total_area == 0:
        return 0.0
    acc = 0.0
    for y, m in zip(y_values, mu_values):
        acc += m
        if acc >= total_area / 2:
            return y
    return 0.0

def decide_control_level(temp, hum, aqi, method="centroid"):
    y, mu = mamdani(temp, hum, aqi)
    if method == "centroid":
        crisp = centroid(y, mu)
    elif method == "mom":
        crisp = mom(y, mu)
    elif method == "bisector":
        crisp = bisector(y, mu)
    else:
        raise ValueError(f"Unknown method '{method}'")

    if crisp >= 85:
        grade = "A"
    elif crisp >= 70:
        grade = "B"
    elif crisp >= 55:
        grade = "C"
    elif crisp >= 40:
        grade = "D"
    else:
        grade = "F"
    return crisp, grade, y, mu

def run_tests_collect():
    test_cases = [
        (0, 0, 0),
        (40, 100, 10),
        (25, 50, 150),
        (10, 80, 200),
        (35, 20, 50),
        (45, 90, 300),
        (30, 40, 75),
        (50, 60, 100),
        (55, 30, 180),
        (50, 70, 250),
    ]
    rows = []
    for i, (t, h, aqi) in enumerate(test_cases):
        c1, g1, _, _ = decide_control_level(t, h, aqi, "centroid")
        c2, _, _, _ = decide_control_level(t, h, aqi, "mom")
        c3, _, _, _ = decide_control_level(t, h, aqi, "bisector")
        rows.append((i+1, t, h, aqi, round(c1,2), round(c2,2), round(c3,2), g1))
    return rows

# ----------------- Tkinter GUI -----------------

class FuzzyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fuzzy Weather Control System — Omar style")
        self.geometry("1000x640")
        self.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        # Inputs
        ttk.Label(left, text="Harorat (°C)").pack(anchor=tk.W)
        self.temp_var = tk.DoubleVar(value=30.0)
        self.temp_scale = ttk.Scale(left, from_=-10, to=60, variable=self.temp_var, orient=tk.HORIZONTAL)
        self.temp_scale.pack(fill=tk.X)
        self.temp_entry = ttk.Entry(left, textvariable=self.temp_var)
        self.temp_entry.pack(fill=tk.X, pady=(0,8))

        ttk.Label(left, text="Namlik (%)").pack(anchor=tk.W)
        self.hum_var = tk.DoubleVar(value=60.0)
        self.hum_scale = ttk.Scale(left, from_=-10, to=110, variable=self.hum_var, orient=tk.HORIZONTAL)
        self.hum_scale.pack(fill=tk.X)
        self.hum_entry = ttk.Entry(left, textvariable=self.hum_var)
        self.hum_entry.pack(fill=tk.X, pady=(0,8))

        ttk.Label(left, text="Havo sifati (AQI, 0..300)").pack(anchor=tk.W)
        self.aqi_var = tk.DoubleVar(value=100.0)
        self.aqi_scale = ttk.Scale(left, from_=-10, to=320, variable=self.aqi_var, orient=tk.HORIZONTAL)
        self.aqi_scale.pack(fill=tk.X)
        self.aqi_entry = ttk.Entry(left, textvariable=self.aqi_var)
        self.aqi_entry.pack(fill=tk.X, pady=(0,8))

        ttk.Label(left, text="Defuzzifikatsiya usuli").pack(anchor=tk.W, pady=(8,0))
        self.method_var = tk.StringVar(value="centroid")
        self.method_cb = ttk.Combobox(left, textvariable=self.method_var, values=["centroid","mom","bisector"], state="readonly")
        self.method_cb.pack(fill=tk.X, pady=(0,8))

        # Buttons
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(8,0))
        self.calc_btn = ttk.Button(btn_frame, text="Hisoblash", command=self.on_calculate)
        self.calc_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        self.test_btn = ttk.Button(btn_frame, text="Run Tests", command=self.on_run_tests)
        self.test_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5,0))

        # Results
        res_frame = ttk.LabelFrame(left, text="Natija", padding=8)
        res_frame.pack(fill=tk.X, pady=(10,0))
        ttk.Label(res_frame, text="Crisp qiymat:").grid(row=0, column=0, sticky=tk.W)
        self.crisp_lbl = ttk.Label(res_frame, text="-")
        self.crisp_lbl.grid(row=0, column=1, sticky=tk.E)
        ttk.Label(res_frame, text="Baholash:").grid(row=1, column=0, sticky=tk.W)
        self.grade_lbl = ttk.Label(res_frame, text="-")
        self.grade_lbl.grid(row=1, column=1, sticky=tk.E)

        # Right side: plot + logs
        right = ttk.Frame(frm)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Log / test output
        log_frame = ttk.LabelFrame(right, text="Log / Tests", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8,0))
        self.logbox = ScrolledText(log_frame, height=8)
        self.logbox.pack(fill=tk.BOTH, expand=True)

    def on_calculate(self):
        try:
            temp = float(self.temp_var.get())
            hum = float(self.hum_var.get())
            aqi = float(self.aqi_var.get())
        except Exception as e:
            messagebox.showerror("Xato", f"Noto'g'ri kiritish: {e}")
            return

        method = self.method_var.get()
        crisp, grade, y, mu = decide_control_level(temp, hum, aqi, method)

        self.crisp_lbl.config(text=f"{crisp:.2f}%")
        self.grade_lbl.config(text=grade)


        self.logbox.insert(tk.END, f"Calculated: Temp={temp}, Hum={hum}, AQI={aqi}, method={method} -> {crisp:.2f}%, Grade={grade}\n")
        self.logbox.see(tk.END)

    def on_run_tests(self):
        rows = run_tests_collect()
        self.logbox.insert(tk.END, "Test # | Temp | Hum | AQI | Centroid | MoM | Bisector | Grade\n")
        self.logbox.insert(tk.END, "-"*72 + "\n")
        for r in rows:
            line = f"{r[0]:6} | {r[1]:4} | {r[2]:3} | {r[3]:3} | {r[4]:8.2f} | {r[5]:6.2f} | {r[6]:8.2f} | {r[7]}\n"
            self.logbox.insert(tk.END, line)
        self.logbox.insert(tk.END, "\n")
        self.logbox.see(tk.END)

if __name__ == "__main__":
    app = FuzzyGUI()
    app.mainloop()
