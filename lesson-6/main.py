import tkinter as tk
from tkinter import ttk
import numpy as np

def tri(x,a,b,c):
    if x<=a or x>=c: return 0.0
    if x<b: return (x-a)/(b-a+1e-12)
    return (c-x)/(c-b+1e-12)

def trap(x,a,b,c,d):
    if x<=a or x>=d: return 0.0
    if x<b: return (x-a)/(b-a+1e-12)
    if x<=c: return 1.0
    return (d-x)/(d-c+1e-12)

def make_new_mus(att_shift=80, exam_shift=80, proj_shift=75):
    def attendance_mu(x):
        return {"low":trap(x,-10,0,40,55),"mid":tri(x,45,60,75),"high":trap(x,att_shift,att_shift+5,100,120)}
    def exam_mu(x):
        return {"weak":trap(x,-10,0,40,55),"avg":tri(x,45,60,75),"strong":trap(x,exam_shift,exam_shift+5,100,120)}
    def project_mu(x):
        return {"poor":trap(x,-10,0,40,55),"good":tri(x,50,65,80),"excellent":trap(x,proj_shift,proj_shift+10,100,120)}
    return attendance_mu, exam_mu, project_mu

attendance_mu, exam_mu, project_mu = make_new_mus()

def lab_mu(x):
    return {"weak":trap(x,-5,0,35,50),"avg":tri(x,40,55,70),"strong":trap(x,65,80,100,110)}

def rating_sets():
    return {
        "low": lambda y: trap(y,-10,0,35,50),
        "fair": lambda y: tri(y,45,55,65),
        "good": lambda y: tri(y,60,72,84),
        "excellent": lambda y: trap(y,80,88,100,110),
        "very_excellent": lambda y: trap(y,90,95,100,110),
    }

RULES_NEW = [
    {"ands":[("attendance","high"),("exam","strong"),("project","excellent"),("lab","strong")],"ors":[],"out":"very_excellent"},
    {"ands":[("attendance","high"),("exam","strong"),("project","good"),("lab","strong")],"ors":[],"out":"excellent"},
    {"ands":[("attendance","mid"),("exam","avg"),("project","poor")],"ors":[],"out":"fair"},
    {"ands":[("attendance","low"),("exam","weak")],"ors":[("lab","weak")],"out":"low"},
]

def mamdani_new(att,ex,proj,lab,y_min=0,y_max=100,y_step=0.5):
    att_mu = attendance_mu(att)
    ex_mu  = exam_mu(ex)
    pr_mu  = project_mu(proj)
    lab_m  = lab_mu(lab)
    y_values = [y_min+i*y_step for i in range(int((y_max-y_min)/y_step)+1)]
    aggregated = [0.0]*len(y_values)
    out_sets = rating_sets()

    def get_mu(var,label):
        return {
            "attendance":att_mu.get(label,0.0),
            "exam":ex_mu.get(label,0.0),
            "project":pr_mu.get(label,0.0),
            "lab":lab_m.get(label,0.0)
        }.get(var,0.0)

    for rule in RULES_NEW:
        ands, ors, out_lbl = rule.get("ands",[]), rule.get("ors",[]), rule["out"]
        alpha_and = min([get_mu(v,l) for v,l in ands]) if ands else 1.0
        alpha_or  = max([get_mu(v,l) for v,l in ors]) if ors else 0.0
        alpha = min(alpha_and, alpha_or) if (ands and ors) else max(alpha_and, alpha_or)
        if alpha<=0: continue
        mu_out = out_sets[out_lbl]
        for i,y in enumerate(y_values):
            aggregated[i] = max(aggregated[i], min(alpha, mu_out(y)))
    return y_values, aggregated

def defuzzify(y_values, mu_values, method):
    if method == "Centroid":
        num = sum(y*m for y,m in zip(y_values,mu_values))
        den = sum(mu_values)
        return num/den if den else 0.0
    elif method == "Mean of Maxima (MoM)":
        max_mu = max(mu_values)
        ys = [y for y,m in zip(y_values,mu_values) if m == max_mu]
        return sum(ys)/len(ys) if ys else 0.0
    elif method == "Bisector":
        total_area = sum(mu_values)
        half_area = total_area / 2
        acc_area = 0
        for y, m in zip(y_values, mu_values):
            acc_area += m
            if acc_area >= half_area:
                return y
        return 0.0
    else:
        return 0.0

def score_to_category(score):
    if score>=90: return "Very Excellent"
    if score>=80: return "Excellent"
    if score>=70: return "Good"
    if score>=55: return "Fair"
    return "Low"

def evaluate(att,ex,proj,lab,method):
    y,mu = mamdani_new(att,ex,proj,lab)
    score = defuzzify(y,mu,method)
    cat = score_to_category(score)
    return round(score,2), cat

INTERPRETATION = {
    "en": {
        "Very Excellent": "Outstanding! The student demonstrates excellent consistency, participation, and mastery.",
        "Excellent": "Strong performance. The student consistently performs above average.",
        "Good": "Solid result with room for growth in some areas.",
        "Fair": "Acceptable, but improvement is recommended.",
        "Low": "Below expected standards. Significant effort is required."
    },
    "uz": {
        "Very Excellent": "Ajoyib! Talaba barqarorlik, ishtirok va mukammallikni namoyon etdi.",
        "Excellent": "Kuchli natija. Talaba doimiy ravishda o‘rtachadan yuqori darajada ishlaydi.",
        "Good": "Yaxshi natija, lekin ayrim jihatlarda o‘sish uchun imkon bor.",
        "Fair": "Qoniqarli, ammo yaxshilanish tavsiya etiladi.",
        "Low": "Kutilgan darajadan past. Jiddiy harakat talab etiladi."
    }
}


class FuzzyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎓 Intelligent Student Rating System")
        self.geometry("950x750")
        self.configure(bg="#f9fafb")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.lang = tk.StringVar(value="en")
        self.method = tk.StringVar(value="Centroid")

        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        left = ttk.LabelFrame(main_frame, text="Inputs (0–100)", padding=10)
        left.pack(side="left", fill="y", padx=(0,10))
        self.vars = {}

        for i,(name,default) in enumerate([
            ("Attendance",82),
            ("Exam",76),
            ("Project",68),
            ("Lab",70)
        ]):
            ttk.Label(left, text=f"{name}:", font=("Segoe UI",10,"bold")).pack(anchor="w", pady=(10,0))
            frame = ttk.Frame(left)
            frame.pack(fill="x", pady=(0,10))
            var = tk.DoubleVar(value=default)
            scale = ttk.Scale(frame, from_=0, to=100, orient="horizontal", variable=var, length=180)
            scale.pack(side="left", padx=4)
            lbl = ttk.Label(frame, text=f"{default:.1f}", width=5)
            lbl.pack(side="left")
            var.trace_add("write", lambda *_, v=var, l=lbl: l.config(text=f"{v.get():.1f}"))
            self.vars[name.lower()] = var

            if name == "Lab":
                ttk.Label(left, text="Defuzzification method:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(5,0))
                ttk.Combobox(left, textvariable=self.method, values=["Centroid", "Mean of Maxima (MoM)", "Bisector"], state="readonly").pack(fill="x", pady=(0,10))

        right = ttk.LabelFrame(main_frame, text="Results", padding=10)
        right.pack(side="left", fill="both", expand=True)

        self.text = tk.Text(right, wrap="word", font=("Consolas",11))
        self.text.pack(fill="both", expand=True, side="left")

        scroll = ttk.Scrollbar(right, orient="vertical", command=self.text.yview)
        scroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scroll.set)

        control = ttk.Frame(self)
        control.pack(fill="x", pady=10)
        ttk.Button(control, text="Evaluate", command=self.on_evaluate).pack(side="left", padx=10)
        ttk.Button(control, text="Clear", command=lambda: self.text.delete("1.0","end")).pack(side="left")

        ttk.Label(control, text="Language:").pack(side="right", padx=(0,5))
        ttk.Combobox(control, textvariable=self.lang, values=["en", "uz"], width=5, state="readonly").pack(side="right", padx=5)
        self.lang.trace_add("write", lambda *_: self.on_evaluate())

        self.on_evaluate()

    def on_evaluate(self):
        att = self.vars["attendance"].get()
        ex  = self.vars["exam"].get()
        proj= self.vars["project"].get()
        lab = self.vars["lab"].get()
        method = self.method.get()
        score, cat = evaluate(att,ex,proj,lab,method)
        lang = self.lang.get()

        self.text.delete("1.0","end")
        self.text.insert("end", f"📊 Input values:\n")
        self.text.insert("end", f"  Attendance: {att:.1f}\n  Exam: {ex:.1f}\n  Project: {proj:.1f}\n  Lab: {lab:.1f}\n")
        self.text.insert("end", f"🧩 Defuzzification method: {method}\n\n")
        self.text.insert("end", f"🏆 Final fuzzy score: {score}\n")
        self.text.insert("end", f"🎯 Category: {cat}\n")

        colors = {
            "Very Excellent": "#4CAF50",
            "Excellent": "#2196F3",
            "Good": "#009688",
            "Fair": "#FFC107",
            "Low": "#F44336"
        }

        color = colors.get(cat, "#000000")
        self.text.insert("end", "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", ("sep",))
        self.text.insert("end", f"✨ Interpretation:\n", ("title",))
        msg = INTERPRETATION[lang][cat]
        self.text.insert("end", msg, ("cat",))
        self.text.tag_config("cat", foreground=color, font=("Segoe UI",11,"bold"))


if __name__ == "__main__":
    FuzzyGUI().mainloop()
