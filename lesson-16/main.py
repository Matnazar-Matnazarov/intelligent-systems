import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ======= MODEL UCHUN MA'LUMOT TAYYORLASH VA O'QITISH =======
np.random.seed(42)  # Har safar bir xil random natija chiqishi uchun
N = 5000  # 5000 ta misol olinadi

# Harorat, vibratsiya, shovqin va yoshi uchun sun'iy ma'lumotlar generatsiya qilinadi
temp = np.random.normal(70, 12, N).clip(30, 120)      # Harorat (o'rtacha 70, std 12, 30-120 oralig'ida)
vibr = np.random.normal(3.5, 1.8, N).clip(0, 12)      # Vibratsiya (o'rtacha 3.5, std 1.8, 0-12 oralig'ida)
noise = np.random.normal(70, 8, N).clip(40, 100)      # Shovqin (o'rtacha 70, std 8, 40-100 oralig'ida)
age = np.random.gamma(shape=3.0, scale=60, size=N).clip(0, 1000)  # Ishlangan kunlar (gamma taqsimotdan)

# Xavf hisoblash uchun sun'iy formullardan foydalaniladi (bu yerda lin va nl — liniy va noliny qismlar):
lin = 0.035*(temp-60) + 0.18*vibr + 0.02*(noise-65) + 0.001*(age-200)
nl = 0.12*np.maximum(0, temp-85) + 0.10*np.maximum(0, vibr-6)

# Sun'iy ravishda xavf logitlari va ularning ustidan sochilish qo'shiladi
risk_logit = lin + nl + np.random.normal(0, 0.25, N)
prob = 1/(1+np.exp(-risk_logit))  # Sigmoid orqali ehtimollik

# Tierli xavf ustunini yasash (low=0, medium=1, high=2)
y = np.where(prob > 0.65, 2, np.where(prob > 0.40, 1, 0))

# Model uchun X ni barcha parametrlar bilan yig'ib chiqamiz
X = np.column_stack([temp, vibr, noise, age])

# Ma'lumotlarni train-testga ajratish (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Ko'p qatlamli perceptron (MLPClassifier) o'qitish
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20
)
mlp.fit(X_train, y_train)  # Neuronal tarmoqni o'qitish

# ======= QOIDA DVIGATELI (EXPLAINABLE LOGIKA) =======
ACTIONS = {
    'monitor': "Monitor qilish (tez-tez kuzatish)",
    'schedule': "24–48 soat ichida reja asosida ta'mir",
    'shutdown': "Zudlik bilan to'xtatish va tekshiruv"
}

def rule_engine(nn_proba, context):
    # Bashorat natijalari (past, o'rta, yuqori xavf ehtimoli)
    p_low, p_med, p_high = nn_proba

    # Parametrlarni kontekstdan olish
    t, v, n = context['temp'], context['vibr'], context['noise']
    reasons = []  # Sabablar ro'yxati
    score = 0.0   # Xavf bali

    # XAVFSIZLIK QOIDALARI (katta xavfli holatlarda darhol shutdown)
    if t >= 95:
        return ACTIONS['shutdown'], ["Harorat ≥95°C → xavfsizlik qoidasi"], 1.0
    if v >= 10:
        return ACTIONS['shutdown'], ["Vibratsiya ≥10 mm/s → xavfsizlik qoidasi"], 1.0
    if n >= 95:
        return ACTIONS['shutdown'], ["Shovqin ≥95 dB → xavfsizlik qoidasi"], 1.0

    # Neyron tarmoq bashorati asosida dastlabki score hisoblanadi
    score = 0.6 * p_high + 0.3 * p_med + 0.1 * p_low

    # Kritiklik (uskunaning ahamiyati) yuqori bo'lsa, score oshiriladi
    crit = context['criticality']
    if crit == 'high':
        score += 0.25
        reasons.append("Kritiklik: YUQORI")
    elif crit == 'med':
        score += 0.12
        reasons.append("Kritiklik: O'RTA")
    else:
        reasons.append("Kritiklik: PAST")

    # Oxirgi ta'mirdan soat ko'p o'tgan bo'lsa, xavf oshadi
    hrs = context['hours_since_maint']
    if hrs > 400:
        score += 0.18
        reasons.append("Oxirgi ta'mirdan 400+ soat o'tgan")
    elif hrs > 200:
        score += 0.08
        reasons.append("Oxirgi ta'mirdan 200+ soat o'tgan")

    # Kafolat bor va NN xavf ehtimoli yuqoriroq — profilaktika uchun qulay
    if context['warranty'] and (p_med + p_high) > 0.45:
        score += 0.07
        reasons.append("Kafolat mavjud → profilaktika qulay")

    # Parametrlar yuqoriligi uchun qo'shimcha ball
    if t > 85:
        score += 0.06
        reasons.append(f"Harorat yuqori ({t:.1f}°C)")
    if v > 6:
        score += 0.06
        reasons.append(f"Vibratsiya yuqori ({v:.1f} mm/s)")
    if n > 85:
        score += 0.04
        reasons.append(f"Shovqin yuqori ({n:.0f} dB)")

    score = np.clip(score, 0.0, 1.0)  # Score 0-1 oralig'ida bo'lishi kerak

    # QAROR QOIDA: umumiy xavfga va ehtimolga asoslanadi
    if score >= 0.75 or p_high >= 0.60:
        action = ACTIONS['shutdown']
        reasons.append("Yuqori xavf darajasi aniqlandi")
    elif score >= 0.45 or p_med >= 0.55:
        action = ACTIONS['schedule']
        reasons.append("O'rta xavf – rejalashtirilgan ta'mir")
    else:
        action = ACTIONS['monitor']
        reasons.append("Xavf darajasi past")

    # Har doim: (tavsiya, sabablar, va yakuniy score) ni chiqaradi
    return action, reasons, round(score, 3)

# ======= GRAFIK ISHLATUVCHI INTERFEYSI (GUI) =======
class EquipmentRiskApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Uskuna Xavfini Baholash")  # Dastur nomi
        self.geometry("800x700")               # Dastur oynasi o'lchami
        self.configure(bg="#ffffff")           # Orqa fon
        self._setup_ui()                       # UI elementlarini yaratish

    def _setup_ui(self):
        # Sarlavha paneli
        header_frame = tk.Frame(self, bg="#2563eb", height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="⚙️ Uskuna Holatini Tahlil Qilish", 
                 font=("Arial", 20, "bold"), bg="#2563eb", fg="white").pack(pady=25)

        # Asosiy konteyner (barcha UI uchun)
        main_frame = tk.Frame(self, bg="#ffffff")
        main_frame.pack(fill="both", expand=True, padx=30, pady=20)

        # Parametrlar uchun kirish maydonlari (temperatura, vibratsiya va h.k.)
        input_frame = tk.LabelFrame(main_frame, text="Parametrlar", 
                                   font=("Arial", 11, "bold"), bg="#ffffff", fg="#1e40af")
        input_frame.pack(fill="x", pady=(0, 15))

        # Parametrlarni saqlash uchun dictionary
        self.vars = {}
        inputs = [
            ("Harorat (°C)", 30, 120, 70.0),
            ("Vibratsiya (mm/s)", 0, 12, 3.5),
            ("Shovqin (dB)", 40, 100, 70.0),
            ("Ishlagan kunlar", 0, 1000, 180.0),
            ("Oxirgi ta'mirdan soatlar", 0, 2000, 250.0),
        ]
        
        # Har bir parametr uchun scale va entry yasash
        for i, (label, min_val, max_val, default) in enumerate(inputs):
            row_frame = tk.Frame(input_frame, bg="#ffffff")
            row_frame.pack(fill="x", padx=15, pady=8)
            
            tk.Label(row_frame, text=label, width=22, anchor="w", 
                    font=("Arial", 10), bg="#ffffff").pack(side="left", padx=(0, 10))
            
            var = tk.DoubleVar(value=default)
            self.vars[label] = var
            
            # Parametrni scale (slayder) orqaligi kiritish
            ttk.Scale(row_frame, from_=min_val, to=max_val, variable=var, 
                     orient="horizontal", length=300).pack(side="left", padx=(0, 10))
            
            # Qo'shimcha: manual entry ham bor
            entry = ttk.Entry(row_frame, textvariable=var, width=10, justify="center")
            entry.pack(side="left")
        
        # Kontekst (qo'shimcha ma'lumotlar) bo'limi: kritiklik, kafolat
        ctx_frame = tk.LabelFrame(main_frame, text="Qo'shimcha ma'lumotlar", 
                                 font=("Arial", 11, "bold"), bg="#ffffff", fg="#1e40af")
        ctx_frame.pack(fill="x", pady=(0, 15))
        
        ctx_inner = tk.Frame(ctx_frame, bg="#ffffff")
        ctx_inner.pack(padx=15, pady=12)
        
        # Kritiklik uchun combobox (past, o'rta, yuqori)
        tk.Label(ctx_inner, text="Kritiklik:", font=("Arial", 10), bg="#ffffff").pack(side="left", padx=(0, 10))
        self.crit_var = tk.StringVar(value="med")
        ttk.Combobox(ctx_inner, textvariable=self.crit_var, values=["low", "med", "high"], 
                    state="readonly", width=12).pack(side="left", padx=(0, 30))
        
        # Kafolat uchun checkbox
        self.warr_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctx_inner, text="Kafolat ostida", variable=self.warr_var, 
                      font=("Arial", 10), bg="#ffffff").pack(side="left")
        
        # Baholash tugmasi
        btn = tk.Button(main_frame, text="🔍 XAVFNI BAHOLASH", font=("Arial", 13, "bold"),
                       bg="#2563eb", fg="white", activebackground="#1d4ed8",
                       command=self.analyze, relief="flat", cursor="hand2", padx=30, pady=12)
        btn.pack(pady=15)
        
        # Natijani ko'rsatish uchun oynacha
        result_frame = tk.LabelFrame(main_frame, text="Natija", 
                                     font=("Arial", 11, "bold"), bg="#ffffff", fg="#1e40af")
        result_frame.pack(fill="both", expand=True)
        
        self.result_text = tk.Text(result_frame, height=12, font=("Consolas", 10), 
                                  bg="#f8fafc", relief="flat", wrap="word", state="disabled")
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def analyze(self):
        try:
            # Foydalanuvchi qiymatlarni oladi
            temp = self.vars["Harorat (°C)"].get()
            vibr = self.vars["Vibratsiya (mm/s)"].get()
            noise = self.vars["Shovqin (dB)"].get()
            age = self.vars["Ishlagan kunlar"].get()
            hrs = int(self.vars["Oxirgi ta'mirdan soatlar"].get())
            
            # Model bashoratini olamiz (MLPClassifier dan)
            X_input = np.array([[temp, vibr, noise, age]])
            proba = mlp.predict_proba(X_input)[0]
            p_low, p_med, p_high = proba
            
            # Kontekst parametrlari — qoida dvigateliga uzatiladi
            ctx = {
                "criticality": self.crit_var.get(),
                "hours_since_maint": hrs,
                "warranty": self.warr_var.get(),
                "temp": temp, "vibr": vibr, "noise": noise
            }
            action, reasons, score = rule_engine(proba, ctx)  # Qoidalar asosida natija

            # Natijani chiqish oynasiga chiroyli qilib yozish
            self.result_text.config(state="normal")
            self.result_text.delete(1.0, tk.END)
            
            self.result_text.tag_config("title", font=("Arial", 12, "bold"), foreground="#1e40af")
            self.result_text.tag_config("action", font=("Arial", 11, "bold"), foreground="#dc2626")
            self.result_text.tag_config("info", foreground="#475569")
            self.result_text.tag_config("reason", foreground="#64748b")
            
            self.result_text.insert(tk.END, "📊 Tahlil natijasi\n\n", "title")
            self.result_text.insert(tk.END, f"Tavsiya qilinadigan amal:\n", "info")
            self.result_text.insert(tk.END, f"{action}\n\n", "action")
            self.result_text.insert(tk.END, f"Umumiy xavf bali: {score:.3f}\n", "info")
            self.result_text.insert(
                tk.END,
                f"NN ehtimolliklari → Past: {p_low:.3f} | O'rta: {p_med:.3f} | Yuqori: {p_high:.3f}\n\n", "info"
            )
            self.result_text.insert(tk.END, "Sabablar:\n", "info")
            for r in reasons:
                self.result_text.insert(tk.END, f"  • {r}\n", "reason")
            
            self.result_text.config(state="disabled")
        except Exception as e:
            # Xatolik bo'lsa, oynachada xabar chiqadi
            messagebox.showerror("Xato", str(e))

if __name__ == "__main__":
    # Ilovani ishga tushirish
    app = EquipmentRiskApp()
    app.mainloop()