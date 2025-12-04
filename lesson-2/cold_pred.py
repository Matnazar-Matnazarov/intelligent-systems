QUESTIONS = { 
    "sneeze": "Aksirish bo'ladimi? (ha/yo'q): ", 
    "itchy_eyes": "Ko'z qichishishi bormi? (ha/yo'q): ", 
    "fever": "Sizda tana harorati ko'tarilganmi? (ha/yo'q): ", 
    "fatigue": "Holzizlik sezilyaptimi? (ha/yo'q): " ,
    "cough": "Yo'tal bormi? (ha/yo'q): ", 
}
RULES = [
    (["fever", "cough", "fatigue"], "shamollash"), 
    (["cough"], "gripp"), 
    (["sneeze", "itchy_eyes"], "allergiya") 
]

def to_bool(ans: str) -> bool | None: 
    ans = ans.strip().lower()
    if ans in ["ha", "ha.", "h", "yes", "1", "hovo"]:
        return True 
    if ans in ["yo'q", "yoq", "y", "no", "n", "0"]: 
        return False
    return None 

def ask_missing_fact(fact_name: str) -> bool: 
    while True: 
        ans = input(QUESTIONS[fact_name])
        val = to_bool(ans) 
        if val is not None: 
            return val 
        print("Iltimos, 'ha' yoki 'yo'q' deb javob bering.") 

def forward_chain(): 
    facts: dict[str, bool] = {} 
    explanation = []

    needed_conditions = []
    for conds, _ in RULES:
        for c in conds:
            if c not in needed_conditions:
                needed_conditions.append(c)

    for c in QUESTIONS.keys():
        if c in needed_conditions:
            facts[c] = ask_missing_fact(c)

    inferred = set() 
    changed = True 
    while changed: 
        changed = False 
        for conds, conclusion in RULES: 
            if conclusion in inferred: 
                continue 
            if all(facts.get(c, False) for c in conds): 
                inferred.add(conclusion) 
                explanation.append((conds, conclusion)) 
                changed = True 
    return inferred, explanation

if __name__ == "__main__": 
    conclusions, trace = forward_chain() 
    if conclusions: 
        print("\n>>> Ehtimoliy tashxis(lar):", ", ".join(conclusions)) 
        print("\nIzoh (qoidalar izi):") 
        for conds, concl in trace: 
            print(f"  Agar {conds} bo'lsa, unda -> {concl}") 
    else:
        print("\n>>> Qoidalar bo'yicha aniq xulosa topilmadi.") 