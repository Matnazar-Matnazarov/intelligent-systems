# ================== HYBRID CHATBOT: INTENTS + ENHANCED RETRIEVAL (TOP-3) ==================
# Req: pip install scikit-learn
import re
import ast
import operator as op
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- 1) Dialog holati (minimal state) --------------------
class DialogState:
    def __init__(self):
        self.user_name: Optional[str] = None
        self.last_intent: Optional[str] = None
        self.slots: Dict[str, str] = {}  # for slot-filling like pending name confirmation

state = DialogState()

# -------------------- 2) Xavfsiz kalkulyator --------------------
allowed_ops = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, 
    ast.Div: op.truediv, ast.Pow: op.pow,
    ast.USub: lambda x: -x
}
def safe_eval(expr: str) -> float:
    def _eval(node):
        if isinstance(node, ast.Num):  # py<3.8
            return node.n
        if isinstance(node, ast.Constant):  # py>=3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers allowed")
        if isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_ops:
                raise ValueError("Unsupported operator")
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_ops:
                raise ValueError("Unsupported unary operator")
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")
    tree = ast.parse(expr, mode='eval')
    return _eval(tree.body)

# -------------------- 3) NLU: intent + entity (regex-based) --------------------
INTENT_PATTERNS = {
    "greet": r"\b(salom|assalomu ?alaykum|hello|hi|hey)\b",
    "bye": r"\b(xayr|salomat bo'ling|bye|goodbye|see you)\b",
    "ask_time": r"\b(soat|time|vaqt|nechchi)\b",
    "ask_date": r"\b(sana|bugun|today|date)\b",
    "set_name": r"\b(ismim|mening ismim|menga)\b",
    "get_name": r"\b(ismim nima|meni kim|men kimman|ismim)\b",
    "calc": r"(hisobla|hisob-kitob|calc|[\d\.\s\+\-\*\/\(\)\^]+)",
    "help": r"\b(yordam|help|nima qila olasan|commands?)\b",
    # Qo‘shimcha mock intentlar (talaba topshirig‘i uchun)
    "joke": r"\b(latifa|joke|hazil)\b",
    "weather": r"\b(havo|ob-havo|weather)\b",
    "translate": r"\b(tarjima|translate)\b",
    "news": r"\b(yangilik|news)\b",
}

# Name extraction (ko‘p variantlarga moslashgan)
NAME_PATTERN = re.compile(r"(?:ismim|mening ismim|menga)\s+([A-Za-zА-Яа-яЁёO‘o‘ʼ’'\-]+)", re.I)

# Date detection (YYYY-MM-DD yoki dd.mm.yyyy yoki '12 dekabr' kabi oddiy match)
DATE_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2})|(\d{1,2}[./]\d{1,2}[./]\d{2,4})|(\b\d{1,2}\s+[A-Za-zА-Яа-яёЁ]+)"
    , re.I
)

def detect_intent(text: str) -> str:
    t = text.lower()
    # priority: explicit calc with only math -> calc
    if re.fullmatch(r"[\d\.\s\+\-\*\/\(\)\^]+", text.strip()):
        return "calc"
    for intent, pat in INTENT_PATTERNS.items():
        if re.search(pat, t):
            return intent
    return "faq_or_fallback"

def extract_name(text: str) -> Optional[str]:
    m = NAME_PATTERN.search(text)
    if m:
        return m.group(1)
    # fallback: "Menga Ali deb murojaat qil" yoki "Ism: Ali"
    m2 = re.search(r"\b([A-ZА-ЯЁ][a-zа-яё']{1,20})\b", text)
    return m2.group(1) if m2 else None

def extract_date(text: str) -> Optional[str]:
    m = DATE_PATTERN.search(text)
    return m.group(0) if m else None

# -------------------- 4) KB: FAQ + retrieval (TF-IDF) with TOP-3 --------------------
KB_QA = [
    ("chatbot nima", "Chatbot — foydalanuvchi bilan matn/suhbat orqali muloqot qiluvchi dastur."),
    ("sen nimalar qila olasan", "Salomlashish, vaqt/sana aytish, sodda hisob-kitob va mini-FAQ javoblari."),
    ("tf-idf nima", "TF-IDF — so‘zning hujjat ichidagi va korpus bo‘yicha ahamiyatini baholovchi usul."),
    ("knn nima", "kNN — yangi namunani eng yaqin k ta qo‘shnining sinfiga ko‘ra tasniflaydi."),
    ("decision tree nima", "Decision Tree — ma’lumotlarni xususiyat bo‘yicha bo‘lib qaror qabul qiladigan model."),
    ("gpu kerakmi", "Oddiy chatbot uchun shart emas, lekin DL modellarni o‘qitishda GPU foydali."),
    ("reinforcement learning nima", "RL — agent mukofot/jazo asosida muhitda siyosatni o‘rganadi."),
    ("python versiya", "Odatda Python 3.9+ tavsiya etiladi; bu kod 3.8+ da ishlaydi."),
    ("kalkulyator misol", "Hisoblash uchun: hisobla 2*(3+4) — shuni sinab ko‘ring."),
    ("faq qanday ishlaydi", "Agar intent topilmasa, retrieval TF-IDF yordamida eng yaqin savolni topamiz."),
]
KB_QUESTIONS = [q for q,_ in KB_QA]
KB_ANSWERS   = [a for _,a in KB_QA]

VECT = TfidfVectorizer(ngram_range=(1,2), min_df=1)
KB_MATRIX = VECT.fit_transform(KB_QUESTIONS)

def retrieve_top_k(user_text: str, k: int = 3) -> List[Tuple[str, float, str]]:
    """
    Returns list of (answer, score, question) sorted by score desc
    """
    q_vec = VECT.transform([user_text])
    sims = cosine_similarity(q_vec, KB_MATRIX)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
    results = []
    for idx, score in ranked:
        results.append((KB_ANSWERS[idx], float(score), KB_QUESTIONS[idx]))
    return results

# -------------------- 5) NLG / Policy: qaysi modul javob beradi --------------------
def handle_intent(intent: str, text: str) -> str:
    state.last_intent = intent

    # Basic rule-based handlers
    if intent == "greet":
        if state.user_name:
            return f"Salom, {state.user_name}! Qanday yordam bera olaman?"
        # push for name in next two messages (slot)
        state.slots.setdefault("ask_name_reminder", "pending")
        return "Salom! Ismingizni aytsangiz, muloqotni shaxsiylashtiraman 🙂"

    if intent == "bye":
        return "Xayr! Yana ko‘rishguncha 👋"

    if intent == "ask_time":
        return f"Hozir soat: {datetime.now().strftime('%H:%M:%S')}."

    if intent == "ask_date":
        today = datetime.now().strftime('%Y-%m-%d (%A)')
        return f"Bugungi sana: {today}."

    if intent == "set_name":
        name = extract_name(text)
        if not name:
            # Ehtimol: "Menga Ali deb murojaat qil" kabi
            m = re.search(r"deb murojaat qil[: ]*\s*([A-Za-zА-Яа-яЁёO‘o‘ʼ’'\-]+)", text, re.I)
            name = m.group(1) if m else None
        if name:
            state.user_name = name
            # clear pending reminder
            state.slots.pop("ask_name_reminder", None)
            return f"Tanishganimdan xursandman, {name}! Qanday yordam bera olaman?"
        return "Ismingizni tushunmadim 🤔 Iltimos: 'Ismim Aziz' kabi yozing."

    if intent == "get_name":
        return f"Sizning ismingiz: {state.user_name}." if state.user_name else "Ismingizni hali bilmayman. 'Ismim ...' deb yozing."

    if intent == "calc":
        # Extract expression more robustly: allow ^ as power
        expr_candidates = re.findall(r"[\d\.\s\+\-\*\/\(\)\^]+", text)
        expr = expr_candidates[0].strip() if expr_candidates else text
        expr = expr.replace("^", "**")
        try:
            val = safe_eval(expr)
            return f"Natija: {val}"
        except SyntaxError:
            return "Hisob ifodasini tushunmadim (syntax). Misol: 2*(3+4)/5"
        except ZeroDivisionError:
            return "Nolga bo‘lish xatosi."
        except Exception as e:
            # more friendly error
            msg = str(e)
            if "unsupported" in msg.lower():
                return "Ifoda ichida ruxsat etilmagan operator bor."
            return "Hisob ifodasini tushunmadim. Misol: hisobla 2*(3+4)"

    if intent == "help":
        return ("Menga shunday yozishingiz mumkin:\n"
                "• 'Salom' — salomlashish\n"
                "• 'Vaqt nechchi?' — hozirgi vaqt\n"
                "• 'Bugungi sana?' — sana\n"
                "• 'Ismim Ali' — ismingizni saqlash\n"
                "• 'Hisobla 2*(3+4)' — kalkulyator\n"
                "• Savol bering: 'kNN nima?' — mini-FAQ\n"
                "Qo‘shimcha: 'latifa' (hazil), 'havo' (mock havo).")

    if intent == "joke":
        return "Latifa: Bir programmist elektron choynakni yuvmoqchi bo'ldi — choynak ishlamadi, chunki u 'boil' ga qo'ng'iroq qildi."

    if intent == "weather":
        # mock response (API dan foydalanmadi — topshiriq bo‘yicha)
        return "Bugun havo taxminan quyoshli, 25°C atrofida (mock ma'lumot)."

    if intent == "translate":
        # very naive mock
        m = re.search(r"tarjima\s+(.+?)\s+qa(y|ga)\s+(\w+)", text, re.I)
        return "Tarjima funksiyasi mock: 'Hello' -> 'Salom'."

    if intent == "news":
        return "Yangiliklar mock: Hozirgi vaqtda yangilik mavjud emas (mock)."

    # Fallback: retrieval-based
    top = retrieve_top_k(text, k=3)
    top1_ans, top1_score, top1_q = top[0]
    # If high confidence → qaytarish
    if top1_score >= 0.55:
        return top1_ans
    # Agar past ishonch → takliflar bilan disambiguation
    suggestions = [f"\"{q}\" ({score:.2f})" for _, score, q in top]
    suggestion_text = " yoki ".join(suggestions)
    return (f"Kechirasiz, savolni aniq tushunmadim. Siz shulardan birini nazarda tutdingizmi?\n"
            f"{suggestion_text}\nBundan birini tanlasangiz, batafsilroq javob beraman.")

# -------------------- 6) Chat loop (konsol) --------------------
def chat_loop():
    print("=== Oddiy Chatbot (hybrid, enhanced) ===")
    print("Yozing (chiqish uchun: quit/exit):")
    while True:
        user = input("> ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("👋 Xayr!")
            break
        # If previous asked for name and user gives one-word reply, try to treat as name
        if state.slots.get("ask_name_reminder") == "pending" and len(user.split()) == 1:
            # assume user replied with name
            intent = "set_name"
        else:
            intent = detect_intent(user)
        reply = handle_intent(intent, user)
        print(reply)

# -------------------- 7) Evaluation / Test scenarios (sample) --------------------
TESTS = [
    # (input, expected_intent, expected_retrieval_answer_contains)
    ("Salom", "greet", None),
    ("Ismim Maftuna", "set_name", None),
    ("Bugungi sana?", "ask_date", None),
    ("Vaqt nechchi?", "ask_time", None),
    ("Hisobla 3*(2+7)", "calc", "Natija"),
    ("kNN nima", "faq_or_fallback", "kNN"),
    ("TF-IDF nima?", "faq_or_fallback", "TF-IDF"),
    ("Latifa ayt", "joke", None),
    ("Havo qanday", "weather", None),
    ("Men kimman?", "get_name", None),
    ("Reinforcement learning nima", "faq_or_fallback", "RL"),
    ("Nima qilasan?", "faq_or_fallback", "Chatbot"),
]

def evaluate(tests=TESTS):
    intent_correct = 0
    intent_total = len(tests)
    retrieval_correct_top1 = 0
    retrieval_correct_top3 = 0
    retrieval_total = 0

    for inp, exp_intent, exp_re in tests:
        pred_intent = detect_intent(inp)
        if pred_intent == exp_intent:
            intent_correct += 1
        # If expected retrieval is provided, check retrieval hits
        if exp_re:
            retrieval_total += 1
            top3 = retrieve_top_k(inp, k=3)
            top1 = top3[0][0].lower()
            top3_qs = [q for _,_,q in top3]
            # simple substring check
            if exp_re.lower() in top1:
                retrieval_correct_top1 += 1
            if any(exp_re.lower() in q.lower() for q in top3_qs):
                retrieval_correct_top3 += 1

    intent_acc = intent_correct / intent_total * 100
    r_top1_acc = (retrieval_correct_top1 / retrieval_total * 100) if retrieval_total else 0
    r_top3_acc = (retrieval_correct_top3 / retrieval_total * 100) if retrieval_total else 0

    print("EVALUATION REPORT")
    print(f"Intent aniqlash: {intent_correct}/{intent_total} = {intent_acc:.1f}%")
    print(f"Retrieval top-1 (on marked tests): {retrieval_correct_top1}/{retrieval_total} = {r_top1_acc:.1f}%")
    print(f"Retrieval top-3 (on marked tests): {retrieval_correct_top3}/{retrieval_total} = {r_top3_acc:.1f}%")
    print("Agar siz batafsil baholash (50 test) xohlasangiz, TESTS ro'yxiga yana savollar qo'shing.")

# -------------------- 8) Agar fayl bevosita ishga tushsa --------------------
if __name__ == "__main__":
    # By default: run a quick evaluation, then start chat loop
    evaluate()
    chat_loop()


"""
token =  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
deep seek r1

"""