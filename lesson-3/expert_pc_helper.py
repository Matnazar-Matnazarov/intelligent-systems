from dataclasses import dataclass
from typing import List, Dict, Tuple

QUESTIONS: Dict[str, str] = {
    "no_power": "⚡ Kompyuter umuman yoqilmayapti (yorug'lik/fan yo'q)? (ha/yo'q): ",
    "no_display": "🖥️ Kompyuter yoqiladi, lekin ekranda tasvir yo'qmi? (ha/yo'q): ",
    "beeps": "🔔 Yoqilganda 'beep' signallari eshitilyaptimi? (ha/yo'q): ",
    "overheating": "🔥 Kompyuter tez qizib ketadimi? (ha/yo'q): ",
    "auto_shutdown": "💤 Ish paytida avtomatik o'chib qoladimi? (ha/yo'q): ",
    "slow_performance": "🐢 Tizim juda sekin ishlayaptimi? (ha/yo'q): ",
    "high_cpu": "📊 Vazifalar dispetcherida CPU doim yuqorimi? (ha/yo'q): ",
    "many_startup": "📦 Avto-ishga tushadigan (startup) dasturlar juda ko'pmi? (ha/yo'q): ",
    "wifi_disconnected": "📡 Wi-Fi tez-tez uziladimi? (ha/yo'q): ",
    "router_ok": "📶 Routerni boshqa qurilmalar normal ishlatyaptimi? (ha/yo'q): ",
    "fan_spins": "🌪️ Yoqilganda fan aylanadimi? (ha/yo'q): ",
    "led_on": "💡 Korpusdagi indikator LED yonadimi? (ha/yo'q): ",
    "strange_noise": "🔊 Kompyuterdan g'alati shovqinlar (g‘ichirlash, taqillash) chiqayaptimi? (ha/yo'q): ",
    "blue_screen": "💻 Ish paytida ko‘k ekran (Blue Screen) xatosi chiqadimi? (ha/yo'q): ",
}


@dataclass
class Rule:
    if_all: List[str]
    then: str
    cf: float = 0.8

RULES: List[Rule] = [
    Rule(["no_power", "fan_spins", "led_on"], "🖥️ Displey yoki videokarta chiqishi muammoli!", 0.6),
    Rule(["no_power"], "⚡ Elektr ta'minoti yoki quvvat bloki muammosi!", 0.9),
    Rule(["no_display", "beeps"], "💾 Operativ xotira (RAM) muammosi!", 0.85),
    Rule(["no_display", "beeps"], "🎮 Videokarta yoki slot aloqasi nosoz!", 0.6),
    Rule(["overheating", "auto_shutdown"], "🔥 Sovutish tizimi yoki termopasta muammosi!", 0.9),
    Rule(["slow_performance", "high_cpu", "many_startup"], "🐢 Dasturiy ortiqcha yuk (startup) muammosi!", 0.85),
    Rule(["wifi_disconnected", "router_ok"], "📡 Wi-Fi adapter yoki drayver muammosi!", 0.8),
    Rule(["strange_noise"], "🔊 Qattiq disk yoki fan mexanik nosozligi!", 0.85),
    Rule(["blue_screen"], "💻 Operatsion tizim yoki drayver xatosi (Blue Screen)!", 0.9),
]


def ask_facts() -> Dict[str, bool]:
    facts: Dict[str, bool] = {}
    print("👉 Savollarga faqat 'ha' yoki 'yo'q' deb javob bering:")
    for key, prompt in QUESTIONS.items():
        while True:
            ans = input(prompt).strip().lower()
            if ans in ("ha", "yo'q", "yoq", "1", "0", "h", "y",):
                facts[key] = (ans in ["ha", "1", "h"])
                break
            else:
                print("⚠️ Iltimos, faqat 'ha' yoki 'yo'q' deb javob bering.")
    return facts

def infer(facts: Dict[str, bool], rules: List[Rule], threshold: float = 0.4) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for r in rules:
        if all(facts.get(p, False) for p in r.if_all):
            scores[r.then] = max(scores.get(r.then, 0.0), r.cf)
    results = [(k, v) for k, v in scores.items() if v >= threshold]
    return sorted(results, key=lambda x: x[1], reverse=True)

RECOMMENDATIONS: Dict[str, str] = {
    "⚡ Elektr ta'minoti yoki quvvat bloki muammosi!":
        "🔌 Tarmoq kabeli va rozetkani tekshiring.\n⚙️ Minimal konfiguratsiyada yoqib ko'ring.",
    "🖥️ Displey yoki videokarta chiqishi muammoli!":
        "🔌 Monitor kabeli/portlarini tekshiring.\n🖥️ Boshqa kabel yoki monitor bilan sinab ko'ring.",
    "💾 Operativ xotira (RAM) muammosi!":
        "🧹 RAM modullarini tozalang yoki qayta joylang.\n🔄 Slotni almashtirib ko'ring.",
    "🎮 Videokarta yoki slot aloqasi nosoz!":
        "🔧 Diskret GPU ni qayta joylang.\n🧹 Kontaktlarni tozalang.\n🖥️ Integratsiyalangan grafikani sinang.",
    "🔥 Sovutish tizimi yoki termopasta muammosi!":
        "🧹 Changni tozalang.\n🌪️ Fan ishini tekshiring.\n❄️ Termopastani yangilang.",
    "🐢 Dasturiy ortiqcha yuk (startup) muammosi!":
        "⚙️ Startup dasturlarni kamaytiring.\n🗑️ Ortiqcha dasturlarni o'chiring.\n🛡️ Antivirus bilan tekshiring.",
    "📡 Wi-Fi adapter yoki drayver muammosi!":
        "💿 Drayverni qayta o'rnating yoki yangilang.\n🖥️ Qurilma boshqaruvchisidan tekshirib ko'ring.",
}

def main():
    print("💻 Ekspert tizimi: Kompyuter muammosini aniqlash")
    facts = ask_facts()
    results = infer(facts, RULES)
    if not results:
        print("\n🤔 Aniq xulosa topilmadi. Ko'proq belgilar yoki qo'shimcha qoidalar kerak.")
        return
    print("\n✅ Ehtimoliy muammolar (ishonchlilik bo'yicha tartiblangan):")
    for i, (diag, cf) in enumerate(results, start=1):
        print(f"{i}) {diag} | 📊 CF={cf:.2f}")
        rec = RECOMMENDATIONS.get(diag)
        if rec:
            print("👉 Tavsiya:\n" + rec)

if __name__ == "__main__":
    main()
