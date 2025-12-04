Talaba = {
    "F.I.Sh": "Anvarov Aziz",
    "Yo'nalish": "Dasturiy injiniring",
    "Kurs": 2,
    "O'rtacha baho": 86
}

Universitet = {
    "Nomi": "TATU",
    "Joylashuv": "Toshkent",
    "Fakultetlar soni": 7
}

Fan = {
    "Nomi": "Intellektual tizimlar",
    "Kredit": 6,
    "O'qituvchi": "Prof. Egamberganova F."
}

BilimBazasi = {
    "Talaba": Talaba,
    "Universitet": Universitet,
    "Fan": Fan
}

def chiqarish():
    print("\n=== Bilim Bazasi ===")
    for nom, freym in BilimBazasi.items():
        print(f"\n{nom} haqida:")
        for slot, qiymat in freym.items():
            print(f"  {slot}: {qiymat}")

def qidiruv(slot_nomi):
    print(f"\nQidiruv natijasi: {slot_nomi}")
    topildi = False
    for nom, freym in BilimBazasi.items():
        if slot_nomi in freym:
            print(f"{nom} -> {freym[slot_nomi]}")
            topildi = True
    if not topildi:
        print("Bunday slot topilmadi.")

menu_tanlash = int(input("Menuni tanlang:\n1.Frame\n2.Qidirish\n>>> "))
if menu_tanlash == 1:
    chiqarish()
elif menu_tanlash == 2:
    soz = input("Slot nomini kiriting: ")
    qidiruv(soz)


