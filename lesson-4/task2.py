Talaba = {
    "F.I.Sh": "Anvarov Aziz",
    "Yo'nalish": "Dasturiy injiniring",
    "Kurs": 2,
    "O'rtacha baho": 86,
    "Yoshi": 20,
    "Telefon raqami": "+998901234567"
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

Kitob = {
    "Nomi": "Sun’iy intellekt asoslari",
    "Muallif": "Sam Altman",
    "Betlar soni": 1150
}

BilimBazasi = {
    "Talaba": Talaba,
    "Universitet": Universitet,
    "Fan": Fan,
    "Kitob": Kitob
}

def chiqarish():
    print("\n=== Bilim Bazasi ===")
    for nom, freym in BilimBazasi.items():
        print(f"\n{nom} haqida:")
        for slot, qiymat in freym.items():
            print(f"  {slot}: {qiymat}")

def yangi_freym_qosh(nomi):
    yangi = {}
    print(f"\nYangi freym uchun {nomi} ma'lumotlarini kiriting:")
    while True:
        slot = input("Slot nomi (chiqish uchun 'stop'): ")
        if slot.lower() == "stop":
            break
        qiymat = input(f"{slot} qiymati: ")
        yangi[slot] = qiymat
    BilimBazasi[nomi] = yangi

chiqarish()

tanlov = input("\nYangi freym qo'shasizmi? (ha/yo'q): ")
if tanlov.lower() == "ha":
    nomi = input("Yangi freym nomini kiriting: ")
    yangi_freym_qosh(nomi)
    chiqarish()
else:
    print("Dastur tugadi.")
