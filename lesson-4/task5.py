Kompyuter = {
    "Markasi": "Dell",
    "Model": "Inspiron 15",
    "RAM": "16GB"
}

Dasturchi = {
    "Ism": "Javlon",
    "Mutaxassislik": "Backend Developer",
    "Tajriba": "3 yil"
}

Kompaniya = {
    "Nomi": "TechSoft",
    "Joylashuv": "Toshkent",
    "Xodimlar soni": 120
}

BilimlarBazasi = {
    "Kompyuter": Kompyuter,
    "Dasturchi": Dasturchi,
    "Kompaniya": Kompaniya
}

def chiqarish():
    print("\n=== Bilim Bazasi ===")
    for nom, freym in BilimlarBazasi.items():
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
    BilimlarBazasi[nomi] = yangi

chiqarish()

tanlov = input("\nYangi freym qo'shasizmi? (ha/yo'q): ")
if tanlov.lower() == "ha":
    nomi = input("Yangi freym nomini kiriting: ")
    yangi_freym_qosh(nomi)
    chiqarish()
else:
    print("Dastur tugadi.")
