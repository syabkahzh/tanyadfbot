import re
from typing import Set

_STRONG_KEYWORDS: Set[str] = {
    'sfood','gfood','grab','shopee','gojek','tokped','tokopedia',
    'voucher','vcr','voc','diskon','promo','cashback','gratis','potongan',
    'idm','indomaret','alfa','alfamart','alfagift',
    'klaim','claim','restock','ristok','nt','abis','habis',
    'gabisa','gaada','g+b+s','gamau','minbel',
    'r+s+t+k','r+s+t+c+k','r+st+ck',
    'cb','kesbek','c+s+h+b+c+k','cash back',
    'luber','pecah','flash','sale','deal','murah','hemat','bonus',
    'ongkir','gratis ongkir',
    'membership','member','mamber',
    # Added from raw data analysis — missing deal-hunter slang
    'tukpo','murce','sopi','tsel','cgv','svip','badut','war','fs',
    'kreator','kopken','chatime','gindaco','solaria','rotio','spx',
    'gopay','neo','tmrw','saqu','seabank','hero',
    'gacor','mantul','nyala','cair','lancar',
}

_WORD_BOUNDARY_KEYWORDS = re.compile(
    r'\b(off|on|aman|work|bs|jp|mm)\b', re.IGNORECASE
)

_SOCIAL_FILLER = re.compile(
    r'^(wkwk|haha|hehe|iya|noted|oke|ok|makasih|thanks|thx|mantap|gas|'
    r'siap|sip|lol|anjir|anjay|btw|oot|gws|semangat)[!.\s]*$',
    re.IGNORECASE
)

def is_worth_checking(text: str) -> bool:
    if not text or not text.strip():
        return False
    t = text.strip().lower()
    if "saya membisukan dia" in t or "@dfautokick_bot" in t:
        return False
    
    words = t.split()
    if len(words) < 2:
        return False

    question_words = {'ga','gak','nggak','apa','gimana','berapa','kapan','dimana','kenapa'}
    if t.endswith('?') and words and words[0] in question_words:
        return False
    if len(words) <= 3 and t.endswith('?'):
        return False
    if _SOCIAL_FILLER.match(t):
        return False
    if any(kw in t for kw in _STRONG_KEYWORDS):
        return True
    if _WORD_BOUNDARY_KEYWORDS.search(t):
        return True
    return False

# --- FROM listener.py ---
INSTANT_PATTERN = re.compile(
    r'\b(on|jp|jackpot|work|aman|luber|pecah|berhasil|gacor|mantul|restock|ristok|aktif|ready|'
    r'nyala|masuk|udah pake|dikirim|cair|done|lancar|'
    r'potongan|idm|alfa|indomaret|ag|alfagift|voc|voucher|minbel|'
    r'r\+s\+t\+k|r\+s\+t\+c\+k|r\+st\+ck|cb|kesbek|c\+s\+h\+b\+c\+k|cash back|'
    r'qr|scan|edc|membership|member|mamber)\b',
    re.IGNORECASE
)
NEG_PATTERN = re.compile(
    r'\b(kapan|kok|ga pernah|tidak|belom|belum|gaada|ngga|ga ada|gak|nggak|bukan|jangan|'
    r'iya|cuma|pas|tadi|gamau|jamber|jambrapa|jamberapa|'
    r'b\+r\+p|brp|berapa|drmana|dimana|dmn|mana|d\+r\+m\+n|'
    r'tunggu|nunggu|nanti|besok|lusa|tar\b|dulu|sore|malem|malam|pagi|'
    r'harusnya|katanya|mungkin|kayaknya|kyknya|sepertinya|entah|'
    r'koid|hangus|refund|batal|balsis|ngebadut|zonk|habis|sold|error|'
    r'coba|nyoba|semoga|mudah.mudahan|insya)\b',
    re.IGNORECASE
)
TRANSIT_NOISE_PATTERN = re.compile(
    r'\b(rute|jalan|macet|kereta|stasiun|paket|kirim|kurir|perjalanan|nyampe)\b',
    re.IGNORECASE
)
FAST_ALLCAPS = re.compile(r'^[^a-z]*[A-Z][^a-z]*$')

def check_fast_path(text: str) -> bool:
    if not text:
        return False
    if FAST_ALLCAPS.match(text) and len(text) > 3:
        return True
    if NEG_PATTERN.search(text):
        return False
    # Transit-noise gate: "aman kak rutenya" is NOT a deal signal
    if 'aman' in text.lower() and TRANSIT_NOISE_PATTERN.search(text):
        return False
    if INSTANT_PATTERN.search(text):
        return True
    return False

# --- SAMPLES BATCH 3 ---
msgs = [
    "kak mau tanya dongg dlu udh pernah buka linknya di email tp blm dipake nah pas aku buka lagi kok udh ke reedem ya",
    "Bentar buka appny dulu lemot",
    "Kena colong kah?soale pas awal2 pubg ad case bgini dia kecolong",
    "Ih kok ak blm buka plz😭",
    "Heykama biasanya sering fs nya di tts tapi gaada komboan voucher😭",
    "pnya kak siapa gtu, ktnya klo udh redeem udh gbisa",
    "kak ini promo rotio apa kak?",
    "Hah gimana?",
    "nggk aku share kmn mn tuh ka tp",
    "Svip ka",
    "Iya mksd w prnh ada kejadian kek gini coba complain deh",
    "Ayooo",
    "komplainnya kmn ya kak klo gini?😭😭",
    "pernah baca disini, dia ke redeem vc nya. krna sempet share atau apa gtu jdi gbisa kepake, gtu loh maksudnya😭",
    "Kuotanya makin dikit kali ya. Awal awal malah masih bisa kalo pagi.",
    "selamat mam roti🫰",
    "Pesanan uda dibayarkan",
    "logika nya gk bisa",
    "Ih tunggu😭",
    "Kamu yg bayar kak? Makasi banyak yaaaa murah rezeki, bntr aku minta qr lg",
    "Kak lagiii aku blm coba",
    "lg pd cek qr qhaqha",
    "Maaci ka nempell",
    "Ehey maaciw ka",
    "ke email ini ka",
    "Pc svip tu harus edc shopee?",
    "Yg email pubg itu deh klo abis isi form kmrn2 kan ada email pubg buat complain klo ad mslh",
    "kobisa yh kena colong tp ga share kmn2🥲😭",
    "Akhirnya ya",
    "Biasanya dianaktirikan buat promo cem hokben 🤣",
    "Scanner sopi iya",
    "pc yg scan kasirnya",
    "Yg mau cek rotio uda kan? Mau aku bayar",
    "Hah apa pubg kasih codenya ada yg sama?",
    "okedehkaaa aku coba dlu😭😭",
    "Pas awal rame2 dulu 2bln pertama tagihan sampe 3jt kak 😭",
    "Akunya sakit ga motong😭",
    "entah sama kaya kopken apa engga tapi kopken ada yg dikirim kode yg sama si",
    "coba ini kak",
    "okey ka aku cobaa",
    "Aman endog,tapi harga di etalasenya di naikin dr 33rb jd 37rb/kg 😵 mana jd preorder 3hari",
    "Astagfirullah jgn smpe kak😭😭😭",
    "dapet kapan? pernah baca kalo dpt awal2 katanya banyak kodenya yg sama, jd 1 kode dikirim ke beberapa user",
    "banyak kok kasusnya di df (entah kopken or cetam) lebih lanjutnya chat ke cs aja coba komplain kl emg blm pake",
    "hahaha miann exp besok mumpung buka grup😭",
    "Siap yg bayar😭",
    "kamu ngisinya pas awal awal ya kak? pas awal awal eventnya muncul, kasusnya sama kaya kopken, kodenya pada samaan makanya tulisannya dah kepake karna kodenya samaa"
]

print(f"{'MESSAGE':<60} | {'FAST-PATH':<10} | {'AI PATH':<10}")
print("-" * 85)
for m in msgs:
    fp = "✅ YES" if check_fast_path(m) else "❌ no"
    ai = "✅ YES" if is_worth_checking(m) else "❌ no"
    print(f"{m[:58]:<60} | {fp:<10} | {ai:<10}")
