"""Uygulamalı Yapay Zeka - Faz 1: IMDb / Metacritic analiz ve ön işleme.

Çalıştırmadan önce bu .py dosyasını üç CSV ile aynı klasöre koyun.
VS Code terminali: pip install pandas matplotlib scikit-learn
Çalıştırma: python main.py

ÖNEMLİ: Bu kod MODEL EĞİTMEZ. IMDb eğitim / doğrulama / test olarak
ayrılır; Metacritic yalnızca hedef alan test verisi olarak hazırlanır.
"""

from pathlib import Path
import html
import re

import matplotlib
matplotlib.use("Agg")  # Grafikleri ekrana açmadan PNG olarak kaydet.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


BASE = Path(__file__).resolve().parent
OUT = BASE / "faz1_ciktilar"
OUT.mkdir(exist_ok=True)

IMDB_PATH = BASE / "IMDB Dataset.csv"
COMMENTS_PATH = BASE / "metacritic_game_user_comments.csv"
GAMES_PATH = BASE / "metacritic_game_info.csv"
SEED = 42


# -------------------- 1. YARDIMCI FONKSİYONLAR --------------------
def temizle_metin(deger):
    """Ortak, hafif temizlik: HTML, satır sonu ve fazladan boşlukları temizle.

    Noktalama, olumsuzluk kelimeleri ve büyük/küçük harfler korunur.
    Böylece TF-IDF ve Transformer için ayrı model-özel işlem yapılabilir.
    """
    if pd.isna(deger):
        return ""
    metin = html.unescape(str(deger))
    metin = re.sub(r"<[^>]*>", " ", metin)  # IMDb <br /> vb.
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


def kontrol_et(df, sutunlar, dosya_adi):
    eksik = set(sutunlar) - set(df.columns)
    if eksik:
        raise ValueError(f"{dosya_adi}: eksik sütun(lar): {sorted(eksik)}")


def sekil_kaydet(dosya_adi):
    plt.tight_layout()
    plt.savefig(OUT / dosya_adi, dpi=180, bbox_inches="tight")
    plt.close()


def cubuk_grafigi(seri, baslik, y_etiketi, dosya):
    fig, ax = plt.subplots(figsize=(8, 5))
    seri.plot(kind="bar", ax=ax)
    ax.set_title(baslik)
    ax.set_ylabel(y_etiketi)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    for i, deger in enumerate(seri.values):
        ax.text(i, deger, f"{int(deger):,}", ha="center", va="bottom", fontsize=9)
    sekil_kaydet(dosya)


def uzunluk_grafigi(seriler):
    """Yorum başına kelime sayıları. %99 üstü yalnızca grafikte kesilir."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for ad, yorumlar in seriler.items():
        uzunluk = yorumlar.fillna("").str.count(r"\S+")
        ust_sinir = max(1, int(uzunluk.quantile(0.99)))
        ax.hist(uzunluk.clip(upper=ust_sinir), bins=50,
                alpha=0.55, density=True, label=f"{ad} (üst sınır: {ust_sinir})")
    ax.set_title("Yorum uzunluğu dağılımı (kelime)")
    ax.set_xlabel("Kelime sayısı (her seri %99 persentilde sınırlandırıldı)")
    ax.set_ylabel("Yoğunluk")
    ax.legend()
    sekil_kaydet("04_yorum_uzunluklari.png")


def sayi_ozeti(ad, df, metin_sutun, etiket_sutun=None):
    uzunluk = df[metin_sutun].fillna("").str.count(r"\S+")
    satirlar = [
        f"{ad}",
        f"  Toplam satır: {len(df):,}",
        f"  Boş yorum: {df[metin_sutun].isna().sum():,}",
        f"  Tamamen aynı yorum metni: {df[metin_sutun].duplicated().sum():,}",
        f"  Ortalama kelime: {uzunluk.mean():.2f}",
        f"  Medyan kelime: {uzunluk.median():.0f}",
        f"  En uzun yorum (kelime): {uzunluk.max():,}",
    ]
    if etiket_sutun is not None:
        satirlar.append("  Sınıf sayıları: " + str(df[etiket_sutun].value_counts(dropna=False).to_dict()))
    return "\n".join(satirlar)


# -------------------- 2. VERİ SETLERİNİ YÜKLE --------------------
for yol in (IMDB_PATH, COMMENTS_PATH, GAMES_PATH):
    if not yol.is_file():
        raise FileNotFoundError(f"Dosya bulunamadı: {yol}\nÜç CSV'yi .py ile aynı klasöre koyun.")

print("Veri setleri okunuyor...")
imdb = pd.read_csv(IMDB_PATH, low_memory=False)
meta = pd.read_csv(COMMENTS_PATH, low_memory=False)
oyunlar = pd.read_csv(GAMES_PATH, low_memory=False)

kontrol_et(imdb, ["review", "sentiment"], IMDB_PATH.name)
kontrol_et(meta, ["Title", "Platform", "Userscore", "Comment"], COMMENTS_PATH.name)
kontrol_et(oyunlar, ["Title", "Platform", "Genre"], GAMES_PATH.name)

# CSV'den gelen gereksiz indeks sütunlarını çıkar.
meta = meta.loc[:, ~meta.columns.str.startswith("Unnamed:")].copy()
oyunlar = oyunlar.loc[:, ~oyunlar.columns.str.startswith("Unnamed:")].copy()

rapor = ["FAZ 1 - VERİ SETİ ANALİZİ VE ÖN İŞLEME RAPORU", "=" * 55]
rapor.append("\n[ÖN İŞLEME ÖNCESİ]")
rapor.append(sayi_ozeti("IMDb", imdb, "review", "sentiment"))
rapor.append(sayi_ozeti("Metacritic yorumları", meta, "Comment"))
rapor.append(f"  Oyun bilgisi kayıt sayısı: {len(oyunlar):,}")
rapor.append(f"  Farklı oyun adı (yorumlar): {meta['Title'].nunique():,}")
rapor.append(f"  (Title, Platform, Comment) tekrar sayısı: {meta.duplicated(['Title', 'Platform', 'Comment']).sum():,}")
rapor.append(f"  Eksik veri - IMDb: {imdb.isna().sum().to_dict()}")
rapor.append(f"  Eksik veri - Metacritic yorumları: {meta.isna().sum().to_dict()}")
rapor.append(f"  Eksik veri - Oyun bilgileri: {oyunlar.isna().sum().to_dict()}")

# -------------------- 3. GÖRSELLEŞTİRMELER --------------------
print("Grafikler oluşturuluyor...")
cubuk_grafigi(imdb["sentiment"].value_counts().reindex(["negative", "positive"], fill_value=0),
               "IMDb duygu sınıfları", "Yorum sayısı", "01_imdb_sinif_dagilimi.png")

meta["Userscore"] = pd.to_numeric(meta["Userscore"], errors="coerce")
puan_sayilari = meta["Userscore"].value_counts().sort_index()
cubuk_grafigi(puan_sayilari, "Metacritic kullanıcı puanları (ham)",
               "Yorum sayısı", "02_metacritic_puan_dagilimi.png")

uzunluk_grafigi({"IMDb": imdb["review"], "Metacritic": meta["Comment"]})

# İsteğe bağlı veri analizi: En sık görülen ana oyun türleri.
ana_tur = oyunlar["Genre"].dropna().astype(str).str.split(";").str[0].str.strip()
fig, ax = plt.subplots(figsize=(10, 5))
ana_tur.value_counts().head(10).sort_values().plot.barh(ax=ax)
ax.set_title("Oyun bilgi tablosunda en yaygın 10 ana tür")
ax.set_xlabel("Oyun/platform kaydı sayısı")
sekil_kaydet("05_oyun_turleri.png")

# -------------------- 4. IMDb ÖN İŞLEME --------------------
print("IMDb temizleniyor ve 70/15/15 bölünüyor...")
imdb["text"] = imdb["review"].map(temizle_metin)
imdb["label"] = imdb["sentiment"].astype("string").str.strip().str.lower()
imdb_temiz = imdb.loc[(imdb["text"] != "") & imdb["label"].isin(["positive", "negative"]), ["text", "label"]].copy()
imdb_once = len(imdb_temiz)
imdb_temiz = imdb_temiz.drop_duplicates(subset=["text"]).reset_index(drop=True)
rapor.append("\n[IMDB ÖN İŞLEME]")
rapor.append(f"  Temizlikten sonra tekrar eden metin nedeniyle çıkarılan: {imdb_once - len(imdb_temiz):,}")
rapor.append(f"  Kalan yorum: {len(imdb_temiz):,}")
rapor.append("  Kaggle 50K CSV dosyasında orijinal train/test ayrımı yoktur.")
rapor.append("  Bunun yerine stratified 70/15/15 train/validation/test bölünmüştür (random_state=42).")

train, kalan = train_test_split(imdb_temiz, test_size=0.30,
                                random_state=SEED, stratify=imdb_temiz["label"])
val, test = train_test_split(kalan, test_size=0.50,
                             random_state=SEED, stratify=kalan["label"])
for ad, df in (("imdb_train", train), ("imdb_val", val), ("imdb_test", test)):
    df.to_csv(OUT / f"{ad}.csv", index=False, encoding="utf-8-sig")
    rapor.append(f"  {ad}: {len(df):,}, dağılım={df['label'].value_counts().to_dict()}")

# -------------------- 5. METACRITIC ÖN İŞLEME --------------------
print("Metacritic temizleniyor ve etiketleniyor...")
# Kullanıcı puanlarından ikili etiket oluştur.
meta["label"] = pd.NA
meta.loc[meta["Userscore"].between(0, 4), "label"] = "negative"
meta.loc[meta["Userscore"].between(7, 10), "label"] = "positive"

notr_sayisi = int(meta["Userscore"].isin([5, 6]).sum())
gecersiz_puan = int((meta["Userscore"].isna() | ~meta["Userscore"].between(0, 10)).sum())
meta = meta.dropna(subset=["label", "Title", "Platform"]).copy()
meta["text"] = meta["Comment"].map(temizle_metin)
bos_yorum = int(meta["text"].eq("").sum())
meta = meta.loc[meta["text"] != ""].copy()

# Aynı oyun + platform + yorum için karşıt etiket varsa bunları dışla;
# diğer tekrarları tekilleştir. Farklı oyunlarda aynı kısa yorum korunur.
anahtar = ["Title", "Platform", "text"]
farkli_etiket = meta.groupby(anahtar)["label"].transform("nunique") > 1
celiskili_kayit = int(farkli_etiket.sum())
meta = meta.loc[~farkli_etiket].copy()
tekrar_once = len(meta)
meta = meta.drop_duplicates(subset=anahtar).reset_index(drop=True)
tekrar_silinen = tekrar_once - len(meta)

meta_cikis = ["text", "label", "Title", "Platform", "Userscore"]
meta[meta_cikis].to_csv(OUT / "metacritic_hedef_test.csv", index=False, encoding="utf-8-sig")

cubuk_grafigi(meta["label"].value_counts().reindex(["negative", "positive"], fill_value=0),
               "Metacritic duygu sınıfları (ön işleme sonrası)",
               "Yorum sayısı", "03_metacritic_sinif_dagilimi.png")

rapor.append("\n[METACRITIC ÖN İŞLEME]")
rapor.append(f"  5-6 puanlı nötr yorum çıkarıldı: {notr_sayisi:,}")
rapor.append(f"  Geçersiz/eksik kullanıcı puanı: {gecersiz_puan:,}")
rapor.append(f"  Boş yorum çıkarıldı (diğer filtrelerden sonra): {bos_yorum:,}")
rapor.append(f"  Aynı oyun-platform-metin olup farklı etikete sahip satırlar çıkarıldı: {celiskili_kayit:,}")
rapor.append(f"  Aynı oyun-platform-metin tekrarları çıkarıldı: {tekrar_silinen:,}")
rapor.append(f"  Hedef alan test satırı: {len(meta):,}")
rapor.append(f"  Son sınıf dağılımı: {meta['label'].value_counts().to_dict()}")
rapor.append("  Hedef alan verisi hiçbir eğitim veya hiperparametre seçiminde kullanılmamalıdır.")
rapor.append("  Yorum İÇİ tekrarlar otomatik değiştirilmedi; ayrı, denetlenebilir inceleme önerilir.")

# -------------------- 6. OYUN BİLGİLERİYLE EŞLEŞME --------------------
oyun_anahtarlari = oyunlar[["Title", "Platform"]].drop_duplicates()
eslesen = meta[["Title", "Platform"]].merge(oyun_anahtarlari,
                            on=["Title", "Platform"], how="left", indicator=True)
rapor.append("\n[OYUN BİLGİSİ EŞLEŞMESİ]")
rapor.append(f"  Eşleşen temiz yorum: {(eslesen['_merge'] == 'both').sum():,}/{len(meta):,}")
rapor.append("  Oyun bilgisi CSV dosyası sınıflandırma girdisi olarak KULLANILMADI.")

(OUT / "rapor_istatistikleri.txt").write_text("\n".join(rapor), encoding="utf-8")
print("\n".join(rapor))
print(f"\nTamamlandı. Grafikler, temiz CSV dosyaları ve özet rapor: {OUT}")
