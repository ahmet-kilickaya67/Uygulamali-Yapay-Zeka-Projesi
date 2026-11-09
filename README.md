# Çapraz Alan Duygu Analizi: IMDb'den Metacritic'e

**Konya Teknik Üniversitesi — Uygulamalı Yapay Zekâ Dersi / Dönem Projesi**

**Proje ekibi:** Oğuzhan Süs, Ahmet Kılıçkaya  
**Proje aşaması:** Faz 1 — Veri analizi, ön işleme ve yöntem planlaması

## Projenin Amacı

Bu projede, IMDb film yorumları üzerinde eğitilecek duygu analizi modellerinin daha önce eğitimde kullanılmamış Metacritic video oyunu yorumlarına ne ölçüde genellenebildiği araştırılmaktadır. Amaç, kaynak alan (film) ile hedef alan (oyun) arasındaki dil ve jargon farklılıklarının sınıflandırma performansına etkisini karşılaştırmaktır.

## Veri Setleri

| Veri seti | Kullanım amacı | Kaynak |
| --- | --- | --- |
| IMDb Dataset of 50K Movie Reviews | Kaynak alan: eğitim, doğrulama ve IMDb testi | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) / [Stanford](https://ai.stanford.edu/~amaas/data/sentiment/) |
| Metacritic Video Game Comments | Hedef alan: oyun yorumları üzerinde test | [Kaggle](https://www.kaggle.com/datasets/dahlia25/metacritic-video-game-comments) |
| Metacritic Game Info | Oyun türleri ve platform bilgileriyle destekleyici analiz | [Kaggle veri seti](https://www.kaggle.com/datasets/dahlia25/metacritic-video-game-comments) |

Büyük CSV dosyaları depoda tutulmamaktadır; veri setleri kaynak bağlantılarından ayrıca indirilmelidir.

## Faz 1'de Yapılan İşlemler

- Kayıt sayısı, sınıf dağılımı, eksik değerler, tekrarlar ve yorum uzunlukları incelendi.
- IMDb ve Metacritic verileri için dağılım grafikleri oluşturuldu; oyun türleri ayrıca görselleştirildi.
- Metinlerdeki HTML etiketleri ve gereksiz boşluklar temizlendi.
- IMDb'de aynı metne sahip yinelenen yorumlar çıkarıldı ve veriler **%70 eğitim / %15 doğrulama / %15 test** olarak sınıf oranları korunarak ayrıldı.
- Metacritic kullanıcı puanları **0–4: negatif**, **7–10: pozitif** biçiminde etiketlendi; **5–6** puanlı yorumlar nötr kabul edilerek çıkarıldı.
- Metacritic'te boş, tekrarlanan veya aynı oyun/platform/metin için çelişkili etiket taşıyan kayıtlar kontrol edilerek temizlendi.
- Metacritic yorumları yalnızca **hedef alan test verisi** olarak hazırlandı; model eğitiminde kullanılmadı.

## Kullanılması Planlanan Modeller

1. TF-IDF + Logistic Regression
2. TF-IDF + Support Vector Machine (SVM)
3. TF-IDF + Random Forest
4. BiLSTM
5. DistilBERT

Planlanan değerlendirme metrikleri: **Accuracy, Precision, Recall, F1-score** ve **Confusion Matrix**. Model eğitimleri ve karşılaştırmalı sonuçlar **Faz 2** kapsamında gerçekleştirilecektir.

## Kurulum ve Çalıştırma

Python 3 ile aşağıdaki adımları uygulayın:

1. Depoyu bilgisayarınıza indirin veya klonlayın.
2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
3. İndirilen CSV dosyalarını **`main.py` dosyasıyla aynı klasöre**, aşağıdaki isimlerle yerleştirin:
   ```text
   Uygulamali-Yapay-Zeka-Projesi/
   ├── main.py
   ├── requirements.txt
   ├── README.md
   ├── IMDB Dataset.csv
   ├── metacritic_game_user_comments.csv
   └── metacritic_game_info.csv
   ```
4. Analizi çalıştırın:
   ```bash
   python main.py
   ```

> **Not:** CSV dosyalarını ve kodun oluşturduğu büyük çıktıları GitHub'a eklemeyin. Kodun mevcut sürümü CSV dosyalarını `main.py` ile **aynı klasörde** aramaktadır.

## Üretilen Çıktılar

Program çalıştırıldığında `faz1_ciktilar/` klasörü oluşur. Bu klasörde:

- `imdb_train.csv`, `imdb_val.csv`, `imdb_test.csv`
- `metacritic_hedef_test.csv`
- `rapor_istatistikleri.txt`
- IMDb sınıf dağılımı, Metacritic puan ve sınıf dağılımı, yorum uzunlukları ve oyun türleri için PNG grafikler

bulunur. **`main.py` model eğitmez; Faz 1 veri hazırlığını gerçekleştirir.**

## Sonraki Aşama

Faz 2'de belirlenen modeller IMDb üzerinde eğitilecek; IMDb test kümesi ve Metacritic hedef alan verisi üzerindeki başarıları karşılaştırılacaktır. Alan değişimi nedeniyle oluşan performans farkları ve hata örnekleri analiz edilecektir.
