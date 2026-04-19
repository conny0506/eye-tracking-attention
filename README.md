# Göz Takibi ve Dikkat Analizi

Gerçek zamanlı göz takibi ve sürücü dikkat analizi uygulaması. MediaPipe Face Mesh ve OpenCV kullanılarak geliştirilmiştir.

## Özellikler

- **Göz Kırpma Tespiti** — EAR (Eye Aspect Ratio) algoritması ile blink sayısı ve dakikadaki blink hızı
- **Bakış Yönü Analizi** — İris konumundan bakışın hangi yöne gittiğini tespit eder (Merkez, Sol, Sağ, Yukarı, Aşağı)
- **Dikkat Skoru** — 0-100 arası anlık ve ortalama dikkat skoru
- **Kişisel Kalibrasyon** — Başlangıçta kullanıcının göz yapısına göre EAR eşiği otomatik ayarlanır
- **Gözlük Tespiti** — Kenar analizi ile kullanıcının gözlük takıp takmadığını tespit eder
- **Bireysel Göz Takibi** — Sol ve sağ gözü ayrı ayrı izler, biri kapandığında uyarı verir
- **Engel Tespiti** — Bir gözün önünde engel olduğunda "Gözlerde engel var" uyarısı gösterir
- **Uyuklama Uyarısı** — Uzun süreli göz kapanmasında ekran uyarısı + sesli alarm

## Kullanılan Teknolojiler

| Teknoloji | Sürüm | Kullanım Amacı |
|-----------|-------|----------------|
| Python | 3.x | Ana programlama dili |
| OpenCV | 4.8+ | Görüntü işleme ve ekran çizimi |
| MediaPipe | 0.10+ | 478 noktalı yüz landmark tespiti |
| NumPy | 1.24+ | Matematiksel hesaplamalar |
| winsound | — | Sesli uyarı (Windows yerleşik) |

## Kurulum

```bash
# Repoyu klonlayın
git clone https://github.com/conny0506/eye-tracking-attention.git
cd eye-tracking-attention

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# MediaPipe model dosyasını indirin
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

## Kullanım

```bash
python main.py
```

Uygulama başladığında **kalibrasyon ekranı** açılır. Kameraya bakarak doğal şekilde göz kırpın — yaklaşık 2 saniye sonra sistem otomatik olarak ana ekrana geçer.

Çıkmak için `q` tuşuna basın.

## Proje Yapısı

```
eye-tracking-attention/
├── main.py                 # Ana uygulama döngüsü
├── eye_tracker.py          # EAR hesabı, iris takibi, gözlük tespiti
├── attention_analyzer.py   # Dikkat skoru, uyarı sistemi, ses
├── utils.py                # Ekran paneli ve görsel çizimler
└── requirements.txt
```

## Ekran Görüntüsü

Sol panelde şu bilgiler anlık olarak gösterilir:

- Gözlük durumu
- Sol / Sağ göz açık-kapalı durumu
- Anlık ve ortalama dikkat skoru (renkli bar ile)
- Blink sayısı ve dakikadaki blink hızı
- Bakış yönü
- EAR değerleri ve kalibrasyon eşiği
- Geçen süre

Tehlikeli durumlarda ekranın altında renkli uyarı kutusu belirir:

| Durum | Uyarı |
|-------|-------|
| Uyuklama | Kırmızı kutu + sesli alarm |
| Tek göz kapalı | Sarı kutu |
| Gözde engel | Sarı kutu |
| Uzun süre blink yok | Sarı kutu |

## Gereksinimler

- Windows işletim sistemi (sesli uyarı için)
- Çalışan bir webcam
- Python 3.8+
