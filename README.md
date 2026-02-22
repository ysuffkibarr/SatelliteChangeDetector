🛰️ Uydu Görüntülerinde Değişim Tespiti

Bu proje, farklı zamanlarda çekilmiş uydu görüntüleri arasında yapısal ve çevresel değişimleri tespit edebilen bir yapay zeka sistemidir. Model, U-Net mimarisi ile geliştirilmiş ve görüntü segmentasyonu yaklaşımı kullanılarak eğitilmiştir.

📌 Proje Amacı
İki farklı tarihte alınmış uydu görüntüleri arasında:

Yeni yapılaşma
Doğal afet etkileri (yangın, sel, vs.)
Tarımsal değişiklikler gibi farkları otomatik olarak tespit edebilmek.

🧠 Kullanılan Teknolojiler
Python 3
TensorFlow / Keras
OpenCV
NumPy
Matplotlib

📁 Veri Kümesi
Proje, CDD (Change Detection Dataset) adlı açık veri kümesiyle eğitilmiştir.

Veri klasör yapısı şu şekildedir:

dataset/
├── train/
│   ├── A/         → Sonraki görüntüler
│   ├── B/         → Önceki görüntüler
│   └── out/       → Maskeler (değişim alanları)
├── val/
└── test/

🏗️ Model Mimarisi
Model, 6 kanallı giriş (önceki + sonraki görüntü) alır.

Çıkış olarak 1 kanallı değişim maskesi üretir.

Kayıp fonksiyonu olarak Binary Crossentropy + Dice Loss kullanılmıştır.

🚀 Kullanım
1. Gerekli kütüphaneleri yükleyin:
  pip install tensorflow opencv-python matplotlib numpy
2. Eğitim:
  python app.py
3. Kendi görsellerinizle test:
  images/before.png ve images/after.png adlı iki uydu görüntüsünü images/ klasörüne yerleştirin.

📌 Notlar
Eğitim süresi donanıma göre değişiklik gösterebilir.

Sistem %100 doğrulukla çalışmamaktadır ve çıktılar mutlaka insan gözetimiyle değerlendirilmelidir. Modelin performansı, daha fazla veri ve optimizasyon ile geliştirilebilir.

📬 İletişim
Bu proje bir iş başvuru test projesi kapsamında geliştirilmiştir.
Her türlü soru için iletişime geçebilirsiniz.
