# Domates Tespiti ve Olgunluk Sınıflandırması

Bu proje, YOLO11n derin öğrenme modeli kullanılarak görüntülerdeki domateslerin tespit edilmesini, sayılmasını ve olgunluk seviyelerinin sınıflandırılmasını amaçlamaktadır.

## Kullanılan Teknolojiler
- Python
- YOLO11n (Ultralytics)
- Streamlit
- OpenCV

## Sınıflar
- Unripe (Olgunlaşmamış)
- Semi-ripe (Yarı olgun)
- Fully-ripe (Tam olgun)

## Model Performansı

Test veri kümesindeki sonuçlar:

| Metrik | Değer |
|---|---:|
| Precision | 65.0% |
| Recall | 51.9% |
| mAP@50 | 48.5% |
| mAP@50-95 | 30.2% |

## Çalıştırma

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

Uygulamayı başlatın:

```bash
streamlit run app.py
```

## Model Dosyası

Eğitilmiş model ağırlıkları `best.pt` dosyasında bulunmaktadır.

## Geliştirilen Uygulama

Streamlit arayüzü üzerinden kullanıcılar bir domates görüntüsü yükleyebilir. Model görüntüdeki domatesleri tespit eder, sınıflarını belirler ve toplam domates sayısını gösterir.