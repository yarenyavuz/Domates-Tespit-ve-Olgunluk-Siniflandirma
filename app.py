
import streamlit as st
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="Domates Tespit Sistemi",
    page_icon="🍅",
    layout="wide"
)


# --------------------------------------------------
# 2. MODELİ YÜKLEME
# --------------------------------------------------

MODEL_YOLU = Path("runs/detect/domates_final/weights/best.pt")


@st.cache_resource
def modeli_yukle():

    if not MODEL_YOLU.exists():
        st.error("Eğitilmiş model dosyası bulunamadı!")
        st.stop()

    model = YOLO(str(MODEL_YOLU))

    return model


model = modeli_yukle()


# Sınıf isimleri
SINIF_ISIMLERI = {
    0: "Olgunlaşmamış",
    1: "Yarı olgun",
    2: "Tam olgun"
}


# Sınıflara ait kutu renkleri (RGB)
SINIF_RENKLERI = {
    0: (50, 150, 255),
    1: (255, 180, 0),
    2: (40, 190, 80)
}


# --------------------------------------------------
# 3. UYGULAMA BAŞLIĞI
# --------------------------------------------------

st.title("🍅 Domates Tespit, Sayım ve Olgunluk Sınıflandırma Sistemi")

st.write(
    "Bu uygulama, YOLO11n derin öğrenme modeli kullanarak "
    "domatesleri tespit eder, sayar ve olgunluk seviyelerine "
    "göre sınıflandırır."
)

st.divider()


# --------------------------------------------------
# 4. TEST VERİ SETİ PERFORMANSI
# --------------------------------------------------

st.subheader("📊 Model Performansı - Test Sonuçları")

st.write(
    "YOLO11n modelinin daha önce ayrılmış test veri seti "
    "üzerindeki değerlendirme sonuçları:"
)

metrik1, metrik2, metrik3, metrik4 = st.columns(4)

metrik1.metric("Precision", "%65,0")
metrik2.metric("Recall", "%51,9")
metrik3.metric("mAP50", "%48,5")
metrik4.metric("mAP50-95", "%30,2")

st.caption(
    "Test veri seti: 55 görüntü ve 843 nesne etiketi. "
    "Sonuçlar eğitim sırasında kullanılmayan test verilerinden elde edilmiştir."
)

with st.expander("Metrikler ne anlama geliyor?"):

    st.write(
        "**Precision:** Modelin domates olarak tespit ettiği "
        "nesnelerin ne kadarının doğru olduğunu gösterir."
    )

    st.write(
        "**Recall:** Gerçek domateslerin ne kadarının model "
        "tarafından tespit edildiğini gösterir."
    )

    st.write(
        "**mAP50:** IoU 0.50 eşik değerinde ortalama "
        "tespit başarısını gösterir."
    )

    st.write(
        "**mAP50-95:** IoU 0.50 ile 0.95 arasındaki farklı "
        "eşiklerde hesaplanan ortalama tespit başarısını gösterir."
    )


# Sınıf bazında test performansı
st.subheader("Sınıflara Göre Test Performansı")

st.write(
    "Modelin test veri setindeki her olgunluk sınıfına "
    "ait performans sonuçları:"
)

st.table({
    "Olgunluk Sınıfı": [
        "Olgunlaşmamış",
        "Yarı olgun",
        "Tam olgun"
    ],
    "Precision": ["%70,0", "%55,7", "%69,3"],
    "Recall": ["%47,3", "%51,7", "%56,7"],
    "mAP50": ["%47,2", "%45,1", "%53,3"]
})

st.caption(
    "Değerler, eğitim sırasında kullanılmayan test veri "
    "seti üzerinde hesaplanmıştır."
)

st.divider()


# --------------------------------------------------
# 5. GÖRÜNTÜ YÜKLEME
# --------------------------------------------------

st.subheader("Domates Görüntüsü Yükle")

yuklenen_dosya = st.file_uploader(
    "Analiz etmek istediğiniz domates fotoğrafını seçin:",
    type=["jpg", "jpeg", "png"]
)


if yuklenen_dosya is not None:

    # Görüntüyü PIL ile oku
    gorsel = Image.open(
        BytesIO(yuklenen_dosya.getvalue())
    ).convert("RGB")

    st.image(
        gorsel,
        caption="Yüklenen domates görüntüsü",
        width="stretch"
    )

    st.write("Görüntü boyutu:", gorsel.size)

    st.divider()


    # --------------------------------------------------
    # 6. MODEL TAHMİNİ
    # --------------------------------------------------

    if st.button("Domatesleri Tespit Et", type="primary"):

        with st.spinner("Görüntü analiz ediliyor..."):

            # PIL RGB görüntüsünü NumPy dizisine dönüştür
            gorsel_dizisi = np.array(gorsel)

            # Ultralytics NumPy girişinde BGR beklediği için
            # RGB görüntüyü BGR formatına dönüştür
            gorsel_bgr = np.ascontiguousarray(
                gorsel_dizisi[:, :, ::-1]
            )

            # Model ile tahmin yap
            sonuclar = model.predict(
                source=gorsel_bgr,
                imgsz=416,
                conf=0.25,
                iou=0.7,
                device=0,
                max_det=300,
                verbose=False
            )

            # İlk görüntünün sonuçlarını al
            sonuc = sonuclar[0]
            print(sonuc.boxes.data[:5].cpu().numpy())

            # Orijinal RGB görüntünün kopyasını oluştur
            tespit_gorseli = gorsel.copy()

            # PIL çizim aracı
            cizim = ImageDraw.Draw(tespit_gorseli)

            # Türkçe karakterleri destekleyen yazı tipi
            font = ImageFont.truetype(
                "C:/Windows/Fonts/arial.ttf",
                14
            )

            # Sınıf sayaçları
            sinif_sayilari = {
                0: 0,
                1: 0,
                2: 0
            }

            # Tespit edilen kutuları al
            kutular = sonuc.boxes

            if len(kutular) > 0:

                koordinatlar = kutular.xyxy.cpu().numpy()
                siniflar = kutular.cls.cpu().numpy().astype(int)
                guvenler = kutular.conf.cpu().numpy()

                # Her tespit için işlem yap
                for koordinat, sinif, guven in zip(
                    koordinatlar,
                    siniflar,
                    guvenler
                ):

                    x1, y1, x2, y2 = map(int, koordinat)

                    sinif_adi = SINIF_ISIMLERI.get(
                        sinif,
                        "Bilinmeyen"
                    )

                    renk = SINIF_RENKLERI.get(
                        sinif,
                        (255, 255, 255)
                    )

                    # Sınıf sayısını artır
                    sinif_sayilari[sinif] += 1

                    # Tespit kutusunu çiz
                    cizim.rectangle(
                        (x1, y1, x2, y2),
                        outline=renk,
                        width=3
                    )

                    # Etiket metni
                    etiket = f"{sinif_adi} %{guven * 100:.1f}"

                    # Etiket konumu
                    metin_konumu = (x1, max(0, y1 - 25))

                    # Etiket arka planının boyutunu hesapla
                    metin_kutusu = cizim.textbbox(
                        metin_konumu,
                        etiket,
                        font=font
                    )

                    # Etiket arka planını çiz
                    cizim.rectangle(
                        metin_kutusu,
                        fill=renk
                    )

                    # Etiket metnini çiz
                    cizim.text(
                        metin_konumu,
                        etiket,
                        fill=(255, 255, 255),
                        font=font
                    )


            # --------------------------------------------------
            # 7. SONUÇLARI GÖSTER
            # --------------------------------------------------

            toplam_domates = sum(sinif_sayilari.values())

            st.subheader("Tespit Sonuçları")

            # Toplam tespit sayısı
            st.metric(
                "Toplam Domates Sayısı",
                toplam_domates
            )

            # Sınıflara göre sayılar
            sutun1, sutun2, sutun3 = st.columns(3)

            sutun1.metric(
                "🟢 Olgunlaşmamış",
                sinif_sayilari[0]
            )

            sutun2.metric(
                "🟡 Yarı Olgun",
                sinif_sayilari[1]
            )

            sutun3.metric(
                "🔴 Tam Olgun",
                sinif_sayilari[2]
            )

            st.divider()

            # Tespit edilmiş görüntü
            st.subheader("Görsel Tespit Sonucu")

            st.image(
                tespit_gorseli,
                caption="YOLO11n domates tespit sonuçları",
                width="stretch"
            )

            # Hiç tespit yoksa bilgilendirme
            if toplam_domates == 0:

                st.warning(
                    "Model bu görüntüde güven eşiğini geçen "
                    "herhangi bir domates tespit edemedi. "
                    "Bu durum modelin yeni görüntülerdeki "
                    "başarısının sınırlı olabileceğini gösterir."
                )

            else:

                st.success(
                    f"Toplam {toplam_domates} domates tespit edildi."
                )
