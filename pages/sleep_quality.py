import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Prediksi Sleep Disorder", layout="wide")
st.title("😴 Prediksi Gangguan Tidur")
st.markdown("Masukkan data pasien untuk memprediksi **Sleep Disorder**")
st.markdown("---")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/adaboost_sleep_model.pkl")

# =========================
# KONSTANTA
# =========================
LABEL_MAP = {0: "Sehat", 1: "Insomnia", 2: "Sleep Apnea"}

# Rentang normal per parameter: (nama tampil, min, max, unit)
NORMAL_RANGES = {
    "Usia":                  (None,  None,  "Tahun"),
    "Durasi Tidur":          (6.0,   9.0,   "Jam"),
    "Kualitas Tidur":        (7,     10,    "Skor"),
    "Aktivitas Fisik":       (30,    60,    "Menit/hari"),
    "Tingkat Stres":         (1,     5,     "Skor"),
    "Heart Rate":            (60,    100,   "bpm"),
    "Daily Steps":           (5000,  10000, "Langkah"),
    "Systolic BP":           (90,    120,   "mmHg"),
    "Diastolic BP":          (60,    80,    "mmHg"),
    "BMI":                   (18.5,  24.9,  "kg/m²"),
}

# Threshold untuk flagging risiko: (nama tampil, kondisi risiko sebagai lambda)
# Diproses terpisah karena logikanya berbeda-beda
RISK_FLAGS_CONFIG = [
    ("Durasi Tidur",     lambda v: v < 6 or v > 9,        "Durasi tidur ideal adalah 6–9 jam/malam."),
    ("Kualitas Tidur",   lambda v: v <= 3,                 "Kualitas tidur sangat rendah — perlu perbaikan rutinitas tidur."),
    ("Tingkat Stres",    lambda v: v >= 8,                 "Tingkat stres sangat tinggi — berdampak besar pada kualitas tidur."),
    ("Heart Rate",       lambda v: v > 100 or v < 60,     "Heart rate di luar rentang normal istirahat (60–100 bpm)."),
    ("Aktivitas Fisik",  lambda v: v < 20,                 "Aktivitas fisik sangat rendah — direkomendasikan setidaknya 30 menit/hari."),
    ("Daily Steps",      lambda v: v < 3000,               "Jumlah langkah harian sangat rendah — target minimal 5.000 langkah/hari."),
    ("Systolic BP",      lambda v: v > 140,                "Tekanan darah sistolik tinggi — konsultasikan dengan dokter."),
    ("Diastolic BP",     lambda v: v > 90,                 "Tekanan darah diastolik tinggi — konsultasikan dengan dokter."),
    ("BMI",              lambda v: v < 18.5 or v > 24.9,   "BMI di luar rentang normal — berisiko mempengaruhi kualitas tidur."),
]

# =========================
# REKOMENDASI MEDIS per hasil prediksi
# =========================
RECOMMENDATIONS = {
    "Sehat": [
        "✅ Hasil prediksi menunjukkan **tidur Anda dalam kondisi sehat**.",
        "🛏️ Pertahankan rutinitas tidur yang konsisten — tidur dan bangun di jam yang sama setiap hari.",
        "🏃 Jaga aktivitas fisik rutin dan hindari olahraga berat menjelang malam.",
        "📱 Kurangi penggunaan layar (HP, laptop) setidaknya 1 jam sebelum tidur.",
        "🧘 Kelola stres dengan teknik relaksasi atau meditasi sebelum tidur.",
        "🥗 Jaga pola makan seimbang dan hindari kafein setelah siang hari.",
        "📋 Tetap lakukan check-up rutin untuk memantau kesehatan tidur Anda.",
    ],
    "Insomnia": [
        "⚠️ Hasil prediksi menunjukkan **indikasi Insomnia**.",
        "🏥 Disarankan untuk **konsultasi dengan dokter atau psikolog** untuk evaluasi lebih lanjut.",
        "🛏️ Terapkan **sleep hygiene** yang ketat: tidur dan bangun di jam yang sama, ruang tidur gelap dan tenang.",
        "📵 Hindari layar elektronik minimal **1–2 jam sebelum tidur**.",
        "☕ Batasi konsumsi **kafein dan alkohol**, terutama setelah jam 14.00.",
        "🧘 Coba teknik relaksasi seperti **progressive muscle relaxation** atau **deep breathing** sebelum tidur.",
        "🌙 Jika sulit tidur lebih dari 20 menit, bangunlah dan lakukan aktivitas ringan hingga merasa ngantuk.",
        "💊 Jangan menggunakan obat tidur tanpa resep dokter.",
    ],
    "Sleep Apnea": [
        "⚠️ Hasil prediksi menunjukkan **indikasi Sleep Apnea**.",
        "🏥 **Segera konsultasikan** dengan dokter spesialis tidur (Sleep Medicine) untuk diagnosis resmi.",
        "🔬 Dokter kemungkinan akan menyarankan **polysomnography (sleep study)** untuk konfirmasi.",
        "😮‍💨 Gejala umum: **mendengkir keras, sesak napas saat tidur, dan mengantuk berlebih di siang hari** — perhatikan hal ini.",
        "⚖️ Menjaga **berat badan ideal** sangat penting — kelebihan berat badan meningkatkan risiko Sleep Apnea.",
        "🛏️ Coba tidur dalam posisi **miring (lateral)** untuk membuka saluran napas.",
        "🍷 Hindari **alkohol dan sedatif** karena dapat memperburuk penyumbatan saluran napas.",
        "💨 Jika sudah didiagnosis, **CPAP therapy** (Continuous Positive Airway Pressure) adalah penanganan utama.",
    ],
}

# =========================
# EDUKASI DISORDER
# =========================
EDUCATION = {
    "Sehat": {
        "apa": "Kondisi tidur Anda termasuk normal dan sehat berdasarkan data yang diinput.",
        "gejala": "Tidak ada gejala gangguan tidur yang terdeteksi.",
        "dampak": "Tidur yang berkualitas mendukung kesehatan fisik, mental, dan daya tahan tubuh secara keseluruhan.",
        "fakta": [
            "Orang dewasa membutuhkan tidur 7–9 jam per malam untuk fungsi optimal.",
            "Tidur yang cukup membantu regenerasi sel dan memperkuat sistem imun.",
            "Kurang tidur secara konsisten dapat meningkatkan risiko penyakit kronis.",
        ],
    },
    "Insomnia": {
        "apa": "Insomnia adalah gangguan tidur yang ditandai dengan kesulitan untuk tidur atau tetap tidur, sehingga waktu dan kualitas tidur tidak mencukupi.",
        "gejala": "Sulit untuk memulai tidur, sering bangun di tengah malam, bangun terlalu pagi, dan merasa lelah/mengantuk di siang hari.",
        "dampak": "Insomnia kronis dapat menyebabkan penurunan konsentrasi, peningkatan risiko depresi dan kecemasan, serta melemahkan daya tahan tubuh.",
        "fakta": [
            "Insomnia adalah salah satu gangguan tidur paling umum — mempengaruhi sekitar 30% orang dewasa.",
            "Insomnia dapat bersifat akut (jangka pendek) atau kronis (berlangsung lebih dari 3 bulan).",
            "Faktor utama: stres, kecemasan, jadwal tidur tidak teratur, dan lingkungan tidur yang tidak nyaman.",
            "Cognitive Behavioral Therapy for Insomnia (CBT-I) adalah terapi lini pertama yang direkomendasikan.",
        ],
    },
    "Sleep Apnea": {
        "apa": "Sleep Apnea adalah gangguan tidur yang ditandai dengan terjadi henti napas berulang saat tidur, menyebabkan tidur terganggu dan tubuh tidak mendapat oksigen yang cukup.",
        "gejala": "Mendengkir keras, napas terasa sesak atau berhenti saat tidur, bangun dengan rasa lelah, dan mengantuk berlebih di siang hari.",
        "dampak": "Jika tidak ditangani, Sleep Apnea meningkatkan risiko hipertensi, penyakit jantung, stroke, dan diabetes tipe 2.",
        "fakta": [
            "Ada dua jenis: Obstructive Sleep Apnea (OSA — paling umum) dan Central Sleep Apnea (CSA).",
            "Faktor risiko utama: kegemukan, usia lanjut, dan anatomi saluran napas.",
            "Pemeriksaan gold standard adalah Polysomnography (sleep study) yang dilakukan di laboratorium tidur.",
            "CPAP (Continuous Positive Airway Pressure) adalah perangkat terapi utama dan paling efektif untuk OSA.",
        ],
    },
}

# =========================
# SARAN GAYA HIDUP DINAMIS (per flagged risk)
# =========================
LIFESTYLE_TIPS = {
    "Durasi Tidur": [
        "⏰ **Durasi Tidur:** Coba tetapkan jadwal tidur dan bangun yang konsisten setiap hari, termasuk akhir pekan.",
        "🌙 Ciptakan rutinitas sebelum tidur — mandi hangat, baca buku ringan, atau lakukan stretching.",
        "📵 Jauhkan gadget dari tempat tidur dan matikan lampu setidaknya 30 menit sebelum tidur.",
    ],
    "Kualitas Tidur": [
        "🛏️ **Kualitas Tidur Rendah:** Pastikan kamar tidur nyaman — suhu sejuk, gelap, dan tenang.",
        "🧸 Gunakan bantal dan kasur yang sesuai kenyamanan tubuh Anda.",
        "🧘 Lakukan relaksasi atau meditasi singkat (5–10 menit) tepat sebelum tidur.",
    ],
    "Tingkat Stres": [
        "🧘 **Stres Tinggi:** Rutin latihan pernapasan dalam (deep breathing) setidaknya 2x sehari.",
        "📝 Coba journaling — tulis pikiran dan perasaan sebelum tidur untuk 'menguras' stres.",
        "🌳 Habiskan waktu di alam atau lakukan aktivitas yang menyenangkan untuk menurunkan cortisol.",
        "🏃 Olahraga teratur (pagi atau sore, bukan malam) terbukti efektif mengurangi stres.",
    ],
    "Heart Rate": [
        "🫀 **Heart Rate Tidak Normal:** Konsultasikan dengan dokter untuk memastikan kondisi jantung Anda.",
        "🧘 Teknik relaksasi dan pernapasan dalam bisa membantu menurunkan heart rate istirahat.",
        "☕ Kurangi konsumsi kafein dan nikotin yang dapat meningkatkan detak jantung.",
    ],
    "Aktivitas Fisik": [
        "🏃 **Aktivitas Fisik Rendah:** Mulai dari olahraga ringan seperti jalan kaki 20–30 menit di pagi hari.",
        "🚴 Tingkatkan secara bertahap — target akhir setidaknya 30 menit aktivitas sedang per hari.",
        "🕐 Hindari olahraga intens menjelang malam karena dapat mengganggu tidur.",
    ],
    "Daily Steps": [
        "🚶 **Langkah Harian Rendah:** Coba gunakan tangga daripada elevator dan jalan kaki untuk jarak dekat.",
        "📱 Gunakan step counter di smartphone untuk memantau dan memotivasi diri setiap hari.",
        "🎯 Target bertahap: mulai dari 3.000 langkah dan tambah 500 setiap minggu hingga mencapai 5.000+.",
    ],
    "Systolic BP": [
        "🩺 **Tekanan Darah Sistolik Tinggi:** Kurangi asupan garam hingga < 2.300 mg/hari.",
        "🍌 Konsumsi makanan kaya kalium: pisang, kentang, bayam untuk membantu menurunkan tekanan darah.",
        "🏥 Konsultasikan dengan dokter untuk pemantauan rutin dan evaluasi lebih lanjut.",
    ],
    "Diastolic BP": [
        "🩺 **Tekanan Darah Diastolik Tinggi:** Jaga pola makan rendah natrium dan tingkatkan konsumsi buah & sayur.",
        "🧘 Kelola stres secara aktif karena stres adalah salah satu penyebab tekanan darah naik.",
        "🏥 Lakukan check-up rutin dan ikuti saran pengobatan dari dokter.",
    ],
    "BMI": [
        "⚖️ **BMI Di Luar Normal:** Konsultasikan dengan ahli gizi untuk rencana makan yang sesuai.",
        "🥗 Fokus pada pola makan seimbang — sayur, buah, protein tanpa lemak, dan batasi makanan olahan.",
        "🏃 Kombinasikan diet sehat dengan olahraga rutin untuk mencapai dan mempertahankan BMI ideal.",
    ],
}

# =========================
# FORM INPUT
# =========================
with st.form("sleep_form"):
    st.subheader("👤 Data Demografis")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Perempuan", "Laki-laki"])
    with col2:
        age = st.number_input("Usia", 10, 100, 30)
    with col3:
        occupation = st.number_input("Kode Pekerjaan", 0, 9, 0)

    # ── BMI Input (baru) ──
    st.subheader("📏 Data Fisik (untuk perhitungan BMI)")
    col_h, col_w = st.columns(2)
    with col_h:
        height_cm = st.number_input("Tinggi Badan (cm)", 100, 250, 170)
    with col_w:
        weight_kg = st.number_input("Berat Badan (kg)", 30.0, 200.0, 65.0)

    st.subheader("😴 Data Tidur")
    col4, col5 = st.columns(2)
    with col4:
        sleep_duration = st.number_input("Durasi Tidur (jam)", 0.0, 12.0, 7.0)
    with col5:
        quality_of_sleep = st.slider("Kualitas Tidur (1–10)", 1, 10, 7)

    st.subheader("🏃 Aktivitas & Stres")
    col6, col7 = st.columns(2)
    with col6:
        physical_activity = st.slider("Aktivitas Fisik (menit/hari)", 0, 120, 50)
    with col7:
        stress_level = st.slider("Tingkat Stres (1–10)", 1, 10, 5)

    st.subheader("🩺 Data Medis")
    col8, col9, col10, col11 = st.columns(4)
    with col8:
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 70)
    with col9:
        daily_steps = st.number_input("Daily Steps", 0, 30000, 8000)
    with col10:
        systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    with col11:
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)

    st.markdown("---")
    submit = st.form_submit_button("🔮 Prediksi", use_container_width=True)

# =========================
# PREDIKSI + SEMUA FITUR
# =========================
if submit:
    # ── Hitung BMI ──
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # ── Bangun DataFrame input untuk model ──
    input_df = pd.DataFrame([[
        1 if gender == "Laki-laki" else 0,
        age,
        occupation,
        sleep_duration,
        quality_of_sleep,
        physical_activity,
        stress_level,
        heart_rate,
        daily_steps,
        systolic_bp,
        diastolic_bp
    ]], columns=[
        'Gender', 'Age', 'Occupation', 'Sleep_Duration', 'Quality_of_Sleep',
        'Physical_Activity', 'Stress_Level', 'Heart_Rate',
        'Daily_Steps', 'Systolic_BP', 'Diastolic_BP'
    ])

    # ── Prediksi & Probabilitas (multiclass) ──
    pred          = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]   # array 3 elemen: [Sehat, Insomnia, Sleep Apnea]
    pred_label    = LABEL_MAP[pred]
    confidence    = probabilities[pred] * 100          # confidence = probabilitas kelas yang dipilih

    # Probabilitas per kelas dalam persen
    prob_dict = {LABEL_MAP[i]: round(probabilities[i] * 100, 2) for i in range(len(probabilities))}

    # ── Label kepercayaan ──
    if confidence >= 80:
        confidence_label = "🟢 Tinggi"
    elif confidence >= 60:
        confidence_label = "🟡 Sedang"
    else:
        confidence_label = "🔴 Rendah"

    # ================================================================
    # A2 — KATEGORI KEPARAHAN
    # ================================================================
    if pred == 0:   # Sehat
        severity = "✅ Sehat"
    else:
        # Logic: gabungan confidence + quality_of_sleep + stress_level
        severity_score = 0
        if confidence >= 80:
            severity_score += 2
        elif confidence >= 60:
            severity_score += 1

        if quality_of_sleep <= 3:
            severity_score += 2
        elif quality_of_sleep <= 5:
            severity_score += 1

        if stress_level >= 8:
            severity_score += 2
        elif stress_level >= 6:
            severity_score += 1

        if sleep_duration < 5:
            severity_score += 1

        # Mapping score -> label
        if severity_score <= 2:
            severity = "🟡 Ringan"
        elif severity_score <= 4:
            severity = "🟠 Sedang"
        else:
            severity = "🔴 Berat"

    # ================================================================
    # B4 — FLAGGING FAKTOR RISIKO
    # ================================================================
    input_values_for_flag = {
        "Durasi Tidur":     sleep_duration,
        "Kualitas Tidur":   quality_of_sleep,
        "Tingkat Stres":    stress_level,
        "Heart Rate":       heart_rate,
        "Aktivitas Fisik":  physical_activity,
        "Daily Steps":      daily_steps,
        "Systolic BP":      systolic_bp,
        "Diastolic BP":     diastolic_bp,
        "BMI":              bmi,
    }

    flagged = []   # list of (nama, deskripsi_risiko)
    for (nama, cond_fn, deskripsi) in RISK_FLAGS_CONFIG:
        val = input_values_for_flag.get(nama)
        if val is not None and cond_fn(val):
            flagged.append((nama, deskripsi))

    total_risk_flags = len(flagged)

    # ================================================================
    # TAMPILAN UTAMA
    # ================================================================
    st.markdown("---")
    st.subheader("📌 Hasil Prediksi")

    if pred == 0:
        st.success(f"Hasil Prediksi: **{pred_label}**")
    else:
        st.error(f"Hasil Prediksi: **{pred_label}** — {severity}")

    # ── Metrik utama ──
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(label="Hasil Prediksi", value=pred_label)
    with col_m2:
        st.metric(label="Kepercayaan", value=f"{confidence:.2f}%", delta=confidence_label)
    with col_m3:
        st.metric(label="Tingkat Keparahan", value=severity)
    with col_m4:
        st.metric(label="Faktor Risiko", value=f"{total_risk_flags} terdeteksi")

    st.progress(confidence / 100, text=f"Kepercayaan Prediksi: {confidence:.2f}%")

    # ================================================================
    # A3 — VISUAL PERBANDINGAN PROBABILITAS 3 KELAS
    # ================================================================
    st.markdown("---")
    st.subheader("Perbandingan Probabilitas 3 Kelas")

    classes = list(prob_dict.keys())
    values  = list(prob_dict.values())
    colors  = ["#27ae60" if c == "Sehat" else "#e67e22" if c == "Insomnia" else "#e74c3c" for c in classes]

    fig = go.Figure(
        data=[go.Bar(
            x=classes,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition="outside",
            width=0.45,
        )]
    )
    fig.update_layout(
        title="Distribusi Probabilitas per Kelas",
        yaxis_title="Probabilitas (%)",
        yaxis=dict(range=[0, 110]),
        xaxis_title="Kategori",
        height=350,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail probabilitas per kelas (metric)
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric(label="🟢 Sehat", value=f"{prob_dict['Sehat']:.2f}%")
        st.progress(prob_dict["Sehat"] / 100)
    with col_p2:
        st.metric(label="🟠 Insomnia", value=f"{prob_dict['Insomnia']:.2f}%")
        st.progress(prob_dict["Insomnia"] / 100)
    with col_p3:
        st.metric(label="🔴 Sleep Apnea", value=f"{prob_dict['Sleep Apnea']:.2f}%")
        st.progress(prob_dict["Sleep Apnea"] / 100)


    # ================================================================
    # B4 — FAKTOR RISIKO TERDETEKSI
    # ================================================================
    st.markdown("---")
    with st.expander("⚠️ Faktor Risiko Terdeteksi", expanded=True):
        if total_risk_flags == 0:
            st.success("✅ Tidak ada faktor risiko yang terdeteksi dari data input Anda.")
        else:
            st.warning(f"Ditemukan **{total_risk_flags} faktor risiko** dari data yang Anda masukkan:")
            st.markdown("---")
            for (nama, deskripsi) in flagged:
                st.markdown(f"  🔺 **{nama}:** {deskripsi}")

    # ================================================================
    # B5 — PERBANDINGAN RENTANG NORMAL
    # ================================================================
    with st.expander("📋 Perbandingan dengan Rentang Normal", expanded=False):
        st.info("Tabel berikut membandingkan nilai input pasien dengan rentang normal standar kesehatan.")

        # Mapping nama -> nilai pasien
        patient_values = {
            "Usia":             age,
            "Durasi Tidur":     sleep_duration,
            "Kualitas Tidur":   quality_of_sleep,
            "Aktivitas Fisik":  physical_activity,
            "Tingkat Stres":    stress_level,
            "Heart Rate":       heart_rate,
            "Daily Steps":      daily_steps,
            "Systolic BP":      systolic_bp,
            "Diastolic BP":     diastolic_bp,
            "BMI":              round(bmi, 1),
        }

        rows = []
        for param, (min_n, max_n, unit) in NORMAL_RANGES.items():
            val = patient_values[param]
            if min_n is not None and max_n is not None:
                normal_str = f"{min_n} – {max_n} {unit}"
                if val < min_n:
                    status = "🔵 Di bawah normal"
                elif val > max_n:
                    status = "🔴 Di atas normal"
                else:
                    status = "🟢 Normal"
            else:
                normal_str = "—"
                status     = "—"
            rows.append({
                "Parameter":        param,
                "Nilai Pasien":     f"{val} {unit}",
                "Rentang Normal":   normal_str,
                "Status":           status,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ================================================================
    # C7 — REKOMENDASI MEDIS
    # ================================================================
    st.markdown("---")
    with st.expander("🏥 Rekomendasi Medis", expanded=True):
        for tip in RECOMMENDATIONS[pred_label]:
            st.markdown(tip)

    # ================================================================
    # C8 — EDUKASI TENTANG DISORDER 
    # ================================================================
    with st.expander(f"📚 Edukasi: Apa itu {pred_label}?", expanded=False):
        edu = EDUCATION[pred_label]

        st.markdown(f"### Definisi\n{edu['apa']}")
        st.markdown(f"### 🔍 Gejala\n{edu['gejala']}")
        st.markdown(f"### ⚡ Dampak Kesehatan\n{edu['dampak']}")

        st.markdown("### 📌 Fakta Penting")
        for i, fakta in enumerate(edu["fakta"], 1):
            st.markdown(f"  {i}. {fakta}")

    # ================================================================
    # C9 — SARAN GAYA HIDUP DINAMIS
    # ================================================================
    with st.expander("💡 Saran Gaya Hidup", expanded=True):
        if total_risk_flags == 0:
            st.success("✅ Data Anda terlihat sehat! Pertahankan gaya hidup positif dan lakukan check-up rutin.")
        else:
            st.markdown("Berikut saran gaya hidup yang disesuaikan berdasarkan faktor risiko yang terdeteksi:\n")
            for (nama, _) in flagged:
                if nama in LIFESTYLE_TIPS:
                    for tip in LIFESTYLE_TIPS[nama]:
                        st.markdown(f"  {tip}")
                    st.markdown("")   # spasi antar grup tips

    # ================================================================
    # RAW INPUT
    # ================================================================
    with st.expander("🔍 Lihat Data Input (Numerik)"):
        st.dataframe(input_df, use_container_width=True)