import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Prediksi Penyakit Jantung")
st.markdown("Masukkan data pasien untuk memprediksi **Heart Disease**")
st.markdown("---")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("models/logistic_regression_model.pkl")

model = load_model()

# =========================
# MAPPING
# =========================
sex_map = {
    "Perempuan": 0,
    "Laki-laki": 1
}

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_map = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

exang_map = {
    "Tidak": 0,
    "Ya": 1
}

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_map = {
    "Fixed Defect": 1,
    "Normal": 2,
    "Reversible Defect": 3
}

label_map = {
    0: "🟢 Tidak Ada Penyakit Jantung",
    1: "🔴 Ada Penyakit Jantung"
}

# =========================
# KONSTANTA: RENTANG NORMAL & RISIKO
# =========================
# Setiap entry: (nama tampil, nilai input, min_normal, max_normal, unit)
# min/max = None berarti tidak ada threshold numerik (fitur kategorikal)
NORMAL_RANGES = {
    "age":                          ("Usia",                          None, None,  "Tahun"),
    "resting_blood_pressure":       ("Tekanan Darah Istirahat",      80,   120,   "mm Hg"),
    "cholesterol":                  ("Kolesterol Serum",             0,    200,   "mg/dl"),
    "max_heart_rate_achieved":      ("Detak Jantung Maksimum",       60,   100,   "bpm"),
    "st_depression":                ("ST Depression (oldpeak)",      0.0,  1.0,   "mm"),
}

# Threshold untuk flagging faktor risiko
RISK_THRESHOLDS = {
    "resting_blood_pressure":       ("Tekanan Darah Istirahat",      120),   # > 120 = risiko
    "cholesterol":                  ("Kolesterol Serum",             200),   # > 200 = risiko
    "fasting_blood_sugar":          ("Gula Darah Puasa > 120",       0),     # == 1 = risiko
    "exercise_induced_angina":      ("Angina Akibat Olahraga",       0),     # == 1 = risiko
    "st_depression":                ("ST Depression",                1.0),   # > 1.0 = risiko
    "num_major_vessels":            ("Pembuluh Darah Utama",         0),     # > 0 = risiko
}

# Kategorikal risiko tambahan (dicek terpisah)
CATEGORICAL_RISK = {
    "chest_pain_type":              ("Tipe Nyeri Dada",              [0]),                     # Typical Angina = risiko
    "restecg_map":                  ("EKG Istirahat",                [1, 2]),                  # Abnormal / LVH = risiko
    "thalassemia":                  ("Thalassemia",                  [1, 3]),                  # Fixed / Reversible = risiko
    "st_slope":                     ("Kemiringan ST",                [1, 2]),                  # Flat / Downsloping = risiko
}

# =========================
# REKOMENDASI MEDIS (static)
# =========================
RECOMMENDATIONS = {
    0: [
        "✅ Hasil prediksi menunjukkan **tidak ada indikasi penyakit jantung**.",
        "📋 Tetap lakukan **check-up rutin** setidaknya 1 kali per tahun untuk memantau kondisi jantung.",
        "🥗 Jaga pola makan seimbang — tingkatkan konsumsi buah, sayur, dan serat.",
        "🏃 Rutinkan olahraga fisik ringan hingga sedang (30 menit/hari, 5 hari/minggu).",
        "🚬 Hindari rokok dan batasi konsumsi alkohol.",
        "😴 Jaga kualitas tidur (7–9 jam/malam) dan kelola stres.",
        "⚖️ Pertahankan berat badan ideal dan pantau tekanan darah secara berkala.",
    ],
    1: [
        "⚠️ Hasil prediksi menunjukkan **adanya indikasi penyakit jantung**.",
        "🏥 **Segera konsultasikan** hasil ini kepada dokter spesialis kardiologi untuk evaluasi lebih lanjut.",
        "📊 Lakukan pemeriksaan lengkap termasuk **angiografi koronari** jika disarankan dokter.",
        "💊 Ikuti regimen pengobatan yang diberikan dokter dan **jangan menghentikan obat** tanpa konsultasi.",
        "🥗 Atur pola makan rendah natrium dan kolesterol — pilih makanan ramah jantung.",
        "🏃 Olahraga ringan boleh dilakukan, tetapi **konsultasikan jenis dan intensitasnya** dengan dokter.",
        "📅 Jadwalkan **follow-up rutin** untuk memantau perkembangan kondisi jantung.",
        "🧘 Kelola stres dengan meditasi atau teknik relaksasi untuk menjaga kestabilan tekanan darah.",
    ]
}

# =========================
# SARAN GAYA HIDUP (dinamis berdasarkan flagged risk)
# =========================
LIFESTYLE_TIPS = {
    "resting_blood_pressure": [
        "**Tekanan Darah Tinggi:** Kurangi asupan garam (natrium) hingga < 2.300 mg/hari.",
        "- Konsumsi makanan kaya kalium seperti pisang, kentang, dan bayam.",
        "- Lakukan relaksasi atau meditasi secara rutin untuk menurunkan stres.",
    ],
    "cholesterol": [
        "**Kolesterol Tinggi:** Ganti lemak jenuh dengan lemak tak jenuh (alpukat, minyak zaitun).",
        "- Konsumsi ikan berlemak (salmon, tuna) 2x per minggu untuk minyak omega-3.",
        "- Tingkatkan serat larut dari oat, kacang-kacangan, dan buah-buahan.",
    ],
    "fasting_blood_sugar": [
        "**Gula Darah Tinggi:** Batasi asupan karbohidrat sederhana dan gula tambahan.",
        "- Perbanyak sayuran hijau dan protein tanpa lemak dalam setiap makanan.",
        "- Makan dengan jadwal yang teratur dan hindari makan larut malam.",
    ],
    "exercise_induced_angina": [
        "**Angina Saat Olahraga:** Konsultasikan jenis olahraga yang aman dengan dokter kardiologi.",
        "- Mulai dari olahraga ringan seperti jalan santai dan tingkatkan secara bertahap.",
        "- Hentikan olahraga segera jika terjadi nyeri dada dan segera cari pertolongan medis.",
    ],
    "st_depression": [
        "- **ST Depression Tinggi:** Ini bisa menjadi tanda iskemia — pastikan sudah ditangani dokter.",
        "- Jangan lewatkan jadwal kontrol dan minum obat sesuai resep.",
    ],
    "num_major_vessels": [
        "**Penyumbatan Pembuluh Darah:** Risiko blockage — konsultasikan dengan kardiologi.",
        "- Jaga diet rendah kolesterol dan rutin olahraga ringan.",
        "- Lakukan monitoring berkala sesuai saran dokter.",
    ],
    "chest_pain_type": [
        "**Typical Angina:** Ini adalah tanda kuat penyakit arteri koronari — segera konsultasikan.",
        "- Dokter kemungkinan akan merekomendasikan obat anti-angina.",
    ],
    "restecg_map": [
        "**Hasil EKG Abnormal:** Pemeriksaan EKG ulang dan echocardiogram sangat direkomendasikan.",
        "- Konsultasikan hasil ini secara langsung kepada dokter.",
    ],
    "thalassemia": [
        "**Thalassemia Abnormal:** Kondisi ini meningkatkan risiko penyakit jantung.",
        "- Pastikan sudah melakukan pemeriksaan nuclear stress test jika disarankan.",
    ],
    "st_slope": [
        "**Kemiringan ST Flat/Down:** Pola ini sering dikaitkan dengan risiko penyakit koronari.",
        "- Lakukan konsultasi untuk pemeriksaan lebih mendalam.",
    ],
}

# =========================
# FORM INPUT
# =========================
with st.form("heart_disease_form"):

    st.subheader("👤 Data Demografis")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Usia", 1, 120, 55)

    with col2:
        sex_display = st.selectbox("Jenis Kelamin", list(sex_map.keys()))
        sex = sex_map[sex_display]

    with col3:
        cp_display = st.selectbox("Tipe Nyeri Dada", list(cp_map.keys()))
        chest_pain_type = cp_map[cp_display]

    st.subheader("🩺 Tekanan Darah & Kolesterol")
    col4, col5 = st.columns(2)

    with col4:
        resting_blood_pressure = st.number_input(
            "Tekanan Darah Istirahat (mm Hg)", 80, 220, 130
        )

    with col5:
        cholesterol = st.number_input(
            "Kolesterol Serum (mg/dl)", 100, 600, 250
        )

    st.subheader("🧪 Pemeriksaan Medis")
    col6, col7, col8 = st.columns(3)

    with col6:
        fasting_blood_sugar = st.selectbox(
            "Gula Darah Puasa > 120 mg/dl?",
            ["Tidak", "Ya"]
        )
        fasting_blood_sugar = 1 if fasting_blood_sugar == "Ya" else 0

    with col7:
        restecg_display = st.selectbox(
            "Hasil EKG Istirahat",
            list(restecg_map.keys())
        )
        resting_electrocardiogram = restecg_map[restecg_display]

    with col8:
        max_heart_rate_achieved = st.number_input(
            "Detak Jantung Maksimum", 60, 220, 150
        )

    st.subheader("🏃 Tes Olahraga")
    col9, col10, col11 = st.columns(3)

    with col9:
        exang_display = st.selectbox(
            "Angina Akibat Olahraga",
            list(exang_map.keys())
        )
        exercise_induced_angina = exang_map[exang_display]

    with col10:
        st_depression = st.number_input(
            "ST Depression (oldpeak)", 0.0, 10.0, 1.2, step=0.1
        )

    with col11:
        slope_display = st.selectbox(
            "Kemiringan ST",
            list(slope_map.keys())
        )
        st_slope = slope_map[slope_display]

    st.subheader("🔬 Pemeriksaan Tambahan")
    col12, col13 = st.columns(2)

    with col12:
        num_major_vessels = st.selectbox(
            "Jumlah Pembuluh Darah Utama",
            [0, 1, 2, 3]
        )

    with col13:
        thal_display = st.selectbox(
            "Status Thalassemia",
            list(thal_map.keys())
        )
        thalassemia = thal_map[thal_display]

    st.markdown("---")
    submit = st.form_submit_button("🔮 Prediksi", use_container_width=True)

# =========================
# PREDIKSI + SEMUA FITUR BARU
# =========================
if submit:
    # ── 0. Bangun DataFrame input ──
    input_data = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_electrocardiogram": resting_electrocardiogram,
        "max_heart_rate_achieved": max_heart_rate_achieved,
        "exercise_induced_angina": exercise_induced_angina,
        "st_depression": st_depression,
        "st_slope": st_slope,
        "num_major_vessels": num_major_vessels,
        "thalassemia": thalassemia
    }
    input_df = pd.DataFrame([input_data])

    # ── Prediksi & Probabilitas ──
    prediction   = model.predict(input_df)[0]
    probabilities    = model.predict_proba(input_df)[0]
    prob_no_disease  = probabilities[0] * 100
    prob_disease     = probabilities[1] * 100
    confidence       = max(prob_no_disease, prob_disease)

    if confidence >= 80:
        confidence_label = "🟢 Tinggi"
    elif confidence >= 60:
        confidence_label = "🟡 Sedang"
    else:
        confidence_label = "🔴 Rendah"

    # ================================================================
    # BAGIAN 2 — FLAGGING FAKTOR RISIKO (numerikal + kategorikal)
    # ================================================================
    flagged_keys = []   # kunci dari LIFESTYLE_TIPS yang akan ditampilkan

    # Numerikal
    for key, (nama, threshold) in RISK_THRESHOLDS.items():
        val = input_data[key]
        if key in ("fasting_blood_sugar", "exercise_induced_angina"):
            if val == 1:
                flagged_keys.append(key)
        elif key == "num_major_vessels":
            if val > 0:
                flagged_keys.append(key)
        else:
            if val > threshold:
                flagged_keys.append(key)

    # Kategorikal
    cat_checks = {
        "chest_pain_type":  (chest_pain_type,                  [0]),
        "restecg_map":      (resting_electrocardiogram,        [1, 2]),
        "thalassemia":      (thalassemia,                      [1, 3]),
        "st_slope":         (st_slope,                         [1, 2]),
    }
    for key, (val, risky_vals) in cat_checks.items():
        if val in risky_vals:
            flagged_keys.append(key)

    total_risk_flags = len(flagged_keys)

    # ================================================================
    # TAMPILAN UTAMA
    # ================================================================
    st.markdown("---")
    st.subheader("📌 Hasil Prediksi")

    if prediction == 1:
        st.error(label_map[prediction])
    else:
        st.success(label_map[prediction])

    # ── Kepercayaan ──
    st.subheader("📊 Tingkat Kepercayaan Prediksi")
    col_conf1, col_conf2, col_conf3 = st.columns(3)

    with col_conf1:
        st.metric(label="Kepercayaan Prediksi", value=f"{confidence:.2f}%", delta=confidence_label)
    with col_conf2:
        st.metric(label="Prediksi Utama", value=label_map[prediction])

    st.progress(confidence / 100, text=f"Kepercayaan: {confidence:.2f}%")

    # ── Detail Probabilitas ──
    st.subheader("📈 Detail Probabilitas")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric(label="🟢 Tidak Ada Penyakit Jantung", value=f"{prob_no_disease:.2f}%")
        st.progress(prob_no_disease / 100)
    with col_p2:
        st.metric(label="🔴 Ada Penyakit Jantung", value=f"{prob_disease:.2f}%")
        st.progress(prob_disease / 100)

    st.markdown("---")

    # ================================================================
    # 1. REKOMENDASI MEDIS
    # ================================================================
    with st.expander("🏥 Rekomendasi Medis", expanded=True):
        for tip in RECOMMENDATIONS[prediction]:
            st.markdown(tip)

    # ================================================================
    # 3. FEATURE IMPORTANCE — Kontribusi Fitur (horizontal bar chart)
    # ================================================================
    with st.expander("📊 Kontribusi Fitur terhadap Prediksi", expanded=False):
        st.info(
            "Chart di bawah menunjukkan **seberapa besar pengaruh setiap fitur** "
            "terhadap prediksi pada data pasien ini. "
            "Nilai positif mendorong ke arah 'Ada Penyakit', negatif ke 'Tidak Ada Penyakit'."
        )

        feature_names = input_df.columns.tolist()
        coefs          = model.coef_[0]                        # koefisien logistic regression
        input_values   = input_df.values[0].astype(float)
        contributions  = coefs * input_values                  # kontribusi per fitur

        # Label tampilan yang lebih readable
        feature_labels = {
            "age":                          "Usia",
            "sex":                          "Jenis Kelamin",
            "chest_pain_type":              "Tipe Nyeri Dada",
            "resting_blood_pressure":       "Tekanan Darah Istirahat",
            "cholesterol":                  "Kolesterol Serum",
            "fasting_blood_sugar":          "Gula Darah Puasa",
            "resting_electrocardiogram":    "EKG Istirahat",
            "max_heart_rate_achieved":      "Detak Jantung Maks",
            "exercise_induced_angina":      "Angina Olahraga",
            "st_depression":                "ST Depression",
            "st_slope":                     "Kemiringan ST",
            "num_major_vessels":            "Pembuluh Darah Utama",
            "thalassemia":                  "Thalassemia",
        }
        labels = [feature_labels.get(f, f) for f in feature_names]

        # Sortir dari terkecil ke terbesar (horizontal bar)
        sorted_idx  = np.argsort(contributions)
        sorted_labels = [labels[i] for i in sorted_idx]
        sorted_vals   = [contributions[i] for i in sorted_idx]

        # Warna: positif = merah (mendorong "sakit"), negatif = hijau (mendorong "sehat")
        colors = ["#e74c3c" if v > 0 else "#27ae60" for v in sorted_vals]

        fig = go.Figure(
            data=[go.Bar(
                x=sorted_vals,
                y=sorted_labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in sorted_vals],
                textposition="outside",
            )]
        )
        fig.update_layout(
            title="Kontribusi Fitur terhadap Prediksi Penyakit Jantung",
            xaxis_title="Kontribusi (positif = risiko ↑)",
            yaxis_title="Fitur",
            height=480,
            margin=dict(l=180, r=60, t=60, b=40),
            template="plotly_white",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

        st.plotly_chart(fig, use_container_width=True)

        # Tabel kontribusi
        contrib_df = pd.DataFrame({
            "Fitur":         labels,
            "Nilai Input":  input_values,
            "Koefisien":    coefs,
            "Kontribusi":   contributions
        }).sort_values("Kontribusi", ascending=False).reset_index(drop=True)
        contrib_df.index = contrib_df.index + 1   # mulai dari 1

        with st.expander("📋 Lihat Tabel Detail Kontribusi"):
            st.dataframe(contrib_df, use_container_width=True)

    # ================================================================
    # 5. PERBANDINGAN RENTANG NORMAL
    # ================================================================
    with st.expander("📋 Perbandingan dengan Rentang Normal", expanded=False):
        st.info("Tabel berikut membandingkan nilai input pasien dengan rentang normal standar medis.")

        comparison_rows = []
        for key, (nama, min_n, max_n, unit) in NORMAL_RANGES.items():
            val = input_data[key]
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
            comparison_rows.append({
                "Parameter":        nama,
                "Nilai Pasien":     f"{val} {unit}",
                "Rentang Normal":   normal_str,
                "Status":           status,
            })

        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ================================================================
    # 6. SARAN GAYA HIDUP (dinamis berdasarkan flagged risk)
    # ================================================================
    with st.expander("💡 Saran Gaya Hidup", expanded=True):
        if total_risk_flags == 0:
            st.success("✅ Data Anda terlihat sehat! Pertahankan gaya hidup positif dan lakukan check-up rutin.")
        else:
            st.markdown("Berikut saran gaya hidup yang disesuaikan berdasarkan faktor risiko yang terdeteksi:\n")
            for key in flagged_keys:
                if key in LIFESTYLE_TIPS:
                    for tip in LIFESTYLE_TIPS[key]:
                        st.markdown(f"  {tip}")
                    st.markdown("")   # spasi antar grup

    # ================================================================
    # RAW INPUT
    # ================================================================
    with st.expander("🔍 Lihat Data Input (Numerik)"):
        st.dataframe(input_df, use_container_width=True)