import streamlit as st

# =========================
# Konfigurasi dasar app
# =========================
st.set_page_config(
    page_title="AI Streamlit App",
    page_icon="🤖",
    layout="centered"
)

# =========================
# Halaman utama (Home)
# =========================
st.title("🤖 AI Streamlit Dashboard")
st.write(
    """
    Selamat datang di aplikasi AI berbasis **Streamlit**.

    Gunakan **sidebar** untuk berpindah halaman:
    - 📊 Prediction
    - 📈 Exploration
    - ℹ️ About
    """
)

st.info("Halaman lain dimuat otomatis dari folder `pages/`")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("© 2026 | AI Streamlit Project")
