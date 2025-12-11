import streamlit as st
import pickle
import joblib
import os

st.title("Language Detection App")

# ===== HELPER LOADER =====
def load_pickle(path):
    if not os.path.exists(path):
        st.error(f"File tidak ditemukan: {path}")
        st.stop()

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Coba load pakai joblib jika pickle gagal
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Gagal load file '{path}': {e}")
            st.stop()

# ===== LOAD FILE =====
vectorizer = load_pickle("lang_vectorizer.pkl")
model = load_pickle("lang_nb_model.pkl")

# ===== UI =====
text_input = st.text_area("Masukkan teks:")

if st.button("Prediksi"):
    if not text_input.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        X = vectorizer.transform([text_input])
        prediction = model.predict(X)[0]
        st.success(f"Prediksi: **{prediction}**")