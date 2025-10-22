import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
CAL_MODEL_PATH = Path("tsunami_best_model.pkl")
RAW_DATA_PATH  = Path("earthquake_data_tsunami.csv")

REQUIRED_COLS = [
    "magnitude","depth","cdi","mmi","sig","nst","dmin","gap",
    "latitude","longitude","Year","Month"
]

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Tsunami Risk Prediction", page_icon="🌊", layout="centered")
st.title("🌊 Tsunami Risk Prediction")
st.caption("Enter the seismic parameters, then click **Predict** to check the potential risk of a tsunami.")

# =========================
# LOAD MODEL + DATA
# =========================
@st.cache_resource
def load_model_and_data():
    if not CAL_MODEL_PATH.exists():
        st.error("❌ The model file `tsunami_best_model.pkl` was not found.")
        st.stop()
    bundle = joblib.load(CAL_MODEL_PATH)
    model = bundle.get("model", None)
    meta  = bundle.get("meta", {})
    df    = pd.read_csv(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else None
    return model, meta, df

model, meta, df = load_model_and_data()

# Feature helper
year_avg_map = {}
global_mag_mean = 5.5
if df is not None and {"Year","magnitude"}.issubset(df.columns):
    year_avg_map = df.groupby("Year")["magnitude"].mean().to_dict()
    global_mag_mean = float(df["magnitude"].mean())

def feature_engineering(df_row: pd.DataFrame) -> pd.DataFrame:
    df_row = df_row.copy()
    for c in REQUIRED_COLS:
        if c not in df_row.columns:
            df_row[c] = np.nan
    df_row["log10_energy"] = 1.5 * df_row["magnitude"] + 4.8
    df_row["shallow_quake"] = (df_row["depth"] < 70).astype(int)
    df_row["yearly_avg_magnitude"] = df_row["Year"].map(year_avg_map).fillna(global_mag_mean)
    return df_row

# =========================
# FORM INPUT
# =========================
st.subheader("🧩 Input Seismic Parameters")

with st.form("single_prediction"):
    col1, col2 = st.columns(2)
    with col1:
        magnitude = st.number_input("Magnitude", value=6.5, min_value=0.0, max_value=10.0, step=0.1)
        depth     = st.number_input("Depth (km)", value=30.0, min_value=0.0, max_value=800.0, step=1.0)
        cdi       = st.number_input("CDI", value=5.0, min_value=0.0, max_value=12.0, step=0.1)
        mmi       = st.number_input("MMI", value=6.0, min_value=0.0, max_value=12.0, step=0.1)
        sig       = st.number_input("sig", value=600.0, min_value=0.0, max_value=2000.0, step=1.0)
    with col2:
        nst       = st.number_input("nst", value=100.0, min_value=0.0, max_value=1000.0, step=1.0)
        dmin      = st.number_input("dmin", value=0.5, min_value=0.0, max_value=100.0, step=0.1, format="%.3f")
        gap       = st.number_input("gap", value=50.0, min_value=0.0, max_value=360.0, step=1.0)
        latitude  = st.number_input("Latitude", value=-3.2, min_value=-90.0, max_value=90.0, step=0.1)
        longitude = st.number_input("Longitude", value=135.0, min_value=-180.0, max_value=180.0, step=0.1)

    c3, c4 = st.columns(2)
    with c3:
        Year  = st.number_input("Year", value=2025, min_value=1900, max_value=2100, step=1)
    with c4:
        Month = st.number_input("Month", value=11, min_value=1, max_value=12, step=1)

    do_predict = st.form_submit_button("🔮 Predict")

# =========================
# PREDICTION LOGIC
# =========================
if do_predict:
    sample = pd.DataFrame([{
        "magnitude": magnitude, "depth": depth, "cdi": cdi, "mmi": mmi, "sig": sig,
        "nst": nst, "dmin": dmin, "gap": gap, "latitude": latitude, "longitude": longitude,
        "Year": Year, "Month": Month
    }])
    sample_fe = feature_engineering(sample)
    try:
        prob = float(model.predict_proba(sample_fe)[:, 1][0])
        thr  = float(meta.get("best_threshold", 0.5))  
        pred = int(prob >= thr)
        st.success("✅ Prediction complete")
        st.metric("Prediction", "🔴 High risk" if pred == 1 else "🟢 Low risk")
        st.metric("Probability (class=1)", f"{prob:.2%}")
        st.caption(f"(Automatic using best threshold from training: {thr:.2f})")
        with st.expander("Check out specially designed features"):
            st.dataframe(sample_fe)
    except Exception as e:
        st.error(f"Fail to predict: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(f"Model: {meta.get('name','Unknown')} | Threshold (train): {meta.get('best_threshold','-')} | Features: {', '.join(meta.get('features', []))}")
st.caption("Educational demonstration — not for real-time tsunami prediction.")