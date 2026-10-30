import streamlit as st
import pandas as pd
import numpy as np
import joblib

rf = joblib.load("used_phone_model.pkl")
model_cols = joblib.load("model_columns.pkl")
defaults = joblib.load("defaults.pkl")
EURO_TO_INR = joblib.load("euro_to_inr.pkl")
phones = joblib.load("phones.pkl")


def predict_price(phone, days_used, new_price_inr):
    row = defaults.copy()
    row["screen_size"] = phone["screen_size"]
    row["rear_camera_mp"] = phone["rear_camera_mp"]
    row["front_camera_mp"] = phone["front_camera_mp"]
    row["internal_memory"] = phone["internal_memory"]
    row["ram"] = phone["ram"]
    row["battery"] = phone["battery"]
    row["weight"] = phone["weight"]
    row["release_year"] = phone["release_year"]
    row["days_used"] = days_used
    row["normalized_new_price"] = np.log(new_price_inr / EURO_TO_INR)
    row["4g_yes"] = 1
    row["5g_yes"] = 1 if phone["g5"] == "yes" else 0

    col = "device_brand_" + phone.get("pred_brand", phone["brand"])
    if col not in row.index:
        col = "device_brand_Others"
    row[col] = 1
    if phone["os"] == "iOS" and "os_iOS" in row.index:
        row["os_iOS"] = 1

    x = pd.DataFrame([row])[model_cols]
    return np.exp(rf.predict(x)[0]) * EURO_TO_INR


def final_price(phone, days_used, new_price_inr,
                camera_ok=True, speaker_ok=True, touch_ok=True):
    price = predict_price(phone, days_used, new_price_inr)
    cap = new_price_inr * max(0.15, 0.85 - 0.55 * days_used / 1095)
    price = min(price, cap)
    if not camera_ok:
        price *= 0.85
    if not speaker_ok:
        price *= 0.92
    if not touch_ok:
        price *= 0.70
    return round(price)


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(236,72,153,0.45), transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(56,189,248,0.40), transparent 40%),
        linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

/* background me do phone */
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    width: 170px; height: 340px;
    border: 6px solid rgba(255,255,255,0.35);
    border-radius: 34px;
    background: linear-gradient(160deg, rgba(255,255,255,0.18), rgba(255,255,255,0.03));
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), inset 0 0 25px rgba(255,255,255,0.12);
    pointer-events: none;
    z-index: 0;
}
.stApp::before { left: 4%; top: 18%; transform: rotate(-14deg); }
.stApp::after  { right: 4%; bottom: 10%; width: 200px; height: 400px; transform: rotate(12deg); }
@media (max-width: 900px) { .stApp::before, .stApp::after { display: none; } }

[data-testid="stHeader"] { background: transparent; }

/* glass card */
.block-container {
    position: relative; z-index: 1;
    max-width: 760px;
    margin-top: 2rem;
    padding: 2.5rem !important;
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 28px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

/* title */
h1 {
    text-align: center;
    font-weight: 800;
    background: linear-gradient(90deg, #facc15, #fb7185, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3, [data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] p {
    color: #ffffff !important;
    font-weight: 600;
}

/* price card */
[data-testid="stAlert"] {
    background: linear-gradient(90deg, #facc15, #fb923c);
    border: none;
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(250,204,21,0.6);
}
[data-testid="stAlert"] p {
    color: #1e1b4b !important;
    font-size: 32px;
    font-weight: 800;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📱 Used Mobile Price Prediction")

brands = sorted(set(p["brand"] for p in phones.values()))
ram_opts = [1, 2, 3, 4, 6, 8, 12, 16]
mem_opts = [16, 32, 64, 128, 256, 512]

brand = st.selectbox("Brand", brands, index=None, placeholder="Brand chuno")

models = [n for n, p in phones.items() if p["brand"] == brand] if brand else []
model_name = st.selectbox(
    "Model", models, index=None,
    placeholder="Model chuno" if brand else "Pehle brand chuno",
    disabled=not brand, key="model_" + str(brand))

c1, c2, c3 = st.columns(3)
ram = c1.selectbox("RAM (GB)", ram_opts, index=None, placeholder="RAM chuno")
rom = c2.selectbox("Storage / ROM (GB)", mem_opts, index=None, placeholder="Storage chuno")
net = c3.selectbox("Network", ["4G", "5G"], index=None, placeholder="Network chuno")

days_used = st.number_input("Kitne din use hua (days)", min_value=0,
                            max_value=1500, value=0, step=10)
new_price = st.number_input("Kharidte time ka price (₹)", min_value=0,
                            value=0, step=500)

st.subheader("Phone ki condition")
camera_ok = st.checkbox("Camera working", value=True)
speaker_ok = st.checkbox("Speaker working", value=True)
touch_ok = st.checkbox("Touch working", value=True)

ready = all([brand, model_name, ram, rom, net]) and new_price > 0

if ready:
    phone = dict(phones[model_name])
    phone["ram"] = ram
    phone["internal_memory"] = rom
    phone["g5"] = "yes" if net == "5G" else "no"

    price = final_price(phone, days_used, new_price,
                        camera_ok, speaker_ok, touch_ok)
    st.success(f"Estimated used price: ₹{price:,}")
else:
    st.caption("Brand, model, RAM, storage, network aur kharid ka price bharo , phir price dikhega.")
