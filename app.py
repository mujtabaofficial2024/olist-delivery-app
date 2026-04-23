import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Olist Delivery Predictor",
    page_icon="📦",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main {
        background-color: #0f0f0f;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
    }
    
    h1 {
        color: #00d4aa !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.2rem !important;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .stNumberInput label, .stSelectbox label {
        color: #ccc !important;
        font-weight: 600;
    }
    
    .result-box {
        background: linear-gradient(135deg, #00d4aa22, #00d4aa11);
        border: 2px solid #00d4aa;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    
    .result-days {
        font-size: 4rem;
        font-weight: 700;
        color: #00d4aa;
    }
    
    .result-label {
        color: #888;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #00a884);
        color: #000;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px #00d4aa44;
    }
    
    .section-header {
        color: #00d4aa;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #00d4aa33;
    }
    
    div[data-testid="stNumberInput"] input {
        background-color: #1e1e2e !important;
        color: #fff !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("delivery_model_small.pkl")
        return model
    except FileNotFoundError:
        return None

model = load_model()

# Header
st.markdown("# 📦 Olist Delivery Predictor")
st.markdown('<p class="subtitle">Order details daalein — delivery time pata karein</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ `delivery_model.pkl` file nahi mili! App folder mein rakhein.")
    st.stop()

# Form
st.markdown('<p class="section-header">💰 Payment Details</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Price (BRL)", min_value=0.0, value=100.0, step=10.0)
with col2:
    freight_value = st.number_input("Freight Value (BRL)", min_value=0.0, value=20.0, step=5.0)

payment_value = st.number_input("Total Payment Value (BRL)", min_value=0.0, value=120.0, step=10.0)

st.markdown('<p class="section-header">📦 Product Dimensions</p>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    weight = st.number_input("Weight (grams)", min_value=0, value=500, step=100)
    length = st.number_input("Length (cm)", min_value=0, value=20, step=1)
with col4:
    height = st.number_input("Height (cm)", min_value=0, value=15, step=1)
    width = st.number_input("Width (cm)", min_value=0, value=10, step=1)

st.markdown('<p class="section-header">📍 Customer Location</p>', unsafe_allow_html=True)
state_mg = st.selectbox(
    "Customer State",
    options=["Other State", "MG (Minas Gerais)"],
    index=0
)
customer_state_MG = 1 if state_mg == "MG (Minas Gerais)" else 0

st.markdown("<br>", unsafe_allow_html=True)

# Predict button
if st.button("🚀 Delivery Time Predict Karein"):
    try:
        # Build input dataframe matching training features
        input_data = pd.DataFrame(columns=model.feature_names_in_)
        input_data.loc[0] = 0

        input_data['price'] = price
        input_data['freight_value'] = freight_value
        input_data['product_weight_g'] = weight
        input_data['product_length_cm'] = length
        input_data['product_height_cm'] = height
        input_data['product_width_cm'] = width
        input_data['payment_value'] = payment_value
        input_data['customer_state_MG'] = customer_state_MG

        prediction = model.predict(input_data)[0]
        days = round(prediction)

        # Result display
        if days <= 5:
            emoji = "⚡"
            msg = "Express Delivery!"
        elif days <= 10:
            emoji = "✅"
            msg = "Standard Delivery"
        else:
            emoji = "🕐"
            msg = "Thodi der lagegi..."

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size: 2rem;">{emoji}</div>
            <div class="result-days">{days}</div>
            <div class="result-label">din mein delivery hogi</div>
            <div style="color: #00d4aa; margin-top: 0.5rem; font-weight: 600;">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("💡 Model ke features check karein — training wale columns match hone chahiye.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color: #444; font-size: 0.8rem;">Olist E-Commerce Dataset | ML Delivery Prediction</p>', unsafe_allow_html=True)