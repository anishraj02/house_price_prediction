import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("house_price_pipeline.joblib")

model = load_model()


# Feature schema
ALL_FEATURES = model.feature_names_in_

# Neighborhood list from dataset
NEIGHBORHOODS = sorted([
    'NAmes','CollgCr','OldTown','Edwards','Somerst','NridgHt','Gilbert',
    'Sawyer','NWAmes','SawyerW','BrkSide','Crawfor','Mitchel','NoRidge',
    'Timber','IDOTRR','ClearCr','StoneBr','SWISU','MeadowV','Blmngtn','Veenker'
])

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction System")

mode = st.sidebar.selectbox("Select Mode", ["Quick Estimate", "Full Prediction", "Model Insights"])

# ================= QUICK MODE =================
if mode == "Quick Estimate":
    st.subheader("Quick Price Estimate")

    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
    st.caption(
        "1–3 = Poor | 4–5 = Below Avg | 6 = Average | "
        "7 = Good | 8 = Very Good | 9–10 = Excellent"
    )

    neighborhood = st.selectbox("Neighborhood (Location)", NEIGHBORHOODS)

    gr_liv_area = st.number_input("Living Area (sq ft)", 300, 6000, 1500)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 6000, 800)
    garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
    year_built = st.number_input("Year Built", 1870, 2023, 2000)

    # Create full schema template
    input_df = pd.DataFrame([{col: np.nan for col in ALL_FEATURES}])

    # Fill selected features
    input_df["OverallQual"] = overall_qual
    input_df["Neighborhood"] = neighborhood
    input_df["GrLivArea"] = gr_liv_area
    input_df["TotalBsmtSF"] = total_bsmt_sf
    input_df["GarageArea"] = garage_area
    input_df["YearBuilt"] = year_built


# ================= FULL MODE =================
elif mode == "Full Prediction":
    st.subheader("Upload Full Feature CSV")

    uploaded_file = st.file_uploader("Upload CSV with all original features", type=["csv"])

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.dataframe(input_df)


# ================= INSIGHTS MODE =================
else:
    st.subheader("Model Insights — What Drives Price")

    st.markdown("""
    ### Key Price Drivers Identified by the Model
    - 🏗 **Overall Quality** — strongest factor
    - 📐 **Living Area (GrLivArea)**
    - 🧱 **Basement Size**
    - 🚗 **Garage Area**
    - 📍 **Neighborhood (Location)**
    - 📅 **Year Built / Renovation**

    ### Business Interpretation
    - Better construction quality increases value sharply  
    - Larger usable space directly boosts pricing  
    - Premium neighborhoods command higher prices  
    - Newer houses attract higher market demand  

    This matches real-world housing market behavior.
    """)


# ================= PREDICTION =================
if st.button("Predict Price"):

    if 'input_df' not in locals():
        st.warning("Please provide input data first.")
    else:
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)

        st.success(f"💰 Estimated House Price: ${pred_price:,.0f}")
