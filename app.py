import xarray as xr
import unidecode
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rf_era5_mekong_pipeline.joblib")

st.title("🌾 Dự đoán năng suất lúa (Mekong Delta)")
st.markdown("Nhập thông tin để dự đoán năng suất lúa:")

province = st.selectbox("Tỉnh", sorted(list(model.named_steps['preproc'].transformers_[1][1].named_steps['onehot'].categories_[0])))
year = st.number_input("Năm", min_value=2000, max_value=2030, value=2015)
month = st.selectbox("Tháng", list(range(1,13)), index=6)
t2m = st.number_input("Nhiệt độ 2m (°C)", value=28.0, format="%.2f")
d2m = st.number_input("Điểm sương (°C)", value=24.0, format="%.2f")
tp = st.number_input("Lượng mưa (mm)", value=5.0, format="%.2f")
ssrd = st.number_input("Bức xạ mặt trời (Wh/m²)", value=200.0, format="%.2f")

if st.button("🔮 Dự đoán"):
    Xnew = pd.DataFrame([{
        't2m': t2m,
        'd2m': d2m,
        'tp': tp,
        'ssrd': ssrd,
        'month': month,
        'year': year,
        'province': province
    }])
    pred = model.predict(Xnew)[0]
    st.success(f"🌾 Dự đoán năng suất: {pred:.3f} (đơn vị cùng với dữ liệu gốc)")
