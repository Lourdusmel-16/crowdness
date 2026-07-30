import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Campus Crowd Predictor", page_icon="🏛️", layout="centered")

st.title("🏛️ Campus Resource & Crowd Density Predictor")
st.write("Predict real-time campus occupancy using machine learning to avoid rush hours.")

@st.cache_resource
def load_model():
    with open("crowd_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.subheader("Select Time Parameters")
hour = st.slider("Hour of the Day", 0, 23, 12)
day_name = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
month = st.slider("Month", 1, 12, 5)
is_weekend = st.selectbox("Is Weekend?", [0, 1])
is_holiday = st.selectbox("Is Holiday?", [0, 1])
temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
is_start_of_semester = st.selectbox("Is Start of Semester?", [0, 1])
is_during_semester = st.selectbox("Is During Semester?", [0, 1])

day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
day_of_week = day_mapping[day_name]

if st.button("Predict Occupancy", type="primary"):
    # Pass all 8 features in the exact column order from your dataset
    features = np.array([[day_of_week, is_weekend, is_holiday, temperature, is_start_of_semester, is_during_semester, month, hour]])
    
    prediction = model.predict(features)[0]
    
    st.markdown("---")
    st.metric(label="Estimated People Count", value=int(prediction))
    
    if prediction > 50:
        st.error("⚠️ **High Congestion Warning:** Consider visiting later.")
    else:
        st.success("✅ **Low Traffic:** Great time to visit!")
