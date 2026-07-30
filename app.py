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

day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
day_of_week = day_mapping[day_name]

if st.button("Predict Occupancy", type="primary"):
    features = np.array([[hour, day_of_week, month]])
    prediction = model.predict(features)[0]
    
    st.markdown("---")
    st.metric(label="Estimated People Count", value=int(prediction))
    
    if prediction > 50:
        st.error("⚠️ **High Congestion Warning:** Consider visiting later.")
    else:
        st.success("✅ **Low Traffic:** Great time to visit!")
