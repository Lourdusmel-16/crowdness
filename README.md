# 🏛️ CampusPulse: Real-Time Campus Crowd Density Predictor

A machine learning web application that predicts campus crowd levels to help students and staff avoid rush hours.

🔗 **Live App:** [View Live Streamlit App]([https://crowdness-predictor.streamlit.app/])

---

## 📊 Overview
CampusPulse uses historical occupancy data and an XGBoost regression model to estimate real-time crowd density based on time, temperature, and historical traffic patterns.

### Key Performance Metrics
* **Model Type:** XGBRegressor
* **$R^2$ Score:** `0.9604`
* **RMSE:** `2.95` occupants (relative to a 145-person peak scale)

---

## ⚙️ Approach & Features
Since live IoT sensors weren't available, historical benchmark data was used to train the model with engineered time-series features:
* **Temporal Features:** Hour, day of the week, and month.
* **Context:** Weekends, holidays, and semester schedules.
* **Lag & Rolling Features:** Past hourly counts (`lag_1`, `lag_24`) and moving averages (`rolling_mean_3`) to capture traffic trends.

---
