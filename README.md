# Day 4 - Support Vector Regression (SVR) ⚡

### 📌 Project: Electricity Consumption Forecasting

In this project, I used **Support Vector Regression (SVR)** with the **RBF kernel** to predict electricity consumption based on **hour of the day** and **temperature**.

---

## 📊 Dataset
I generated a **synthetic dataset** with:
- Hour of the day (0–23.5, step 0.5)
- Temperature (daily sinusoidal variation with noise)
- Electricity consumption (higher during evening & hotter hours)

---

## ⚙️ Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## 🧠 Algorithm
- **SVR with RBF kernel**: Captures non-linear relationships between features (time, temperature) and electricity consumption.  
- Scaled data before training using `StandardScaler`.  
- Evaluated with **R² Score** and **MSE**.

---

## ✅ Results
- **R² Score:** ~0.93  
- **MSE:** Small error compared to variance in data  
- SVR captured **peak consumption during evenings** and higher usage at higher temperatures.

---

## 📈 Visualization
Scatter plot comparing **actual vs. predicted consumption**.

---

## 🔗 Learning
SVR is a **powerful regression algorithm** when data has **non-linear patterns**, and scaling features is critical for good performance.

