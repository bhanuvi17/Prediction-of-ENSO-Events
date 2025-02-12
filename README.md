# 🌊 ENSO Events Prediction using LSTM

This project performs **time series prediction** of the **Oceanic Niño Index (ONI)** using **Long Short-Term Memory (LSTM) networks**. The ONI time series is extracted from the dataset `ENSO.csv`, and the model forecasts the ONI values for the upcoming months based on historical data.

---

## 📌 Features

✅ Predicts future ONI values using LSTM neural networks  
✅ Interactive dashboard built with **Streamlit**  
✅ Data visualization using **Matplotlib** and **Plotly**  
✅ Model trained using **TensorFlow/Keras**  
✅ Deployed on **Streamlit Cloud**  

---

## 📂 Project Structure

```
ENSO_EVENTS_PREDICTION/
│── .devcontainer/              # Development container configuration
│── enso_env/                   # Virtual environment setup
│── .gitignore                   # Git ignore file
│── dashboard.py                 # Streamlit dashboard
│── enso_prediction.ipynb        # Jupyter Notebook for LSTM model training
│── ENSO.csv                     # Dataset used for training and testing
│── model_lstm.keras             # Saved LSTM model
│── Procfile                     # Deployment configuration for Streamlit
│── README.md                    # Project documentation
│── requirements.txt              # Dependencies list
```

---

## 🎯 Technologies Used

- **Machine Learning:** TensorFlow, Keras, scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Plotly  
- **Web Application:** Streamlit  
- **Deployment:** Streamlit Cloud  

---

## 🔧 Installation & Setup

### 🔹 Clone the Repository
```bash
git clone https://github.com/bhanuvi17/Prediction-of-ENSO-Events.git
cd Prediction-of-ENSO-Events
```

### 🔹 Create a Virtual Environment
```bash
python -m venv enso_env
source enso_env/bin/activate  # On Mac/Linux
enso_env\Scripts\activate  # On Windows
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Run the Streamlit App
```bash
streamlit run dashboard.py
```
🔗 Open in your browser: **http://localhost:8501/**  

---

## 🚀 Live Dashboard  
🔗 **[Streamlit App](https://bhanuvi17-prediction-of-enso-events-dashboard-w3nzfx.streamlit.app/)**

---

## 🚀 Deploying on Streamlit Cloud  

### 1️⃣ **Push Code to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2️⃣ **Deploy on Streamlit:**
- Go to [Streamlit Cloud](https://streamlit.io/cloud)
- Click **New App**  
- Connect to your GitHub repository  
- Set **Main file path:** `dashboard.py`  
- Click **Deploy** 🎉  

---

## 📸 Screenshots

### 🔹 **Web Interface**
![Web App Screenshot](https://github.com/bhanuvi17/Prediction-of-ENSO-Events/blob/ba223734e58fa39d2f4a48bef880722fa8acb655/Screenshot%202025-02-12%20232014.png)

---

## 🏆 Future Enhancements  

✅ Add support for multiple time series forecasting models (e.g., ARIMA, Prophet)  
✅ Improve UI with more interactive charts  
✅ Allow user input for different datasets  

---

## 📜 License  
This project is **open-source** under the [MIT License](LICENSE).  

---

### 💡 **Need Help?**  
Feel free to **open an issue** or **contribute** to improve this project! 😊  
⭐ If you like this project, give it a **star** on GitHub! ⭐

