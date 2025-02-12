✨ ENSO Events Prediction using LSTM

This is an ENSO (El Niño-Southern Oscillation) Time Series Prediction web application built using Streamlit and LSTM Neural Networks. The model predicts the Oceanic Niño Index (ONI) based on historical ENSO data.

---

📌 Features

✅ Predict ENSO events based on historical ONI values  
✅ Interactive dashboard using Streamlit  
✅ Time series forecasting using LSTM (Long Short-Term Memory) networks  
✅ Data visualization using Plotly and Matplotlib  
✅ Deployed on Streamlit Cloud  

---

📂 Project Structure

```
enso_prediction/
│── ENSO.csv                    # Dataset (ONI values)
│── model_lstm.keras            # Trained LSTM model
│── enso_prediction.ipynb       # Jupyter Notebook for training
│── dashboard.py                # Streamlit dashboard (main file)
│── requirements.txt            # Dependencies
│── Procfile                    # Streamlit deployment config
│── README.md                   # Project info
│── .gitignore                  # Ignore unnecessary files
│── enso_env/                   # Virtual environment
│── .devcontainer/              # Development container settings
```

---

🛠 Technologies Used

- **Backend:** TensorFlow, Keras  
- **Frontend:** Streamlit  
- **Machine Learning:** LSTM, Scikit-Learn, Pandas, NumPy  
- **Visualization:** Matplotlib, Plotly  
- **Deployment:** Streamlit Cloud  

---

🔧 Installation & Setup

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

### 🔹 Run the Streamlit Dashboard
```bash
streamlit run dashboard.py
```

🔗 Open in your browser: **http://localhost:8501/**  

---

🚀 Live Website  
<https://bhanuvi17-prediction-of-enso-events-dashboard-w3nzfx.streamlit.app/>  

---

🚀 Deploying on Streamlit Cloud  

### 1️⃣ **Push Code to GitHub**  
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2️⃣ **Deploy on Streamlit:**  
- Go to <https://share.streamlit.io/>  
- Click **"New App"**  
- Connect to your GitHub repository  
- Set the **Main file path:** `dashboard.py`  
- Click **Deploy** 🎉  

---

🖼 Screenshots  

### 🔹 **Web Interface**  
![Home Page](https://github.com/bhanuvi17/Prediction-of-ENSO-Events/blob/ba223734e58fa39d2f4a48bef880722fa8acb655/Screenshot%202025-02-12%20232014.png)  

---

🏆 Future Enhancements  

✅ Train with larger datasets for improved accuracy  
✅ Implement additional time series forecasting models (e.g., ARIMA, Prophet)  
✅ Add real-time ONI data updates  

---

🐝 License  

This project is **open-source** under the [MIT License](LICENSE).  

---

### 💡 **Need Help?**  
Feel free to **open an issue** or **contribute** to improve this project! 😊  
🌟 If you like this project, give it a **star** on GitHub! 🌟  

