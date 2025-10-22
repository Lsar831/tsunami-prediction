
# 🌊 Tsunami Risk Prediction using Seismic Data

**Short description:** An end-to-end ML project that predicts whether an earthquake will trigger a tsunami (`tsunami = 1`) using seismic features (magnitude, depth, intensity, geometry, and time).  
Includes a notebook, trained model export and the dataset.

---

## 📁 Repository Structure
- `tsunami_risk_prediction.ipynb` — end-to-end training & EDA
- `streamlit_app_v2.py` — demo app to make on-the-fly predictions
- `earthquake_data_tsunami.csv` — dataset (13 columns, 782 rows)
- `tsunami_best_model.pkl` — saved model after running the notebook
- - `requirement.txt` — saved required libraries to run the notebook

---

## 🚀 Quickstart (Local)

1. **Create env & install deps**
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install numpy pandas scikit-learn matplotlib joblib streamlit
```

2. **Train the model (or use existing)**
- Open and run `tsunami_risk_prediction.ipynb` to generate `tsunami_best_model.pkl`  
  (If running locally, update the paths to your local folder.)

3. **Run the app**
```bash
streamlit run tsunami_app.py
```

---

## 🧪 What’s inside the Notebook
- Data loading & sanity checks
- EDA with matplotlib-only plots (target distribution, depth vs magnitude, yearly trends)
- Feature engineering: `energy`, `shallow_quake`, `yearly_avg_magnitude`
- Modeling with `LogisticRegression` and `RandomForest` (`class_weight="balanced"`)
- Best-model selection (ROC-AUC), confusion matrix & ROC curve
- 5-fold Stratified CV (ROC-AUC)
- Model export via `joblib` + inference helper

---

## 🖥️ Streamlit Demo
The app loads `tsunami_best_model.pkl` and provides a simple form to input seismic features.  
It mirrors the notebook’s feature engineering and outputs:
- **Binary prediction** (High/Low tsunami risk)
- **Class probability** (if available)
- **Engineered row** preview (for transparency)

> ⚠️ **Disclaimer:** This demo is for educational and dummy project purposes only and **not** for real-world emergency decisions.

---

## 📎 Dataset Columns (13)
The dataset from kaggle https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset
magnitude, cdi, mmi, sig, nst, dmin, gap, depth, latitude, longitude, Year, Month, tsunami
---
