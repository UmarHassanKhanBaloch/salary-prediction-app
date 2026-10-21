# 💼 Salary Prediction App

An end-to-end Machine Learning project: a regression model that predicts **salary from years of experience**, served through an interactive **Streamlit** web app.

**Live app:** [https://salary-prediction-app-plrdjg6zm7ajy3epeojhft.streamlit.app/]

## Project description
The model is trained in a Jupyter notebook (data exploration, cleaning, training, evaluation), saved with `joblib`, and loaded by the Streamlit app (`app.py`) to make predictions. The app does not retrain the model on every run.

**App features:** clear title and description, input for years of experience, predicted salary display, input validation (warning when the value is outside the training range), regression-line chart, prediction history, dataset information tab and model performance tab.

## Dataset
Salary dataset provided in class (`data/Salary_Data.csv`): 40 rows, 2 columns.

| Column | Description |
|---|---|
| Experience Years | Years of experience (feature) |
| Salary | Salary (target) |

No missing values or duplicates. Correlation between experience and salary is about 0.98.

## Model
**Linear Regression** (scikit-learn), 80:20 train/test split (`random_state=42`).

| Metric | Value |
|---|---|
| R² (test set) | 0.907 |
| R² (5-fold cross validation, mean) | 0.947 |
| MAE | ≈ 6,420 |
| RMSE | ≈ 6,934 |
| MSE | ≈ 48,077,731 |

Equation: `Salary = 26,716 + 9,408 × Years of experience`

> Only 40 rows and one feature are used, so the test score (8 rows) can vary; the cross validation score is more reliable. The model should not be trusted outside 1.1–10.5 years of experience.

## Project structure
```
ML_Project/
├── data/Salary_Data.csv
├── model/
│   ├── model.pkl          # trained model (joblib)
│   └── metrics.json       # evaluation metrics used by the app
├── notebooks/model_training.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally
```bash
git clone <your-repo-url>
cd ML_Project

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
streamlit run app.py
```
To retrain the model, run all cells of `notebooks/model_training.ipynb` (it overwrites `model/model.pkl` and `model/metrics.json`).

## Tech stack
Python 3.12, pandas, NumPy, scikit-learn, joblib, matplotlib, seaborn (notebook), Streamlit.
