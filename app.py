import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
DATA_PATH = BASE_DIR / "data" / "Salary_Data.csv"
CURRENCY = "$"          # change this if the salaries are in another currency


# ---------------------------------------------------------------
# Load the saved model, metrics and data (cached, so it only happens once)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)       # the model is NOT trained again here


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
metrics = load_metrics()
df = load_data()
min_exp, max_exp = metrics["min_experience"], metrics["max_experience"]

# prediction history is kept for the current session
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------
# Title and description
# ---------------------------------------------------------------
st.title("💼 Salary Prediction App")
st.write(
    "Enter the years of experience of an employee and the app predicts the expected salary "
    "using a **Linear Regression** model trained on the Salary dataset "
    f"({len(df)} employees)."
)

tab_predict, tab_data, tab_model = st.tabs(["🔮 Predict", "📊 Dataset", "📈 Model performance"])

# ---------------------------------------------------------------
# Tab 1: prediction
# ---------------------------------------------------------------
with tab_predict:
    st.subheader("Predict salary")
    years = st.number_input(
        "Years of experience",
        min_value=0.0, max_value=50.0, value=5.0, step=0.5,
        help=f"The model was trained on {min_exp} to {max_exp} years of experience.",
    )

    if st.button("Predict salary", type="primary"):
        input_df = pd.DataFrame({"Experience Years": [years]})   # same column name as in training
        predicted = float(model.predict(input_df)[0])
        st.session_state.history.append({"Experience (years)": years, "Predicted salary": round(predicted, 2)})

        st.metric("Predicted salary", f"{CURRENCY}{predicted:,.0f}")
        st.caption(
            f"Typical error of the model is about ±{CURRENCY}{metrics['mae']:,.0f} (mean absolute error on the test set)."
        )

        # input validation: warn when the value is outside the training range
        if years < min_exp or years > max_exp:
            st.warning(
                f"{years} years is outside the range the model was trained on ({min_exp}–{max_exp} years). "
                "The prediction is an extrapolation and may not be reliable."
            )

        # plot the regression line and the user's point
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.scatter(df["Experience Years"], df["Salary"], alpha=0.7, label="Employees in dataset")
        line = pd.DataFrame({"Experience Years": [min(min_exp, years), max(max_exp, years)]})
        ax.plot(line["Experience Years"], model.predict(line), color="red", label="Regression line")
        ax.scatter([years], [predicted], color="green", s=120, zorder=5, label="Your prediction")
        ax.set_title("Experience vs. Salary")
        ax.set_xlabel("Experience (years)")
        ax.set_ylabel(f"Salary ({CURRENCY})")
        ax.legend()
        st.pyplot(fig)

    if st.session_state.history:
        st.subheader("Prediction history")
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch", hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------
# Tab 2: dataset information
# ---------------------------------------------------------------
with tab_data:
    st.subheader("Dataset information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Correlation", f"{df['Experience Years'].corr(df['Salary']):.2f}")
    st.write("**Summary statistics**")
    st.dataframe(df.describe().round(2), width="stretch")
    st.write("**Data**")
    st.dataframe(df, width="stretch", hide_index=True)

# ---------------------------------------------------------------
# Tab 3: model performance
# ---------------------------------------------------------------
with tab_model:
    st.subheader("Model performance")
    st.write(f"**Model:** {metrics['model']}  |  **Train rows:** {metrics['train_rows']}  |  **Test rows:** {metrics['test_rows']}")
    m1, m2 = st.columns(2)
    m1.metric("R² (test set)", f"{metrics['r2']:.3f}")
    m2.metric("R² (5-fold cross validation)", f"{metrics['cv_r2_mean']:.3f}")
    m3, m4 = st.columns(2)
    m3.metric("MAE", f"{CURRENCY}{metrics['mae']:,.0f}")
    m4.metric("RMSE", f"{CURRENCY}{metrics['rmse']:,.0f}")
    st.write(
        f"**Equation:** Salary = {metrics['intercept']:,.0f} + {metrics['coefficient']:,.0f} × Years of experience"
    )
    st.info(
        f"On average, each additional year of experience adds about {CURRENCY}{metrics['coefficient']:,.0f} to the predicted salary. "
        "The model uses only experience, so it does not consider job role, education or location."
    )
