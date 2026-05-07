import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Carbon Footprint Predictor", layout="wide")

st.title(" Personal Carbon Footprint Predictor")
st.markdown("Predict daily carbon footprint using your trained ML model.")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\MINI_PROJECT\personal_carbon_footprint_behavior.csv")

df = load_data()

# -----------------------------
# Load saved model files
# -----------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("best_model.pkl")
    best_model_name = joblib.load("best_model_name.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, best_model_name, scaler, label_encoders, feature_columns

model, best_model_name, scaler, label_encoders, feature_columns = load_model_files()

# -----------------------------
# Load real model insights
# -----------------------------
@st.cache_data
def load_model_insights():
    comparison_df = pd.read_csv("model_comparison.csv")
    coef_df = pd.read_csv("linear_model_coefficients.csv")
    return comparison_df, coef_df

comparison_df, coef_df = load_model_insights()

# -----------------------------
# Helper: dataset defaults
# -----------------------------
def get_default_values(df):
    defaults = {}

    for col in df.columns:
        if col in ["carbon_footprint_kg", "carbon_impact_level", "user_id"]:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            mode_vals = df[col].mode(dropna=True)
            defaults[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else ""

    return defaults

defaults = get_default_values(df)

# -----------------------------
# Feature engineering function
# -----------------------------
def create_features(df_input):
    df_feat = df_input.copy()

    # transport intensity
    if "distance_km" in df_feat.columns and "transport_mode" in df_feat.columns:
        transport_map = {
            "Car": 2.0,
            "Bus": 0.8,
            "EV": 0.3,
            "Bike": 0.0,
            "Walk": 0.0
        }
        df_feat["transport_intensity"] = (
            df_feat["distance_km"] * df_feat["transport_mode"].map(transport_map).fillna(0)
        )

        df_feat["long_distance"] = (df_feat["distance_km"] > 15).astype(int)
        df_feat["zero_emission_transport"] = df_feat["transport_mode"].isin(["Walk", "Bike"]).astype(int)

    # energy level
    if "electricity_kwh" in df_feat.columns:
        df_feat["energy_level"] = pd.cut(
            df_feat["electricity_kwh"],
            bins=[0, 4, 7, 1000],
            labels=["Low", "Medium", "High"],
            include_lowest=True
        )

    # renewable champion
    if "renewable_usage_pct" in df_feat.columns:
        df_feat["renewable_champion"] = (df_feat["renewable_usage_pct"] >= 75).astype(int)

    # dirty energy
    if "electricity_kwh" in df_feat.columns and "renewable_usage_pct" in df_feat.columns:
        df_feat["dirty_energy_kwh"] = (
            df_feat["electricity_kwh"] * (1 - df_feat["renewable_usage_pct"] / 100)
        )

    # waste per screen hour
    if "waste_generated_kg" in df_feat.columns and "screen_time_hours" in df_feat.columns:
        df_feat["waste_per_screen_hour"] = (
            df_feat["waste_generated_kg"] / (df_feat["screen_time_hours"] + 1)
        )

    # food score
    if "food_type" in df_feat.columns:
        food_map = {
            "Vegan": 0.5,
            "Vegetarian": 0.8,
            "Pescatarian": 1.0,
            "Omnivore": 1.5,
            "Non-Vegetarian": 1.5
        }
        df_feat["diet_emission_score"] = df_feat["food_type"].map(food_map).fillna(1.0)

    return df_feat

# -----------------------------
# Prepare input for model
# -----------------------------
def prepare_input(user_input_dict):
    input_df = pd.DataFrame([user_input_dict])

    # apply same feature engineering
    input_df = create_features(input_df)

    # encode categorical columns using saved encoders
    categorical_cols = input_df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = input_df[col].astype(str)
            known_classes = list(le.classes_)

            input_df[f"{col}_encoded"] = input_df[col].apply(
                lambda x: le.transform([x])[0] if x in known_classes else -1
            )

    # drop original categorical columns
    input_df.drop(columns=categorical_cols, inplace=True, errors="ignore")

    # keep only numeric columns
    input_df = input_df.select_dtypes(include=[np.number]).copy()

    # add missing columns expected by model
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # exact column order
    input_df = input_df[feature_columns]

    # final fill
    input_df = input_df.fillna(0)

    return input_df

# -----------------------------
# Predict helper
# -----------------------------
def predict_carbon(user_input_dict):
    model_input = prepare_input(user_input_dict)

    if best_model_name == "Linear Regression":
        input_data = scaler.transform(model_input)
    else:
        input_data = model_input

    prediction = model.predict(input_data)[0]
    prediction = max(prediction, 0)

    return prediction, model_input

# -----------------------------
# Sidebar navigation
# -----------------------------
page = st.sidebar.radio(
    "Choose Section",
    ["Home", "EDA Dashboard", "Full Prediction", "Category-wise Prediction", "Model Insights"]
)

# -----------------------------
# Home
# -----------------------------
if page == "Home":
    st.subheader("Project Overview")
    st.write("""
    This app predicts a user's daily carbon footprint using the trained machine learning model.
    It supports full prediction and category-wise prediction.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Connected Model", best_model_name)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# -----------------------------
# EDA Dashboard
# -----------------------------
elif page == "EDA Dashboard":
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Carbon Footprint Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["carbon_footprint_kg"], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Carbon Footprint (kg CO₂/day)")
        st.pyplot(fig)

    with col2:
        if "transport_mode" in df.columns:
            st.markdown("### Transport Mode Usage")
            fig, ax = plt.subplots(figsize=(7, 4))
            df["transport_mode"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("Transport Mode")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    st.markdown("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="YlGnBu", annot=False, ax=ax)
    st.pyplot(fig)

# -----------------------------
# Full Prediction
# -----------------------------
elif page == "Full Prediction":
    st.subheader("Full Carbon Footprint Prediction")
    st.write(f" Connected model: **{best_model_name}**")

    input_data = {}

    usable_cols = [
        col for col in df.columns
        if col not in ["carbon_footprint_kg", "carbon_impact_level", "user_id"]
    ]

    for col in usable_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(defaults[col])

            if min_val == max_val:
                max_val = min_val + 1

            input_data[col] = st.number_input(
                f"{col}",
                min_value=min_val,
               # max_value=max_val,
                value=default_val
            )
        else:
            options = sorted(df[col].dropna().astype(str).unique().tolist())
            default_option = str(defaults[col]) if str(defaults[col]) in options else options[0]
            input_data[col] = st.selectbox(f"{col}", options, index=options.index(default_option))

    if st.button("Predict Full Carbon Footprint"):
        prediction, model_input = predict_carbon(input_data)

        st.success(f"Predicted Carbon Footprint: {prediction:.2f} kg CO₂/day")

        if prediction < 8:
            st.info("Impact Level: Low")
        elif prediction < 15:
            st.warning("Impact Level: Medium")
        else:
            st.error("Impact Level: High")

        st.write("### Model Input Used")
        st.dataframe(model_input)

# -----------------------------
# Category-wise Prediction
# -----------------------------
elif page == "Category-wise Prediction":
    st.subheader("Category-wise Carbon Prediction")
    st.write("Select one category and enter only related values. Other inputs are filled automatically using dataset defaults.")

    category_choice = st.selectbox(
        "Choose Category",
        ["Food", "Transport", "Energy", "Lifestyle", "Waste"]
    )

    user_input = {}

    for col in df.columns:
        if col not in ["carbon_footprint_kg", "carbon_impact_level", "user_id"]:
            user_input[col] = defaults[col]

    if category_choice == "Food":
        if "food_type" in df.columns:
            food_options = sorted(df["food_type"].dropna().astype(str).unique().tolist())
            user_input["food_type"] = st.selectbox("Select Food Type", food_options)
        else:
            st.error("food_type column not found in dataset.")

    elif category_choice == "Transport":
        if "transport_mode" in df.columns:
            transport_options = sorted(df["transport_mode"].dropna().astype(str).unique().tolist())
            user_input["transport_mode"] = st.selectbox("Select Transport Mode", transport_options)

        if "distance_km" in df.columns:
            user_input["distance_km"] = st.number_input(
                "Distance Travelled (km)",
                min_value=float(df["distance_km"].min()),
             #   max_value=float(df["distance_km"].max()),
                value=float(defaults["distance_km"])
            )

    elif category_choice == "Energy":
        if "electricity_kwh" in df.columns:
            user_input["electricity_kwh"] = st.number_input(
                "Electricity Usage (kWh)",
                min_value=float(df["electricity_kwh"].min()),
            #    max_value=float(df["electricity_kwh"].max()),
                value=float(defaults["electricity_kwh"])
            )

        if "renewable_usage_pct" in df.columns:
            user_input["renewable_usage_pct"] = st.number_input(
                "Renewable Usage (%)",
                min_value=float(df["renewable_usage_pct"].min()),
             #   max_value=float(df["renewable_usage_pct"].max()),
                value=float(defaults["renewable_usage_pct"])
            )

    elif category_choice == "Lifestyle":
        if "screen_time_hours" in df.columns:
            user_input["screen_time_hours"] = st.number_input(
                "Screen Time Hours",
                min_value=float(df["screen_time_hours"].min()),
             #   max_value=float(df["screen_time_hours"].max()),
                value=float(defaults["screen_time_hours"])
            )

        if "eco_actions" in df.columns:
            user_input["eco_actions"] = st.number_input(
                "Eco Actions",
                min_value=float(df["eco_actions"].min()),
              #  max_value=float(df["eco_actions"].max()),
                value=float(defaults["eco_actions"])
            )

    elif category_choice == "Waste":
        if "waste_generated_kg" in df.columns:
            user_input["waste_generated_kg"] = st.number_input(
                "Waste Generated (kg)",
                min_value=float(df["waste_generated_kg"].min()),
              #  max_value=float(df["waste_generated_kg"].max()),
                value=float(defaults["waste_generated_kg"])
            )

    if st.button("Predict Category-wise Carbon Footprint"):
        prediction, model_input = predict_carbon(user_input)

        st.success(f"Predicted Carbon Footprint: {prediction:.2f} kg CO₂/day")

        if prediction < 8:
            st.info("Impact Level: Low")
        elif prediction < 15:
            st.warning("Impact Level: Medium")
        else:
            st.error("Impact Level: High")

        st.write("### Category Selected")
        st.write(category_choice)

        st.write("### Model Input Used")
        st.dataframe(model_input)

# -----------------------------
# Model Insights
# -----------------------------
elif page == "Model Insights":
    st.subheader("Model Insights")

    st.write(f"Connected model: **{best_model_name}**")

    st.markdown("### Real Model Performance")
    st.dataframe(comparison_df)

    st.markdown("### Linear Regression Coefficients")
    st.dataframe(coef_df)

    fig, ax = plt.subplots(figsize=(9, 6))
    top_coef = coef_df.head(15)
    sns.barplot(data=top_coef, x="Coefficient", y="Feature", ax=ax)
    ax.set_title("Top Linear Regression Coefficients")
    st.pyplot(fig)

    st.markdown("### Coefficient Meaning")
    st.write("""
    - Positive coefficient → increases predicted carbon footprint
    - Negative coefficient → decreases predicted carbon footprint
    - Larger absolute value → stronger effect on prediction
    """)
