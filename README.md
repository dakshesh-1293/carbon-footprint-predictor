# Personal Carbon Footprint Predictor

A machine learning and Streamlit-based project that predicts a user's daily carbon footprint from lifestyle and consumption behavior.

## Project Overview

This project uses a behavioral carbon-footprint dataset and a trained machine learning model to estimate daily `carbon_footprint_kg`. The app also includes basic EDA visuals and model insights.

### Main features
- Predicts daily carbon footprint in **kg CO₂/day**
- Streamlit web app with 4 sections:
  - Home
  - EDA Dashboard
  - Prediction
  - Model Insights
- Loads saved preprocessing and model artifacts using `joblib`
- Applies the same feature engineering used during training
- Shows transport usage, target distribution, and correlation heatmap

## Dataset

The main dataset used by the project is:
- `personal_carbon_footprint_behavior.csv`

### Main columns
- `user_id`
- `day_type`
- `transport_mode`
- `distance_km`
- `electricity_kwh`
- `renewable_usage_pct`
- `food_type`
- `screen_time_hours`
- `waste_generated_kg`
- `eco_actions`
- `carbon_footprint_kg`
- `carbon_impact_level`

## Feature Engineering Used

The app creates these engineered features before prediction:
- `transport_intensity`
- `long_distance`
- `zero_emission_transport`
- `energy_level`
- `renewable_champion`
- `dirty_energy_kwh`
- `waste_per_screen_hour`

## Machine Learning Models Mentioned in Training
The notebook/report shows comparison of these regression models:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- LightGBM Regressor

Saved deployment artifacts used by the app:
- `best_model.pkl`
- `best_model_name.pkl`
- `scaler.pkl`
- `label_encoders.pkl`
- `feature_columns.pkl`

## Project Structure

```text
.
├── app.py
├── Mini_Project(2).ipynb
├── personal_carbon_footprint_behavior.csv
├── carbon_dataset_1.csv
├── best_model.pkl
├── best_model_name.pkl
├── scaler.pkl
├── label_encoders.pkl
├── feature_columns.pkl
├── Carbon_Footprint_Report.docx
├── Carbon_Footprint_Predictor.pptx
└── README.md
```

## Libraries Required

Your uploaded files show two levels of dependencies:

### 1) Required to run the Streamlit app
- `streamlit`
- `pandas`
- `numpy`
- `joblib`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### 2) Required to run the full training notebook
- `streamlit`
- `pandas`
- `numpy`
- `joblib`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `xgboost`

## Installation

### Step 1: Clone the repository
```bash
git clone <your-github-repo-link>
cd <your-repo-folder>
```

### Step 2: Create a virtual environment (recommended)
#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install libraries

#### For app only
```bash
pip install streamlit pandas numpy joblib matplotlib seaborn scikit-learn
```

#### For full project including notebook training
```bash
pip install streamlit pandas numpy joblib matplotlib seaborn scikit-learn lightgbm xgboost
```

## How to Run the Project

### Run the Streamlit app
```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:
```text
http://localhost:8501
```

## Important Note Before Running

In the uploaded `app.py`, the dataset path is hardcoded as:

```python
pd.read_csv(r"D:\Mini-Project\personal_carbon_footprint_behavior.csv")
```

If someone else downloads this project, that path will not exist on their system.

### Recommended fix
Change it to:

```python
pd.read_csv("personal_carbon_footprint_behavior.csv")
```

This lets the app read the dataset directly from the project folder.

## How the App Works

1. Loads dataset using `pandas`
2. Loads trained artifacts using `joblib`
3. Takes user inputs such as:
   - transport mode
   - distance travelled
   - electricity usage
   - renewable usage
   - screen time
   - waste generated
   - eco actions
4. Applies feature engineering
5. Encodes categorical values using saved label encoders
6. Aligns columns with training-time feature order
7. Applies scaler if the selected model is Linear Regression
8. Predicts daily carbon footprint
9. Displays impact level as Low / Medium / High

## Streamlit Sections

### Home
- Project overview
- Dataset size
- Unique users
- Dataset preview

### EDA Dashboard
- Carbon footprint distribution
- Transport mode usage
- Correlation heatmap

### Prediction
- User input form
- Carbon footprint prediction
- Impact level output
- Processed input preview

### Model Insights
- Model comparison table
- Feature importance style chart

## Suggested `requirements.txt`

Create a `requirements.txt` file with:

```txt
streamlit
pandas
numpy
joblib
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
```

## Example Git Commands

```bash
git init
git add .
git commit -m "Initial commit - Personal Carbon Footprint Predictor"
git branch -M main
git remote add origin <your-github-repo-link>
git push -u origin main
```

## Author

**Kaushik Ajani**  
M.Sc. Data Science, Marwadi University

## Acknowledgement

Project team:
- Kaushik Ajani
- Diya Kukadia
- Dakshesh Parmar

Guide:
- Nikunj Bosamiya
