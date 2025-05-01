import streamlit as st
import pandas as pd
import joblib
import json

# Load Original Models
knn = joblib.load('Models/knn_model.pkl')
tree = joblib.load('Models/decision_tree_model.pkl')
kmeans = joblib.load('Models/kmeans_model.pkl')

# Load New Balanced Models
log_model = joblib.load('Models/log_model.pkl')
rf_model = joblib.load('Models/rf_model.pkl')
xgb_model = joblib.load('Models/xgb_model.pkl')

# Load Scaler and Risk Mapping
scaler = joblib.load('Models/scaler.pkl')
with open('Models/risk_mapping.json', 'r') as f:
    risk_mapping = json.load(f)

# App Title
st.title("NBA Player Injury Risk Predictor")

st.write("""
Input player statistics below and predict their injury risk category based on machine learning models!
""")

# Sidebar for Inputs
st.sidebar.header("Player Stats Input")

# Model page selector
page = st.sidebar.radio("Select Model Set", ["Original Models", "Balanced Models"])

# User Input
games_played = st.sidebar.number_input("Games Played This Season", min_value=0, max_value=200, value=20)
avg_minutes = st.sidebar.number_input("Average Minutes Per Game", min_value=0.0, max_value=48.0, value=28.0)
avg_points = st.sidebar.number_input("Average Points Per Game", min_value=0.0, max_value=50.0, value=15.0)
avg_assists = st.sidebar.number_input("Average Assists Per Game", min_value=0.0, max_value=30.0, value=4.0)
avg_rebounds = st.sidebar.number_input("Average Rebounds Per Game", min_value=0.0, max_value=30.0, value=5.0)
rolling5_minutes = st.sidebar.number_input("Rolling 5-Game Avg Minutes", min_value=0.0, max_value=48.0, value=30.0)
rolling5_points = st.sidebar.number_input("Rolling 5-Game Avg Points", min_value=0.0, max_value=50.0, value=18.0)
back_to_back_games = st.sidebar.number_input("Back-to-Back Games Played", min_value=0, max_value=50, value=1)
injury_count = st.sidebar.number_input("Past Injury Count", min_value=0, max_value=100, value=0)
height_inches = st.sidebar.number_input("Height (in inches)", min_value=50.0, max_value=100.0, value=78.0)
age = st.sidebar.number_input("Age", min_value=15, max_value=50, value=28)
weight = st.sidebar.number_input("Weight (in pounds)", min_value=100.0, max_value=400.0, value=220.0)

# Collect Input into DataFrame
new_player = pd.DataFrame({
    'games_played': [games_played],
    'avg_minutes': [avg_minutes],
    'avg_points': [avg_points],
    'avg_assists': [avg_assists],
    'avg_rebounds': [avg_rebounds],
    'rolling5_minutes': [rolling5_minutes],
    'rolling5_points': [rolling5_points],
    'back_to_back_games': [back_to_back_games],
    'injury_count': [injury_count],
    'height_inches': [height_inches],
    'age': [age],
    'weight': [weight]
})



# Original Models
if page == "Original Models":
    st.subheader("Original Model Predictions (KNN, Decision Tree, KMeans)")

    if st.button('Predict Injury Risk'):
        
        new_player_ordered = new_player[scaler.feature_names_in_]
        new_player_scaled = scaler.transform(new_player_ordered)


        knn_pred = knn.predict(new_player_scaled)[0]
        tree_pred = tree.predict(new_player_scaled)[0]
        kmeans_pred = kmeans.predict(new_player_scaled)[0]

        knn_risk = risk_mapping.get(str(knn_pred), 'Unknown')
        tree_risk = risk_mapping.get(str(tree_pred), 'Unknown')
        kmeans_risk = risk_mapping.get(str(kmeans_pred), 'Unknown')

        st.write(f"**KNN Model Prediction:** {knn_risk} (Cluster {knn_pred})")
        st.write(f"**Decision Tree Prediction:** {tree_risk} (Cluster {tree_pred})")
        st.write(f"**KMeans Group Prediction:** {kmeans_risk} (Cluster {kmeans_pred})")

        st.success(f"Overall Predicted Risk Level: **{tree_risk}**")

# Balanced Models
elif page == "Balanced Models":
    st.subheader("Balanced Model Predictions (Logistic Regression, Random Forest, XGBoost)")

    if st.button('Predict Injury Risk (Balanced Models)'):\
      
        new_player_ordered = new_player[scaler.feature_names_in_]
        new_player_scaled = scaler.transform(new_player_ordered)

        log_pred = log_model.predict(new_player_scaled)[0]
        rf_pred = rf_model.predict(new_player_scaled)[0]
        xgb_pred = xgb_model.predict(new_player_scaled)[0]

        log_proba = log_model.predict_proba(new_player_scaled)[0][1]
        rf_proba = rf_model.predict_proba(new_player_scaled)[0][1]
        xgb_proba = xgb_model.predict_proba(new_player_scaled)[0][1]
        Avg = (log_proba + rf_proba + xgb_proba) / 2

        st.write(f"**Logistic Regression Prediction:**(Probability: {log_proba:.2f})")
        st.write(f"**Random Forest Prediction:** (Probability: {rf_proba:.2f})")
        st.write(f"**XGBoost Prediction:** (Probability: {xgb_proba:.2f})")

        st.success(f"Average Predication from all models: **{Avg:.2f}**")
