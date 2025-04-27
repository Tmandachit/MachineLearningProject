# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# --- Load models ---
knn = joblib.load('Models/knn_model.pkl')
tree = joblib.load('Models/decision_tree_model.pkl')
kmeans = joblib.load('Models/kmeans_model.pkl')
scaler = joblib.load('Models/scaler.pkl')

# --- Load Risk Mapping (Optional if you saved it) ---
risk_mapping = {
    0: 'High Risk',         # from your real cluster analysis
    1: 'Very High Risk',
    2: 'Low Risk',
    3: 'Medium Risk'
}

# --- App Title ---
st.title("🏀 NBA Player Injury Risk Predictor")

st.write("""
Input player weekly statistics below and predict their injury risk category based on dynamic player workload models!
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Player Stats Input")

games_played = st.sidebar.number_input("Games Played This Season", min_value=0, max_value=200, value=20)
avg_minutes = st.sidebar.number_input("Average Minutes Per Game", min_value=0.0, max_value=48.0, value=28.0)
avg_points = st.sidebar.number_input("Average Points Per Game", min_value=0.0, max_value=50.0, value=15.0)
avg_assists = st.sidebar.number_input("Average Assists Per Game", min_value=0.0, max_value=30.0, value=4.0)
avg_rebounds = st.sidebar.number_input("Average Rebounds Per Game", min_value=0.0, max_value=30.0, value=5.0)
rolling5_minutes = st.sidebar.number_input("Rolling 5-Game Avg Minutes", min_value=0.0, max_value=48.0, value=30.0)
rolling5_points = st.sidebar.number_input("Rolling 5-Game Avg Points", min_value=0.0, max_value=50.0, value=18.0)
back_to_back_games = st.sidebar.number_input("Back-to-Back Games Played", min_value=0, max_value=50, value=1)
injury_count = st.sidebar.number_input("Past Injury Count", min_value=0, max_value=100, value=0)

# --- Collect Input into DataFrame ---
new_player = pd.DataFrame({
    'games_played': [games_played],
    'avg_minutes': [avg_minutes],
    'avg_points': [avg_points],
    'avg_assists': [avg_assists],
    'avg_rebounds': [avg_rebounds],
    'rolling5_minutes': [rolling5_minutes],
    'rolling5_points': [rolling5_points],
    'back_to_back_games': [back_to_back_games],
    'injury_count': [injury_count]
})

# --- Prediction Button ---
if st.button('Predict Injury Risk'):
    # Scale input
    new_player_scaled = scaler.transform(new_player)

    # Predict
    knn_pred = knn.predict(new_player_scaled)[0]
    tree_pred = tree.predict(new_player_scaled)[0]
    kmeans_pred = kmeans.predict(new_player_scaled)[0]

    # Map predictions to risk levels
    knn_risk = risk_mapping.get(knn_pred, 'Unknown')
    tree_risk = risk_mapping.get(tree_pred, 'Unknown')
    kmeans_risk = risk_mapping.get(kmeans_pred, 'Unknown')

    # --- Output Results ---
    st.subheader("🧠 Injury Risk Predictions")

    st.write(f"**KNN Model Prediction:** {knn_risk} (Cluster {knn_pred})")
    st.write(f"**Decision Tree Prediction:** {tree_risk} (Cluster {tree_pred})")
    st.write(f"**KMeans Group Prediction:** {kmeans_risk} (Cluster {kmeans_pred})")

    st.success(f"🔮 Overall Predicted Risk Level: **{tree_risk}**")
