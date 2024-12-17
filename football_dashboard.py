import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample football data (you can replace this with actual data from an API or database)
# Simulated player and team data
player_data = {
    'player_name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
    'team': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B'],
    'shots_on_target': [3, 1, 2, 4, 1],
    'assists': [1, 0, 1, 2, 0],
    'pass_accuracy': [85, 75, 90, 80, 88],
    'goals': [1, 0, 2, 1, 0],
    'team_logo_url': [
        'https://upload.wikimedia.org/wikipedia/commons/4/4d/FC_Barcelona_%28crest%29.svg',  # Team A logo
        'https://upload.wikimedia.org/wikipedia/commons/2/2f/Arsenal_FC.svg',  # Team B logo
        'https://upload.wikimedia.org/wikipedia/commons/0/04/Real_Madrid_CF.svg',  # Team C logo
        'https://upload.wikimedia.org/wikipedia/commons/4/4d/FC_Barcelona_%28crest%29.svg',  # Team A logo
        'https://upload.wikimedia.org/wikipedia/commons/2/2f/Arsenal_FC.svg',  # Team B logo
    ]
}

# Create DataFrame for player data
df = pd.DataFrame(player_data)

# Train a Random Forest model to predict goals based on player performance data
X = df[['shots_on_target', 'assists', 'pass_accuracy']]  # Features
y = df['goals']  # Target variable (goals)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Mean Squared Error: {mse:.2f}")

# Title and description for the dashboard
st.title('Football Prediction Dashboard')
st.markdown("## Live Match Data")

# Placeholder for live matches (simulated for now)
live_matches = pd.DataFrame({
    'match_id': [1, 2, 3],
    'home_team': ['Team A', 'Team B', 'Team C'],
    'away_team': ['Team D', 'Team E', 'Team F'],
    'status': ['Live', 'Live', 'Live'],
})

# Display live matches
if not live_matches.empty:
    for _, match in live_matches.iterrows():
        st.write(f"**{match['home_team']}** vs **{match['away_team']}** - Status: {match['status']}")
else:
    st.write("No live matches data available.")

# Player Performance Prediction Section
st.markdown("## Player Performance Prediction")

# Input for player performance (shots, assists, pass accuracy)
shots_on_target = st.number_input("Shots on Target", min_value=0, max_value=10, value=0)
assists = st.number_input("Assists", min_value=0, max_value=5, value=0)
pass_accuracy = st.number_input("Pass Accuracy (%)", min_value=0, max_value=100, value=50)

# Prediction based on input data
input_features = np.array([[shots_on_target, assists, pass_accuracy]])
predicted_goals = model.predict(input_features)

# Display predicted results
st.write(f"Predicted Goals: {predicted_goals[0]:.2f}")

# Display player details, team logos, and performance
st.markdown("### Player Details and Performance")

for i, row in df.iterrows():
    # Display player name, team name, and team logo
    st.write(f"**{row['player_name']}** - Team: {row['team']}")
    st.image(row['team_logo_url'], width=50)  # Displaying team logo (URL-based image)
    st.write(f"**Shots on Target**: {row['shots_on_target']}, **Assists**: {row['assists']}, **Pass Accuracy**: {row['pass_accuracy']}%")
    st.write(f"**Goals Scored**: {row['goals']}")
    st.write("----")

# You can also add additional analysis, graphs, and charts here
