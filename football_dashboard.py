import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# Example API (replace with actual API URL)
API_KEY = 'your_api_key'
BASE_URL = 'https://api.football-data.org/v2'

# Function to fetch live matches data from the API
def fetch_live_matches():
    url = f"{BASE_URL}/matches"
    headers = {'X-Auth-Token': API_KEY}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        matches = response.json().get("matches", [])
        return matches
    return None

# Function to display live match data
def display_live_matches():
    matches = fetch_live_matches()
    
    if matches:
        for match in matches:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            score = match['score']['fullTime']
            st.write(f"**{home_team}** vs **{away_team}**")
            st.write(f"Score: {score['homeTeam']} - {score['awayTeam']}")
            
            # Display player stats if available
            # Assuming the API returns player stats (check your API documentation)
            players = match.get('players', [])
            for player in players:
                st.write(f"Player: {player['name']}, Goals: {player['goals']}, Assists: {player['assists']}")
    else:
        st.write("No live matches data available.")

# Sample player data (replace with actual dataset)
player_data = pd.DataFrame({
    'shots': [5, 3, 4, 2, 6],
    'assists': [1, 2, 1, 0, 1],
    'pass_accuracy': [85, 90, 88, 80, 92],
    'goals': [1, 0, 2, 0, 1]  # This is the target variable
})

# Features (X) and target (y)
X = player_data[['shots', 'assists', 'pass_accuracy']]
y = player_data['goals']

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Function to predict if a player will score based on stats
def predict_goal(shots, assists, pass_accuracy):
    prediction = model.predict([[shots, assists, pass_accuracy]])
    return "Goal Predicted" if prediction[0] == 1 else "No Goal Predicted"

# Function for player performance prediction
def player_performance_prediction():
    st.header("Player Performance Prediction")
    
    # Inputs for player stats
    shots = st.number_input("Shots on Target", min_value=0, max_value=10)
    assists = st.number_input("Assists", min_value=0, max_value=5)
    pass_accuracy = st.number_input("Pass Accuracy (%)", min_value=50, max_value=100)
    
    # Predict and display result
    if st.button("Predict Player Goal"):
        prediction = predict_goal(shots, assists, pass_accuracy)
        st.write(f"Prediction: {prediction}")

# Full dashboard integration
def main():
    st.title("Football Prediction Dashboard")
    
    # Display live match data
    st.header("Live Match Data")
    display_live_matches()  # Display live match data
    
    # Predict individual player performance
    player_performance_prediction()  # Predict player outcomes

if __name__ == "__main__":
    main()
