import pandas as pd
import requests
import random
import streamlit as st

# Your API configuration
api_key = "3acf305c864c1140733b63d0e970d52f"
base_url = "https://v3.football.api-sports.io"
headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "v3.football.api-sports.io"}

# Function to fetch live match data
def fetch_live_matches():
    url = f"{base_url}/fixtures?live=all"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        matches = response.json().get("response", [])
        return matches
    return None

# Function to save live match data to CSV
def save_live_matches_to_csv():
    matches = fetch_live_matches()
    if matches:
        df = pd.DataFrame(matches)
        df.to_csv('live_matches.csv', mode='a', header=False, index=False)  # Append data to the CSV
        st.write("Live matches data saved!")
    else:
        st.write("No live matches found.")

# Step 2: Load Data from CSV and Display Historical Data
def load_live_match_data():
    try:
        data = pd.read_csv("live_matches.csv")
        return data
    except FileNotFoundError:
        st.write("No historical data found.")
        return pd.DataFrame()

# Display Historical Data
st.subheader("Historical Match Data")
historical_data = load_live_match_data()

# Show the dataframe of historical data
if not historical_data.empty:
    st.dataframe(historical_data)
else:
    st.write("No historical data available.")

# Load the saved live match data
def load_live_matches():
    try:
        live_matches = pd.read_csv("live_matches.csv")
        return live_matches
    except FileNotFoundError:
        st.write("No live matches data found.")
        return None

# Improved player performance prediction logic
def predict_player_performance(player_name):
    # Simulate some realistic stats for now (you can replace this with actual logic)
    shots_on_target = random.randint(0, 5)
    assists = random.randint(0, 2)
    pass_accuracy = random.randint(40, 100)

    # Simulate goal prediction based on shots and assists
    predicted_goal = "No Goal Predicted" if shots_on_target == 0 else f"Goal Predicted: {shots_on_target} goals"
    
    return {
        "shots_on_target": shots_on_target,
        "assists": assists,
        "pass_accuracy": pass_accuracy,
        "predicted_goal": predicted_goal
    }

# Dashboard
st.title("Football Prediction Dashboard")

# Option to fetch live matches
if st.button("Fetch Live Matches"):
    save_live_matches_to_csv()

# Load live matches from CSV
live_matches = load_live_matches()

if live_matches is not None:
    st.subheader("Live Matches")
    for index, match in live_matches.iterrows():
        st.write(f"**{match['teams']['home']['name']}** vs **{match['teams']['away']['name']}**")
        st.write(f"Date: {match['fixture']['date']}")
        st.write(f"Score: {match['goals']['home']} - {match['goals']['away']}")
        st.write("---")
else:
    st.write("No live matches available.")

# Player performance prediction
st.subheader("Player Performance Prediction")
player_name = st.text_input("Enter Player Name")

if player_name:
    player_stats = predict_player_performance(player_name)
    st.write(f"Shots on Target: {player_stats['shots_on_target']}")
    st.write(f"Assists: {player_stats['assists']}")
    st.write(f"Pass Accuracy: {player_stats['pass_accuracy']}%")
    st.write(f"Prediction: {player_stats['predicted_goal']}")
