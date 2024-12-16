import pandas as pd
import requests
import time
import streamlit as st
import random

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
        print("Live matches data saved!")
    else:
        print("No live matches found.")

# Automate the process (run it every 10 minutes for example)
while True:
    save_live_matches_to_csv()
    time.sleep(600)  # Wait 10 minutes before fetching again

# Step 2: Load Data from CSV and Display Historical Data

# Load the saved live match data from CSV
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

# Prediction logic
def predict_match(home_team, away_team):
    home_score = random.randint(0, 5)
    away_score = random.randint(0, 5)
    return f"Predicted Score: {home_team} {home_score} - {away_team} {away_score}"

# Dashboard
st.title("Football Prediction Dashboard")

# Load live matches from CSV
live_matches = load_live_matches()

if live_matches is not None:
    st.subheader("Live Matches")
    for index, match in live_matches.iterrows():
        st.write(f"**{match['home_team']}** vs **{match['away_team']}**")
        st.write(f"Date: {match['date']}")
        st.write(f"Score: {match['home_score']} - {match['away_score']}")
        st.write("---")
else:
    st.write("No live matches available.")

# Predict match outcome
st.subheader("Predict Match Outcome")
home_team = st.selectbox("Select Home Team", live_matches["home_team"].unique())
away_team = st.selectbox("Select Away Team", live_matches["away_team"].unique())

if st.button("Predict Outcome"):
    prediction = predict_match(home_team, away_team)
    st.write(prediction)


