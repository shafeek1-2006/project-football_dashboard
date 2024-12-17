import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# -------------------------- API Configuration -------------------------- #
API_KEY = "3acf305c864c1140733b63d0e970d52f"
BASE_URL = "https://v3.football.api-sports.io/"
headers = {"x-apisports-key": API_KEY}

# -------------------------- Helper Functions -------------------------- #
# Fetch live match data
@st.cache
def fetch_live_matches():
    response = requests.get(f"{BASE_URL}fixtures?live=all", headers=headers)
    if response.status_code == 200:
        return response.json().get("response", [])
    return []

# Fetch player statistics
@st.cache
def fetch_player_stats(team_id):
    response = requests.get(f"{BASE_URL}players?team={team_id}&season=2023", headers=headers)
    if response.status_code == 200:
        return response.json().get("response", [])
    return []

# Fetch team information
def fetch_team_logo(team_id):
    response = requests.get(f"{BASE_URL}teams?id={team_id}", headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data["response"]:
            return data["response"][0]["team"]["logo"]
        else:
            return None
    return None

# Train Random Forest model
def train_prediction_model(data, model_option):
    features = data[['shots_on_target', 'assists', 'pass_accuracy']]
    target = data['goals_scored']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    if model_option == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_option == "Linear Regression":
        model = LinearRegression()
    else:
        model = DecisionTreeRegressor(random_state=42)
        
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return model, mse

# -------------------------- Streamlit Dashboard -------------------------- #
st.title("Football Prediction Dashboard")

# Live Match Data
st.header("Live Match Data")
live_matches = fetch_live_matches()

if live_matches:
    for match in live_matches:
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        home_logo = fetch_team_logo(match['teams']['home']['id'])
        away_logo = fetch_team_logo(match['teams']['away']['id'])
        st.image([home_logo, away_logo], width=80, caption=[home_team, away_team])
        st.write(f"**{home_team}** vs **{away_team}** - Status: Live")
else:
    st.write("No live matches available.")

# Player Performance Data
st.header("Player Performance Prediction")
team_id = st.text_input("Enter a Team ID to Fetch Player Stats (Example: 33 for Manchester United):")
position_filter = st.selectbox("Select Position", ["All", "Forward", "Midfielder", "Defender", "Goalkeeper"])

if team_id:
    player_stats = fetch_player_stats(team_id)
    if player_stats:
        # Filter players by position
        filtered_player_data = []
        for player in player_stats:
            player_position = player['statistics'][0]['games']['position']
            if position_filter == "All" or player_position == position_filter:
                stats = player['statistics'][0]
                filtered_player_data.append({
                    'Player': player['player']['name'],
                    'Position': player_position,
                    'Team': stats['team']['name'],
                    'Shots on Target': stats['shots']['on'],
                    'Assists': stats['goals']['assists'],
                    'Pass Accuracy': stats['passes']['accuracy'],
                    'Goals Scored': stats['goals']['total']
                })
        
        player_df = pd.DataFrame(filtered_player_data).fillna(0)
        st.dataframe(player_df)

        # Train the model
        st.subheader("Train Prediction Model")
        model_option = st.selectbox("Select Prediction Model", ["Random Forest", "Linear Regression", "Decision Tree"])
        model, mse = train_prediction_model(player_df, model_option)
        st.write(f"**Model Mean Squared Error:** {mse:.2f}")

        # Predict goals for all players
        player_df['Predicted Goals'] = model.predict(player_df[['Shots on Target', 'Assists', 'Pass Accuracy']])
        st.dataframe(player_df[['Player', 'Team', 'Predicted Goals']])

        # Graphs: Shots on Target vs Goals
        st.subheader("Shots on Target vs Goals Scored")
        fig, ax = plt.subplots()
        ax.bar(player_df['Player'], player_df['Shots on Target'], color='blue', label='Shots on Target')
        ax.bar(player_df['Player'], player_df['Goals Scored'], color='green', label='Goals Scored')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

        # Interactive Visualization using Plotly
        st.subheader("Assists vs Goals Scored (Interactive)")
        fig = px.bar(player_df, x='Player', y=['Assists', 'Goals Scored'], barmode='group', title="Assists vs Goals Scored")
        st.plotly_chart(fig)
    else:
        st.write("No player statistics available for this team.")

# Footer
st.write("Data provided by API-FOOTBALL.")
