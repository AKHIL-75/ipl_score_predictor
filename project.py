import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse

# Load and preprocess data
df = pd.read_csv(r'C:\\Users\\akhil\\OneDrive\\Desktop\\AKHIL\\ADM-PROJECT\\ipl_data.csv')
df.drop(labels=['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'], axis=1, inplace=True)
consistent_teams = [
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
df = df[df['overs'] >= 5.0]
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

# One-hot encoding
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
encoded_df = encoded_df[[
    'date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
    'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
    'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
    'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
    'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
    'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
    'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]
y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

# Model training
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

decision_regressor = DecisionTreeRegressor()
decision_regressor.fit(X_train, y_train)

random_regressor = RandomForestRegressor()
random_regressor.fit(X_train, y_train)

# Team mapping
team_map = {
    'CSK': 'Chennai Super Kings',
    'DD': 'Delhi Daredevils',
    'KXIP': 'Kings XI Punjab',
    'KKR': 'Kolkata Knight Riders',
    'MI': 'Mumbai Indians',
    'RR': 'Rajasthan Royals',
    'RCB': 'Royal Challengers Bangalore',
    'SRH': 'Sunrisers Hyderabad'
}

def predict_score(batting_team='CSK', bowling_team='MI', overs=5.1, runs=50, wickets=0, runs_in_prev_5=50, wickets_in_prev_5=0):
    temp_array = []
    teams = list(team_map.values())
    batting_team_encoding = [1 if team_map[batting_team] == team else 0 for team in teams]
    bowling_team_encoding = [1 if team_map[bowling_team] == team else 0 for team in teams]
    temp_array.extend(batting_team_encoding)
    temp_array.extend(bowling_team_encoding)
    temp_array.extend([overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5])
    return int(linear_regressor.predict([np.array(temp_array)])[0])

# Streamlit UI
st.title("IPL Score Predictor")

st.sidebar.header("Input Match Details")
batting_team = st.sidebar.selectbox("Select Batting Team", options=team_map.keys())
bowling_team = st.sidebar.selectbox("Select Bowling Team", options=team_map.keys())
overs = st.sidebar.slider("Overs Completed", 5.0, 20.0, step=0.1)
runs = st.sidebar.number_input("Current Runs", min_value=0)
wickets = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10)
runs_last_5 = st.sidebar.number_input("Runs Scored in Last 5 Overs", min_value=0)
wickets_last_5 = st.sidebar.number_input("Wickets Lost in Last 5 Overs", min_value=0, max_value=10)

if st.sidebar.button("Predict Score"):
    final_score = predict_score(batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5)
    st.subheader("🏏 Predicted Final Score")
    st.success(f"The predicted final score is most likely between {final_score - 10} and {final_score + 5}")

# Sample Predictions Section
st.markdown("## 🧾 Sample Predictions from IPL 2018")

final_score_1 = predict_score('KKR', 'DD', 9.2, 79, 2, 60, 1)
st.markdown(f"""
### 🎯 Prediction 1
- **Date:** 16th April 2018  
- **IPL:** Season 11  
- **Match number:** 13  
- **Teams:** Kolkata Knight Riders vs. Delhi Daredevils  
- **First Innings Final Score:** 200/9  
- **Predicted Score Range:** {final_score_1 - 10} to {final_score_1 + 5}
""")

final_score_2 = predict_score('SRH', 'RCB', 10.5, 67, 3, 29, 1)
st.markdown(f"""
### 🎯 Prediction 2
- **Date:** 7th May 2018  
- **IPL:** Season 11  
- **Match number:** 39  
- **Teams:** Sunrisers Hyderabad vs. Royal Challengers Bangalore  
- **First Innings Final Score:** 146/10  
- **Predicted Score Range:** {final_score_2 - 10} to {final_score_2 + 5}
""")

final_score_3 = predict_score('MI', 'KXIP', 14.1, 136, 4, 50, 0)
st.markdown(f"""
### 🎯 Prediction 3
- **Date:** 17th May 2018  
- **IPL:** Season 11  
- **Match number:** 50  
- **Teams:** Mumbai Indians vs. Kings XI Punjab  
- **First Innings Final Score:** 186/8  
- **Predicted Score Range:** {final_score_3 - 10} to {final_score_3 + 5}
""")
