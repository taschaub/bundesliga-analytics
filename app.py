import streamlit as st
import pandas as pd
import plotly.express as px
from src.table_calculator import TableCalculator
import joblib
from src.feature_engineering import FeatureEngineer

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('data/bundesliga_matches_full.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    return model

def display_match_history(df: pd.DataFrame, home_team: str, away_team: str, before_date: pd.Timestamp):
    """Display recent matches for both teams before a specific date."""
    # Get last 5 matches for each team before the match date
    before_date = pd.to_datetime(before_date)  # Ensure before_date is datetime
    
    home_matches = df[
        ((df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)) &
        (df['Date'] < before_date)
    ].sort_values('Date', ascending=False).head(5)
    
    away_matches = df[
        ((df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)) &
        (df['Date'] < before_date)
    ].sort_values('Date', ascending=False).head(5)
    
    # Display home team's last matches
    st.write(f"\n{home_team}'s last 5 matches:")
    for _, match in home_matches.iterrows():
        if match['HomeTeam'] == home_team:
            result = f"{match['FTHG']} - {match['FTAG']} vs {match['AwayTeam']}"
            if match['FTHG'] > match['FTAG']:
                result += " (W)"
            elif match['FTHG'] < match['FTAG']:
                result += " (L)"
            else:
                result += " (D)"
        else:
            result = f"{match['FTAG']} - {match['FTHG']} vs {match['HomeTeam']}"
            if match['FTAG'] > match['FTHG']:
                result += " (W)"
            elif match['FTAG'] < match['FTHG']:
                result += " (L)"
            else:
                result += " (D)"
        st.write(f"{pd.to_datetime(match['Date']).strftime('%Y-%m-%d')}: {result}")
    
    # Display away team's last matches
    st.write(f"\n{away_team}'s last 5 matches:")
    for _, match in away_matches.iterrows():
        if match['HomeTeam'] == away_team:
            result = f"{match['FTHG']} - {match['FTAG']} vs {match['AwayTeam']}"
            if match['FTHG'] > match['FTAG']:
                result += " (W)"
            elif match['FTHG'] < match['FTAG']:
                result += " (L)"
            else:
                result += " (D)"
        else:
            result = f"{match['FTAG']} - {match['FTHG']} vs {match['HomeTeam']}"
            if match['FTAG'] > match['FTHG']:
                result += " (W)"
            elif match['FTAG'] < match['FTHG']:
                result += " (L)"
            else:
                result += " (D)"
        st.write(f"{pd.to_datetime(match['Date']).strftime('%Y-%m-%d')}: {result}")

def display_head_to_head(df: pd.DataFrame, home_team: str, away_team: str, before_date: pd.Timestamp):
    """Display previous encounters between the teams."""
    h2h_matches = df[
        (
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
            ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
        ) &
        (df['Date'] < before_date)
    ].sort_values('Date', ascending=False).head(5)
    
    if not h2h_matches.empty:
        st.write("\nPrevious encounters:")
        for _, match in h2h_matches.iterrows():
            date_str = pd.to_datetime(match['Date']).strftime('%Y-%m-%d')
            result = f"{match['HomeTeam']} {match['FTHG']} - {match['FTAG']} {match['AwayTeam']}"
            st.write(f"{date_str}: {result}")
    else:
        st.write("\nNo previous encounters found.")

def main():
    st.title("Bundesliga Match Analysis")
    
    # Load data and model
    df = load_data()
    model = load_model()
    calculator = TableCalculator(df)
    
    # Season selection
    seasons = sorted(df['Season'].unique(), reverse=True)
    selected_season = st.selectbox('Select Season', seasons)
    
    # Matchday selection
    season_data = df[df['Season'] == selected_season]
    max_matchday = season_data['Matchday'].max()
    selected_matchday = st.slider('Select Matchday', 1, max_matchday, max_matchday)
    
    # Get matches for selected matchday
    matchday_data = season_data[season_data['Matchday'] == selected_matchday]
    
    # Match selection
    matches = [f"{row['HomeTeam']} vs {row['AwayTeam']}" for _, row in matchday_data.iterrows()]
    selected_match = st.selectbox('Select Match', matches)
    
    if selected_match:
        home_team, away_team = selected_match.split(' vs ')
        match = matchday_data[
            (matchday_data['HomeTeam'] == home_team) & 
            (matchday_data['AwayTeam'] == away_team)
        ].iloc[0]
        match_date = pd.to_datetime(match['Date'])
        
        # Compare with current date
        current_date = pd.Timestamp.now()
        is_past_match = match_date <= current_date
        
        # Display match details
        st.subheader("Match Details")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.write(f"Home: {home_team}")
        with col2:
            if match_date <= pd.Timestamp.now():
                st.write(f"{match['FTHG']} - {match['FTAG']}")
                if match['FTHG'] > match['FTAG']:
                    actual_result = "Home Win"
                elif match['FTHG'] < match['FTAG']:
                    actual_result = "Away Win"
                else:
                    actual_result = "Draw"
                st.write(f"Result: {actual_result}")
            else:
                st.write("vs")
        with col3:
            st.write(f"Away: {away_team}")
        
        st.write(f"Date: {match_date.strftime('%Y-%m-%d')}")
        
        # Make prediction
        feat_eng = FeatureEngineer(df)
        match_features = feat_eng.create_match_features(home_team, away_team, match_date)
        probs = model.predict_proba(match_features)[0]
        
        st.subheader("Match Prediction")
        if match_date <= pd.Timestamp.now():
            col1, col2 = st.columns(2)
            with col1:
                st.write("Predicted Result:")
                st.write(f"Home Win: {probs[2]:.1%}")
                st.write(f"Draw: {probs[1]:.1%}")
                st.write(f"Away Win: {probs[0]:.1%}")
            with col2:
                st.write("Actual Result:")
                st.write(actual_result)
        else:
            st.write(f"Home Win: {probs[2]:.1%}")
            st.write(f"Draw: {probs[1]:.1%}")
            st.write(f"Away Win: {probs[0]:.1%}")
        
        # Display team form
        st.subheader("Recent Form")
        display_match_history(df, home_team, away_team, match_date)
        
        # Display head-to-head history
        st.subheader("Head-to-Head History")
        display_head_to_head(df, home_team, away_team, match_date)
        
        # Display league table before this match
        st.subheader("League Table Before Match")
        table = calculator.calculate_table(selected_season, match_date)
        st.dataframe(table)

if __name__ == "__main__":
    main() 