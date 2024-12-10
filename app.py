import streamlit as st
import pandas as pd
import plotly.express as px
from src.table_calculator import TableCalculator

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/bundesliga_matches_full.csv')

def display_match_history(df: pd.DataFrame, home_team: str, away_team: str, before_date: pd.Timestamp):
    """Display recent matches for both teams before a specific date."""
    # Get last 5 matches for each team before the match date
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

def main():
    st.title("Bundesliga Match Analysis")
    
    # Load data
    df = load_data()
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
        
        # Display match details
        st.subheader("Match Details")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.write(f"Home: {home_team}")
        with col2:
            if pd.to_datetime(match['Date']) <= pd.Timestamp.now():
                st.write(f"{match['FTHG']} - {match['FTAG']}")
            else:
                st.write("vs")
        with col3:
            st.write(f"Away: {away_team}")
        
        st.write(f"Date: {pd.to_datetime(match['Date']).strftime('%Y-%m-%d')}")
        
        # Display team form
        st.subheader("Recent Form")
        display_match_history(df, home_team, away_team, match['Date'])
        
        # Display league table before this match
        st.subheader("League Table Before Match")
        table = calculator.calculate_table(selected_season, pd.to_datetime(match['Date']))
        st.dataframe(table)
        
        # If the match has betting odds, show them
        if 'B365H' in match and 'B365D' in match and 'B365A' in match:
            st.subheader("Betting Odds")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Home Win", f"{match['B365H']:.2f}")
            with col2:
                st.metric("Draw", f"{match['B365D']:.2f}")
            with col3:
                st.metric("Away Win", f"{match['B365A']:.2f}")

if __name__ == "__main__":
    main() 