import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from src.data_preparation import DataPreparator
from src.feature_engineering import FeatureEngineer

# Load model and data
@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

@st.cache_data
def load_data():
    data_prep = DataPreparator('data/bundesliga_matches.csv')
    return data_prep.prepare_data()

def display_league_table(df: pd.DataFrame, current_date: pd.Timestamp):
    """Display the league table up to the given date."""
    feat_eng = FeatureEngineer(df)
    table = feat_eng.calculate_league_table(current_date)
    
    # Format table for display
    display_table = table[['Position', 'Points', 'Matches', 'Wins', 'Draws', 'Losses', 'GF', 'GA', 'GD', 'PPG']]
    display_table = display_table.round(2)
    
    st.subheader("Current League Table")
    st.dataframe(display_table)

def main():
    st.title("Bundesliga Match Predictor")
    
    # Load model and data
    model, feature_names = load_model()
    df = load_data()
    
    # Team selection
    teams = sorted(df['HomeTeam'].unique())
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox('Select Home Team', teams)
    with col2:
        away_team = st.selectbox('Select Away Team', teams)
    
    if home_team and away_team:
        # Display recent form
        st.subheader("Recent Form")
        
        # Get last 5 matches for each team
        home_matches = df[
            (df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)
        ].tail(5)
        
        away_matches = df[
            (df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)
        ].tail(5)
        
        # Create form plots
        fig1 = px.line(home_matches, x='Date', y='FTHG', title=f"{home_team} Recent Form")
        fig2 = px.line(away_matches, x='Date', y='FTAG', title=f"{away_team} Recent Form")
        
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        
        # Make prediction
        feat_eng = FeatureEngineer(df)
        X, _ = feat_eng.get_feature_matrix()
        latest_features = X.iloc[-1:]  # Get latest feature values
        
        probs = model.predict_proba(latest_features)[0]
        
        st.subheader("Match Prediction")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Home Win", f"{probs[2]:.1%}")
        with col2:
            st.metric("Draw", f"{probs[1]:.1%}")
        with col3:
            st.metric("Away Win", f"{probs[0]:.1%}")
        
        # Display current league table
        latest_date = df['Date'].max()
        display_league_table(df, latest_date)

if __name__ == "__main__":
    main() 