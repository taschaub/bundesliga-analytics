import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.table_calculator import TableCalculator

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/bundesliga_matches_full.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_betting_history():
    try:
        # Load the most recent betting history file
        import glob
        history_files = glob.glob('match_history_season_*.csv')
        if not history_files:
            return None
        latest_file = max(history_files)
        return pd.read_csv(latest_file)
    except Exception:
        return None

def display_betting_analysis(betting_history: pd.DataFrame):
    """Display betting analysis and visualizations."""
    st.markdown("### 📊 Betting Performance Analysis")
    
    if betting_history is None or len(betting_history) == 0:
        st.warning("⚠️ No betting history available.")
        return
    
    # Convert date column
    betting_history['Date'] = pd.to_datetime(betting_history['Date'])
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Bankroll Evolution", "📊 Bet Analysis", "📝 Match Details"])
    
    with tab1:
        # Bankroll evolution chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=betting_history['Date'],
            y=betting_history['BankrollAfter'],
            mode='lines+markers',
            name='Bankroll',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=6)
        ))
        fig.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Initial Bankroll")
        fig.update_layout(
            title="💰 Bankroll Evolution Over Time",
            xaxis_title="Date",
            yaxis_title="Bankroll ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Bet type distribution pie chart
            bet_matches = betting_history[betting_history['BetPlaced']]
            bet_type_counts = bet_matches['BetType'].value_counts()
            fig_pie = px.pie(
                values=bet_type_counts.values,
                names=bet_type_counts.index,
                title="🎯 Bet Type Distribution",
                labels={'index': 'Bet Type', 'value': 'Count'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Win/Loss distribution
            win_loss = bet_matches['BetResult'].value_counts()
            fig_win_loss = px.pie(
                values=win_loss.values,
                names=['Won', 'Lost'],
                title="✅❌ Betting Outcomes",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig_win_loss, use_container_width=True)
        
        # Bet amount distribution
        fig_hist = px.histogram(
            bet_matches,
            x='BetAmount',
            title="💰 Bet Amount Distribution",
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        # Detailed match history table
        st.markdown("### 📝 Match History")
        bet_matches = betting_history[betting_history['BetPlaced']].copy()
        bet_matches['ROI'] = (bet_matches['Profit'] / bet_matches['BetAmount'] * 100).round(2)
        
        # Format the table
        display_df = bet_matches[[
            'Date', 'HomeTeam', 'AwayTeam', 'BetType', 'BetAmount',
            'Edge', 'BetResult', 'Profit', 'ROI'
        ]].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Edge'] = (display_df['Edge'] * 100).round(2).astype(str) + '%'
        display_df['ROI'] = display_df['ROI'].astype(str) + '%'
        display_df['BetAmount'] = display_df['BetAmount'].round(2)
        display_df['Profit'] = display_df['Profit'].round(2)
        
        st.dataframe(
            display_df.style.apply(lambda x: ['background-color: #2ecc71' if v else 'background-color: #e74c3c' 
                                            for v in x == 'True'], subset=['BetResult'])
        )

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
    st.title("⚽ Bundesliga Match Analysis")
    
    # Load data
    df = load_data()
    betting_history = load_betting_history()
    calculator = TableCalculator(df)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["🎯 Match Analysis", "💰 Betting Analysis"])
    
    with tab1:
        st.markdown("### 📅 Select Match")
        
        col1, col2 = st.columns(2)
        with col1:
            # Season selection
            seasons = sorted(df['Season'].unique(), reverse=True)
            selected_season = st.selectbox('Season', seasons, key='season_match')
        
        with col2:
            # Matchday selection
            season_data = df[df['Season'] == selected_season]
            max_matchday = season_data['Matchday'].max()
            selected_matchday = st.slider('Matchday', 1, max_matchday, max_matchday, key='matchday_match')
        
        # Get matches for selected matchday
        matchday_data = season_data[season_data['Matchday'] == selected_matchday]
        
        # Match selection
        matches = [f"{row['HomeTeam']} vs {row['AwayTeam']}" for _, row in matchday_data.iterrows()]
        selected_match = st.selectbox('Match', matches, key='match_select')
        
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
            st.markdown("### ⚔️ Match Details")
            
            # Create a styled match header
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(28, 131, 225, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 10px 0;">
                    <h2>{home_team} 🆚 {away_team}</h2>
                    <p style="font-size: 1.2em;">📅 {match_date.strftime('%Y-%m-%d')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.markdown(f"**🏠 Home:** {home_team}")
            with col2:
                if match_date <= pd.Timestamp.now():
                    st.markdown(f"**⚽ {match['FTHG']} - {match['FTAG']} ⚽**")
                    if match['FTHG'] > match['FTAG']:
                        actual_result = "🏆 Home Win"
                    elif match['FTHG'] < match['FTAG']:
                        actual_result = "🏆 Away Win"
                    else:
                        actual_result = "🤝 Draw"
                    st.markdown(f"**{actual_result}**")
                else:
                    st.markdown("**VS**")
            with col3:
                st.markdown(f"**🏃 Away:** {away_team}")
            
            # If this match exists in betting history, show the betting details
            if betting_history is not None:
                # Convert match date to date only for comparison
                match_date_only = pd.to_datetime(match_date).date()
                betting_history['Date'] = pd.to_datetime(betting_history['Date'])
                
                match_bet = betting_history[
                    (betting_history['Date'].dt.date == match_date_only) &
                    (betting_history['HomeTeam'] == home_team) &
                    (betting_history['AwayTeam'] == away_team)
                ]
                
                if not match_bet.empty and match_bet['BetPlaced'].iloc[0]:
                    st.markdown("### 💰 Betting Information")
                    bet = match_bet.iloc[0]
                    
                    # Display odds and predictions comparison
                    st.markdown("#### 📊 Odds & Predictions Comparison")
                    odds_col1, odds_col2, odds_col3 = st.columns(3)
                    
                    # Calculate implied probabilities (1/odds)
                    home_implied_prob = (1 / bet['HomeOdds']) * 100
                    draw_implied_prob = (1 / bet['DrawOdds']) * 100
                    away_implied_prob = (1 / bet['AwayOdds']) * 100
                    
                    # Calculate total to normalize probabilities
                    total_implied = home_implied_prob + draw_implied_prob + away_implied_prob
                    
                    # Normalize implied probabilities
                    home_implied_prob = (home_implied_prob / total_implied) * 100
                    draw_implied_prob = (draw_implied_prob / total_implied) * 100
                    away_implied_prob = (away_implied_prob / total_implied) * 100
                    
                    with odds_col1:
                        st.markdown("**🏠 Home Win**")
                        st.markdown(f"Odds: {bet['HomeOdds']:.2f}")
                        st.markdown(f"Implied: {home_implied_prob:.1f}%")
                        model_prob = bet['HomePred'] * 100
                        diff = model_prob - home_implied_prob
                        color = "green" if diff > 0 else "red"
                        st.markdown(f"Model: <span style='color: {color}'>{model_prob:.1f}% ({diff:+.1f}%)</span>", unsafe_allow_html=True)
                    
                    with odds_col2:
                        st.markdown("**🤝 Draw**")
                        st.markdown(f"Odds: {bet['DrawOdds']:.2f}")
                        st.markdown(f"Implied: {draw_implied_prob:.1f}%")
                        model_prob = bet['DrawPred'] * 100
                        diff = model_prob - draw_implied_prob
                        color = "green" if diff > 0 else "red"
                        st.markdown(f"Model: <span style='color: {color}'>{model_prob:.1f}% ({diff:+.1f}%)</span>", unsafe_allow_html=True)
                    
                    with odds_col3:
                        st.markdown("**🏃 Away Win**")
                        st.markdown(f"Odds: {bet['AwayOdds']:.2f}")
                        st.markdown(f"Implied: {away_implied_prob:.1f}%")
                        model_prob = bet['AwayPred'] * 100
                        diff = model_prob - away_implied_prob
                        color = "green" if diff > 0 else "red"
                        st.markdown(f"Model: <span style='color: {color}'>{model_prob:.1f}% ({diff:+.1f}%)</span>", unsafe_allow_html=True)
                    
                    # Display bet details
                    st.markdown("#### 🎲 Bet Details")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**📍 Type:** {bet['BetType']}")
                        st.markdown(f"**💵 Amount:** ${bet['BetAmount']:.2f}")
                    with col2:
                        st.markdown(f"**📊 Edge:** {bet['Edge']*100:.1f}%")
                        result_emoji = "✅" if bet['BetResult'] else "❌"
                        st.markdown(f"**{result_emoji} Result:** {'Won' if bet['BetResult'] else 'Lost'}")
                    with col3:
                        # Display odds based on bet type
                        if bet['BetType'] == 'H':
                            odds = bet['HomeOdds']
                            st.markdown(f"**📈 Home Odds:** {odds:.2f}")
                        elif bet['BetType'] == 'D':
                            odds = bet['DrawOdds']
                            st.markdown(f"**📈 Draw Odds:** {odds:.2f}")
                        elif bet['BetType'] == 'A':
                            odds = bet['AwayOdds']
                            st.markdown(f"**📈 Away Odds:** {odds:.2f}")
                        st.markdown(f"**📊 ROI:** {(bet['Profit']/bet['BetAmount']*100):+.1f}%")
                    
                    # Display bankroll information last
                    st.markdown("#### 🏦 Bankroll")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**💵 Before:** ${bet['BankrollBefore']:.2f}")
                    with col2:
                        profit_color = "green" if bet['Profit'] > 0 else "red"
                        change_emoji = "📈" if bet['Profit'] > 0 else "📉"
                        st.markdown(f"**{change_emoji} Change:** <span style='color: {profit_color}'>${bet['Profit']:.2f}</span>", unsafe_allow_html=True)
                        st.markdown(f"<span style='color: {profit_color}'>({(bet['Profit']/bet['BankrollBefore']*100):+.1f}%)</span>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"**💰 After:** ${bet['BankrollAfter']:.2f}")
            
            # Display team form and head-to-head history side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Recent Form")
                # Home team form
                st.markdown(f"**🏠 {home_team}'s last 5 matches:**")
                home_matches = df[
                    ((df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)) &
                    (df['Date'] < match_date)
                ].sort_values('Date', ascending=False).head(5)
                
                for _, match in home_matches.iterrows():
                    date_str = pd.to_datetime(match['Date']).strftime('%Y-%m-%d')
                    if match['HomeTeam'] == home_team:
                        result = f"{match['FTHG']} - {match['FTAG']} vs {match['AwayTeam']}"
                        if match['FTHG'] > match['FTAG']:
                            result += " (✅ W)"
                        elif match['FTHG'] < match['FTAG']:
                            result += " (❌ L)"
                        else:
                            result += " (🤝 D)"
                    else:
                        result = f"{match['FTAG']} - {match['FTHG']} vs {match['HomeTeam']}"
                        if match['FTAG'] > match['FTHG']:
                            result += " (✅ W)"
                        elif match['FTAG'] < match['FTHG']:
                            result += " (❌ L)"
                        else:
                            result += " (🤝 D)"
                    st.markdown(f"📅 {date_str}: {result}")
                
                # Away team form
                st.markdown(f"\n**🏃 {away_team}'s last 5 matches:**")
                away_matches = df[
                    ((df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)) &
                    (df['Date'] < match_date)
                ].sort_values('Date', ascending=False).head(5)
                
                for _, match in away_matches.iterrows():
                    date_str = pd.to_datetime(match['Date']).strftime('%Y-%m-%d')
                    if match['HomeTeam'] == away_team:
                        result = f"{match['FTHG']} - {match['FTAG']} vs {match['AwayTeam']}"
                        if match['FTHG'] > match['FTAG']:
                            result += " (✅ W)"
                        elif match['FTHG'] < match['FTAG']:
                            result += " (❌ L)"
                        else:
                            result += " (🤝 D)"
                    else:
                        result = f"{match['FTAG']} - {match['FTHG']} vs {match['HomeTeam']}"
                        if match['FTAG'] > match['FTHG']:
                            result += " (✅ W)"
                        elif match['FTAG'] < match['FTHG']:
                            result += " (❌ L)"
                        else:
                            result += " (🤝 D)"
                    st.markdown(f"📅 {date_str}: {result}")
            
            with col2:
                st.markdown("### ⚔️ Head-to-Head History")
                h2h_matches = df[
                    (
                        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
                    ) &
                    (df['Date'] < match_date)
                ].sort_values('Date', ascending=False).head(5)
                
                if not h2h_matches.empty:
                    st.markdown("**Previous encounters:**")
                    for _, match in h2h_matches.iterrows():
                        date_str = pd.to_datetime(match['Date']).strftime('%Y-%m-%d')
                        # Highlight the teams in the result
                        if match['HomeTeam'] == home_team:
                            result = f"**{match['HomeTeam']}** {match['FTHG']} - {match['FTAG']} {match['AwayTeam']}"
                            if match['FTHG'] > match['FTAG']:
                                result += " (🏠 Home Win)"
                            elif match['FTHG'] < match['FTAG']:
                                result += " (🏃 Away Win)"
                            else:
                                result += " (🤝 Draw)"
                        else:
                            result = f"{match['HomeTeam']} {match['FTHG']} - {match['FTAG']} **{match['AwayTeam']}**"
                            if match['FTHG'] > match['FTAG']:
                                result += " (🏠 Home Win)"
                            elif match['FTHG'] < match['FTAG']:
                                result += " (🏃 Away Win)"
                            else:
                                result += " (🤝 Draw)"
                        st.markdown(f"📅 {date_str}: {result}")
                else:
                    st.markdown("🤷 No previous encounters found.")
            
            # Display league table before this match
            st.markdown("### 🏆 League Table Before Match")
            table = calculator.calculate_table(selected_season, match_date)
            st.dataframe(table)
    
    with tab2:
        display_betting_analysis(betting_history)

if __name__ == "__main__":
    main() 