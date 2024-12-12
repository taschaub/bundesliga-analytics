import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BetResult:
    match_date: pd.Timestamp
    home_team: str
    away_team: str
    predicted_probs: Dict[str, float]
    actual_result: str
    odds: Dict[str, float]
    bet_amount: float
    bet_type: str
    profit_loss: float
    value: float

class BettingStrategy:
    def __init__(self, bankroll: float = 1000, kelly_fraction: float = 0.02):
        """Initialize betting strategy with moderate parameters."""
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction  # Use 2% of Kelly
        self.bets: List[BetResult] = []
        self.min_edge = 0.05  # Minimum 5% edge to place a bet
        self.min_prob = 0.2   # Minimum 20% probability to consider a bet
        self.max_bet_size = 0.05  # Maximum 5% of bankroll per bet
        self.max_odds = 5.0  # Don't bet on odds higher than this
        self.min_odds = 1.2  # Don't bet on odds lower than this
        
        # Track consecutive losses
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5  # Stop betting after 5 consecutive losses
        
        # Daily betting limits
        self.daily_bet_count = {}
        self.max_daily_bets = 5
        
        # Track profit/loss streaks
        self.current_streak = 0
        self.stop_loss_threshold = -0.25  # Stop betting if we lose 25% of bankroll
    
    def calculate_edge(self, our_prob: float, odds: float) -> float:
        """Calculate betting edge with bookmaker margin consideration."""
        # Typical bookmaker margin is 5%
        margin = 0.05
        market_prob = (1 / odds) * (1 - margin)
        edge = our_prob - market_prob
        return edge
    
    def kelly_criterion(self, prob: float, odds: float) -> float:
        """Calculate Kelly criterion bet size with conservative adjustments."""
        q = 1 - prob
        b = odds - 1
        f = (prob * b - q) / b
        
        # Apply very conservative Kelly
        kelly_bet = max(0, f * self.kelly_fraction)
        
        # Additional safety: cap at max_bet_size
        return min(kelly_bet, self.max_bet_size)
    
    def should_stop_betting(self) -> bool:
        """Determine if we should stop betting based on various criteria."""
        # Stop if we've lost too much
        if self.bankroll < self.initial_bankroll * (1 + self.stop_loss_threshold):
            return True
        
        # Stop if we've had too many consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True
        
        return False
    
    def analyze_betting_opportunity(
        self,
        match_date: pd.Timestamp,
        home_team: str,
        away_team: str,
        predicted_probs: Dict[str, float],
        odds: Dict[str, float]
    ) -> List[Tuple[str, float, float]]:
        """Analyze betting opportunity with strict criteria."""
        if self.should_stop_betting():
            return []
        
        # Check daily bet limit
        date_str = match_date.date().isoformat()
        if self.daily_bet_count.get(date_str, 0) >= self.max_daily_bets:
            return []
        
        opportunities = []
        
        for outcome in ['H', 'D', 'A']:
            prob = predicted_probs[outcome]
            odd = odds[outcome]
            
            # Skip if odds are outside our acceptable range
            if odd < self.min_odds or odd > self.max_odds:
                continue
            
            # Calculate edge
            edge = self.calculate_edge(prob, odd)
            
            # Only bet if we have a significant edge and sufficient probability
            if edge > self.min_edge and prob > self.min_prob:
                # Calculate bet size using Kelly criterion
                bet_size = self.kelly_criterion(prob, odd)
                bet_amount = self.bankroll * bet_size
                
                # Minimum bet size of $10
                if bet_amount >= 10:
                    opportunities.append((outcome, bet_amount, edge))
        
        # Update daily bet count
        if opportunities:
            self.daily_bet_count[date_str] = self.daily_bet_count.get(date_str, 0) + 1
        
        return opportunities
    
    def place_bet(
        self,
        match_date: pd.Timestamp,
        home_team: str,
        away_team: str,
        predicted_probs: Dict[str, float],
        actual_result: str,
        odds: Dict[str, float],
        bet_type: str,
        bet_amount: float,
        edge: float
    ) -> None:
        """Place a bet and record the result with updated tracking."""
        # Calculate profit/loss
        if bet_type == actual_result:
            profit = bet_amount * (odds[bet_type] - 1)
            self.consecutive_losses = 0
            self.current_streak = max(0, self.current_streak + 1)
        else:
            profit = -bet_amount
            self.consecutive_losses += 1
            self.current_streak = min(0, self.current_streak - 1)
        
        # Update bankroll
        self.bankroll += profit
        
        # Record bet
        bet_result = BetResult(
            match_date=match_date,
            home_team=home_team,
            away_team=away_team,
            predicted_probs=predicted_probs,
            actual_result=actual_result,
            odds=odds,
            bet_amount=bet_amount,
            bet_type=bet_type,
            profit_loss=profit,
            value=edge
        )
        self.bets.append(bet_result)
    
    def get_betting_summary(self) -> pd.DataFrame:
        """Generate summary of betting performance."""
        if not self.bets:
            return pd.DataFrame()
        
        # Convert bets to DataFrame
        df = pd.DataFrame([vars(bet) for bet in self.bets])
        
        # Calculate key metrics
        total_bets = len(df)
        winning_bets = len(df[df['profit_loss'] > 0])
        total_profit = df['profit_loss'].sum()
        roi = (total_profit / df['bet_amount'].sum()) * 100
        
        # Calculate profit over time
        df['cumulative_profit'] = df['profit_loss'].cumsum()
        df['bankroll'] = self.initial_bankroll + df['cumulative_profit']
        
        # Calculate monthly profits
        df['month'] = df['match_date'].dt.to_period('M')
        monthly_profits = df.groupby('month')['profit_loss'].sum()
        
        # Calculate win rate by bet type
        win_rates = {}
        for bet_type in ['H', 'D', 'A']:
            type_bets = df[df['bet_type'] == bet_type]
            if len(type_bets) > 0:
                win_rate = len(type_bets[type_bets['profit_loss'] > 0]) / len(type_bets) * 100
                win_rates[bet_type] = win_rate
        
        summary = {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate': (winning_bets / total_bets * 100) if total_bets > 0 else 0,
            'total_profit': total_profit,
            'roi': roi,
            'final_bankroll': self.bankroll,
            'max_bankroll': df['bankroll'].max(),
            'min_bankroll': df['bankroll'].min(),
            'monthly_profits': monthly_profits,
            'win_rates_by_type': win_rates
        }
        
        return summary
    
    def plot_performance(self) -> None:
        """Plot betting performance over time."""
        if not self.bets:
            print("No bets to plot")
            return
        
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame([vars(bet) for bet in self.bets])
        df['cumulative_profit'] = df['profit_loss'].cumsum()
        df['bankroll'] = self.initial_bankroll + df['cumulative_profit']
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['match_date'], df['bankroll'], label='Bankroll')
        plt.axhline(y=self.initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
        
        plt.title('Betting Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Bankroll')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('betting_performance.png')
        plt.close() 