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
        """
        Initialize betting strategy.
        
        Args:
            bankroll: Initial bankroll
            kelly_fraction: Fraction of Kelly criterion to use (conservative approach)
        """
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.bets: List[BetResult] = []
        self.min_edge = 0.05  # Minimum 5% edge to place a bet
        self.min_prob = 0.2   # Minimum 20% probability to consider a bet
        
    def calculate_edge(self, our_prob: float, odds: float) -> float:
        """Calculate betting edge based on our probability vs. market odds."""
        market_prob = 1 / odds
        edge = our_prob - market_prob
        return edge
    
    def kelly_criterion(self, prob: float, odds: float) -> float:
        """Calculate Kelly criterion bet size."""
        q = 1 - prob
        b = odds - 1
        f = (prob * b - q) / b
        return max(0, f * self.kelly_fraction)  # Conservative Kelly
    
    def analyze_betting_opportunity(
        self,
        match_date: pd.Timestamp,
        home_team: str,
        away_team: str,
        predicted_probs: Dict[str, float],
        odds: Dict[str, float]
    ) -> List[Tuple[str, float, float]]:
        """
        Analyze betting opportunity and return list of recommended bets.
        Returns: List of (bet_type, bet_amount, edge)
        """
        opportunities = []
        
        # Check each possible outcome
        for outcome in ['H', 'D', 'A']:
            prob = predicted_probs[outcome]
            odd = odds[outcome]
            
            # Calculate edge
            edge = self.calculate_edge(prob, odd)
            
            # Only bet if we have a significant edge and sufficient probability
            if edge > self.min_edge and prob > self.min_prob:
                # Calculate bet size using Kelly criterion
                bet_size = self.kelly_criterion(prob, odd)
                bet_amount = self.bankroll * bet_size
                
                if bet_amount > 0:
                    opportunities.append((outcome, bet_amount, edge))
        
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
        """Place a bet and record the result."""
        # Calculate profit/loss
        if bet_type == actual_result:
            profit = bet_amount * (odds[bet_type] - 1)
        else:
            profit = -bet_amount
        
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