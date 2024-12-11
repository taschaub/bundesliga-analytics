class BettingStrategy:
    def __init__(self, confidence_threshold=0.6, min_odds=1.5):
        self.confidence_threshold = confidence_threshold
        self.min_odds = min_odds
        
    def evaluate_bet(self, probs, odds, actual_result):
        """Evaluate if we should bet and calculate returns."""
        max_prob = max(probs)
        pred_result = np.argmax(probs)
        
        # Only bet if confidence is high enough and odds are attractive
        if max_prob >= self.confidence_threshold and odds[pred_result] >= self.min_odds:
            if pred_result == actual_result:
                return (odds[pred_result] - 1) * 1  # Win
            return -1  # Loss
        return 0  # No bet
    
    def simulate_season(self, predictions, odds, actual_results):
        """Simulate betting strategy over a season."""
        total_bets = 0
        wins = 0
        losses = 0
        profit = 0
        
        for pred, odd, result in zip(predictions, odds, actual_results):
            bet_result = self.evaluate_bet(pred, odd, result)
            if bet_result != 0:
                total_bets += 1
                if bet_result > 0:
                    wins += 1
                    profit += bet_result
                else:
                    losses += 1
                    profit += bet_result
        
        return {
            'total_bets': total_bets,
            'win_rate': wins / max(1, total_bets),
            'profit': profit,
            'roi': profit / max(1, total_bets)
        } 