import random
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class RPSMain:
    """
    Adaptive Rock-Paper-Scissors bot that combines multiple prediction
    strategies and learns which ones work best against a given opponent.
    """

    # Strategy weight profiles by game phase (moves played)
    PHASE_WEIGHTS = {
        'early':  {'markov': 0.12, 'frequency': 0.22, 'psychology': 0.38, 'pattern': 0.13, 'reaction': 0.15},
        'mid':    {'markov': 0.25, 'frequency': 0.25, 'psychology': 0.25, 'pattern': 0.10, 'reaction': 0.15},
        'late':   {'markov': 0.25, 'frequency': 0.15, 'psychology': 0.15, 'pattern': 0.25, 'reaction': 0.20},
    }

    PHASE_THRESHOLDS = {
        'early': 2,
        'mid': 5,
    }

    # Algorithm parameters
    EXPLORATION_RATE = 0.10          # Probability of random move to avoid exploitation
    ADAPTATION_SPEED = 0.30          # How fast transition matrix updates
    BASE_CONFIDENCE = 0.10           # Starting confidence floor
    CONFIDENCE_PER_MOVE = 0.15       # Confidence increase per move played
    MAX_CONFIDENCE = 0.90            # Confidence ceiling
    MIN_STRATEGY_ATTEMPTS = 3        # Minimum tries before adjusting strategy weights
    RECENT_WINDOW = 10               # Moves considered for frequency analysis
    MAX_PATTERN_MEMORY = 50          # Total distinct patterns to keep
    TOP_PATTERNS_TO_KEEP = 10        # Patterns to retain when pruning
    WIN_RESULT = 1
    TIE_RESULT = 0
    LOSS_RESULT = -1

    def __init__(self) -> None:
        self.moves = ['rock', 'paper', 'scissors']
        self.move_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.beats = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
        self.loses_to = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}

        # Game history
        self.opponent_history: List[str] = []
        self.my_history: List[str] = []
        self.results_history: List[int] = []

        # Markov model
        self.transition_matrix = np.ones((3, 3)) / 3

        # Pattern memory: maps tuple of previous moves -> list of following moves
        self.last_n_transitions: Dict[Tuple[str, ...], List[str]] = defaultdict(list)

        # Frequency tracking
        self.frequency_table = Counter()
        self.opponent_repeats = 0
        self.opponent_losses = 0
        self.exploration_rate = self.EXPLORATION_RATE
        self.adapt_speed = self.ADAPTATION_SPEED
        self.confidence = 0.0

        # Strategy performance tracking
        self.strategy_successes = {
            'markov': 0,
            'frequency': 0,
            'psychology': 0,
            'pattern': 0,
            'reaction': 0,
        }
        self.strategy_attempts = {
            'markov': 0,
            'frequency': 0,
            'psychology': 0,
            'pattern': 0,
            'reaction': 0,
        }
        self.last_strategy_used: Optional[str] = None

    def get_move(self) -> str:
        """
        Select the bot's next move.

        With probability ``exploration_rate`` returns a random move to
        avoid being exploited. Otherwise predicts the opponent's next
        move and counters it.
        """
        moves_seen = len(self.opponent_history)
        self.confidence = min(
            self.MAX_CONFIDENCE,
            self.BASE_CONFIDENCE + self.CONFIDENCE_PER_MOVE * moves_seen
        )

        if not self.opponent_history:
            return random.choice(self.moves)

        if random.random() < self.exploration_rate:
            return random.choice(self.moves)

        predicted_move = self._make_prediction()
        counter_move = self.loses_to[predicted_move]
        return counter_move

    def _make_prediction(self) -> str:
        """
        Combine all strategies into a weighted vote and return the
        predicted opponent move.
        """
        predictions = {
            'markov': self._markov_prediction(),
            'frequency': self._frequency_prediction(),
            'psychology': self._psychology_prediction(),
            'pattern': self._pattern_prediction(),
            'reaction': self._reaction_prediction(),
        }

        moves_played = len(self.opponent_history)
        if moves_played <= self.PHASE_THRESHOLDS['early']:
            weights = dict(self.PHASE_WEIGHTS['early'])
        elif moves_played <= self.PHASE_THRESHOLDS['mid']:
            weights = dict(self.PHASE_WEIGHTS['mid'])
        else:
            weights = dict(self.PHASE_WEIGHTS['late'])

        # Adjust weights based on historical strategy performance
        for strategy in weights:
            attempts = self.strategy_attempts[strategy]
            if attempts > self.MIN_STRATEGY_ATTEMPTS:
                success_rate = self.strategy_successes[strategy] / attempts
                weights[strategy] *= (1 + success_rate)

        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Weighted vote
        vote: Dict[str, float] = defaultdict(float)
        for strat, pred in predictions.items():
            vote[pred] += weights[strat]

        best_prediction = max(vote, key=vote.get)

        # Determine which strategy contributed most to the winning prediction
        # so we can credit/blame it after seeing the result.
        top_strategy: Optional[str] = None
        top_strategy_weight = -1.0
        for strat, pred in predictions.items():
            if pred == best_prediction and weights[strat] > top_strategy_weight:
                top_strategy_weight = weights[strat]
                top_strategy = strat
        self.last_strategy_used = top_strategy

        return best_prediction

    def _markov_prediction(self) -> str:
        """Predict using a first-order Markov transition matrix."""
        if not self.opponent_history:
            return random.choice(self.moves)

        last_move = self.opponent_history[-1]
        last_idx = self.move_idx[last_move]
        probs = self.transition_matrix[last_idx]
        predicted_idx = int(np.argmax(probs))
        return self.moves[predicted_idx]

    def _frequency_prediction(self) -> str:
        """
        Predict based on the opponent's most common recent move,
        weighting more recent moves higher via exponential decay.
        """
        if not self.opponent_history:
            return random.choice(self.moves)

        window = self.opponent_history[-self.RECENT_WINDOW:]
        weighted = Counter()
        decay = 0.85
        for i, move in enumerate(window):
            weight = decay ** (len(window) - 1 - i)
            weighted[move] += weight

        prediction = weighted.most_common(1)[0][0]
        return prediction

    def _psychology_prediction(self) -> str:
        """
        Predict based on human psychological tendencies.

        Heuristics:
        - After 2+ losses, players often repeat their last move (tilt).
        - After 2+ repeats, players often switch to the move that beats
          their repeated move.
        - After the bot loses, the player may repeat what beat the bot.
        - Win-stay lose-shift: players tend to repeat moves that won and
          switch away from moves that lost.
        """
        if not self.opponent_history:
            return random.choice(self.moves)

        last_move = self.opponent_history[-1]

        # Tilt detection: player lost last round(s)
        if self.results_history and self.results_history[-1] == self.WIN_RESULT:
            self.opponent_losses += 1
            if self.opponent_losses >= 2:
                return last_move
        else:
            self.opponent_losses = 0

        # Repeat detection: player played same move 2+ times in a row
        if len(self.opponent_history) >= 2 and last_move == self.opponent_history[-2]:
            self.opponent_repeats += 1
            if self.opponent_repeats >= 2:
                return self.loses_to[last_move]
        else:
            self.opponent_repeats = 0

        # Bot lost last round: player may stick with what beat us
        if self.results_history and self.results_history[-1] == self.LOSS_RESULT:
            return self.loses_to[self.my_history[-1]]

        # Win-stay lose-shift: if player won, they likely repeat; if they
        # lost, they shift to what would have beaten their last move.
        if len(self.results_history) >= 2:
            # Check if the player's last move was a win or loss for them
            # results_history stores from bot's perspective, so invert
            player_result = -self.results_history[-1]
            if player_result == self.WIN_RESULT:
                # Player won: likely repeats
                return last_move
            elif player_result == self.LOSS_RESULT:
                # Player lost: likely shifts to what beats their last move
                return self.loses_to[last_move]

        # Fallback: random
        return random.choice(self.moves)

    def _pattern_prediction(self) -> str:
        """
        Predict based on observed patterns in the opponent's move history.

        Checks the last 2-move and 3-move sequences and picks the most
        common follow-up move.
        """
        if len(self.opponent_history) < 2:
            return random.choice(self.moves)

        pattern_2 = tuple(self.opponent_history[-2:])
        follows_2 = Counter(self.last_n_transitions.get(pattern_2, []))
        if follows_2:
            return follows_2.most_common(1)[0][0]

        if len(self.opponent_history) >= 3:
            pattern_3 = tuple(self.opponent_history[-3:])
            follows_3 = Counter(self.last_n_transitions.get(pattern_3, []))
            if follows_3:
                return follows_3.most_common(1)[0][0]

        return random.choice(self.moves)

    def _reaction_prediction(self) -> str:
        """
        Predict based on the opponent reacting to *our* previous moves.

        Many players unconsciously try to beat whatever the bot just played.
        This strategy detects that tendency and exploits it.
        """
        if len(self.my_history) < 2 or len(self.opponent_history) < 2:
            return random.choice(self.moves)

        # Count how often the opponent played the move that beats our previous move
        reaction_count = 0
        total_checked = 0
        for i in range(1, min(len(self.opponent_history), self.RECENT_WINDOW + 1)):
            our_prev = self.my_history[-(i + 1)]
            their_next = self.opponent_history[-i]
            if their_next == self.loses_to[our_prev]:
                reaction_count += 1
            total_checked += 1

        # Only commit if there's a clear reaction pattern (>60% of recent moves)
        if total_checked >= 3 and reaction_count / total_checked > 0.60:
            # If they try to beat our last move, predict that counter
            predicted = self.loses_to[self.my_history[-1]]
            return predicted

        return random.choice(self.moves)

    def update(self, my_move: str, opponent_move: str) -> None:
        """
        Record the result of a round and update all internal models.
        """
        self.my_history.append(my_move)
        self.opponent_history.append(opponent_move)

        # Determine result from bot's perspective
        if my_move == opponent_move:
            result = self.TIE_RESULT
        elif self.beats[my_move] == opponent_move:
            result = self.WIN_RESULT
        else:
            result = self.LOSS_RESULT
        self.results_history.append(result)

        # Update frequency table
        self.frequency_table[opponent_move] += 1

        # Credit/blame the strategy that led to this move
        if self.last_strategy_used:
            self.strategy_attempts[self.last_strategy_used] += 1
            if result == self.WIN_RESULT:
                self.strategy_successes[self.last_strategy_used] += 1

        # Update Markov transition matrix
        if len(self.opponent_history) > 1:
            prev_idx = self.move_idx[self.opponent_history[-2]]
            curr_idx = self.move_idx[opponent_move]

            for i in range(3):
                if i == curr_idx:
                    self.transition_matrix[prev_idx][i] += self.adapt_speed
                else:
                    self.transition_matrix[prev_idx][i] *= (1 - self.adapt_speed / 2)

            row_sum = sum(self.transition_matrix[prev_idx])
            self.transition_matrix[prev_idx] = self.transition_matrix[prev_idx] / row_sum

        # Update pattern memory
        if len(self.opponent_history) >= 3:
            pattern_2 = tuple(self.opponent_history[-3:-1])
            self.last_n_transitions[pattern_2].append(opponent_move)

        if len(self.opponent_history) >= 4:
            pattern_3 = tuple(self.opponent_history[-4:-1])
            self.last_n_transitions[pattern_3].append(opponent_move)

        # Prune pattern memory if it grows too large
        if len(self.last_n_transitions) > self.MAX_PATTERN_MEMORY:
            self._prune_patterns()

    def _prune_patterns(self) -> None:
        """Keep only the most frequently matched patterns."""
        pattern_counts = Counter({k: len(v) for k, v in self.last_n_transitions.items()})
        # most_common() returns descending order; keep the top N
        keep = pattern_counts.most_common(self.TOP_PATTERNS_TO_KEEP)
        keep_keys = {k for k, _ in keep}
        for key in list(self.last_n_transitions.keys()):
            if key not in keep_keys:
                del self.last_n_transitions[key]
