import random
import numpy as np
from collections import defaultdict, Counter

class RPSMain:
    def __init__(self):
        self.moves = ['rock', 'paper', 'scissors']
        self.moveIdx = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.beats = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
        self.losesTo = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}

        self.opponent_history = []
        self.my_history = []
        self.results_history = []

        self.transition_matrix = np.ones((3, 3)) / 3
        self.last_n_transitions = defaultdict(list)
        self.frequency_table = Counter()

        self.opponent_repeats = 0
        self.opponent_losses = 0

        self.exploration_rate = 0.1
        self.adapt_speed = 0.3
        self.confidence = 0

    def get_move(self):
        if not self.opponent_history:
            return random.choice(self.moves)

        self.confidence = min(0.9, 0.1 + 0.15 * len(self.opponent_history))
        if random.random() < self.exploration_rate:
            return random.choice(self.moves)

        predicted = self._make_prediction()
        counter_move = self.losesTo[predicted]
        return counter_move

    def _make_prediction(self):
        strategies = {
            'markov': self._markov_prediction(),
            'frequency': self._frequency_prediction(),
            'psychology': self._psychology_prediction(),
            'pattern': self._pattern_prediction()
        }

        moves_played = len(self.opponent_history)
        if moves_played <= 2:
            weights = {'markov': 0.1, 'frequency': 0.3, 'psychology': 0.5, 'pattern': 0.1}
        elif moves_played <= 5:
            weights = {'markov': 0.3, 'frequency': 0.2, 'psychology': 0.3, 'pattern': 0.2}
        else:
            weights = {'markov': 0.4, 'frequency': 0.1, 'psychology': 0.2, 'pattern': 0.3}

        vote = defaultdict(float)
        for strat, pred in strategies.items():
            vote[pred] += weights[strat]

        best_prediction = max(vote, key=vote.get)
        return best_prediction

    def _markov_prediction(self):
        last_move = self.opponent_history[-1]
        last_idx = self.moveIdx[last_move]
        probs = self.transition_matrix[last_idx]
        predicted_idx = np.argmax(probs)
        return self.moves[predicted_idx]

    def _frequency_prediction(self):
        if not self.frequency_table:
            return random.choice(self.moves)
        return self.frequency_table.most_common(1)[0][0]

    def _psychology_prediction(self):
        last_move = self.opponent_history[-1]

        if self.results_history and self.results_history[-1] == 1:
            self.opponent_losses += 1
            if self.opponent_losses >= 2:
                return last_move
        else:
            self.opponent_losses = 0

        if len(self.opponent_history) >= 2 and last_move == self.opponent_history[-2]:
            self.opponent_repeats += 1
            if self.opponent_repeats >= 2:
                return self.losesTo[last_move]
        else:
            self.opponent_repeats = 0

        return self.beats[last_move]

    def _pattern_prediction(self):
        if len(self.opponent_history) < 3:
            return random.choice(self.moves)

        pattern = tuple(self.opponent_history[-2:])
        if pattern in self.last_n_transitions:
            follows = Counter(self.last_n_transitions[pattern])
            if follows:
                return follows.most_common(1)[0][0]

        return random.choice(self.moves)

    def update(self, my_move, opponent_move):
        self.my_history.append(my_move)
        self.opponent_history.append(opponent_move)

        if my_move == opponent_move:
            result = 0
        elif self.beats[my_move] == opponent_move:
            result = 1
        else:
            result = -1
        self.results_history.append(result)

        self.frequency_table[opponent_move] += 1

        if len(self.opponent_history) > 1:
            prev_idx = self.moveIdx[self.opponent_history[-2]]
            curr_idx = self.moveIdx[opponent_move]

            for i in range(3):
                if i == curr_idx:
                    self.transition_matrix[prev_idx][i] += self.adapt_speed
                else:
                    self.transition_matrix[prev_idx][i] *= (1 - self.adapt_speed / 2)

            row_sum = sum(self.transition_matrix[prev_idx])
            self.transition_matrix[prev_idx] = self.transition_matrix[prev_idx] / row_sum

        if len(self.opponent_history) >= 3:
            pattern2 = tuple(self.opponent_history[-3:-1])
            self.last_n_transitions[pattern2].append(opponent_move)

        if len(self.opponent_history) >= 4:
            pattern3 = tuple(self.opponent_history[-4:-1])
            self.last_n_transitions[pattern3].append(opponent_move)

            if len(self.last_n_transitions) > 50:
                least_common = Counter({k: len(v) for k, v in self.last_n_transitions.items()}).most_common()[:-10]
                for k, _ in least_common:
                    del self.last_n_transitions[k]
