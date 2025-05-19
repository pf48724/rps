import random
import pytest
from rps_main import RPSMain


class TestRPSMainBasics:
    def test_init(self):
        bot = RPSMain()
        assert bot.opponent_history == []
        assert bot.my_history == []
        assert bot.confidence == 0.0
        assert bot.last_strategy_used is None
        for s in bot.strategy_attempts:
            assert bot.strategy_attempts[s] == 0
            assert bot.strategy_successes[s] == 0

    def test_first_move_is_random(self):
        bot = RPSMain()
        move = bot.get_move()
        assert move in bot.moves
        # get_move shouldn't set strategy on first call (no history)
        assert bot.last_strategy_used is None

    def test_confidence_grows(self):
        bot = RPSMain()
        for i in range(10):
            bot_move = bot.get_move()
            bot.update(bot_move, random.choice(bot.moves))
            # confidence is based on moves_seen BEFORE this get_move call
            expected = min(
                RPSMain.MAX_CONFIDENCE,
                RPSMain.BASE_CONFIDENCE + RPSMain.CONFIDENCE_PER_MOVE * i
            )
            assert bot.confidence == expected

    def test_last_strategy_used_set_after_prediction(self):
        bot = RPSMain()
        bot.exploration_rate = 0.0  # Disable random exploration for this test

        # First move: no history, random, strategy not set
        bot.get_move()
        assert bot.last_strategy_used is None

        # After update, next get_move should set strategy
        bot.update('rock', 'paper')
        bot.get_move()
        assert bot.last_strategy_used in bot.strategy_attempts


class TestStrategyTracking:
    def test_strategy_attempts_increment_on_win(self):
        bot = RPSMain()
        bot.exploration_rate = 0.0  # Disable exploration for deterministic strategy use
        for _ in range(30):
            bot_move = bot.get_move()
            # Force a win: player plays what bot beats
            user_move = bot.beats[bot_move]
            prev_attempts = dict(bot.strategy_attempts)
            prev_successes = dict(bot.strategy_successes)
            bot.update(bot_move, user_move)
            if bot.last_strategy_used:
                assert bot.strategy_attempts[bot.last_strategy_used] == prev_attempts[bot.last_strategy_used] + 1
                assert bot.strategy_successes[bot.last_strategy_used] == prev_successes[bot.last_strategy_used] + 1

    def test_strategy_attempts_increment_on_loss(self):
        bot = RPSMain()
        bot.exploration_rate = 0.0  # Disable exploration for deterministic strategy use
        for _ in range(30):
            bot_move = bot.get_move()
            # Force a loss: player plays what beats bot
            user_move = bot.loses_to[bot_move]
            prev_attempts = dict(bot.strategy_attempts)
            prev_successes = dict(bot.strategy_successes)
            bot.update(bot_move, user_move)
            if bot.last_strategy_used:
                assert bot.strategy_attempts[bot.last_strategy_used] == prev_attempts[bot.last_strategy_used] + 1
                # Success should NOT increment on loss
                assert bot.strategy_successes[bot.last_strategy_used] == prev_successes[bot.last_strategy_used]

    def test_strategy_attempts_increment_on_tie(self):
        bot = RPSMain()
        bot.exploration_rate = 0.0  # Disable exploration for deterministic strategy use
        for _ in range(30):
            bot_move = bot.get_move()
            # Force a tie
            user_move = bot_move
            prev_attempts = dict(bot.strategy_attempts)
            prev_successes = dict(bot.strategy_successes)
            bot.update(bot_move, user_move)
            if bot.last_strategy_used:
                assert bot.strategy_attempts[bot.last_strategy_used] == prev_attempts[bot.last_strategy_used] + 1
                # Success should NOT increment on tie
                assert bot.strategy_successes[bot.last_strategy_used] == prev_successes[bot.last_strategy_used]


class TestMarkovPrediction:
    def test_markov_predicts_based_on_transitions(self):
        bot = RPSMain()
        # Train bot: after rock, player always plays paper
        for _ in range(10):
            bot.get_move()
            bot.update('rock', 'paper')

        # After another rock, markov should predict paper
        pred = bot._markov_prediction()
        assert pred == 'paper'

    def test_markov_random_on_empty_history(self):
        bot = RPSMain()
        pred = bot._markov_prediction()
        assert pred in bot.moves


class TestFrequencyPrediction:
    def test_frequency_predicts_most_common(self):
        bot = RPSMain()
        for _ in range(5):
            bot.get_move()
            bot.update('rock', 'paper')
        for _ in range(2):
            bot.get_move()
            bot.update('rock', 'scissors')

        pred = bot._frequency_prediction()
        # Paper should be most common
        assert pred == 'paper'

    def test_frequency_random_on_empty_history(self):
        bot = RPSMain()
        # frequency_table is empty initially, but opponent_history is also empty
        # The method checks opponent_history indirectly via frequency_table
        # Actually frequency_table is populated by update, so on fresh bot it's empty
        pred = bot._frequency_prediction()
        assert pred in bot.moves

    def test_frequency_weights_recent_moves(self):
        bot = RPSMain()
        # Fill history with old rocks
        for _ in range(8):
            bot.get_move()
            bot.update('rock', 'rock')
        # Recent papers should outweigh old rocks due to decay
        for _ in range(5):
            bot.get_move()
            bot.update('rock', 'paper')

        pred = bot._frequency_prediction()
        assert pred == 'paper'


class TestPsychologyPrediction:
    def test_tilt_detection_after_losses(self):
        bot = RPSMain()
        bot.opponent_history = ['rock', 'rock']
        bot.results_history = [RPSMain.WIN_RESULT, RPSMain.WIN_RESULT]
        bot.opponent_losses = 2
        pred = bot._psychology_prediction()
        assert pred == 'rock'

    def test_repeat_detection(self):
        bot = RPSMain()
        bot.opponent_history = ['rock', 'rock', 'rock']
        bot.opponent_repeats = 2
        pred = bot._psychology_prediction()
        # After 2+ repeats, expect player to switch to what beats repeated move
        assert pred == bot.loses_to['rock']

    def test_win_stay(self):
        bot = RPSMain()
        # Player won last round (bot lost)
        bot.opponent_history = ['paper']
        bot.my_history = ['rock']
        bot.results_history = [RPSMain.LOSS_RESULT]
        pred = bot._psychology_prediction()
        # Player should repeat what beat us
        assert pred == 'paper'

    def test_lose_shift(self):
        bot = RPSMain()
        # Player lost last round
        bot.opponent_history = ['rock']
        bot.my_history = ['paper']
        bot.results_history = [RPSMain.WIN_RESULT]
        pred = bot._psychology_prediction()
        # Player should shift to what beats their last move
        assert pred == bot.loses_to['rock']


class TestPatternPrediction:
    def test_pattern_2_match(self):
        bot = RPSMain()
        # Train pattern: after (rock, paper), player plays scissors
        for _ in range(5):
            bot.get_move()
            bot.update('rock', 'rock')
            bot.get_move()
            bot.update('rock', 'paper')
            bot.get_move()
            bot.update('rock', 'scissors')

        bot.opponent_history = ['rock', 'paper']
        pred = bot._pattern_prediction()
        assert pred == 'scissors'

    def test_pattern_3_fallback(self):
        bot = RPSMain()
        # Train pattern: after (rock, paper, scissors), player plays rock
        for _ in range(5):
            for move in ['rock', 'paper', 'scissors']:
                bot.get_move()
                bot.update('rock', move)
            bot.get_move()
            bot.update('rock', 'rock')

        bot.opponent_history = ['rock', 'paper', 'scissors']
        pred = bot._pattern_prediction()
        assert pred == 'rock'

    def test_pattern_random_on_short_history(self):
        bot = RPSMain()
        bot.opponent_history = ['rock']
        pred = bot._pattern_prediction()
        assert pred in bot.moves


class TestReactionPrediction:
    def test_detects_counter_strategy(self):
        bot = RPSMain()
        # Player always plays what beats our PREVIOUS move.
        # Round i: our move is my[i], their move is what beats my[i-1]
        my_moves =     ['rock', 'rock', 'rock', 'rock', 'rock', 'rock', 'rock']
        their_moves =  ['rock', 'paper', 'paper', 'paper', 'paper', 'paper', 'paper']
        for m, t in zip(my_moves, their_moves):
            bot.get_move()
            bot.update(m, t)

        pred = bot._reaction_prediction()
        # Our last move was rock, so reaction predicts paper (what beats rock)
        assert pred == 'paper'

    def test_no_reaction_on_weak_signal(self):
        bot = RPSMain()
        # Random moves - no clear reaction pattern
        for _ in range(10):
            bot.get_move()
            bot.update(random.choice(bot.moves), random.choice(bot.moves))

        pred = bot._reaction_prediction()
        assert pred in bot.moves

    def test_reaction_random_on_short_history(self):
        bot = RPSMain()
        pred = bot._reaction_prediction()
        assert pred in bot.moves


class TestUpdateLogic:
    def test_markov_matrix_updates(self):
        bot = RPSMain()
        initial = bot.transition_matrix.copy()
        bot.update('rock', 'paper')
        # Matrix shouldn't change on first update (no previous opponent move)
        assert (bot.transition_matrix == initial).all()

        bot.update('rock', 'scissors')
        # Now matrix should have updated for transition paper -> scissors
        paper_idx = bot.move_idx['paper']
        scissors_idx = bot.move_idx['scissors']
        assert bot.transition_matrix[paper_idx][scissors_idx] > initial[paper_idx][scissors_idx]

    def test_pattern_memory_populated(self):
        bot = RPSMain()
        for move in ['rock', 'paper', 'scissors', 'rock']:
            bot.get_move()
            bot.update('rock', move)

        # Should have pattern (rock, paper) -> scissors
        assert ('rock', 'paper') in bot.last_n_transitions
        assert 'scissors' in bot.last_n_transitions[('rock', 'paper')]

    def test_pattern_pruning(self):
        bot = RPSMain()
        # Fill with many distinct 4-move patterns
        moves = bot.moves
        for i in range(200):
            bot.get_move()
            seq = tuple(moves[(i // (3 ** j)) % 3] for j in range(4))
            bot.last_n_transitions[seq].append(moves[(i + 2) % 3])

        assert len(bot.last_n_transitions) > RPSMain.MAX_PATTERN_MEMORY
        bot._prune_patterns()
        assert len(bot.last_n_transitions) <= RPSMain.TOP_PATTERNS_TO_KEEP

    def test_frequency_table_updated(self):
        bot = RPSMain()
        bot.update('rock', 'paper')
        assert bot.frequency_table['paper'] == 1
        bot.update('rock', 'paper')
        assert bot.frequency_table['paper'] == 2


class TestResultCalculation:
    def test_win_result(self):
        bot = RPSMain()
        bot.update('rock', 'scissors')  # rock beats scissors
        assert bot.results_history[-1] == RPSMain.WIN_RESULT

    def test_loss_result(self):
        bot = RPSMain()
        bot.update('rock', 'paper')  # paper beats rock
        assert bot.results_history[-1] == RPSMain.LOSS_RESULT

    def test_tie_result(self):
        bot = RPSMain()
        bot.update('rock', 'rock')
        assert bot.results_history[-1] == RPSMain.TIE_RESULT


class TestExploration:
    def test_exploration_rate(self):
        bot = RPSMain()
        bot.exploration_rate = 1.0  # Always explore
        bot.opponent_history = ['rock']  # Force past first-move branch
        moves = [bot.get_move() for _ in range(50)]
        assert all(m in bot.moves for m in moves)

    def test_no_exploration_when_rate_zero(self):
        bot = RPSMain()
        bot.exploration_rate = 0.0
        # Train a clear pattern
        for _ in range(20):
            bot.get_move()
            bot.update('rock', 'paper')
        # Markov should predict paper, so bot should play scissors
        bot_move = bot.get_move()
        assert bot_move == 'scissors'


class TestPhaseWeights:
    def test_early_phase_weights(self):
        bot = RPSMain()
        bot.opponent_history = ['rock']  # 1 move = early
        bot.get_move()
        # Just verify it doesn't crash
        assert bot.last_strategy_used in bot.strategy_attempts or bot.last_strategy_used is None

    def test_mid_phase_weights(self):
        bot = RPSMain()
        for _ in range(4):
            bot.get_move()
            bot.update('rock', 'paper')
        bot.get_move()
        assert bot.last_strategy_used in bot.strategy_attempts

    def test_late_phase_weights(self):
        bot = RPSMain()
        for _ in range(10):
            bot.get_move()
            bot.update('rock', 'paper')
        bot.get_move()
        assert bot.last_strategy_used in bot.strategy_attempts


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
