import os
import uuid
import time
import logging
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from rps_main import RPSMain

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSION_MAX = 500
INACTIVE_TIME = 60 * 60  # 1 hour
CLEANUP_INTERVAL = 60    # Only run cleanup once per 60 seconds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    logger.warning("SECRET_KEY not set; using a random key (sessions will not survive restarts).")
    app.secret_key = os.urandom(24)

app.config['SESSION_PERMANENT'] = False

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
game_states: Dict[str, Dict[str, Any]] = {}
last_activity: Dict[str, float] = {}
_last_cleanup_time: float = 0.0


def _ensure_user_id() -> str:
    """Return the current session's user_id, creating one if needed."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


def _get_or_create_state(user_id: str) -> Dict[str, Any]:
    """Return existing game state or initialise a fresh one."""
    if user_id not in game_states:
        game_states[user_id] = {
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'bot': RPSMain()
        }
    return game_states[user_id]


def _clean_old_sessions() -> None:
    """Remove expired or excess sessions. Called automatically inside routes."""
    global _last_cleanup_time

    now = time.time()
    if len(game_states) <= SESSION_MAX:
        return

    if now - _last_cleanup_time < CLEANUP_INTERVAL:
        return
    _last_cleanup_time = now

    expired: list[str] = []
    for user_id, last_time in list(last_activity.items()):
        if now - last_time > INACTIVE_TIME:
            expired.append(user_id)

    if len(game_states) - len(expired) > SESSION_MAX:
        active_sessions = [(uid, t) for uid, t in last_activity.items() if uid not in expired]
        active_sessions.sort(key=lambda x: x[1])
        to_remove = len(game_states) - SESSION_MAX
        for i in range(min(to_remove, len(active_sessions))):
            expired.append(active_sessions[i][0])

    for user_id in expired:
        game_states.pop(user_id, None)
        last_activity.pop(user_id, None)

    if expired:
        logger.info("Cleaned up %d session(s). Remaining: %d", len(expired), len(game_states))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index() -> str:
    user_id = _ensure_user_id()
    _get_or_create_state(user_id)
    last_activity[user_id] = time.time()
    _clean_old_sessions()
    return render_template('index.html')


@app.route('/play', methods=['POST'])
def play() -> Any:
    user_id = _ensure_user_id()
    state = _get_or_create_state(user_id)
    last_activity[user_id] = time.time()
    _clean_old_sessions()

    data: Optional[dict] = request.get_json(silent=True)
    if not isinstance(data, dict):
        logger.warning("Invalid JSON body from %s", user_id)
        return jsonify({'error': 'Invalid JSON body'}), 400

    user_move = data.get('move')
    if user_move not in ['rock', 'paper', 'scissors']:
        return jsonify({'error': 'Invalid move'}), 400

    bot = state['bot']
    bot_move = bot.get_move()

    if user_move == bot_move:
        result = 'tie'
        state['ties'] += 1
    elif (user_move == 'rock' and bot_move == 'scissors') or \
         (user_move == 'paper' and bot_move == 'rock') or \
         (user_move == 'scissors' and bot_move == 'paper'):
        result = 'win'
        state['wins'] += 1
    else:
        result = 'loss'
        state['losses'] += 1

    bot.update(bot_move, user_move)

    return jsonify({
        'bot_move': bot_move,
        'result': result,
        'score': {
            'wins': state['wins'],
            'losses': state['losses'],
            'ties': state['ties']
        }
    })


@app.route('/reset', methods=['POST'])
def reset() -> Any:
    user_id = _ensure_user_id()
    game_states[user_id] = {
        'wins': 0, 'losses': 0, 'ties': 0,
        'bot': RPSMain()
    }
    last_activity[user_id] = time.time()
    return jsonify({
        'status': 'reset',
        'score': {'wins': 0, 'losses': 0, 'ties': 0}
    })


@app.errorhandler(400)
def bad_request(error: Any) -> Any:
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(500)
def internal_error(error: Any) -> Any:
    logger.exception("Unhandled 500 error")
    return jsonify({'error': 'Internal server error'}), 500


# ---------------------------------------------------------------------------
# Local dev
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
