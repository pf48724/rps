import os
import uuid
import time
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from rps_main import RPSMain

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or os.urandom(24)
app.config['SESSION_PERMANENT'] = False

game_states = {}
last_activity = {}
SESSION_MAX = 500
INACTIVE_TIME = 60 * 60


@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    user_id = session['user_id']
    
    game_states[user_id] = {
        'wins': 0,
        'losses': 0, 
        'ties': 0,
        'bot': RPSMain()
    }
    last_activity[user_id] = time.time()
    
    return render_template('index.html')


@app.route('/play', methods=['POST'])
def play():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    user_id = session['user_id']
    
    if user_id not in game_states:
        game_states[user_id] = {
            'wins': 0, 'losses': 0, 'ties': 0,
            'bot': RPSMain()
        }
    
    data = request.get_json()
    user_move = data.get('move')
    if user_move not in ['rock', 'paper', 'scissors']:
        return jsonify({'error': 'Invalid move'}), 400
    
    state = game_states[user_id]
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
    last_activity[user_id] = time.time()
    
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
def reset():
    if 'user_id' in session and session['user_id'] in game_states:
        user_id = session['user_id']
        game_states[user_id] = {
            'wins': 0, 'losses': 0, 'ties': 0,
            'bot': RPSMain()
        }
        last_activity[user_id] = time.time()
    
    return jsonify({
        'status': 'reset',
        'score': {'wins': 0, 'losses': 0, 'ties': 0}
    })


@app.before_request
def clean_old_sessions():
    now = time.time()
    
    if len(game_states) > SESSION_MAX:
        expired = []
        
        for user_id, last_time in last_activity.items():
            if now - last_time > INACTIVE_TIME:
                expired.append(user_id)
        
        if len(game_states) - len(expired) > SESSION_MAX:
            active_sessions = [(uid, time) for uid, time in last_activity.items() 
                              if uid not in expired]
            active_sessions.sort(key=lambda x: x[1])
            
            to_remove = len(game_states) - SESSION_MAX
            for i in range(min(to_remove, len(active_sessions))):
                expired.append(active_sessions[i][0])
        
        for user_id in expired:
            if user_id in game_states:
                del game_states[user_id]
            if user_id in last_activity:
                del last_activity[user_id]


if __name__ == '__main__':
    app.run(debug=True)