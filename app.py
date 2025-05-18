import os
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from rps_main import RPSMain

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['SESSION_PERMANENT'] = False

@app.route('/')
def index():
    session.clear()
    session['wins'] = 0
    session['losses'] = 0
    session['ties'] = 0
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    bot = RPSMain()
    if 'bot_state' in session:
        bot.opponent_history = session['bot_state'].get('opponent_history', [])
        bot.my_history = session['bot_state'].get('my_history', [])
        bot.results_history = session['bot_state'].get('results_history', [])
    
    data = request.get_json()
    user_move = data.get('move')

    if user_move not in ['rock', 'paper', 'scissors']:
        return jsonify({'error': 'Invalid move'}), 400

    bot_move = bot.get_move()

    if user_move == bot_move:
        result = 'tie'
        session['ties'] = session.get('ties', 0) + 1
    elif (user_move == 'rock' and bot_move == 'scissors') or \
         (user_move == 'paper' and bot_move == 'rock') or \
         (user_move == 'scissors' and bot_move == 'paper'):
        result = 'win'
        session['wins'] = session.get('wins', 0) + 1
    else:
        result = 'loss'
        session['losses'] = session.get('losses', 0) + 1

    bot.update(bot_move, user_move)
    
    session['bot_state'] = {
        'opponent_history': bot.opponent_history,
        'my_history': bot.my_history,
        'results_history': bot.results_history
    }
    
    session.modified = True

    return jsonify({
        'bot_move': bot_move,
        'result': result,
        'score': {
            'wins': session.get('wins', 0),
            'losses': session.get('losses', 0),
            'ties': session.get('ties', 0)
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
