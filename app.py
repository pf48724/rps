from flask import Flask, request, jsonify, render_template
from rps_main import RPSMain

app = Flask(__name__)

bot = RPSMain()
wins, losses, ties = 0, 0, 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    global wins, losses, ties
    data = request.get_json()
    user_move = data.get('move')

    if user_move not in ['rock', 'paper', 'scissors']:
        return jsonify({'error': 'Invalid move'}), 400

    bot_move = bot.get_move()

    # Determine result from user's perspective
    if user_move == bot_move:
        result = 'tie'
        ties += 1
    elif (user_move == 'rock' and bot_move == 'scissors') or \
         (user_move == 'paper' and bot_move == 'rock') or \
         (user_move == 'scissors' and bot_move == 'paper'):
        result = 'win'
        wins += 1
    else:
        result = 'loss'
        losses += 1

    bot.update(bot_move, user_move)

    return jsonify({
        'bot_move': bot_move,
        'result': result,
        'score': {'wins': wins, 'losses': losses, 'ties': ties}
    })

if __name__ == '__main__':
    app.run(debug=True)
