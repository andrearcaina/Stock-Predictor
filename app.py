from flask import Flask, render_template, jsonify, request
from src.prediction import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        data = request.get_json()
        symbol = data['symbol']
        predictions = predict(symbol)
        return jsonify(predictions=predictions)
    else:
        return "Please send a POST request to this endpoint."


if __name__ == '__main__':
    app.run(debug=True)
