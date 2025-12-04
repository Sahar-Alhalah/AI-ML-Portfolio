from flask import Flask, request, make_response, jsonify, render_template
from pickle import load

app = Flask(__name__)
covid_model = load(open('models/covid_model.pkl', 'rb'))


@app.route("/")
def hello_world():
    return render_template('form.html', title='Covid Deaths Predictor')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        week = int(data['week'])
        cases = int(data['cases'])
        if week > int(365 / 7) or week < 1:
            return make_response(jsonify({'deaths': 0, 'error': 'Invalid week number'}), 200)
        elif cases > 21260000 or cases < 0:  # 21260000 is the Brazil's population
            return make_response(jsonify({'deaths': 0, 'error': 'Invalid cases number'}), 200)
        deaths = int(covid_model.predict([[week, cases]])[0])
        return make_response(jsonify({'deaths': deaths, 'error': ''}), 200)
    except Exception as ex:
        return make_response(jsonify({'deaths': 0, 'error': ex}), 200)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7000, debug=True)
