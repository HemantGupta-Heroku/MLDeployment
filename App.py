from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('ML_LogReg.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = request.form['Age']
    prediction = model.predict([[int(input_features)]])[0]
    return render_template('index.html', Prediction_Text='Customer would buy insurance {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
