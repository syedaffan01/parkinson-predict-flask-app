import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('parkinson_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
                 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
                 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
                 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
                 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form[feat]) for feat in feature_names]
    scaled_input = scaler.transform([values])
    prediction = model.predict(scaled_input)
    result = 'Parkinson Positive' if prediction[0] == 1 else 'Healthy'
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
