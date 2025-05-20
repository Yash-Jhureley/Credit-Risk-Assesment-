from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict([np.array(data)])
    return jsonify({'prediction': 'Approved' if prediction[0] == 1 else 'Rejected'})

if __name__ == '__main__':
    app.run(debug=True)
