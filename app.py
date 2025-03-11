from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    data = np.array(data).reshape(1, 50, 1)
    prediction = model.predict(data)
    return jsonify({'predicted_price': scaler.inverse_transform(prediction).tolist()})

if __name__ == '__main__':
    app.run(debug=True)
