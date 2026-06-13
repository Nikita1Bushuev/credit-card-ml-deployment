import numpy as np
from flask import Flask, request, jsonify
from .model_handler import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_end():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        version = data.get('model_version', 'v1')
        prediction, probability = predict(features, version)
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'model_version': version
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)