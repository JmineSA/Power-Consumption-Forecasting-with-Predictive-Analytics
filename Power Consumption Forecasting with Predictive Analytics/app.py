from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'zone1_power_model.pkl')
model = joblib.load(model_path)

EXPECTED_FEATURES = 8

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get('features')

    # Validate presence and length
    if features is None:
        return jsonify({'error': 'No features provided. Please send features in JSON.'}), 400
    if not isinstance(features, list):
        return jsonify({'error': 'Features should be a list.'}), 400
    if len(features) != EXPECTED_FEATURES:
        return jsonify({'error': f'Expected {EXPECTED_FEATURES} features, but got {len(features)}.'}), 400

    try:
        prediction = model.predict([features])
        return jsonify({'Zone 1 Power Consumption Prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
