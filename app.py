from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)

# Set CORS to allow requests from specific origins (e.g., localhost)
CORS(app,origins="*")

# Load the model
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json

        # Convert data to a pandas DataFrame
        input_df = pd.DataFrame(data)

        # Use the model to make a prediction
        prediction = model.predict(input_df)

        return jsonify({'predicted_price': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
