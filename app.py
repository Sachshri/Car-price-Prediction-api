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

with open('kms-model2.pkl', 'rb') as f:
    model2 = pickle.load(f)

with open('Year-model.pkl', 'rb') as f:
    model3 = pickle.load(f)

# Define a route for prediction
@app.route('/predict-price', methods=['POST'])
def predictPrice():
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



@app.route('/predict-kms', methods=['POST'])
def predictKms():
    try:
        # Get the data from the request
        data = request.json

        # Convert data to a pandas DataFrame
        input_df = pd.DataFrame(data)

        # Use the model to make a prediction
        prediction = model2.predict(input_df)

        return jsonify({'kms_driven': prediction.tolist()})
        # return jsonify({'kms_driven is fetched':"ok"})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/predict-year', methods=['POST'])
def predictYear():
    try:
        # Get the data from the request
        data = request.json

        # Convert data to a pandas DataFrame
        input_df = pd.DataFrame(data)

        # Use the model to make a prediction
        prediction = model3.predict(input_df)

        return jsonify({'year': prediction.tolist()})
        # return jsonify({'kms_driven is fetched':"ok"})
    
    except Exception as e:
        return jsonify({'error': str(e)})
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
