from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = './Crop_recommendation.pkl'
with open(model_path, 'rb') as file:
    data = pickle.load(file)
    loaded_model = data["model"]
    loaded_encoder = data["encoder"]
    
@app.route('/')
def index():
    return render_template('index.html')  # Load HTML template

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = loaded_model.predict([data])
    crop_name = loaded_encoder.inverse_transform(prediction)[0]
    return render_template('index.html', prediction_text = f"Recommended crop: {crop_name}")
if __name__ == '__main__':
    app.run(debug=True)