from flask import Flask, render_template, request, jsonify
from model.predict import load_model, predict_receipts
import logging

# Initialize Flask application
app = Flask(__name__)

# Set up logging to help with debugging
logging.basicConfig(level=logging.INFO)

# Load the trained model and normalization parameters
model, mean, std = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract the input values from the form
            last_3_months = [
                float(request.form['month1']),
                float(request.form['month2']),
                float(request.form['month3'])
            ]
            # Log the input values for debugging
            logging.info(f"Input values: {last_3_months}")
            
            # Make a prediction using the loaded model
            prediction = predict_receipts(model, mean, std, last_3_months)
            # Log the prediction for debugging
            logging.info(f"Predicted value: {prediction}")
            
            # Return the prediction as JSON
            return jsonify({'prediction': prediction})
        except Exception as e:
            # Log any errors that occur during prediction
            logging.error(f"Error in prediction: {str(e)}")
            # Return the error message to the client
            return jsonify({'error': str(e)}), 400
    # For GET requests, render the HTML template
    return render_template('index.html')

# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    # Start the Flask development server
    # Debug mode is set to True for development purposes
    app.run(debug=True)