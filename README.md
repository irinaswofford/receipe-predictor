Receipt Prediction Application

This application predicts future receipt counts based on historical data using a machine learning model.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Docker 

## Setup

1. Clone the repository: git clone https://github.com/irinaswofford/receipe-predictor.git

2. Enter Folder ReceiptPredictor: cd ReceiptPredictor

3. Create a virtual environment (optional but recommended): python3 -m venv venv (on linux), On Windows use: venv\Scripts\activate

4. Install the required packages: pip install -r requirements.txt

## Running the Application: Docker Lower

1. Ensure your data file `receipt_data.csv` is in the root directory of the project, if you have another csv file, make sure
to insert it in the root directory and ensure that the first row is "# Date, Receipt_Count". You also need to change the line 36
df = pd.read_csv('receipt_data.csv', parse_dates=['# Date']) in Fetch/model/train.py to include your csv file instead.

2. Train the model: 
This will train the model on the historical data and save it as `model/receipt_predictor.pth`.

3. Start the Flask application: python -m app.main

The application will be available at `http://localhost:5000`.

4. Use the web interface to input the receipt counts for the last three months and get a prediction for the next month .
NOTE: make sure that you are using the net sum amount for each month, the Monthly receipts range for the default dataset ranges from 
220033460 to 309948684

## Docker Deployment

To run the application in a Docker container:

1. Build the Docker image (After entering root directory): docker build -t receipt-predictor .

2. Run the Docker container: docker run -p 5000:5000 receipt-predictor

The application will be available at `http://localhost:5000`.

## Project Structure

- `model/train.py`: Script to train the machine learning model
- `model/predict.py`: Functions for loading the model and making predictions
- `app/main.py`: Flask application for serving predictions
- `app/templates/index.html`: HTML template for the web interface
- `Dockerfile`: Instructions for building the Docker image
- `requirements.txt`: List of Python package dependencies
- `receipt_data.csv`: Historical data for training the model

## Troubleshooting

- If you encounter any issues, check the console output for error messages.
- Ensure that the `data_daily (1).csv` file is present in the root directory of the project.
- Verify that all required packages are installed correctly.
- If using Docker, make sure Docker is running on your system.

## Model Details

The prediction model uses an LSTM (Long Short-Term Memory) neural network to predict receipt counts. It takes the last three months of data as input to predict the next month's receipt count.
