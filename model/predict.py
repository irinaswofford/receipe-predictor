import torch
import numpy as np
from model.train import ReceiptPredictor

def load_model():
    #loads the model from the checkpoint file and returns the model, mean, and std
    checkpoint = torch.load('model/receipt_predictor.pth')
    model = ReceiptPredictor(input_size=1, hidden_size=16, output_size=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['mean'], checkpoint['std']

def predict_receipts(model, mean, std, last_3_months):
    # Convert the input to a float type Numpy array
    last_3_months = np.array(last_3_months, dtype=float)
    
    # Check if the input is already normalized
    if np.abs(last_3_months).max() < 10:  # This assumes normalized data is typically between -3 and 3
        normalized_input = last_3_months
    else:
        normalized_input = (last_3_months - mean) / std
    
    input_tensor = torch.FloatTensor(normalized_input).unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Denormalize the prediction
    denormalized_prediction = prediction.item() * std + mean
    
    # Ensure the prediction is not negative
    return max(0, round(denormalized_prediction))