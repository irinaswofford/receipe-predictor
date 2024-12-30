import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Define the neural network model
class ReceiptPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ReceiptPredictor, self).__init__()
        # LSTM layer with specified input and hidden sizes
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Fully connected layer to produce final output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Pass input through LSTM layer
        out, _ = self.lstm(x)
        # Use only the last output of the sequence
        out = self.fc(out[:, -1, :])
        # Remove extra dimensions
        return out.squeeze(-1)

# Function to create input-output sequences from data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        # Input sequence
        xs.append(data[i:(i+seq_length)])
        # Target value (next value after sequence)
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

def train_model():
    # Load data from CSV file
    df = pd.read_csv('receipt_data.csv', parse_dates=['# Date'])
    df.set_index('# Date', inplace=True)
    # Resample daily data to monthly and sum receipt counts
    monthly_receipts = df.resample('M')['Receipt_Count'].sum().values

    print(f"Monthly receipts range: {monthly_receipts.min()} to {monthly_receipts.max()}")
    
    # Normalize the data
    mean = monthly_receipts.mean()
    std = monthly_receipts.std()
    normalized_receipts = (monthly_receipts - mean) / std

    print(f"Normalized receipts range: {normalized_receipts.min()} to {normalized_receipts.max()}")

    # Create sequences for LSTM input
    seq_length = 3
    X, y = create_sequences(normalized_receipts, seq_length)
    X = torch.FloatTensor(X).unsqueeze(-1)  # Add extra dimension for LSTM input
    y = torch.FloatTensor(y)

    # Initialize the model, loss function, and optimizer
    model = ReceiptPredictor(input_size=1, hidden_size=16, output_size=1)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X)  # Forward pass
        loss = criterion(outputs, y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model and normalization parameters
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std
    }, 'model/receipt_predictor.pth')

    print("Model trained and saved as receipt_predictor.pth")
    print(f"Mean: {mean}, Std: {std}")

if __name__ == "__main__":
    train_model()
