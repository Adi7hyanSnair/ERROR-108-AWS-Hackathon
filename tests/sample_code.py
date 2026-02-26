"""
Sample Python code for testing NeuroTidy analysis.
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """A simple neural network for demonstration."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(model, data_loader, epochs=10):
    """
    Train the model using the provided data loader.
    
    Args:
        model: Neural network model
        data_loader: DataLoader with training data
        epochs: Number of training epochs
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Create model
    model = SimpleModel(input_size=784, hidden_size=128, output_size=10)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
