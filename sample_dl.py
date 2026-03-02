import torch
import torch.nn as nn
import torch.optim as optim

# A simple neural network
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Dummy dataset
data = torch.randn(64, 10)
targets = torch.randn(64, 1)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train():
    # TIP: model.train() is missing here
    for epoch in range(5):
        # ERROR: optimizer.zero_grad() is missing!
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

def evaluate():
    # TIP: torch.no_grad() and model.eval() are missing here
    output = model(data)
    print("Evaluation complete")

if __name__ == "__main__":
    train()
    evaluate()
