# tests/eval_dataset.py
EVAL_DATASET = [
    {
        "id": "eval_001_missing_zero_grad",
        "description": "PyTorch training loop missing optimizer.zero_grad()",
        "endpoint": "/optimize",
        "code": """
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(5):
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 2)
    
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
""",
        "expected_findings": ["zero_grad"]
    },
    {
        "id": "eval_002_missing_eval_mode",
        "description": "PyTorch evaluation loop missing model.eval() and torch.no_grad()",
        "endpoint": "/optimize",
        "code": """
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
# Missing model.eval()
# Missing with torch.no_grad():
inputs = torch.randn(32, 10)
outputs = model(inputs)
""",
        "expected_findings": ["eval()", "no_grad"]
    },
    {
        "id": "eval_003_device_mismatch_debug",
        "description": "Debugging a device mismatch error",
        "endpoint": "/debug",
        "code": """
import torch
import torch.nn as nn

model = nn.Linear(10, 2).cuda()
inputs = torch.randn(32, 10)
outputs = model(inputs)
""",
        "error_message": "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!",
        "expected_findings": ["inputs.to('cuda')", "inputs.cuda()", "to(device)"]
    },
    {
        "id": "eval_004_bare_except",
        "description": "Static analysis for bare except",
        "endpoint": "/analyze",
        "code": """
def load_data():
    try:
        data = open("file.txt").read()
    except:
        pass
""",
        "expected_findings": ["bare except", "PY004"]
    },
    {
        "id": "eval_005_softmax_cross_entropy",
        "description": "Applying softmax before CrossEntropyLoss",
        "endpoint": "/optimize",
        "code": """
# Bypassing cache 3
import torch
import torch.nn as nn
import torch.nn.functional as F

model = nn.Linear(10, 5)
criterion = nn.CrossEntropyLoss()

inputs = torch.randn(32, 10)
targets = torch.randint(0, 5, (32,))

outputs = model(inputs)
probs = F.softmax(outputs, dim=1)
loss = criterion(probs, targets)
""",
        "expected_findings": ["softmax", "CrossEntropyLoss applies Softmax internally"]
    }
]
