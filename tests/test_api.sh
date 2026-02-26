#!/bin/bash

# Test script for NeuroTidy API

# Replace with your actual API endpoint
API_ENDPOINT="${1:-https://your-api-gateway-url/prod/analyze}"

echo "🧪 Testing NeuroTidy API..."
echo "Endpoint: $API_ENDPOINT"
echo ""

# Test 1: Simple function (Beginner mode)
echo "Test 1: Simple function - Beginner mode"
curl -X POST "$API_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b):\n    return a + b",
    "mode": "beginner"
  }' | jq '.'

echo ""
echo "---"
echo ""

# Test 2: Class with methods (Intermediate mode)
echo "Test 2: Class - Intermediate mode"
curl -X POST "$API_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def multiply(self, a, b):\n        return a * b",
    "mode": "intermediate"
  }' | jq '.'

echo ""
echo "---"
echo ""

# Test 3: ML code (Advanced mode)
echo "Test 3: ML Training Loop - Advanced mode"
curl -X POST "$API_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d @- << 'EOF' | jq '.'
{
  "code": "import torch\nimport torch.nn as nn\n\ndef train(model, data, epochs):\n    optimizer = torch.optim.Adam(model.parameters())\n    criterion = nn.CrossEntropyLoss()\n    for epoch in range(epochs):\n        for batch in data:\n            optimizer.zero_grad()\n            output = model(batch)\n            loss = criterion(output, batch.labels)\n            loss.backward()\n            optimizer.step()",
  "mode": "advanced"
}
EOF

echo ""
echo "✅ Tests complete!"
