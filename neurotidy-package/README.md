# NeuroTidy

**AI-Powered Python & Deep Learning Code Analyzer**

NeuroTidy analyzes your Python and PyTorch/TensorFlow code using AI. It can explain, analyze, optimize, and debug your code — all from the terminal.

## Install

```bash
pip install neurotidy
```

## Usage

```bash
# Explain code for beginners
neurotidy explain train.py --mode beginner

# Static code analysis
neurotidy analyze model.py

# DL-specific optimization suggestions
neurotidy optimize train.py

# Debug an error message
neurotidy debug --error "RuntimeError: mat1 and mat2 shapes cannot be multiplied"

# Inline code
neurotidy explain --code "def add(a, b): return a + b" --mode beginner
```

## Features

- **Explain** — AI-powered code explanations at beginner, intermediate, or advanced level
- **Analyze** — Static analysis with Python best-practice rules (PY001–PY010)
- **Optimize** — Deep learning specific optimizations (NT001–NT027) with RAG-enhanced analysis
- **Debug** — Paste an error message, get root cause analysis and step-by-step fixes

## How It Works

NeuroTidy calls a serverless AWS backend:
- **API Gateway** → **Lambda** → **Amazon Bedrock** (Claude AI)
- Results cached in **DynamoDB**, stored in **S3**
- Zero infrastructure to manage — just `pip install` and go

## Configuration

By default, NeuroTidy connects to the hosted API. To use a custom endpoint:

```bash
export NEUROTIDY_API_ENDPOINT=https://your-api.execute-api.us-east-1.amazonaws.com/prod
```

## Built by

**ERROR-108** — AWS Hackathon 2026

## License

MIT
