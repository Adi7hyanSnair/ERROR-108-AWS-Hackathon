# 🧠 NeuroTidy — AI-Powered Python Code Analyzer

> **AWS Hackathon 2024** · Powered by **Amazon Bedrock**

NeuroTidy explains, analyzes, optimizes, and debugs your Python & Deep Learning code — serverless on AWS.

---

## ✨ Features

| Mode | Endpoint | Description |
|------|----------|-------------|
| 📖 **Explain** | `POST /explain` | Multi-level code explanations (beginner → advanced) |
| 🔍 **Analyze** | `POST /analyze` | 17+ static analysis rules + ML-specific patterns |
| ⚡ **Optimize** | `POST /optimize` | DL performance optimizer (PyTorch / TensorFlow) |
| 🐛 **Debug** | `POST /debug` | Root-cause error analysis with step-by-step fixes |

---

## ⚡ Quick Start

### Step 1 — Fill in `config.env`

Open **`config.env`** in the project root and fill in:

```env
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_ACCOUNT_ID=123456789012
AWS_REGION=us-east-1

# Pick a Bedrock model (must enable it in AWS Console first):
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

> 💡 See the **Model Selection** section below for all available models.

### Step 2 — Enable Bedrock Model Access

1. Open [AWS Bedrock Console](https://console.aws.amazon.com/bedrock)
2. Go to **Model Access** → Request access to Claude 3 Sonnet (or your chosen model)
3. Wait ~1 min for activation

### Step 3 — Deploy

```powershell
# Windows (PowerShell)
cd infrastructure
.\deploy.ps1
```

```bash
# Linux / Mac
cd infrastructure
./deploy.sh
```

The script prints your API URLs at the end. Copy the `BaseApiUrl` value.

### Step 4 — Update `config.env` with the API URL

```env
NEUROTIDY_API_ENDPOINT=https://abc123.execute-api.us-east-1.amazonaws.com/prod
```

### Step 5 — Test it!

```powershell
# PowerShell
cd tests
.\test_api.ps1
```

```bash
# Linux/Mac
cd tests
./test_api.sh https://your-api-url/prod
```

---

## 🔑 Model Selection (config.env)

Uncomment exactly ONE model in `config.env`:

| Model | ID | Speed | Quality | Cost |
|-------|----|-------|---------|------|
| **Claude 3 Sonnet** ✅ (default) | `anthropic.claude-3-sonnet-20240229-v1:0` | Fast | Excellent | Medium |
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` | Fastest | Good | Lowest |
| Claude 3 Opus | `anthropic.claude-3-opus-20240229-v1:0` | Slow | Best | Highest |
| Llama 3 70B | `meta.llama3-70b-instruct-v1:0` | Medium | Very Good | Low |
| Llama 3 8B | `meta.llama3-8b-instruct-v1:0` | Fastest | Good | Lowest |
| Mistral Large | `mistral.mistral-large-2402-v1:0` | Fast | Very Good | Medium |

---

## 📁 Project Structure

```
ERROR-108-AWS-Hackathon/
│
├── config.env               ← ⭐ YOUR CREDENTIALS & SETTINGS (edit this!)
├── config.example.env       ← Template (safe to commit)
│
├── lambda/                  ← AWS Lambda source code
│   ├── handler.py           ← Main router (4 endpoints)
│   ├── code_explainer.py    ← Code explanation via Bedrock
│   ├── bug_explainer.py     ← Error & stack trace analysis
│   ├── dl_optimizer.py      ← DL performance rule engine
│   ├── static_analyzer.py   ← Python static analysis (17 rules)
│   └── requirements.txt     ← Lambda dependencies (boto3)
│
├── infrastructure/
│   ├── template.yaml        ← SAM/CloudFormation template
│   ├── deploy.ps1           ← Windows deploy script
│   └── deploy.sh            ← Linux/Mac deploy script
│
├── cli/
│   └── neurotidy.py         ← Command-line interface
│
├── frontend/
│   ├── index.html           ← Web UI
│   ├── style.css            ← Premium dark design
│   └── app.js               ← Frontend API integration
│
└── tests/
    ├── test_local.py        ← Unit tests (no AWS needed)
    ├── test_api.ps1         ← PowerShell API tests
    ├── test_api.sh          ← Bash API tests
    └── sample_code.py       ← Sample ML code
```

---

## 🖥️ CLI Usage

```bash
# Install deps (only for CLI, not Lambda)
pip install requests

# Explain code
python cli/neurotidy.py explain myfile.py --mode beginner
python cli/neurotidy.py explain myfile.py --mode advanced

# Analyze code quality
python cli/neurotidy.py analyze myfile.py

# Find DL optimizations
python cli/neurotidy.py optimize train.py

# Debug an error
python cli/neurotidy.py debug --error "RuntimeError: mat1 shapes cannot be multiplied"
python cli/neurotidy.py debug myfile.py --error "NameError: name 'model' is not defined"
```

---

## 🌐 API Reference

All endpoints accept `POST` with `Content-Type: application/json`.

### `POST /explain`
```json
{
  "code": "def add(a, b): return a + b",
  "mode": "beginner"
}
```
`mode` options: `beginner` | `intermediate` | `advanced`

### `POST /analyze`
```json
{
  "code": "...",
  "use_ai": true
}
```

### `POST /optimize`
```json
{
  "code": "import torch\nfor batch in loader:\n    ...",
  "use_ai": true
}
```

### `POST /debug`
```json
{
  "error": "NameError: name 'x' is not defined",
  "stack_trace": "File train.py line 14...",
  "code": "optional source code"
}
```

---

## 🧪 Local Tests (No AWS Required)

```bash
python tests/test_local.py
```

Runs 20 unit tests on the static analyzer, DL optimizer, and bug explainer — no credentials needed.

---

## 🏗️ Architecture

```
User → API Gateway → Lambda → Amazon Bedrock (Claude / Llama / Mistral)
                           → S3 (store results)
                           → DynamoDB (cache metadata)
```

- **Lambda**: Python 3.11, 512 MB, 60s timeout
- **API Gateway**: 4 POST routes + CORS
- **S3**: Results stored for 30 days (configurable)
- **DynamoDB**: Metadata with 24h TTL (configurable)

---

## 💡 Static Analysis Rules

| Rule | Severity | Description |
|------|----------|-------------|
| PY001 | LOW | Missing function docstring |
| PY004 | HIGH | Bare `except:` clause |
| PY005 | HIGH | Mutable default argument |
| PY009 | MEDIUM | `== None` instead of `is None` |
| NT007 | HIGH | Missing `optimizer.zero_grad()` |
| NT008 | HIGH | `CrossEntropyLoss` + `sigmoid` mismatch |
| NT005 | LOW | Missing `pin_memory=True` in DataLoader |
| NT006 | MEDIUM | Missing `num_workers` in DataLoader |
| NT017 | MEDIUM | No random seed set |
| NT019 | MEDIUM | Saving full model instead of `state_dict()` |
| … | … | 17+ rules total |

---

## 🔒 Security

- `config.env` is in `.gitignore` — **never committed**
- All code stays in your own AWS account
- Lambda IAM role has least-privilege permissions
- No API keys are exposed in the codebase
