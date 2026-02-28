# NeuroTidy — AI-Powered Python Code Analyzer

> **AWS Hackathon 2024** · Powered by **Amazon Bedrock** (Claude 3.5 Haiku)

NeuroTidy explains, analyzes, optimizes, debugs your Python & Deep Learning code — and reviews your GitHub PRs — all serverless on AWS.

---

## Endpoints

| Mode | Endpoint | Description |
|------|----------|-------------|
| **Explain** | `POST /explain` | Multi-level code explanations (beginner → advanced) with 5-section Markdown output |
| **Analyze** | `POST /analyze` | 17+ static analysis rules + ML-specific anti-pattern detection |
| **Optimize** | `POST /optimize` | DL performance optimizer (PyTorch / TensorFlow) with RAG-enhanced analysis |
| **Debug** | `POST /debug` | Root-cause error analysis with step-by-step fixes + `confidence_level` |
| **Review** | `POST /review` | 🤖 GitHub PR Review Bot — auto-analyses pull requests and posts inline comments |

---

## Quick Start

### 1. Configure
Edit `config.env` and fill in:
```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ACCOUNT_ID=...
GITHUB_TOKEN=ghp_...         # For /review endpoint
GITHUB_WEBHOOK_SECRET=...    # Generate: python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Deploy
```powershell
.\infrastructure\deploy.ps1
# The API URL is printed at the end — paste it into config.env as NEUROTIDY_API_ENDPOINT
```

### 3. Test locally (no AWS needed)
```bash
python tests/test_local.py         # 68 tests
python tests/test_github_review.py # 50 tests
```

### 4. Test deployed API
```powershell
.\tests\test_api.ps1   # 8 end-to-end tests against live endpoints
```

---

## CLI Usage

```bash
python cli/neurotidy.py explain  train.py --mode beginner
python cli/neurotidy.py analyze  model.py
python cli/neurotidy.py optimize train.py
python cli/neurotidy.py debug    --error "NameError: name 'x' is not defined"
python cli/neurotidy.py review   --diff changes.diff
python cli/neurotidy.py review   --repo myorg/myrepo --pr 42
```

---

## GitHub PR Review Bot Setup

1. Deploy (step 2 above)
2. In GitHub → **Settings → Webhooks → Add webhook**
   - Payload URL: `https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/review`
   - Content type: `application/json`
   - Secret: your `GITHUB_WEBHOOK_SECRET`
   - Events: ✅ Pull requests

The bot will automatically analyse every opened/updated PR and post inline review comments with severity badges and fix suggestions.

---

## Architecture

```
Developer → curl/CLI/GitHub Webhook
              ↓
         API Gateway  (5 routes)
              ↓
         Lambda Function  (handler.py)
         ├─ /explain   → CodeExplainer  + Bedrock (cached)
         ├─ /analyze   → StaticAnalyzer + Bedrock
         ├─ /optimize  → RAGOptimizer   + Bedrock KB + DLOptimizer
         ├─ /debug     → BugExplainer   + Bedrock (cached)
         └─ /review    → GitHubReviewer + StaticAnalyzer + Bedrock
              ↓                ↓
         DynamoDB cache    S3 (raw results)
```

---

## Key Features

- **Exponential backoff** — auto-retries Bedrock on throttle errors (configurable via env vars)  
- **DynamoDB caching** — SHA-256 code-hash key, 24h TTL — identical code never hits Bedrock twice  
- **GitHub PR bot** — HMAC-SHA256 webhook validation, per-file diff parsing, inline review comments  
- **Zero hardcoded values** — all config via environment variables  
- **118 local unit tests** — run without AWS credentials
