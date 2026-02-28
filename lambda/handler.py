"""
NeuroTidy Lambda Handler
Supports 5 endpoints: /explain  /analyze  /optimize  /debug  /review
All config is read from environment variables (set via config.env + template.yaml).
"""

import json
import os
import uuid
from datetime import datetime
import boto3

from code_explainer import CodeExplainer
from bug_explainer import BugExplainer
from dl_optimizer import DLOptimizer
from static_analyzer import StaticAnalyzer
from rag_optimizer import RAGOptimizer
from github_reviewer import GitHubReviewer

# ─── AWS Clients ───────────────────────────────────────────────────────────
REGION = os.environ.get("AWS_REGION", "us-east-1")

s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=REGION)

# ─── Environment Variables ─────────────────────────────────────────────────
S3_BUCKET = os.environ.get("S3_BUCKET", "neurotidy-results")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "neurotidy-cache")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")

table = dynamodb.Table(DYNAMODB_TABLE)

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Hub-Signature-256",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


# ─── Response helpers ───────────────────────────────────────────────────────

def _success(body: dict) -> dict:
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(body, default=str)}


def _error(status: int, message: str) -> dict:
    return {"statusCode": status, "headers": CORS_HEADERS,
            "body": json.dumps({"error": message})}


def _save_to_s3(analysis_id: str, action: str, payload: dict) -> str:
    key = f"analyses/{datetime.utcnow().strftime('%Y/%m/%d')}/{action}/{analysis_id}.json"
    s3_client.put_object(
        Bucket=S3_BUCKET, Key=key,
        Body=json.dumps(payload, indent=2, default=str),
        ContentType="application/json",
    )
    return key


def _cache_metadata(analysis_id: str, action: str, s3_key: str,
                    code_length: int, mode: str = "") -> None:
    table.put_item(Item={
        "analysis_id": analysis_id,
        "action": action, "mode": mode,
        "timestamp": datetime.utcnow().isoformat(),
        "s3_location": f"s3://{S3_BUCKET}/{s3_key}",
        "code_length": code_length,
        "ttl": int(datetime.utcnow().timestamp()) + CACHE_TTL,
    })


# ─── Main handler ───────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Route API Gateway requests to the correct analysis component.

    Supported paths:
      POST /explain  → Code explanation (beginner / intermediate / advanced)
      POST /analyze  → Static code quality analysis
      POST /optimize → DL performance optimization suggestions
      POST /debug    → Bug / error explanation from stack trace
      POST /review   → GitHub PR review bot (webhook endpoint)
    """
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    path = event.get("path", "/explain").rstrip("/")
    action = path.split("/")[-1]

    # ── /review is handled separately (raw body for HMAC, JSON for payload) ──
    if action == "review":
        return _handle_review(event)

    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _error(400, "Invalid JSON body")

    code = body.get("code", "").strip()

    if not code and action != "debug":
        return _error(400, "Field 'code' is required")

    analysis_id = str(uuid.uuid4())

    try:
        if action == "explain":
            mode = body.get("mode", "intermediate")
            # Pass the DynamoDB table so CodeExplainer can use cache
            explainer = CodeExplainer(bedrock_client, BEDROCK_MODEL_ID, dynamodb_table=table)
            result = explainer.explain_code(code, mode)
            payload = {"analysis_id": analysis_id, "action": "explain",
                       "mode": mode, "code": code, "explanation": result,
                       "timestamp": datetime.utcnow().isoformat()}
            s3_key = _save_to_s3(analysis_id, action, payload)
            _cache_metadata(analysis_id, action, s3_key, len(code), mode)
            return _success({"analysis_id": analysis_id, "explanation": result,
                             "s3_location": f"s3://{S3_BUCKET}/{s3_key}"})

        elif action == "analyze":
            use_ai = body.get("use_ai", True)
            analyzer = StaticAnalyzer(bedrock_client, BEDROCK_MODEL_ID)
            result = analyzer.analyze(code, use_ai=use_ai)
            payload = {"analysis_id": analysis_id, "action": "analyze",
                       "code": code, "result": result,
                       "timestamp": datetime.utcnow().isoformat()}
            s3_key = _save_to_s3(analysis_id, action, payload)
            _cache_metadata(analysis_id, action, s3_key, len(code))
            return _success({"analysis_id": analysis_id, **result,
                             "s3_location": f"s3://{S3_BUCKET}/{s3_key}"})

        elif action == "optimize":
            use_ai = body.get("use_ai", True)
            mode = body.get("mode", "intermediate")
            use_rag = body.get("use_rag", True)

            if use_rag:
                optimizer = RAGOptimizer(bedrock_client, bedrock_agent_client, BEDROCK_MODEL_ID, KNOWLEDGE_BASE_ID)
                result = optimizer.analyze(code, mode=mode, use_ai=use_ai)
            else:
                optimizer = DLOptimizer(bedrock_client, BEDROCK_MODEL_ID)
                result = optimizer.analyze(code, use_ai=use_ai)

            payload = {"analysis_id": analysis_id, "action": "optimize",
                       "mode": mode, "code": code, "result": result,
                       "timestamp": datetime.utcnow().isoformat()}
            s3_key = _save_to_s3(analysis_id, action, payload)
            _cache_metadata(analysis_id, action, s3_key, len(code), mode)
            return _success({"analysis_id": analysis_id, **result,
                             "s3_location": f"s3://{S3_BUCKET}/{s3_key}"})

        elif action == "debug":
            error_msg = body.get("error", "")
            stack_trace = body.get("stack_trace", "")
            if not error_msg and not code:
                return _error(400, "Provide 'error' and optionally 'code' and 'stack_trace'")
            # Pass the DynamoDB table so BugExplainer can use cache
            explainer = BugExplainer(bedrock_client, BEDROCK_MODEL_ID, dynamodb_table=table)
            result = explainer.explain_error(error_msg, stack_trace, code)
            payload = {"analysis_id": analysis_id, "action": "debug",
                       "error": error_msg, "result": result,
                       "timestamp": datetime.utcnow().isoformat()}
            s3_key = _save_to_s3(analysis_id, action, payload)
            _cache_metadata(analysis_id, action, s3_key, len(code))
            return _success({"analysis_id": analysis_id, **result,
                             "s3_location": f"s3://{S3_BUCKET}/{s3_key}"})

        else:
            return _error(404, f"Unknown action '{action}'. Use: /explain /analyze /optimize /debug /review")

    except Exception as e:
        print(f"[ERROR] {action}: {e}")
        return _error(500, f"Internal error: {str(e)}")


def _handle_review(event: dict) -> dict:
    """
    Handle GitHub PR review webhook.

    Validates HMAC-SHA256 signature then delegates to GitHubReviewer.
    """
    if not GITHUB_TOKEN:
        return _error(503, "GitHub integration not configured: GITHUB_TOKEN missing")

    raw_body = event.get("body") or ""
    if isinstance(raw_body, str):
        raw_body_bytes = raw_body.encode("utf-8")
    else:
        raw_body_bytes = raw_body

    # Validate webhook signature when a secret is configured
    if GITHUB_WEBHOOK_SECRET:
        signature = (event.get("headers") or {}).get("X-Hub-Signature-256", "")
        if not GitHubReviewer.verify_webhook_signature(raw_body_bytes, signature, GITHUB_WEBHOOK_SECRET):
            return _error(401, "Invalid webhook signature")

    # Parse event type — only handle pull_request events
    github_event = (event.get("headers") or {}).get("X-GitHub-Event", "")
    if github_event and github_event != "pull_request":
        return _success({"status": "ignored", "reason": f"Event type '{github_event}' not handled"})

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return _error(400, "Invalid JSON in webhook body")

    try:
        reviewer = GitHubReviewer(
            github_token=GITHUB_TOKEN,
            bedrock_client=bedrock_client,
            model_id=BEDROCK_MODEL_ID,
            dynamodb_table=table,
        )
        result = reviewer.handle_pr_event(payload)
        return _success(result)

    except Exception as exc:
        print(f"[ERROR] review: {exc}")
        return _error(500, f"Review failed: {str(exc)}")
