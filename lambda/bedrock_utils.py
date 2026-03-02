"""
bedrock_utils.py — NeuroTidy shared Bedrock helper.

Provides:
  - call_bedrock_with_retry: invoke Bedrock with exponential backoff on throttle/service errors
  - hash_code: deterministic SHA-256 hash of code + action for cache keys
  - get_cached_result: look up a recent DynamoDB cache entry by (code_hash, action)
  - put_cached_result: write a new cache entry to DynamoDB

NOTE: boto3 and botocore are imported lazily inside each function so this module
can be safely imported in unit-test environments that do not have the AWS SDK installed.
"""

import hashlib
import json
import logging
import os
import time
from typing import Optional

# botocore is only available inside Lambda / when boto3 is installed.
# We stub ClientError so the module can still be imported during local unit tests.
try:
    from botocore.exceptions import ClientError  # noqa: F401
except ImportError:  # pragma: no cover
    class ClientError(Exception):  # type: ignore[no-redef]
        """Stub for botocore.exceptions.ClientError (boto3 not installed)."""
        def __init__(self, *args, **kwargs):
            self.response = {"Error": {"Code": "Stub"}}
            super().__init__(*args, **kwargs)

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Retry configuration (all tuneable via env vars so nothing is hard-coded)
# ---------------------------------------------------------------------------
MAX_RETRIES = int(os.environ.get("BEDROCK_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("BEDROCK_RETRY_BASE_DELAY_SEC", "1.0"))  # seconds
RETRY_MAX_DELAY = float(os.environ.get("BEDROCK_RETRY_MAX_DELAY_SEC", "30.0"))   # seconds

# Bedrock error codes that are safe to retry
_RETRYABLE_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "TooManyRequestsException",
    "InternalServerException",
}


# ---------------------------------------------------------------------------
# Core: Bedrock invocation with exponential backoff
# ---------------------------------------------------------------------------

def call_bedrock_with_retry(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Invoke a model via the Amazon Bedrock Converse API with automatic
    exponential-backoff retries. This is model-agnostic.

    Args:
        bedrock_client: boto3 bedrock-runtime client
        model_id:       Bedrock model ID
        prompt:         User message text
        max_tokens:     Max response tokens
        temperature:    Sampling temperature
        top_p:          Top-p sampling
        system_prompt:  Optional system instruction

    Returns:
        The text content of the model response.
    """
    system = [{"text": system_prompt}] if system_prompt else []
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    
    inference_config = {
        "maxTokens": max_tokens,
        "temperature": temperature,
        "topP": top_p,
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = bedrock_client.converse(
                modelId=model_id,
                messages=messages,
                system=system,
                inferenceConfig=inference_config
            )
            return response["output"]["message"]["content"][0]["text"]

        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in _RETRYABLE_CODES:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                logger.warning(
                    "Bedrock %s on attempt %d/%d — retrying in %.1fs",
                    error_code, attempt, MAX_RETRIES, delay,
                )
                time.sleep(delay)
                last_error = exc
            else:
                logger.error("Non-retryable Bedrock error: %s", exc)
                raise
        except Exception as exc:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            logger.warning("Unexpected error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)
            time.sleep(delay)
            last_error = exc

    raise RuntimeError(f"Bedrock failed after {MAX_RETRIES} retries. Last: {last_error}")


# ---------------------------------------------------------------------------
# Cache helpers — DynamoDB code-hash caching
# ---------------------------------------------------------------------------

def hash_code(code: str, action: str = "") -> str:
    """
    Return a deterministic SHA-256 hex digest for a (code, action) pair.
    Used as the cache key so identical code + action always hits the same entry.
    """
    raw = f"{action}::{code}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_result(table, code_hash: str, action: str) -> Optional[dict]:
    """
    Look up a cached result in DynamoDB.

    The table is keyed on `code_hash` (HASH) and stores `action` as an attribute.
    TTL is managed by DynamoDB native TTL on `ttl` attribute.

    Returns:
        The cached `result` dict if found and not expired, else None.
    """
    try:
        resp = table.get_item(Key={"code_hash": code_hash})
        item = resp.get("Item")
        if item and item.get("action") == action:
            # Double-check TTL in case DynamoDB TTL deletion is slightly delayed
            ttl = int(item.get("ttl", 0))
            if ttl == 0 or ttl > int(time.time()):
                logger.info("Cache HIT for hash=%s action=%s", code_hash[:12], action)
                return item.get("result")
    except ClientError as exc:
        logger.warning("DynamoDB get_item error (cache miss): %s", exc)
    return None


def put_cached_result(
    table,
    code_hash: str,
    action: str,
    result: dict,
    ttl_seconds: int = 86400,
) -> None:
    """
    Write a result to the DynamoDB cache.

    Args:
        table:       boto3 DynamoDB Table resource
        code_hash:   SHA-256 of (action + code) — primary key
        action:      endpoint name (explain, debug, analyze, optimize)
        result:      serialisable dict to cache
        ttl_seconds: seconds until entry expires (default 24 h)
    """
    try:
        table.put_item(Item={
            "code_hash": code_hash,
            "action": action,
            "result": result,
            "cached_at": int(time.time()),
            "ttl": int(time.time()) + ttl_seconds,
        })
        logger.info("Cache WRITE for hash=%s action=%s", code_hash[:12], action)
    except ClientError as exc:
        # Cache writes are best-effort — never block the main response
        logger.warning("DynamoDB put_item error (cache write skipped): %s", exc)
