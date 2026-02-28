"""
bug_explainer.py — NeuroTidy error analysis component.

Analyses Python errors and stack traces using Amazon Bedrock (Claude).
Integrates with bedrock_utils for:
  - Exponential-backoff retries on Bedrock throttling
  - DynamoDB code-hash response caching
"""

import json
import os
import re
from typing import Optional

from bedrock_utils import (
    call_bedrock_with_retry,
    get_cached_result,
    hash_code,
    put_cached_result,
)

# Configurable from Lambda environment — no hard-coded values
MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "2000"))
TEMPERATURE = float(os.environ.get("BEDROCK_TEMPERATURE", "0.1"))
TOP_P = float(os.environ.get("BEDROCK_TOP_P", "0.9"))
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))

_SYSTEM_PROMPT = (
    "You are NeuroTidy, an expert Python debugging assistant specialising in helping "
    "students and junior developers understand and fix errors. "
    "You are especially knowledgeable about Deep Learning frameworks (PyTorch, TensorFlow, JAX). "
    "Always respond with valid, parseable JSON — no markdown fences, no extra prose."
)


class BugExplainer:
    """
    Analyses Python error messages and stack traces to produce
    human-friendly explanations with root-cause analysis and fix suggestions.
    Results are cached in DynamoDB by (error_hash, action) to skip redundant Bedrock calls.
    """

    COMMON_ERRORS = {
        "NameError": "A variable or function is referenced before it was defined.",
        "TypeError": "An operation was applied to a value of the wrong type.",
        "IndexError": "A list/tuple index is out of range.",
        "KeyError": "A dictionary key does not exist.",
        "AttributeError": "An object does not have the referenced attribute or method.",
        "ValueError": "A function received an argument of the right type but wrong value.",
        "ImportError": "A module could not be imported.",
        "ModuleNotFoundError": "The specified module is not installed or not found.",
        "ZeroDivisionError": "Division (or modulo) by zero was attempted.",
        "RecursionError": "Maximum recursion depth was exceeded.",
        "MemoryError": "The program ran out of memory.",
        "FileNotFoundError": "A file or directory was not found.",
        "PermissionError": "Operating system denied access to a file.",
        "RuntimeError": "A general runtime error occurred.",
        "IndentationError": "The code has incorrect indentation.",
        "SyntaxError": "The code has invalid Python syntax.",
        "StopIteration": "An iterator has no more items.",
        "AssertionError": "An assert statement failed.",
        "OverflowError": "A numeric value is too large to represent.",
        # Deep Learning specific
        "RuntimeError: CUDA": "A GPU/CUDA operation failed — often a tensor device mismatch.",
        "RuntimeError: size": "Tensor dimension mismatch — shapes are incompatible.",
        "RuntimeError: Expected": "Tensor type or shape mismatch.",
        "RuntimeError: mat1": "Matrix multiplication shape mismatch.",
        "RuntimeError: stack": "Tensors being stacked have different shapes.",
    }

    def __init__(self, bedrock_client, model_id: str, dynamodb_table=None):
        """
        Args:
            bedrock_client:  boto3 bedrock-runtime client
            model_id:        Bedrock model ID (read from env, never hardcoded here)
            dynamodb_table:  DynamoDB Table resource for caching (optional)
        """
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.table = dynamodb_table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_error(
        self,
        error_msg: str,
        stack_trace: str = "",
        code: str = "",
    ) -> dict:
        """
        Explain a Python error with root-cause analysis and suggested fixes.

        Args:
            error_msg:   The error message string (e.g. "NameError: name 'x' is not defined")
            stack_trace: Full stack trace text (optional)
            code:        Python source code that caused the error (optional)

        Returns:
            dict with keys: error_type, original_error, root_cause, faulty_lines,
                            explanation (from AI), learning_tips, suggested_fixes,
                            is_dl_error, confidence_level, related_concepts
        """
        error_type = self._extract_error_type(error_msg)
        faulty_lines = self._identify_faulty_lines(stack_trace)
        quick_desc = self._get_quick_description(error_msg, error_type)
        is_dl = self._is_dl_error(error_msg, stack_trace)

        # Cache key is derived from the combined error context
        cache_input = f"{error_msg}::{stack_trace}::{code}"
        cache_key = hash_code(cache_input, "debug")
        ai_explanation: dict = {}

        # 1. Check DynamoDB cache
        if self.table:
            cached = get_cached_result(self.table, cache_key, "debug")
            if cached:
                ai_explanation = cached.get("ai_explanation", {})

        # 2. Call Bedrock if not cached
        if not ai_explanation and self.bedrock_client:
            prompt = self._build_prompt(error_msg, stack_trace, code, error_type, is_dl)
            raw_text = call_bedrock_with_retry(
                self.bedrock_client,
                self.model_id,
                prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                system_prompt=_SYSTEM_PROMPT,
            )
            ai_explanation = self._parse_ai_response(raw_text)

            # 3. Write to cache (best-effort)
            if self.table:
                put_cached_result(
                    self.table,
                    cache_key,
                    "debug",
                    {"ai_explanation": ai_explanation},
                    ttl_seconds=CACHE_TTL,
                )

        return {
            "error_type": error_type,
            "original_error": error_msg,
            "root_cause": quick_desc,
            "faulty_lines": faulty_lines,
            "explanation": ai_explanation,
            "learning_tips": self._get_learning_tips(error_type),
            "suggested_fixes": self._get_suggested_fixes(error_type, error_msg),
            "is_dl_error": is_dl,
            "confidence_level": ai_explanation.get("confidence_level", "medium"),
            "related_concepts": ai_explanation.get("related_concepts", []),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_error_type(self, error_msg: str) -> str:
        """Extract error class name from message."""
        match = re.match(
            r"^(\w+(?:Error|Exception|Warning|StopIteration|SystemExit))", error_msg
        )
        if match:
            return match.group(1)
        if "RuntimeError" in error_msg:
            return "RuntimeError"
        return "UnknownError"

    def _identify_faulty_lines(self, stack_trace: str) -> list:
        """Extract line numbers referenced in a stack trace."""
        if not stack_trace:
            return []
        matches = re.findall(r'File ".*?", line (\d+)', stack_trace)
        return [int(m) for m in matches]

    def _get_quick_description(self, error_msg: str, error_type: str) -> str:
        """Return a one-sentence root-cause description from the local lookup table."""
        for pattern, desc in self.COMMON_ERRORS.items():
            if pattern in error_msg:
                return desc
        return self.COMMON_ERRORS.get(error_type, f"A {error_type} occurred.")

    def _is_dl_error(self, error_msg: str, stack_trace: str) -> bool:
        """Detect if the error is Deep Learning / tensor related."""
        dl_keywords = [
            "torch", "tensorflow", "cuda", "tensor", "gradient",
            "backward", "device", "cuda:0", "RuntimeError: mat1",
            "RuntimeError: Expected", "RuntimeError: size mismatch",
            "jax", "keras", "autograd", "nn.Module",
        ]
        combined = (error_msg + " " + stack_trace).lower()
        return any(kw.lower() in combined for kw in dl_keywords)

    def _get_learning_tips(self, error_type: str) -> list:
        """Return educational tips for preventing this error type."""
        tips = {
            "NameError": [
                "Always define variables before using them.",
                "Check for typos in variable names — Python is case-sensitive.",
                "Ensure the variable is in the correct scope (function vs global).",
            ],
            "TypeError": [
                "Use type() or isinstance() to check types before operations.",
                "Read function documentation to understand expected argument types.",
                "Use type hints to catch type errors early.",
            ],
            "IndexError": [
                "Always check len(list) before accessing by index.",
                "Use negative indexing carefully: list[-1] is the last element.",
                "Use enumerate() instead of manual index counters.",
            ],
            "KeyError": [
                "Use dict.get(key, default) to safely access dictionary values.",
                "Check 'if key in dict' before accessing.",
                "Use collections.defaultdict for dictionaries with default values.",
            ],
            "RuntimeError": [
                "For tensor errors: print tensor shapes before operations with tensor.shape.",
                "Ensure all tensors are on the same device (CPU vs GPU).",
                "Use try/except blocks around GPU operations for better error messages.",
            ],
            "ImportError": [
                "Install the missing package with: pip install <package-name>",
                "Check if you're using the correct virtual environment.",
                "Verify the package name (e.g. 'sklearn' is installed as 'scikit-learn').",
            ],
            "AttributeError": [
                "Use dir(object) to inspect available attributes and methods.",
                "Check for None values before calling methods on objects.",
                "Verify the object type matches what you expect.",
            ],
            "ValueError": [
                "Validate input data before passing it to functions.",
                "Check function documentation for valid value ranges.",
                "Add assertions to catch invalid values early in your code.",
            ],
        }
        return tips.get(error_type, [
            "Read the full error message carefully — it usually tells you exactly what's wrong.",
            "Add print() statements to inspect variable values right before the error.",
            "Use Python's built-in help() function to learn about built-in functions.",
        ])

    def _get_suggested_fixes(self, error_type: str, error_msg: str) -> list:
        """Return concrete fix suggestions."""
        fixes = {
            "NameError": [
                "Define the variable before the line that uses it.",
                "Check for spelling mistakes in the variable name.",
            ],
            "TypeError": [
                "Convert the value to the expected type (e.g. int(), str(), float()).",
                "Check function signature with help(function_name).",
            ],
            "IndexError": [
                "Use: if index < len(my_list): my_list[index]",
                "Iterate with 'for item in list' instead of index-based access.",
            ],
            "KeyError": [
                "Use: value = my_dict.get('key', 'default_value')",
                "Use: if 'key' in my_dict: value = my_dict['key']",
            ],
            "RuntimeError": [
                "Print tensor shapes: print(tensor_a.shape, tensor_b.shape)",
                "Move all tensors to same device: tensor.to(device)",
                "Reshape tensors: tensor.view(batch_size, -1)",
            ],
            "ModuleNotFoundError": [
                "Run: pip install <module_name>",
                "Activate your virtual environment first: source venv/bin/activate",
            ],
            "AttributeError": [
                "Check spelling: use dir(object) to list available attributes.",
                "Guard against None: if obj is not None: obj.method()",
            ],
            "ValueError": [
                "Validate inputs before passing them to the function.",
                "Check the function's documentation for expected value ranges.",
            ],
        }
        return fixes.get(error_type, [
            "Add debugging print statements to trace the issue.",
            "Read the full stack trace carefully.",
        ])

    def _build_prompt(
        self,
        error_msg: str,
        stack_trace: str,
        code: str,
        error_type: str,
        is_dl: bool,
    ) -> str:
        """Build an enhanced, richly-structured debug prompt for Bedrock."""
        dl_note = (
            "\n⚠️  DL/ML CONTEXT: This appears to be a Deep Learning error "
            "(PyTorch / TensorFlow / JAX). Include tensor shape analysis, "
            "device placement tips, and gradient-related guidance where relevant."
            if is_dl else ""
        )

        return f"""Analyse this Python error and return a JSON object (no markdown, no prose outside JSON).
{dl_note}

ERROR MESSAGE:
{error_msg}

STACK TRACE:
{stack_trace if stack_trace else "(not provided)"}

CODE:
```python
{code if code else "(not provided)"}
```

Return EXACTLY this JSON schema — all fields are required:
{{
  "simple_explanation": "One sentence explaining what went wrong in plain English",
  "detailed_explanation": "2-3 paragraphs: root cause, why Python raises this, how execution reaches it",
  "step_by_step_fix": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "example_fix": "Show corrected code snippet if the original code was provided, else show a generic fix pattern",
  "prevention_tip": "One actionable tip to avoid this error class in the future",
  "confidence_level": "high | medium | low — how confident you are in the root-cause diagnosis",
  "related_concepts": ["concept_1", "concept_2"]
}}"""

    def _parse_ai_response(self, raw_text: str) -> dict:
        """Parse the AI JSON response, with graceful fallback."""
        try:
            # Prefer a clean JSON block
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        # Fallback: return raw text under simple_explanation
        return {
            "simple_explanation": raw_text,
            "confidence_level": "low",
            "related_concepts": [],
        }
