"""
code_explainer.py — NeuroTidy AI code explanation component.

Uses Amazon Bedrock (Claude) to produce structured, mode-aware explanations.
Integrates with bedrock_utils for:
  - Exponential-backoff retries on Bedrock throttling
  - DynamoDB code-hash response caching
"""

import ast
import os

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
    "You are NeuroTidy, an expert Python educator and code analyst. "
    "Your explanations are clear, accurate, and tailored to the learner's level. "
    "Always respond in well-structured Markdown."
)

_MODE_INSTRUCTIONS = {
    "beginner": (
        "Explain this Python code to a COMPLETE BEGINNER who is just starting to learn programming. "
        "Use plain, everyday language. Avoid jargon. Use analogies where helpful. "
        "Walk through the code line by line if necessary."
    ),
    "intermediate": (
        "Explain this Python code to an INTERMEDIATE developer who understands basic syntax. "
        "Focus on algorithmic logic, design patterns, data flow, and code organisation. "
        "Highlight non-obvious language features or idioms used."
    ),
    "advanced": (
        "Explain this Python code to an ADVANCED developer. "
        "Emphasise performance characteristics, complexity (time/space), design trade-offs, "
        "potential optimisation opportunities, and any security or concurrency concerns."
    ),
}


class CodeExplainer:
    """
    Produces structured, mode-aware explanations of Python source code.
    Supports 'beginner', 'intermediate', and 'advanced' explanation levels.
    Results are cached in DynamoDB by (code_hash, mode) to avoid redundant Bedrock calls.
    """

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

    def explain_code(self, code: str, mode: str = "intermediate") -> str:
        """
        Explain Python code using AI with mode-specific depth.

        Args:
            code: Python source code to explain
            mode: 'beginner' | 'intermediate' | 'advanced'

        Returns:
            Structured Markdown explanation string.
        """
        mode = mode if mode in _MODE_INSTRUCTIONS else "intermediate"
        cache_key = hash_code(code, f"explain:{mode}")

        # 1. Check DynamoDB cache first
        if self.table:
            cached = get_cached_result(self.table, cache_key, f"explain:{mode}")
            if cached:
                return cached.get("explanation", "")

        # 2. Analyse code structure
        structure = self._analyze_structure(code)

        # 3. Build prompt and call Bedrock (with automatic retries)
        prompt = self._build_prompt(code, mode, structure)
        explanation = call_bedrock_with_retry(
            self.bedrock_client,
            self.model_id,
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            system_prompt=_SYSTEM_PROMPT,
        )

        # 4. Write to cache (best-effort, errors are swallowed in put_cached_result)
        if self.table:
            put_cached_result(
                self.table,
                cache_key,
                f"explain:{mode}",
                {"explanation": explanation},
                ttl_seconds=CACHE_TTL,
            )

        return explanation

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyze_structure(self, code: str) -> dict:
        """Extract top-level structure via AST for richer prompt context."""
        try:
            tree = ast.parse(code)
            structure: dict = {
                "functions": [],
                "classes": [],
                "imports": [],
                "decorators": [],
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure["functions"].append(node.name)
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Name):
                            structure["decorators"].append(dec.id)
                        elif isinstance(dec, ast.Attribute):
                            structure["decorators"].append(
                                f"{dec.value.id}.{dec.attr}"
                                if isinstance(dec.value, ast.Name) else dec.attr
                            )
                elif isinstance(node, ast.ClassDef):
                    structure["classes"].append(node.name)
                elif isinstance(node, ast.Import):
                    structure["imports"].extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    structure["imports"].append(node.module)
            return structure
        except Exception:
            return {"functions": [], "classes": [], "imports": [], "decorators": []}

    def _build_prompt(self, code: str, mode: str, structure: dict) -> str:
        """
        Build a richly structured explanation prompt.

        The prompt asks for five clearly labelled sections so the response
        is always consistent and easy to display in the frontend.
        """
        instruction = _MODE_INSTRUCTIONS[mode]

        funcs = ", ".join(structure["functions"]) or "none"
        classes = ", ".join(structure["classes"]) or "none"
        imports = ", ".join(dict.fromkeys(structure["imports"])) or "none"  # deduplicated
        decorators = ", ".join(dict.fromkeys(structure["decorators"])) or "none"

        return f"""{instruction}

---

## Code to explain

```python
{code}
```

**Detected structure:**
| Element    | Names |
|------------|-------|
| Functions  | {funcs} |
| Classes    | {classes} |
| Imports    | {imports} |
| Decorators | {decorators} |

---

## Instructions

Produce a Markdown response with **exactly these five sections** (use the exact headings):

### 1. What does it do?
A 2–3 sentence high-level summary of the code's purpose.

### 2. Key concepts
Bullet-point list of important programming concepts, libraries, or patterns used.
For each concept, give a one-line definition at the right level for the target audience.

### 3. Step-by-step walkthrough
Walk through the logic in order of execution. For {mode} level, adjust depth accordingly.
Use numbered steps and include short inline code snippets where helpful.

### 4. Gotchas & edge cases
Describe potential pitfalls, edge cases, or common mistakes related to this code.
If none are obvious, say so briefly.

### 5. Example usage
Show a short, runnable example of how to use the main function/class from this code.
Include expected output in a comment.

---

Begin your response now:"""
