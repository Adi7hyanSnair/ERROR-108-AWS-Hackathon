"""
Bug Explainer component for NeuroTidy.
Analyzes Python errors and stack traces using Amazon Bedrock.
"""

import json
import ast
import re
from typing import Optional


class BugExplainer:
    """
    Analyzes Python error messages and stack traces to produce
    human-friendly explanations with root-cause analysis and fix suggestions.
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

    def __init__(self, bedrock_client, model_id: str):
        self.bedrock_client = bedrock_client
        self.model_id = model_id

    def explain_error(self, error_msg: str, stack_trace: str = "", code: str = "") -> dict:
        """
        Explain a Python error with root cause analysis and suggested fixes.

        Args:
            error_msg: The error message string (e.g. "NameError: name 'x' is not defined")
            stack_trace: Full stack trace text (optional)
            code: The Python source code that caused the error (optional)

        Returns:
            dict with keys: error_type, root_cause, faulty_lines,
                            explanation, learning_tips, suggested_fixes, ai_explanation
        """
        error_type = self._extract_error_type(error_msg)
        faulty_lines = self._identify_faulty_lines(stack_trace)
        quick_desc = self._get_quick_description(error_msg, error_type)

        # Build AI prompt
        prompt = self._build_prompt(error_msg, stack_trace, code, error_type)
        ai_explanation = self._call_bedrock(prompt)

        return {
            "error_type": error_type,
            "original_error": error_msg,
            "root_cause": quick_desc,
            "faulty_lines": faulty_lines,
            "explanation": ai_explanation,
            "learning_tips": self._get_learning_tips(error_type),
            "suggested_fixes": self._get_suggested_fixes(error_type, error_msg),
            "is_dl_error": self._is_dl_error(error_msg, stack_trace),
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_error_type(self, error_msg: str) -> str:
        """Extract error class name from message."""
        match = re.match(r'^(\w+(?:Error|Exception|Warning|StopIteration|SystemExit))', error_msg)
        if match:
            return match.group(1)
        # Handle "RuntimeError: CUDA …"
        if "RuntimeError" in error_msg:
            return "RuntimeError"
        return "UnknownError"

    def _identify_faulty_lines(self, stack_trace: str) -> list:
        """Extract line numbers referenced in a stack trace."""
        if not stack_trace:
            return []
        pattern = r'File ".*?", line (\d+)'
        matches = re.findall(pattern, stack_trace)
        return [int(m) for m in matches]

    def _get_quick_description(self, error_msg: str, error_type: str) -> str:
        """Return a one-sentence root-cause description."""
        # Check DL-specific patterns first
        for pattern, desc in self.COMMON_ERRORS.items():
            if pattern in error_msg:
                return desc
        return self.COMMON_ERRORS.get(error_type, f"A {error_type} occurred.")

    def _is_dl_error(self, error_msg: str, stack_trace: str) -> bool:
        """Detect if the error is Deep Learning / tensor related."""
        dl_keywords = [
            "torch", "tensorflow", "cuda", "tensor", "gradient",
            "backward", "device", "cuda:0", "RuntimeError: mat1",
            "RuntimeError: Expected", "RuntimeError: size mismatch"
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
        }
        if error_type in fixes:
            return fixes[error_type]
        return ["Add debugging print statements to trace the issue.", "Read the full stack trace carefully."]

    def _build_prompt(self, error_msg: str, stack_trace: str, code: str, error_type: str) -> str:
        """Build detailed prompt for Bedrock."""
        is_dl = self._is_dl_error(error_msg, stack_trace)
        dl_note = "\nNote: This appears to be a Deep Learning / PyTorch / TensorFlow error. Include tensor shape analysis and device placement tips." if is_dl else ""

        prompt = f"""You are NeuroTidy, an expert Python debugging assistant for students and developers.
{dl_note}

Analyze this Python error and provide a clear, educational explanation:

ERROR MESSAGE:
{error_msg}

STACK TRACE:
{stack_trace if stack_trace else "(not provided)"}

CODE:
```python
{code if code else "(not provided)"}
```

Provide your response in this EXACT JSON format:
{{
  "simple_explanation": "One sentence explaining what went wrong in plain English",
  "detailed_explanation": "2-3 paragraphs explaining the root cause and how Python handles this",
  "step_by_step_fix": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "example_fix": "Show corrected code if possible",
  "prevention_tip": "One tip to avoid this error in the future"
}}
"""
        return prompt

    def _call_bedrock(self, prompt: str) -> dict:
        """Call Amazon Bedrock and return parsed response."""
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            text = response_body['content'][0]['text']

            # Try to parse JSON response
            try:
                # Find JSON block in response
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

            return {"simple_explanation": text}

        except Exception as e:
            return {
                "simple_explanation": f"Could not get AI explanation: {str(e)}",
                "details": "Check your Bedrock model access and AWS credentials."
            }
