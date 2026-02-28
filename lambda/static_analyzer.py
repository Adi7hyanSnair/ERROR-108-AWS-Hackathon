"""
Static Analyzer component for NeuroTidy.
Applies rule-based code quality checks with ML/DL-specific rules.
"""

import ast
import json
import re
from typing import List, Dict


SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


class StaticAnalyzer:
    """
    Rule-based static analysis for Python code quality, best practices,
    and ML/DL-specific anti-patterns.
    """

    # Static analysis rules: rule_id -> (description, severity, category)
    RULES = {
        # General Python quality
        "PY001": ("Function has no docstring", "LOW", "documentation"),
        "PY002": ("Function too long (>50 lines)", "MEDIUM", "complexity"),
        "PY003": ("Too many function arguments (>7)", "MEDIUM", "complexity"),
        "PY004": ("Bare except clause (catches all exceptions)", "HIGH", "error-handling"),
        "PY005": ("mutable default argument in function signature", "HIGH", "bug"),
        "PY006": ("print() used instead of logging", "LOW", "best-practice"),
        "PY007": ("Magic number (unexplained literal) used in code", "LOW", "readability"),
        "PY008": ("Variable name too short (single character)", "LOW", "readability"),
        "PY009": ("Comparison to None using == instead of 'is'", "MEDIUM", "bug"),
        "PY010": ("Comparison to True/False using == instead of 'is'", "LOW", "readability"),
        # ML/DL specific
        "NT014": ("Training loop missing explicit loss logging", "LOW", "ml-practice"),
        "NT015": ("Model not set to training mode (model.train())", "MEDIUM", "ml-practice"),
        "NT016": ("Hardcoded learning rate without variable", "LOW", "ml-practice"),
        "NT017": ("No random seed set — results not reproducible", "MEDIUM", "reproducibility"),
        "NT018": ("torch.tensor() used where torch.as_tensor() would avoid copy", "LOW", "performance"),
        "NT019": ("Model saved without state_dict (saving full model)", "MEDIUM", "best-practice"),
        "NT020": ("DataLoader batch_size not configurable (hardcoded)", "LOW", "ml-practice"),
        "NT021": ("Missing dropout layer in deep network", "MEDIUM", "ml-practice"),
        "NT022": ("BatchNorm used without proper training/eval mode switching", "HIGH", "bug"),
        "NT023": ("Learning rate not decayed during training", "LOW", "ml-practice"),
        "NT024": ("No validation set used during training", "MEDIUM", "ml-practice"),
        "NT025": ("Model weights not initialized properly", "MEDIUM", "ml-practice"),
        "NT026": ("Missing early stopping mechanism", "LOW", "ml-practice"),
        "NT027": ("Gradient accumulation without zero_grad at right step", "HIGH", "bug"),
    }

    def __init__(self, bedrock_client=None, model_id: str = ""):
        self.bedrock_client = bedrock_client
        self.model_id = model_id

    def analyze(self, code: str, use_ai: bool = True) -> dict:
        """
        Run full static analysis on Python source code.

        Args:
            code: Python source code string
            use_ai: Whether to include AI-enhanced analysis

        Returns:
            dict with violations, metrics, ai_insights, and summary
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "violations": [],
                "metrics": {},
                "ai_insights": {},
                "summary": f"❌ Syntax Error: {e}",
                "error": str(e)
            }

        violations = self._run_rules(code, tree)
        violations.sort(key=lambda v: SEVERITY_ORDER.get(v['severity'], 99))
        metrics = self._compute_metrics(code, tree)
        ai_insights = {}
        if use_ai and self.bedrock_client:
            ai_insights = self._get_ai_insights(code, violations)

        return {
            "violation_count": len(violations),
            "violations": violations,
            "metrics": metrics,
            "ai_insights": ai_insights,
            "summary": self._build_summary(violations, metrics),
        }

    # -------------------------------------------------------------------------
    # Rule checks
    # -------------------------------------------------------------------------

    def _run_rules(self, code: str, tree: ast.AST) -> List[dict]:
        violations = []
        lines = code.split('\n')

        # PY004 – bare except
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append(self._make_violation("PY004", node.lineno,
                    snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else ""))

        # PY005 – mutable default args
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        violations.append(self._make_violation("PY005", node.lineno,
                            detail=f"Function '{node.name}' has a mutable default argument."))

        # PY001 – missing docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant)):
                    violations.append(self._make_violation("PY001", node.lineno,
                        detail=f"Function '{node.name}' has no docstring."))

        # PY002 – function too long
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_len = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_len > 50:
                    violations.append(self._make_violation("PY002", node.lineno,
                        detail=f"Function '{node.name}' is {func_len} lines (limit: 50)."))

        # PY003 – too many args
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                arg_count = len(node.args.args)
                if arg_count > 7:
                    violations.append(self._make_violation("PY003", node.lineno,
                        detail=f"Function '{node.name}' has {arg_count} arguments (max: 7)."))

        # PY009 – comparison to None with ==
        for i, line in enumerate(lines, 1):
            if re.search(r'==\s*None|None\s*==', line):
                violations.append(self._make_violation("PY009", i, snippet=line.strip()))

        # PY010 – comparison to True/False with ==
        for i, line in enumerate(lines, 1):
            if re.search(r'==\s*True|==\s*False|True\s*==|False\s*==', line):
                violations.append(self._make_violation("PY010", i, snippet=line.strip()))

        # PY006 – print() instead of logging
        for i, line in enumerate(lines, 1):
            if re.search(r'\bprint\s*\(', line) and 'def ' not in line:
                violations.append(self._make_violation("PY006", i, snippet=line.strip()))

        # NT017 – no random seed
        has_seed = any(kw in code for kw in [
            'random.seed', 'torch.manual_seed', 'np.random.seed',
            'tf.random.set_seed', 'seed('])
        has_training = any(kw in code for kw in ['model.train()', 'optimizer', 'criterion'])
        if has_training and not has_seed:
            violations.append(self._make_violation("NT017", 0,
                detail="No random seed set. Add torch.manual_seed(42) for reproducibility."))

        # NT019 – saving full model instead of state_dict
        if 'torch.save' in code and 'state_dict' not in code:
            for i, line in enumerate(lines, 1):
                if 'torch.save' in line:
                    violations.append(self._make_violation("NT019", i, snippet=line.strip()))
                    break

        # NT015 – model not set to training mode
        has_training = any(kw in code for kw in ['optimizer', 'loss.backward', 'criterion'])
        has_train_mode = 'model.train()' in code or '.train()' in code
        if has_training and not has_train_mode:
            violations.append(self._make_violation("NT015", 0,
                detail="Training code detected but model.train() not called."))

        # NT016 – hardcoded learning rate
        for i, line in enumerate(lines, 1):
            if re.search(r'lr\s*=\s*0\.\d+|learning_rate\s*=\s*0\.\d+', line):
                if 'optim.' in code and not re.search(r'lr\s*=\s*\w+|learning_rate\s*=\s*\w+', line):
                    violations.append(self._make_violation("NT016", i, snippet=line.strip()))
                    break

        # NT018 – torch.tensor() instead of torch.as_tensor()
        for i, line in enumerate(lines, 1):
            if 'torch.tensor(' in line and 'torch.as_tensor' not in code:
                violations.append(self._make_violation("NT018", i, snippet=line.strip()))
                break

        # NT020 – hardcoded batch_size in DataLoader
        for i, line in enumerate(lines, 1):
            if 'DataLoader' in line and re.search(r'batch_size\s*=\s*\d+', line):
                violations.append(self._make_violation("NT020", i, snippet=line.strip()))
                break

        # NT021 – missing dropout in deep network
        has_multiple_layers = code.count('nn.Linear') > 2 or code.count('nn.Conv') > 2
        has_dropout = 'nn.Dropout' in code or 'dropout' in code.lower()
        if has_multiple_layers and not has_dropout:
            violations.append(self._make_violation("NT021", 0,
                detail="Deep network detected without dropout layers for regularization."))

        # NT022 – BatchNorm without proper mode switching
        if 'BatchNorm' in code:
            has_eval = '.eval()' in code
            has_train = '.train()' in code
            if not (has_eval and has_train):
                violations.append(self._make_violation("NT022", 0,
                    detail="BatchNorm used but model.train()/model.eval() not properly called."))

        # NT023 – no learning rate decay
        has_optimizer = 'optimizer' in code.lower()
        has_lr_scheduler = any(kw in code for kw in ['scheduler', 'StepLR', 'ReduceLROnPlateau', 'lr_scheduler'])
        if has_optimizer and 'epoch' in code and not has_lr_scheduler:
            violations.append(self._make_violation("NT023", 0,
                detail="Training loop detected without learning rate scheduler."))

        # NT024 – no validation set
        has_training_loop = any(kw in code for kw in ['for epoch', 'optimizer.step', 'loss.backward'])
        has_validation = any(kw in code for kw in ['valid', 'val_', 'validation', 'evaluate'])
        if has_training_loop and not has_validation:
            violations.append(self._make_violation("NT024", 0,
                detail="Training loop without validation set monitoring."))

        # NT025 – weights not initialized
        has_model_class = 'nn.Module' in code
        has_init_weights = any(kw in code for kw in ['init.', 'kaiming', 'xavier', 'normal_', 'uniform_'])
        if has_model_class and not has_init_weights:
            violations.append(self._make_violation("NT025", 0,
                detail="Model class without explicit weight initialization."))

        # NT026 – missing early stopping
        has_long_training = 'epochs' in code and any(str(i) in code for i in range(50, 1000))
        has_early_stop = 'early' in code.lower() and 'stop' in code.lower()
        if has_long_training and not has_early_stop:
            violations.append(self._make_violation("NT026", 0,
                detail="Long training detected without early stopping mechanism."))

        # NT027 – gradient accumulation without proper zero_grad
        if 'accumulation' in code.lower() or 'accum_steps' in code:
            has_conditional_zero_grad = re.search(r'if.*zero_grad|zero_grad.*if', code)
            if not has_conditional_zero_grad:
                violations.append(self._make_violation("NT027", 0,
                    detail="Gradient accumulation detected without conditional zero_grad()."))

        # NT014 – missing loss logging
        has_loss_backward = 'loss.backward()' in code
        has_logging = any(kw in code for kw in ['print', 'logger', 'wandb', 'tensorboard', 'log'])
        if has_loss_backward and not has_logging:
            violations.append(self._make_violation("NT014", 0,
                detail="Training loop without explicit loss logging."))

        return violations

    def _make_violation(self, rule_id: str, line: int,
                        snippet: str = "", detail: str = "") -> dict:
        desc, severity, category = self.RULES[rule_id]
        return {
            "rule_id": rule_id,
            "severity": severity,
            "category": category,
            "description": detail if detail else desc,
            "line_number": line,
            "snippet": snippet,
        }

    # -------------------------------------------------------------------------
    # Code metrics
    # -------------------------------------------------------------------------

    def _compute_metrics(self, code: str, tree: ast.AST) -> dict:
        lines = code.split('\n')
        non_empty = [l for l in lines if l.strip()]
        comment_lines = [l for l in lines if l.strip().startswith('#')]

        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        return {
            "total_lines": len(lines),
            "code_lines": len(non_empty),
            "comment_lines": len(comment_lines),
            "function_count": len(functions),
            "class_count": len(classes),
            "comment_ratio": round(len(comment_lines) / max(len(non_empty), 1) * 100, 1),
        }

    def _build_summary(self, violations: List[dict], metrics: dict) -> str:
        if not violations:
            return f"✅ Code looks clean! ({metrics.get('code_lines', 0)} lines analyzed)"
        high = sum(1 for v in violations if v['severity'] in ('CRITICAL', 'HIGH'))
        medium = sum(1 for v in violations if v['severity'] == 'MEDIUM')
        low = sum(1 for v in violations if v['severity'] in ('LOW', 'INFO'))
        return (f"Found {len(violations)} issues in {metrics.get('code_lines', 0)} lines: "
                f"{high} critical/high, {medium} medium, {low} low.")

    # -------------------------------------------------------------------------
    # AI-enhanced analysis
    # -------------------------------------------------------------------------

    def _get_ai_insights(self, code: str, violations: List[dict]) -> dict:
        """Get AI-powered code quality insights from Bedrock."""
        violation_summary = "\n".join([
            f"- [{v['rule_id']}] {v['description']}" for v in violations[:5]
        ]) or "None found."

        prompt = f"""You are NeuroTidy, a Python code quality expert.
Analyze this code for quality, readability, and best practices.

Already found by static analysis:
{violation_summary}

Code:
```python
{code[:2000]}
```

Return JSON:
{{
  "readability_score": <0-100>,
  "maintainability_score": <0-100>,
  "additional_issues": ["issue 1", "issue 2"],
  "strengths": ["strength 1", "strength 2"],
  "top_recommendation": "single most important improvement"
}}
"""
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 600,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            text = json.loads(response['body'].read())['content'][0]['text']
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            return {"error": str(e)}
        return {}
