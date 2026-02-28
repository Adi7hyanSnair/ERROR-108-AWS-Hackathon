"""
Deep Learning Optimizer component for NeuroTidy.
Detects inefficient patterns in ML/DL Python code and suggests optimizations.
"""

import ast
import json
import re
from typing import List, Dict, Optional


class DLOptimizer:
    """
    Analyzes Python ML/DL code for performance anti-patterns and
    suggests concrete, measurable optimizations.
    """

    # Rule definitions: (rule_id, description, severity, fix_hint)
    RULES = {
        "NT001": ("Missing torch.no_grad() in inference", "HIGH",
                  "Wrap inference code with: with torch.no_grad(): ..."),
        "NT002": ("Missing model.eval() before inference", "HIGH",
                  "Call model.eval() before running inference to disable dropout/batchnorm training behavior."),
        "NT003": ("CPU-to-GPU transfer in training loop", "MEDIUM",
                  "Move .to(device) calls outside the training loop to avoid repeated transfers."),
        "NT004": ("Detaching tensors with .cpu() inside loop", "MEDIUM",
                  "Use .detach().item() instead of .cpu().numpy() inside tight loops."),
        "NT005": ("Missing pin_memory=True in DataLoader", "LOW",
                  "Add pin_memory=True to DataLoader for faster GPU data transfer."),
        "NT006": ("Missing num_workers in DataLoader", "MEDIUM",
                  "Add num_workers=4 (or more) to DataLoader to parallelize data loading."),
        "NT007": ("optimizer.zero_grad() missing in training loop", "HIGH",
                  "Call optimizer.zero_grad() before loss.backward() to clear accumulated gradients."),
        "NT008": ("Loss function mismatch: CrossEntropyLoss with sigmoid output", "HIGH",
                  "CrossEntropyLoss expects raw logits, not sigmoid probabilities. Remove sigmoid or use BCELoss."),
        "NT009": ("Unnecessary .clone() before in-place operation", "LOW",
                  "Remove .clone() if you immediately overwrite the tensor."),
        "NT010": ("Missing mixed precision training (torch.cuda.amp)", "LOW",
                  "Use torch.cuda.amp.autocast() and GradScaler for 2x faster training on supported GPUs."),
        "NT011": ("Large model parameters computed inside loop", "MEDIUM",
                  "Move model.parameters() or .named_parameters() calls outside of loops."),
        "NT012": ("Python list used instead of torch.tensor for gradients", "MEDIUM",
                  "Create tensors directly with torch.tensor() instead of converting Python lists inside loops."),
        "NT013": ("Missing gradient clipping for RNN/LSTM", "LOW",
                  "Add torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) before optimizer.step()."),
        "NT014": ("model.training not checked before model.eval()", "LOW",
                  "Check if model.training is True before switching modes to avoid unnecessary state changes."),
    }

    def __init__(self, bedrock_client=None, model_id: str = ""):
        """
        Args:
            bedrock_client: Optional Bedrock client for AI-enhanced analysis.
            model_id: Bedrock model ID string.
        """
        self.bedrock_client = bedrock_client
        self.model_id = model_id

    def analyze(self, code: str, use_ai: bool = True) -> dict:
        """
        Detect optimization opportunities in ML/DL Python code.

        Args:
            code: Python source code string.
            use_ai: Whether to also call Bedrock for additional insights.

        Returns:
            dict with 'violations', 'optimizations', 'ai_insights', 'summary'
        """
        violations = self._run_static_rules(code)
        ai_insights = {}
        if use_ai and self.bedrock_client:
            ai_insights = self._get_ai_optimizations(code)

        return {
            "violation_count": len(violations),
            "violations": violations,
            "ai_insights": ai_insights,
            "summary": self._build_summary(violations),
            "is_dl_code": self._is_dl_code(code),
        }

    # -------------------------------------------------------------------------
    # Static rule engine
    # -------------------------------------------------------------------------

    def _run_static_rules(self, code: str) -> List[dict]:
        """Apply all static rules to the code and return violations."""
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        lines = code.split('\n')

        violations += self._check_no_grad(tree, code)
        violations += self._check_model_eval(tree, code)
        violations += self._check_zero_grad(tree, code)
        violations += self._check_dataloader(tree, code)
        violations += self._check_inside_loop(tree, code)
        violations += self._check_loss_mismatch(code)
        violations += self._check_mixed_precision(code)
        violations += self._check_gradient_clipping(tree, code)

        return violations

    def _check_no_grad(self, tree: ast.AST, code: str) -> List[dict]:
        """NT001: Check for missing torch.no_grad() in inference code."""
        results = []
        has_inference_keywords = any(kw in code for kw in ['predict', 'inference', 'eval', 'test'])
        has_no_grad = 'no_grad' in code

        if has_inference_keywords and not has_no_grad:
            results.append(self._make_violation("NT001", 0))
        return results

    def _check_model_eval(self, tree: ast.AST, code: str) -> List[dict]:
        """NT002: Check for missing model.eval() before predictions."""
        results = []
        if ('model(' in code or '.forward(' in code) and '.eval()' not in code and 'train(' not in code:
            # likely inference without eval mode
            if any(kw in code for kw in ['predict', 'test', 'inference', 'val']):
                results.append(self._make_violation("NT002", 0))
        return results

    def _check_zero_grad(self, tree: ast.AST, code: str) -> List[dict]:
        """NT007: Check for missing optimizer.zero_grad() in training loops."""
        results = []
        has_backward = 'backward()' in code
        has_zero_grad = 'zero_grad()' in code
        if has_backward and not has_zero_grad:
            # find line of backward() call
            for i, line in enumerate(code.split('\n')):
                if 'backward()' in line:
                    results.append(self._make_violation("NT007", i + 1))
                    break
        return results

    def _check_dataloader(self, tree: ast.AST, code: str) -> List[dict]:
        """NT005+NT006: Check DataLoader for missing optimizations."""
        results = []
        if 'DataLoader' in code:
            if 'pin_memory' not in code:
                results.append(self._make_violation("NT005", 0))
            if 'num_workers' not in code:
                results.append(self._make_violation("NT006", 0))
        return results

    def _check_inside_loop(self, tree: ast.AST, code: str) -> List[dict]:
        """NT003+NT004: Check for expensive ops inside loops."""
        results = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_src = ast.get_source_segment(code, node) or ""
                if '.to(device)' in loop_src or '.cuda()' in loop_src:
                    results.append(self._make_violation("NT003", getattr(node, 'lineno', 0)))
        return results

    def _check_loss_mismatch(self, code: str) -> List[dict]:
        """NT008: Detect CrossEntropyLoss + sigmoid anti-pattern."""
        results = []
        if 'CrossEntropyLoss' in code and ('sigmoid' in code or 'Sigmoid' in code):
            results.append(self._make_violation("NT008", 0))
        return results

    def _check_mixed_precision(self, code: str) -> List[dict]:
        """NT010: Check for missing mixed precision training."""
        results = []
        has_training = any(kw in code for kw in ['optimizer.step', 'loss.backward', 'model.train'])
        has_cuda = 'cuda' in code or '.to(device)' in code or 'gpu' in code.lower()
        has_amp = 'autocast' in code or 'GradScaler' in code or 'torch.cuda.amp' in code
        
        if has_training and has_cuda and not has_amp:
            results.append(self._make_violation("NT010", 0))
        return results

    def _check_gradient_clipping(self, tree: ast.AST, code: str) -> List[dict]:
        """NT013: Check for missing gradient clipping in RNN/LSTM."""
        results = []
        has_rnn = any(kw in code for kw in ['LSTM', 'GRU', 'RNN', 'nn.LSTM', 'nn.GRU', 'nn.RNN'])
        has_clipping = 'clip_grad' in code or 'clip_grad_norm' in code or 'clip_grad_value' in code
        
        if has_rnn and not has_clipping:
            results.append(self._make_violation("NT013", 0))
        return results

    def _is_dl_code(self, code: str) -> bool:
        """Detect if the code contains ML/DL patterns."""
        dl_indicators = ['torch', 'tensorflow', 'keras', 'nn.Module', 'DataLoader',
                         'nn.Linear', 'Conv2d', 'backward()', 'optimizer', 'criterion']
        return any(ind in code for ind in dl_indicators)

    def _make_violation(self, rule_id: str, line: int) -> dict:
        """Build a violation dict from a rule ID."""
        desc, severity, fix = self.RULES[rule_id]
        return {
            "rule_id": rule_id,
            "severity": severity,
            "description": desc,
            "line_number": line,
            "suggested_fix": fix,
        }

    def _build_summary(self, violations: List[dict]) -> str:
        """Build a human-readable summary of findings."""
        if not violations:
            return "✅ No optimization issues detected."
        high = sum(1 for v in violations if v['severity'] == 'HIGH')
        medium = sum(1 for v in violations if v['severity'] == 'MEDIUM')
        low = sum(1 for v in violations if v['severity'] == 'LOW')
        return (f"Found {len(violations)} optimization opportunities: "
                f"{high} HIGH, {medium} MEDIUM, {low} LOW severity.")

    # -------------------------------------------------------------------------
    # AI-powered analysis
    # -------------------------------------------------------------------------

    def _get_ai_optimizations(self, code: str) -> dict:
        """Call Bedrock for additional AI-powered optimization insights."""
        prompt = f"""You are NeuroTidy, a deep learning performance expert.
Analyze this Python ML/DL code for performance optimization opportunities.

```python
{code}
```

Return JSON with this exact structure:
{{
  "performance_score": <0-100, where 100 is perfectly optimized>,
  "top_issues": ["issue 1", "issue 2", "issue 3"],
  "quick_wins": ["easy fix 1", "easy fix 2"],
  "advanced_optimizations": ["advanced optimization 1", "advanced optimization 2"],
  "estimated_speedup": "e.g. 20-40% faster with suggested changes"
}}
"""
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 800,
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
