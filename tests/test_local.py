# -*- coding: utf-8 -*-
"""
NeuroTidy Local Unit Tests (Extended)
Run without AWS credentials — tests static analysis, DL optimizer,
bug explainer, code explainer prompt structure, bedrock_utils cache helpers,
and GitHub reviewer diff parsing.

Usage:
    python tests/test_local.py
"""

import sys
import os
import hashlib
import time
import unittest.mock as mock

# Add lambda folder to path so we can import the modules
LAMBDA_DIR = os.path.join(os.path.dirname(__file__), '..', 'lambda')
sys.path.insert(0, LAMBDA_DIR)

from static_analyzer import StaticAnalyzer
from dl_optimizer import DLOptimizer
from bug_explainer import BugExplainer
from code_explainer import CodeExplainer
from bedrock_utils import hash_code, get_cached_result, put_cached_result
from github_reviewer import GitHubReviewer

# ─── Test helpers ─────────────────────────────────────────────────────────────
PASS = '\033[92m[PASS]\033[0m'
FAIL = '\033[91m[FAIL]\033[0m'
passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ''):
    global passed, failed
    if condition:
        print(f"  {PASS} {name}")
        passed += 1
    else:
        print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ''))
        failed += 1


# ─── StaticAnalyzer Tests ─────────────────────────────────────────────────────
print("\nNeuroTidy Local Tests (no AWS required)\n")
print("--- Static Analyzer ---")
analyzer = StaticAnalyzer()  # No Bedrock client needed

result = analyzer.analyze("def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b", use_ai=False)
test("Clean code: no violations", len(result['violations']) == 0)
test("Clean code: returns metrics", 'metrics' in result)
test("Clean code: has summary", 'summary' in result)

bad_code = "def process(data=[]):\n    return data"
result = analyzer.analyze(bad_code, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects PY005 (mutable default)", 'PY005' in rule_ids)

bare = "try:\n    x = 1\nexcept:\n    pass"
result = analyzer.analyze(bare, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects PY004 (bare except)", 'PY004' in rule_ids)

none_cmp = "x = None\nif x == None:\n    pass"
result = analyzer.analyze(none_cmp, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects PY009 (== None)", 'PY009' in rule_ids)

syntax_err = analyzer.analyze("def broken(:\n    pass", use_ai=False)
test("Handles syntax error gracefully", 'error' in syntax_err)

print_code = "def process():\n    print('Processing data')\n    return True"
result = analyzer.analyze(print_code, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects PY006 (print instead of logging)", 'PY006' in rule_ids)

no_train_mode = "optimizer = torch.optim.Adam(model.parameters())\nloss.backward()\noptimizer.step()"
result = analyzer.analyze(no_train_mode, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT015 (missing model.train)", 'NT015' in rule_ids)

deep_no_dropout = "class Net(nn.Module):\n    def __init__(self):\n        self.fc1 = nn.Linear(10, 20)\n        self.fc2 = nn.Linear(20, 30)\n        self.fc3 = nn.Linear(30, 10)"
result = analyzer.analyze(deep_no_dropout, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT021 (missing dropout)", 'NT021' in rule_ids)

no_validation = "for epoch in range(100):\n    optimizer.step()\n    loss.backward()"
result = analyzer.analyze(no_validation, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT024 (no validation set)", 'NT024' in rule_ids)

# ─── DLOptimizer Tests ────────────────────────────────────────────────────────
print("\n--- DL Optimizer ---")
optimizer = DLOptimizer()

no_zero_grad = "for epoch in range(10):\n    output = model(data)\n    loss = criterion(output, target)\n    loss.backward()\n    optimizer.step()"
result = optimizer.analyze(no_zero_grad, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT007 (missing zero_grad)", 'NT007' in rule_ids)

dl_code = "loader = DataLoader(dataset, batch_size=32)"
result = optimizer.analyze(dl_code, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT005 (missing pin_memory)", 'NT005' in rule_ids)
test("Detects NT006 (missing num_workers)", 'NT006' in rule_ids)

mismatch = "criterion = nn.CrossEntropyLoss()\noutput = torch.sigmoid(output)\nloss = criterion(output, labels)"
result = optimizer.analyze(mismatch, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT008 (sigmoid + CrossEntropyLoss)", 'NT008' in rule_ids)

test("Detects DL code", optimizer._is_dl_code("import torch; model = nn.Module()"))
test("Non-DL code not flagged", not optimizer._is_dl_code("x = 1 + 2"))

no_amp = "model.train()\nfor data in loader:\n    output = model(data.cuda())\n    loss.backward()\n    optimizer.step()"
result = optimizer.analyze(no_amp, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT010 (missing mixed precision)", 'NT010' in rule_ids)

# ─── BugExplainer Tests ────────────────────────────────────────────────────────
print("\n--- Bug Explainer ---")
explainer = BugExplainer(None, "")

test("Extracts NameError type",
     explainer._extract_error_type("NameError: name 'x' is not defined") == "NameError")
test("Extracts TypeError type",
     explainer._extract_error_type("TypeError: unsupported operand") == "TypeError")
test("Extracts RuntimeError type",
     explainer._extract_error_type("RuntimeError: CUDA error") == "RuntimeError")

trace = 'File "train.py", line 14, in train\n  File "model.py", line 7, in forward'
lines = explainer._identify_faulty_lines(trace)
test("Extracts faulty lines from trace", lines == [14, 7])

test("Detects DL error (cuda)", explainer._is_dl_error("RuntimeError: CUDA error", ""))
test("No false positive DL error", not explainer._is_dl_error("NameError: name 'x'", ""))

learning_tips = explainer._get_learning_tips("NameError")
test("Returns learning tips for NameError", len(learning_tips) > 0)

# New: confidence_level in returned dict
result = explainer.explain_error("NameError: name 'x' is not defined", "", "")
test("explain_error contains confidence_level key", 'confidence_level' in result)
test("explain_error contains related_concepts key", 'related_concepts' in result)
test("explain_error contains is_dl_error key", 'is_dl_error' in result)

# ─── CodeExplainer Prompt Structure Tests ─────────────────────────────────────
print("\n--- Code Explainer (prompt structure, no Bedrock) ---")
ce = CodeExplainer(None, "")

for mode in ('beginner', 'intermediate', 'advanced'):
    structure = ce._analyze_structure("def add(a, b):\n    return a + b")
    prompt = ce._build_prompt("def add(a, b):\n    return a + b", mode, structure)
    test(f"Prompt ({mode}): contains 'What does it do'",
         "What does it do" in prompt)
    test(f"Prompt ({mode}): contains 'Step-by-step walkthrough'",
         "Step-by-step walkthrough" in prompt)
    test(f"Prompt ({mode}): contains 'Key concepts'",
         "Key concepts" in prompt)
    test(f"Prompt ({mode}): contains 'Gotchas'",
         "Gotchas" in prompt)
    test(f"Prompt ({mode}): contains 'Example usage'",
         "Example usage" in prompt)

# Structure analysis
structure = ce._analyze_structure(
    "import torch\nfrom torch import nn\n\nclass MyNet(nn.Module):\n    def forward(self, x): return x"
)
test("Detects imports in structure", 'torch' in structure['imports'])
test("Detects class in structure", 'MyNet' in structure['classes'])
test("Detects method in structure", 'forward' in structure['functions'])

# ─── bedrock_utils Cache Tests ─────────────────────────────────────────────────
print("\n--- bedrock_utils Cache Helpers ---")

# hash_code determinism
h1 = hash_code("def add(a, b): return a+b", "explain")
h2 = hash_code("def add(a, b): return a+b", "explain")
h3 = hash_code("def add(a, b): return a+b", "debug")
test("hash_code is deterministic", h1 == h2)
test("hash_code differs for different actions", h1 != h3)
test("hash_code is 64-char hex",
     len(h1) == 64 and all(c in '0123456789abcdef' for c in h1))

# DynamoDB cache mock tests
mock_table = mock.MagicMock()

# Cache miss (get_item returns no Item)
mock_table.get_item.return_value = {}
result = get_cached_result(mock_table, "somehash", "explain")
test("Cache miss returns None when no item found", result is None)

# Cache hit (valid item with future TTL)
mock_table.get_item.return_value = {
    'Item': {
        'code_hash': 'somehash',
        'action': 'explain',
        'ttl': int(time.time()) + 3600,
        'result': {'explanation': 'Hello world'}
    }
}
result = get_cached_result(mock_table, "somehash", "explain")
test("Cache hit returns result dict", result == {'explanation': 'Hello world'})

# Cache hit expired (TTL in the past)
mock_table.get_item.return_value = {
    'Item': {
        'code_hash': 'somehash',
        'action': 'explain',
        'ttl': int(time.time()) - 100,  # expired
        'result': {'explanation': 'Stale result'}
    }
}
result = get_cached_result(mock_table, "somehash", "explain")
test("Cache ignores expired items (TTL in past)", result is None)

# put_cached_result calls put_item
mock_table.put_item.reset_mock()
put_cached_result(mock_table, "myhash", "explain", {"explanation": "test"}, ttl_seconds=100)
test("put_cached_result calls DynamoDB put_item", mock_table.put_item.called)
call_args = mock_table.put_item.call_args[1]['Item']
test("put_cached_result stores code_hash", call_args.get('code_hash') == 'myhash')
test("put_cached_result stores action", call_args.get('action') == 'explain')
test("put_cached_result stores result", call_args.get('result') == {'explanation': 'test'})
test("put_cached_result sets ttl > now", call_args.get('ttl', 0) > int(time.time()))

# ─── GitHub Reviewer (no network) ─────────────────────────────────────────────
print("\n--- GitHub Reviewer (offline) ---")

# Webhook signature verification
import hashlib as _hl, hmac as _hmac
secret = "mysecret"
body = b'{"action": "opened"}'
correct_sig = "sha256=" + _hmac.new(secret.encode(), body, _hl.sha256).hexdigest()
bad_sig = "sha256=deadbeef"

test("verify_webhook_signature: valid sig returns True",
     GitHubReviewer.verify_webhook_signature(body, correct_sig, secret))
test("verify_webhook_signature: bad sig returns False",
     not GitHubReviewer.verify_webhook_signature(body, bad_sig, secret))
test("verify_webhook_signature: missing header returns False",
     not GitHubReviewer.verify_webhook_signature(body, "", secret))
test("verify_webhook_signature: missing secret returns False",
     not GitHubReviewer.verify_webhook_signature(body, correct_sig, ""))

# Diff parsing
sample_diff = """\
diff --git a/train.py b/train.py
--- a/train.py
+++ b/train.py
@@ -1,3 +1,5 @@
+import torch
+import torch.nn as nn
+
 x = 1
 y = 2
+print(x + y)
"""
reviewer = GitHubReviewer.__new__(GitHubReviewer)  # skip __init__
chunks = GitHubReviewer._parse_diff(sample_diff)
test("Diff parser finds train.py", any(f == 'train.py' for f, _, _ in chunks))
py_chunk = next((c for c in chunks if c[0] == 'train.py'), None)
test("Diff parser captured added lines", py_chunk is not None and len(py_chunk[1]) > 0)
test("Diff parser captured added code", py_chunk is not None and 'torch' in py_chunk[2])

# Violation comment formatting
violation = {
    'rule_id': 'PY006',
    'severity': 'LOW',
    'category': 'best-practice',
    'description': "print() used instead of logging",
    'snippet': "print('hello')",
}
comment_body = GitHubReviewer._format_violation_comment(violation)
test("Violation comment contains rule_id", 'PY006' in comment_body)
test("Violation comment contains severity emoji", '🔵' in comment_body)
test("Violation comment contains snippet", "print('hello')" in comment_body)

# ignored action
reviewer2 = GitHubReviewer.__new__(GitHubReviewer)
reviewer2.token = "faketoken"
reviewer2.bedrock_client = None
reviewer2.model_id = ""
reviewer2.table = None
result = reviewer2.handle_pr_event({"action": "closed"})
test("handle_pr_event ignores 'closed' action", result.get('status') == 'ignored')

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "-" * 50)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print('All tests passed! [OK]')
else:
    print('Some tests FAILED. See above for details.')
    sys.exit(1)
