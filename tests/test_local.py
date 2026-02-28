# -*- coding: utf-8 -*-
"""
NeuroTidy Local Unit Tests
Run without AWS credentials -- tests static analysis components only.
Usage: python tests/test_local.py
"""

import sys
import os

# Add lambda folder to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambda'))

from static_analyzer import StaticAnalyzer
from dl_optimizer import DLOptimizer
from bug_explainer import BugExplainer

PASS = '\033[92m[PASS]\033[0m'
FAIL = '\033[91m[FAIL]\033[0m'

passed = 0
failed = 0


def test(name: str, condition: bool):
    global passed, failed
    if condition:
        print(f"  {PASS} {name}")
        passed += 1
    else:
        print(f"  {FAIL} {name}")
        failed += 1


print("\nNeuroTidy Local Tests (no AWS required)\n")

# ─── StaticAnalyzer Tests ─────────────────────────────────
print("--- Static Analyzer ---")
analyzer = StaticAnalyzer()  # No Bedrock client

# Clean code
result = analyzer.analyze("def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b", use_ai=False)
test("Clean code: no violations", len(result['violations']) == 0)
test("Clean code: returns metrics", 'metrics' in result)
test("Clean code: has summary", 'summary' in result)

# Mutable default arg
bad_code = "def process(data=[]):\n    return data"
result = analyzer.analyze(bad_code, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects PY005 (mutable default)", 'PY005' in rule_ids)

# Bare except
bare = "try:\n    x = 1\nexcept:\n    pass"
result = analyzer.analyze(bare, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects PY004 (bare except)", 'PY004' in rule_ids)

# None comparison
none_cmp = "x = None\nif x == None:\n    pass"
result = analyzer.analyze(none_cmp, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects PY009 (== None)", 'PY009' in rule_ids)

# Syntax error handling
syntax_err = analyzer.analyze("def broken(:\n    pass", use_ai=False)
test("Handles syntax error gracefully", 'error' in syntax_err)

# PY006 – print() instead of logging (NEW TEST)
print_code = "def process():\n    print('Processing data')\n    return True"
result = analyzer.analyze(print_code, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects PY006 (print instead of logging)", 'PY006' in rule_ids)

# NT015 – missing model.train() (NEW TEST)
no_train_mode = "optimizer = torch.optim.Adam(model.parameters())\nloss.backward()\noptimizer.step()"
result = analyzer.analyze(no_train_mode, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT015 (missing model.train)", 'NT015' in rule_ids)

# NT021 – missing dropout in deep network (NEW TEST)
deep_no_dropout = "class Net(nn.Module):\n    def __init__(self):\n        self.fc1 = nn.Linear(10, 20)\n        self.fc2 = nn.Linear(20, 30)\n        self.fc3 = nn.Linear(30, 10)"
result = analyzer.analyze(deep_no_dropout, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT021 (missing dropout)", 'NT021' in rule_ids)

# NT024 – no validation set (NEW TEST)
no_validation = "for epoch in range(100):\n    optimizer.step()\n    loss.backward()"
result = analyzer.analyze(no_validation, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT024 (no validation set)", 'NT024' in rule_ids)

# ─── DLOptimizer Tests ──────────────────────────────────
print("\n--- DL Optimizer ---")
optimizer = DLOptimizer()

# Missing zero_grad
no_zero_grad = "for epoch in range(10):\n    output = model(data)\n    loss = criterion(output, target)\n    loss.backward()\n    optimizer.step()"
result = optimizer.analyze(no_zero_grad, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT007 (missing zero_grad)", 'NT007' in rule_ids)

# Missing DataLoader optimizations
dl_code = "loader = DataLoader(dataset, batch_size=32)"
result = optimizer.analyze(dl_code, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT005 (missing pin_memory)", 'NT005' in rule_ids)
test("Detects NT006 (missing num_workers)", 'NT006' in rule_ids)

# CrossEntropyLoss + sigmoid mismatch
mismatch = "criterion = nn.CrossEntropyLoss()\noutput = torch.sigmoid(output)\nloss = criterion(output, labels)"
result = optimizer.analyze(mismatch, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT008 (sigmoid + CrossEntropyLoss)", 'NT008' in rule_ids)

# DL code detection
test("Detects DL code", optimizer._is_dl_code("import torch; model = nn.Module()"))
test("Non-DL code not flagged", not optimizer._is_dl_code("x = 1 + 2"))

# NT010 – missing mixed precision (NEW TEST)
no_amp = "model.train()\nfor data in loader:\n    output = model(data.cuda())\n    loss.backward()\n    optimizer.step()"
result = optimizer.analyze(no_amp, use_ai=False)
rule_ids = [v['rule_id'] for v in result['violations']]
test("Detects NT010 (missing mixed precision)", 'NT010' in rule_ids)

# ─── BugExplainer Tests ──────────────────────────────────
print("\n--- Bug Explainer ---")
explainer = BugExplainer(None, "")

test("Extracts NameError type", explainer._extract_error_type("NameError: name 'x' is not defined") == "NameError")
test("Extracts TypeError type", explainer._extract_error_type("TypeError: unsupported operand") == "TypeError")
test("Extracts RuntimeError type", explainer._extract_error_type("RuntimeError: CUDA error") == "RuntimeError")

# Stack trace line detection
trace = 'File "train.py", line 14, in train\n  File "model.py", line 7, in forward'
lines = explainer._identify_faulty_lines(trace)
test("Extracts faulty lines from trace", lines == [14, 7])

test("Detects DL error (cuda)", explainer._is_dl_error("RuntimeError: CUDA error", ""))
test("No false positive DL error", not explainer._is_dl_error("NameError: name 'x'", ""))

learning_tips = explainer._get_learning_tips("NameError")
test("Returns learning tips for NameError", len(learning_tips) > 0)

# ─── Summary ─────────────────────────────────────────────
print("\n" + "-"*40)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("\033[92mAll tests passed!\033[0m")
else:
    print("\033[91mSome tests failed.\033[0m")
    sys.exit(1)
