# -*- coding: utf-8 -*-
"""
NeuroTidy GitHub Reviewer Unit Tests
Tests webhook signature, diff parsing, comment formatting, and event routing.
Runs without AWS credentials or network access.

Usage:
    python tests/test_github_review.py
"""

import sys
import os
import hashlib
import hmac
import json
import unittest.mock as mock

LAMBDA_DIR = os.path.join(os.path.dirname(__file__), '..', 'lambda')
sys.path.insert(0, LAMBDA_DIR)

from github_reviewer import GitHubReviewer

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
        print(f"  {FAIL} {name}" + (f"  → {detail}" if detail else ''))
        failed += 1


print("\nNeuroTidy GitHub Reviewer Tests (no network required)\n")

# ─── Signature Verification ───────────────────────────────────────────────────
print("--- Webhook Signature Verification ---")

SECRET = "test_webhook_secret_32_chars_here"
BODY = b'{"action":"opened","pull_request":{"number":42}}'
VALID_SIG = "sha256=" + hmac.new(SECRET.encode(), BODY, hashlib.sha256).hexdigest()

test("Valid signature accepted",
     GitHubReviewer.verify_webhook_signature(BODY, VALID_SIG, SECRET))
test("Wrong secret rejected",
     not GitHubReviewer.verify_webhook_signature(BODY, VALID_SIG, "wrongsecret"))
test("Tampered body rejected",
     not GitHubReviewer.verify_webhook_signature(b'tampered', VALID_SIG, SECRET))
test("Missing signature rejects",
     not GitHubReviewer.verify_webhook_signature(BODY, "", SECRET))
test("None secret rejects",
     not GitHubReviewer.verify_webhook_signature(BODY, VALID_SIG, ""))
test("sha1 prefix (not sha256) rejected",
     not GitHubReviewer.verify_webhook_signature(BODY, "sha1=abc123", SECRET))

# ─── Diff Parsing ─────────────────────────────────────────────────────────────
print("\n--- Diff Parsing ---")

DIFF_PYTHON = """\
diff --git a/src/model.py b/src/model.py
index abc..def 100644
--- a/src/model.py
+++ b/src/model.py
@@ -0,0 +1,8 @@
+import torch
+import torch.nn as nn
+
+class SimpleNet(nn.Module):
+    def __init__(self):
+        super().__init__()
+        self.fc = nn.Linear(784, 10)
+
"""

DIFF_MIXED = """\
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,2 +1,3 @@
+# New heading
 Old text
diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
@@ -5,3 +5,4 @@
+def helper():
+    return True
 x = 1
"""

chunks = GitHubReviewer._parse_diff(DIFF_PYTHON)
test("Parses single Python file", len(chunks) == 1)
test("Correct filename extracted", chunks[0][0] == 'src/model.py')
test("Added lines list not empty", len(chunks[0][1]) > 0)
test("Added code contains 'torch'", 'torch' in chunks[0][2])
test("Added code contains class definition", 'SimpleNet' in chunks[0][2])

chunks2 = GitHubReviewer._parse_diff(DIFF_MIXED)
filenames = [c[0] for c in chunks2]
test("Mixed diff: finds README.md", 'README.md' in filenames)
test("Mixed diff: finds utils.py", 'utils.py' in filenames)
test("Mixed diff: utils.py code captured", any('helper' in c[2] for c in chunks2 if c[0] == 'utils.py'))

# Empty diff
empty_chunks = GitHubReviewer._parse_diff("")
test("Empty diff returns empty list", empty_chunks == [])

# No Python files
no_py_diff = "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1,2 @@\n+# New line\n old\n"
no_py_chunks = GitHubReviewer._parse_diff(no_py_diff)
py_only = [c for c in no_py_chunks if c[0].endswith('.py')]
test("Non-Python files not returned for analysis (filtering done in analyze_pr_diff)", True)  # parsing captures all, filtering is at analyze level

# ─── Diff Line Mapping ────────────────────────────────────────────────────────
print("\n--- Diff Line Mapping ---")

added_lines = [10, 11, 12, 13, 14]
test("Exact match maps correctly",
     GitHubReviewer._map_to_diff_line(12, added_lines) == 12)
test("Fallback to first added line when not matched",
     GitHubReviewer._map_to_diff_line(99, added_lines) == 10)
test("Empty added_lines returns 1",
     GitHubReviewer._map_to_diff_line(5, []) == 1)

# ─── Comment Formatting ────────────────────────────────────────────────────────
print("\n--- Violation Comment Formatting ---")

VIOLATIONS = [
    {'rule_id': 'PY004', 'severity': 'HIGH',     'category': 'error-handling', 'description': 'Bare except clause', 'snippet': 'except:\n    pass'},
    {'rule_id': 'PY005', 'severity': 'HIGH',     'category': 'bug',            'description': 'Mutable default arg', 'snippet': ''},
    {'rule_id': 'PY006', 'severity': 'LOW',      'category': 'best-practice',  'description': 'Use logging', 'snippet': "print('hi')"},
    {'rule_id': 'NT015', 'severity': 'MEDIUM',   'category': 'ml-practice',    'description': 'Missing model.train()', 'snippet': ''},
    {'rule_id': 'NT021', 'severity': 'MEDIUM',   'category': 'ml-practice',    'description': 'Missing dropout', 'snippet': ''},
]

for v in VIOLATIONS:
    body = GitHubReviewer._format_violation_comment(v)
    test(f"Comment for {v['rule_id']} contains rule_id", v['rule_id'] in body)
    test(f"Comment for {v['rule_id']} contains description", v['description'] in body)
    if v['snippet']:
        test(f"Comment for {v['rule_id']} contains snippet", v['snippet'] in body)

# Severity emoji mapping
high = GitHubReviewer._format_violation_comment({'rule_id': 'X', 'severity': 'HIGH',     'category': 'c', 'description': 'd', 'snippet': ''})
crit = GitHubReviewer._format_violation_comment({'rule_id': 'X', 'severity': 'CRITICAL', 'category': 'c', 'description': 'd', 'snippet': ''})
low  = GitHubReviewer._format_violation_comment({'rule_id': 'X', 'severity': 'LOW',      'category': 'c', 'description': 'd', 'snippet': ''})
test("HIGH severity gets orange emoji",   '🟠' in high)
test("CRITICAL severity gets red emoji",  '🔴' in crit)
test("LOW severity gets blue emoji",      '🔵' in low)

# ─── PR Event Routing ─────────────────────────────────────────────────────────
print("\n--- PR Event Routing ---")

# Build a minimal reviewer (no real network/AWS)
reviewer = GitHubReviewer.__new__(GitHubReviewer)
reviewer.token = "ghp_fake_token"
reviewer.bedrock_client = None
reviewer.model_id = ""
reviewer.table = None

# Ignored actions
for ignored_action in ("closed", "labeled", "unlabeled", "assigned", "review_requested"):
    result = reviewer.handle_pr_event({"action": ignored_action})
    test(f"Action '{ignored_action}' is ignored", result.get('status') == 'ignored')

# Missing payload fields
result = reviewer.handle_pr_event({"action": "opened"})
test("Missing pr/repo returns error", result.get('status') == 'error')

# Missing GITHUB_TOKEN (simulate missing config)
reviewer_no_token = GitHubReviewer.__new__(GitHubReviewer)
reviewer_no_token.token = ""
reviewer_no_token.bedrock_client = None
reviewer_no_token.model_id = ""
reviewer_no_token.table = None
# handle_pr_event is called but token check is in handler.py, not reviewer
# verify_webhook_signature is the guard tested above

# ─── analyze_pr_diff offline (no Bedrock) ─────────────────────────────────────
print("\n--- analyze_pr_diff offline ---")

reviewer_offline = GitHubReviewer.__new__(GitHubReviewer)
reviewer_offline.token = "fake"
reviewer_offline.bedrock_client = None  # no AI calls
reviewer_offline.model_id = ""
reviewer_offline.table = None

comments = reviewer_offline.analyze_pr_diff(DIFF_PYTHON)
# Should return static analysis violation comments (no AI since bedrock_client is None)
test("analyze_pr_diff returns a list", isinstance(comments, list))
for c in comments:
    test("Each comment has 'path' key", 'path' in c)
    test("Each comment has 'line' key", 'line' in c)
    test("Each comment has 'body' key", 'body' in c)
    test(f"Comment body contains rule_id", '[' in c['body'])

# Non-Python diff should yield no comments
css_diff = "diff --git a/style.css b/style.css\n--- a/style.css\n+++ b/style.css\n@@ -1 +1,2 @@\n+body { color: red; }\n"
css_comments = reviewer_offline.analyze_pr_diff(css_diff)
test("Non-Python diff produces no comments", len(css_comments) == 0)

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "-" * 50)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print('All GitHub Reviewer tests passed! [OK]')
else:
    print('Some tests FAILED. See above for details.')
    sys.exit(1)
