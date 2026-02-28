"""
github_reviewer.py — NeuroTidy GitHub PR Review Bot.

Handles GitHub Pull Request webhook events:
  1. Validates the HMAC-SHA256 webhook signature
  2. Fetches the PR diff via GitHub REST API
  3. Splits the diff into per-file Python chunks
  4. Runs StaticAnalyzer + CodeExplainer on changed Python code
  5. Posts structured inline review comments back to GitHub

All credentials and settings come from environment variables — nothing is hardcoded.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Environment variables (set via config.env → template.yaml → Lambda)
GITHUB_API_BASE = os.environ.get("GITHUB_API_BASE", "https://api.github.com")
GITHUB_MAX_FILES = int(os.environ.get("GITHUB_MAX_FILES_PER_REVIEW", "10"))
GITHUB_MAX_LINES = int(os.environ.get("GITHUB_MAX_LINES_PER_FILE", "500"))
GITHUB_POST_COMMENTS = os.environ.get("GITHUB_POST_COMMENTS", "true").lower() == "true"


class GitHubReviewer:
    """
    Fetches a PR diff from GitHub, analyses changed Python files,
    and posts structured inline review comments.
    """

    def __init__(
        self,
        github_token: str,
        bedrock_client,
        model_id: str,
        dynamodb_table=None,
    ):
        """
        Args:
            github_token:    GitHub Personal Access Token (repo + pull_requests scope)
            bedrock_client:  boto3 bedrock-runtime client
            model_id:        Bedrock model ID (from env, not hardcoded)
            dynamodb_table:  DynamoDB Table resource for caching (optional)
        """
        if not github_token:
            raise ValueError("GITHUB_TOKEN is required for GitHubReviewer")
        self.token = github_token
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.table = dynamodb_table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_pr_event(self, payload: dict) -> dict:
        """
        Process a GitHub Pull Request webhook payload.

        Expected GitHub event actions: 'opened', 'synchronize', 'reopened'.
        Other actions (closed, labeled, etc.) are ignored gracefully.

        Args:
            payload: decoded JSON body from the GitHub webhook request

        Returns:
            dict with keys: status, pr_number, repo, files_reviewed, comments_posted
        """
        action = payload.get("action", "")
        if action not in ("opened", "synchronize", "reopened"):
            logger.info("Ignoring PR event action=%s", action)
            return {"status": "ignored", "reason": f"action '{action}' not reviewed"}

        pr = payload.get("pull_request", {})
        pr_number = pr.get("number")
        repo_full = payload.get("repository", {}).get("full_name", "")
        diff_url = pr.get("diff_url", "")
        head_sha = pr.get("head", {}).get("sha", "")

        if not pr_number or not repo_full:
            return {"status": "error", "reason": "Missing PR number or repo in payload"}

        logger.info("Reviewing PR #%s on %s (sha=%s)", pr_number, repo_full, head_sha[:8])

        # 1. Fetch the diff
        diff_text = self._fetch_url(diff_url)
        if not diff_text:
            return {"status": "error", "reason": "Could not fetch PR diff"}

        # 2. Analyse the diff
        comments = self.analyze_pr_diff(diff_text, repo_full, pr_number, head_sha)

        # 3. Post a consolidated review
        comments_posted = 0
        if comments and GITHUB_POST_COMMENTS:
            comments_posted = self.post_review(repo_full, pr_number, head_sha, comments)

        return {
            "status": "reviewed",
            "pr_number": pr_number,
            "repo": repo_full,
            "files_reviewed": len({c["path"] for c in comments}),
            "comments_posted": comments_posted,
        }

    def analyze_pr_diff(
        self,
        diff_text: str,
        repo: str = "",
        pr_number: Optional[int] = None,
        head_sha: str = "",
    ) -> list:
        """
        Parse a unified diff and analyse each changed Python file.

        Returns:
            List of comment dicts: {path, line, body}
        """
        # Import here to keep the module importable without AWS
        from static_analyzer import StaticAnalyzer
        from code_explainer import CodeExplainer

        analyzer = StaticAnalyzer(self.bedrock_client, self.model_id)
        explainer = CodeExplainer(self.bedrock_client, self.model_id, self.table)

        file_chunks = self._parse_diff(diff_text)
        comments = []
        files_processed = 0

        for file_path, added_lines, raw_added_code in file_chunks:
            if not file_path.endswith(".py"):
                continue
            if files_processed >= GITHUB_MAX_FILES:
                logger.info("Reached GITHUB_MAX_FILES=%d — skipping remaining files", GITHUB_MAX_FILES)
                break

            code = raw_added_code[:GITHUB_MAX_LINES * 100]  # char limit safety
            if not code.strip():
                continue

            logger.info("Analysing %s (%d added lines)", file_path, len(added_lines))

            # Static analysis violations
            analysis = analyzer.analyze(code, use_ai=False)
            for violation in analysis.get("violations", []):
                # Map violation line relative to diff to absolute line in PR
                diff_line = self._map_to_diff_line(violation.get("line_number", 0), added_lines)
                body = self._format_violation_comment(violation)
                comments.append({"path": file_path, "line": diff_line, "body": body})

            # AI summary comment on the file (posted as a general comment)
            if self.bedrock_client:
                try:
                    summary = explainer.explain_code(code, mode="intermediate")
                    summary_comment = (
                        f"### 🤖 NeuroTidy — Code Summary for `{file_path}`\n\n"
                        f"{summary[:2000]}"  # truncate to keep reviews readable
                        f"\n\n---\n*Generated by NeuroTidy AI Code Reviewer*"
                    )
                    comments.append({"path": file_path, "line": added_lines[0] if added_lines else 1,
                                     "body": summary_comment, "is_summary": True})
                except Exception as exc:
                    logger.warning("Could not generate AI summary for %s: %s", file_path, exc)

            files_processed += 1

        return comments

    def post_review(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        comments: list,
    ) -> int:
        """
        Post a PR review with inline comments to GitHub.

        Uses the GitHub Pull Request Reviews API (single API call for all comments).

        Returns:
            Number of comments successfully posted.
        """
        # Separate summary comments from inline comments
        inline = [c for c in comments if not c.get("is_summary")]
        summaries = [c for c in comments if c.get("is_summary")]

        # Build review body from AI summaries
        review_body = "## 🤖 NeuroTidy AI Code Review\n\n"
        for s in summaries:
            review_body += s["body"] + "\n\n"
        if not summaries:
            review_body += (
                f"Reviewed {len(inline)} issue(s) in the changed Python files.\n\n"
                "*Generated by [NeuroTidy](https://github.com) PR Review Bot*"
            )

        # Build inline comment payload (GitHub requires position in diff, not line number)
        review_comments = []
        for c in inline:
            review_comments.append({
                "path": c["path"],
                "line": c["line"],
                "body": c["body"],
                "side": "RIGHT",
            })

        payload = {
            "commit_id": head_sha,
            "body": review_body,
            "event": "COMMENT",  # COMMENT = no approve/request changes
            "comments": review_comments[:50],  # GitHub API limit
        }

        url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}/reviews"
        try:
            self._github_post(url, payload)
            posted = len(review_comments[:50])
            logger.info("Posted review with %d inline comments on PR #%s", posted, pr_number)
            return posted
        except Exception as exc:
            logger.error("Failed to post review: %s", exc)
            return 0

    @staticmethod
    def verify_webhook_signature(body: bytes, signature_header: str, secret: str) -> bool:
        """
        Validate a GitHub webhook X-Hub-Signature-256 header.

        Args:
            body:             Raw request body bytes
            signature_header: Value of X-Hub-Signature-256 header (e.g. "sha256=abc123...")
            secret:           The GITHUB_WEBHOOK_SECRET configured on the webhook

        Returns:
            True if valid, False otherwise.
        """
        if not signature_header or not secret:
            return False
        if not signature_header.startswith("sha256="):
            return False
        expected_sig = "sha256=" + hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_sig, signature_header)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_url(self, url: str) -> str:
        """Fetch content from a URL using the GitHub token for auth."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"token {self.token}",
                    "Accept": "application/vnd.github.v3.diff",
                    "User-Agent": "NeuroTidy-PR-Bot/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            logger.error("HTTP %s fetching %s: %s", exc.code, url, exc.reason)
        except Exception as exc:
            logger.error("Error fetching %s: %s", url, exc)
        return ""

    def _github_post(self, url: str, data: dict) -> dict:
        """POST JSON data to the GitHub API."""
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
                "User-Agent": "NeuroTidy-PR-Bot/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _parse_diff(diff_text: str) -> list:
        """
        Parse a unified diff into (file_path, added_line_numbers, added_code) tuples.

        Returns:
            List of (file_path: str, added_lines: list[int], added_code: str)
        """
        results = []
        current_file = None
        current_added_lines = []
        current_code_lines = []
        current_line_num = 0

        for raw_line in diff_text.splitlines():
            # Detect new file header: "diff --git a/foo.py b/foo.py"
            new_file_match = re.match(r"^\+\+\+ b/(.+)$", raw_line)
            if new_file_match:
                if current_file and current_code_lines:
                    results.append((current_file, current_added_lines, "\n".join(current_code_lines)))
                current_file = new_file_match.group(1)
                current_added_lines = []
                current_code_lines = []
                current_line_num = 0
                continue

            # Hunk header: @@ -old_start,old_count +new_start,new_count @@
            hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
            if hunk_match:
                current_line_num = int(hunk_match.group(1))
                continue

            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                # Added line
                current_added_lines.append(current_line_num)
                current_code_lines.append(raw_line[1:])  # strip the leading '+'
                current_line_num += 1
            elif not raw_line.startswith("-"):
                # Context line (unchanged)
                current_line_num += 1

        # Flush last file
        if current_file and current_code_lines:
            results.append((current_file, current_added_lines, "\n".join(current_code_lines)))

        return results

    @staticmethod
    def _map_to_diff_line(violation_line: int, added_lines: list) -> int:
        """Map a violation source line to the nearest line in the diff hunk."""
        if not added_lines:
            return 1
        if violation_line in added_lines:
            return violation_line
        # Fall back to the first added line in the file
        return added_lines[0]

    @staticmethod
    def _format_violation_comment(violation: dict) -> str:
        """Format a static analysis violation as a GitHub review comment."""
        severity_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🔵",
            "INFO": "⚪",
        }.get(violation.get("severity", "INFO"), "⚪")

        rule_id = violation.get("rule_id", "")
        severity = violation.get("severity", "")
        category = violation.get("category", "")
        description = violation.get("description", "")
        snippet = violation.get("snippet", "")

        body = (
            f"{severity_emoji} **NeuroTidy [{rule_id}]** — {severity} | `{category}`\n\n"
            f"{description}"
        )
        if snippet:
            body += f"\n\n```python\n{snippet}\n```"

        return body
