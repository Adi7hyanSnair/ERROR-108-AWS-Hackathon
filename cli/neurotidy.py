#!/usr/bin/env python3
"""
NeuroTidy CLI — Analyze Python code from your terminal.

Usage:
  python neurotidy.py explain myfile.py --mode beginner
  python neurotidy.py analyze myfile.py
  python neurotidy.py optimize myfile.py
  python neurotidy.py debug --error "NameError: name 'x' is not defined"
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import urllib.request
import urllib.error


# ─── Config ─────────────────────────────────────────────────────────────────
def load_config() -> str:
    """Load API endpoint from config.env or environment."""
    # 1. Try environment variable
    endpoint = os.environ.get('NEUROTIDY_API_ENDPOINT', '').strip()
    if endpoint:
        return endpoint.rstrip('/')

    # 2. Try reading from config.env in the project root
    config_path = Path(__file__).parent.parent / 'config.env'
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith('NEUROTIDY_API_ENDPOINT=') and not line.startswith('#'):
                val = line.split('=', 1)[1].strip()
                if val and not val.startswith('<'):
                    return val.rstrip('/')

    return ""


def call_api(endpoint: str, path: str, payload: dict) -> dict:
    """POST to the NeuroTidy API and return parsed JSON."""
    url = f"{endpoint}/{path.lstrip('/')}"
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return json.loads(body)
        except Exception:
            return {'error': f"HTTP {e.code}: {body[:200]}"}
    except urllib.error.URLError as e:
        return {'error': f"Connection error: {e.reason}"}


# ─── Pretty Printers ─────────────────────────────────────────────────────────
BOLD   = '\033[1m'
CYAN   = '\033[96m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
RESET  = '\033[0m'


def _h(text: str, color: str = CYAN) -> str:
    return f"{color}{BOLD}{text}{RESET}"


def print_banner():
    print(f"""
{CYAN}╔══════════════════════════════════════╗
║   NeuroTidy — AI Code Analyzer v1.0  ║
╚══════════════════════════════════════╝{RESET}
""")


def print_explanation(result: dict):
    print(_h("📖 Code Explanation"))
    explanation = result.get('explanation', '')
    if isinstance(explanation, dict):
        explanation = json.dumps(explanation, indent=2)
    print(textwrap.fill(str(explanation), width=90))
    print()


def print_analysis(result: dict):
    print(_h("🔍 Static Analysis Report"))
    print(f"  {result.get('summary', '')}")
    violations = result.get('violations', [])
    if violations:
        print(f"\n  {_h('Violations:', YELLOW)}")
        for v in violations:
            severity = v.get('severity', '')
            color = RED if severity in ('CRITICAL', 'HIGH') else YELLOW if severity == 'MEDIUM' else RESET
            line = f"  L{v.get('line_number', '?')}" if v.get('line_number') else ""
            print(f"    {color}[{severity}]{RESET} [{v.get('rule_id')}]{line} {v.get('description')}")
    ai = result.get('ai_insights', {})
    if ai and not ai.get('error'):
        print(f"\n  {_h('AI Insights:', CYAN)}")
        if 'readability_score' in ai:
            print(f"    Readability: {ai['readability_score']}/100  |  Maintainability: {ai.get('maintainability_score', '?')}/100")
        if 'top_recommendation' in ai:
            print(f"    💡 Top recommendation: {ai['top_recommendation']}")
    print()


def print_optimization(result: dict):
    print(_h("⚡ DL Optimization Report"))
    print(f"  {result.get('summary', '')}")
    violations = result.get('violations', [])
    if violations:
        print(f"\n  {_h('Issues Found:', YELLOW)}")
        for v in violations:
            severity = v.get('severity', '')
            color = RED if severity == 'HIGH' else YELLOW if severity == 'MEDIUM' else RESET
            print(f"    {color}[{severity}]{RESET} [{v.get('rule_id')}] {v.get('description')}")
            print(f"          → {GREEN}{v.get('suggested_fix', '')}{RESET}")
    ai = result.get('ai_insights', {})
    if ai and not ai.get('error'):
        print(f"\n  {_h('AI Performance Insights:', CYAN)}")
        if 'performance_score' in ai:
            print(f"    Performance score: {ai['performance_score']}/100")
        if 'estimated_speedup' in ai:
            print(f"    Estimated speedup: {ai['estimated_speedup']}")
        for tip in ai.get('quick_wins', []):
            print(f"    ✓ {tip}")
    print()


def print_debug(result: dict):
    print(_h("🐛 Bug Analysis Report"))
    print(f"  Error type: {RED}{result.get('error_type', 'Unknown')}{RESET}")
    print(f"  Root cause: {result.get('root_cause', '')}")
    faulty_lines = result.get('faulty_lines', [])
    if faulty_lines:
        print(f"  Faulty lines: {', '.join(str(l) for l in faulty_lines)}")
    tips = result.get('learning_tips', [])
    if tips:
        print(f"\n  {_h('Learning Tips:', CYAN)}")
        for tip in tips:
            print(f"    • {tip}")
    fixes = result.get('suggested_fixes', [])
    if fixes:
        print(f"\n  {_h('Suggested Fixes:', GREEN)}")
        for fix in fixes:
            print(f"    → {fix}")
    ai = result.get('explanation', {})
    if isinstance(ai, dict) and 'simple_explanation' in ai:
        print(f"\n  {_h('AI Explanation:', CYAN)}")
        print(f"    {ai['simple_explanation']}")
        steps = ai.get('step_by_step_fix', [])
        if steps:
            print(f"\n  {_h('Step-by-Step Fix:', GREEN)}")
            for s in steps:
                print(f"    {s}")
    print()


# ─── Commands ───────────────────────────────────────────────────────────────
def cmd_explain(args, endpoint: str):
    code = Path(args.file).read_text() if args.file else args.code
    if not code:
        print(f"{RED}Error: provide a file or --code{RESET}")
        sys.exit(1)
    print(f"  Explaining code ({args.mode} mode)…\n")
    result = call_api(endpoint, 'explain', {'code': code, 'mode': args.mode})
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_explanation(result)


def cmd_analyze(args, endpoint: str):
    code = Path(args.file).read_text() if args.file else args.code
    if not code:
        print(f"{RED}Error: provide a file or --code{RESET}")
        sys.exit(1)
    print(f"  Analyzing code quality…\n")
    result = call_api(endpoint, 'analyze', {'code': code, 'use_ai': not args.no_ai})
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_analysis(result)


def cmd_optimize(args, endpoint: str):
    code = Path(args.file).read_text() if args.file else args.code
    if not code:
        print(f"{RED}Error: provide a file or --code{RESET}")
        sys.exit(1)
    print(f"  Checking for DL optimizations…\n")
    result = call_api(endpoint, 'optimize', {'code': code, 'use_ai': not args.no_ai})
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_optimization(result)


def cmd_debug(args, endpoint: str):
    code = ""
    if args.file:
        code = Path(args.file).read_text()
    print(f"  Debugging error…\n")
    result = call_api(endpoint, 'debug', {
        'error': args.error or '',
        'stack_trace': args.stack_trace or '',
        'code': code,
    })
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_debug(result)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog='neurotidy',
        description='NeuroTidy — AI-powered Python & DL Code Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python neurotidy.py explain train.py --mode beginner
          python neurotidy.py analyze model.py
          python neurotidy.py optimize train.py
          python neurotidy.py debug --error "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
        """)
    )
    parser.add_argument('--endpoint', help='API endpoint URL (overrides config.env)')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # explain
    p_explain = subparsers.add_parser('explain', help='Explain Python code')
    p_explain.add_argument('file', nargs='?', help='Python file to explain')
    p_explain.add_argument('--code', help='Inline code string')
    p_explain.add_argument('--mode', choices=['beginner', 'intermediate', 'advanced'],
                           default='intermediate')

    # analyze
    p_analyze = subparsers.add_parser('analyze', help='Static code quality analysis')
    p_analyze.add_argument('file', nargs='?', help='Python file to analyze')
    p_analyze.add_argument('--code', help='Inline code string')
    p_analyze.add_argument('--no-ai', action='store_true', help='Skip AI-enhanced analysis')

    # optimize
    p_optimize = subparsers.add_parser('optimize', help='DL performance optimization')
    p_optimize.add_argument('file', nargs='?', help='Python file to optimize')
    p_optimize.add_argument('--code', help='Inline code string')
    p_optimize.add_argument('--no-ai', action='store_true', help='Skip AI-enhanced analysis')

    # debug
    p_debug = subparsers.add_parser('debug', help='Explain a Python error/bug')
    p_debug.add_argument('file', nargs='?', help='Python file where error occurred')
    p_debug.add_argument('--error', help='Error message string')
    p_debug.add_argument('--stack-trace', help='Full stack trace text')

    args = parser.parse_args()

    print_banner()

    endpoint = args.endpoint or load_config()
    if not endpoint:
        print(f"{RED}❌ No API endpoint configured!{RESET}")
        print(f"\nSet NEUROTIDY_API_ENDPOINT in config.env or export it as an environment variable:")
        print(f"  {YELLOW}export NEUROTIDY_API_ENDPOINT=https://your-api.execute-api.us-east-1.amazonaws.com/prod{RESET}")
        sys.exit(1)

    dispatch = {
        'explain': cmd_explain,
        'analyze': cmd_analyze,
        'optimize': cmd_optimize,
        'debug': cmd_debug,
    }
    dispatch[args.command](args, endpoint)


if __name__ == '__main__':
    main()
