#!/usr/bin/env python3
import os; os.environ.setdefault('PYTHONUTF8', '1')  # Force UTF-8 on Windows
"""
NeuroTidy CLI — Analyze Python code from your terminal.

Usage:
  neurotidy explain myfile.py --mode beginner
  neurotidy analyze myfile.py
  neurotidy optimize myfile.py
  neurotidy debug --error "NameError: name 'x' is not defined"
  
  neurotidy review --diff path/to/changes.diff
  neurotidy review --repo owner/repo --pr 42
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import urllib.request
import urllib.error

from neurotidy import DEFAULT_API_ENDPOINT, __version__

# Fix Windows encoding so emoji/unicode prints correctly
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ─── Config ───────────────────────────────────────────────────────────────────
def load_config() -> str:
    """Load API endpoint from configuration files or environment."""
    # 1. Try environment variable
    endpoint = os.environ.get('NEUROTIDY_API_ENDPOINT', '').strip()
    if endpoint:
        return endpoint.rstrip('/')

    # 2. Try reading from config files in the current working directory
    for config_name in ('config.env', 'config.eng'):
        config_path = Path.cwd() / config_name
        if config_path.exists():
            try:
                # Force UTF-8 for reading config files on Windows
                content = config_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = config_path.read_text() # Fallback

            for line in content.splitlines():
                line = line.strip()
                if line.startswith('NEUROTIDY_API_ENDPOINT=') and not line.startswith('#'):
                    val = line.split('=', 1)[1].strip()
                    if val and not val.startswith('<'):
                        return val.rstrip('/')

    # 3. Use built-in default endpoint
    return DEFAULT_API_ENDPOINT


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


# ─── Theme ────────────────────────────────────────────────────────────────────
RESET   = '\033[0m'
BOLD    = '\033[1m'
DIM     = '\033[2m'
ITALIC  = '\033[3m'

# CodeRabbit-inspired palette: deep purple/indigo + mint green accents
PURPLE  = '\033[38;5;135m'   # primary brand
INDIGO  = '\033[38;5;105m'   # secondary
MINT    = '\033[38;5;121m'   # success / positive
AMBER   = '\033[38;5;220m'   # warning
ROSE    = '\033[38;5;204m'   # error / critical
SKY     = '\033[38;5;117m'   # info / headings
GREY    = '\033[38;5;245m'   # muted text
WHITE   = '\033[38;5;255m'   # primary text

BG_DARK = '\033[48;5;234m'   # subtle dark bg for panels

# Legacy aliases kept for internal use
CYAN    = SKY
GREEN   = MINT
YELLOW  = AMBER
RED     = ROSE


# ─── UI Primitives ────────────────────────────────────────────────────────────
_W = 72  # terminal width target

def _rule(char='─', color=GREY):
    return f"{color}{char * _W}{RESET}"

def _badge(text, color=PURPLE):
    return f"{color}{BOLD} {text} {RESET}"

def _label(text, color=SKY):
    return f"{color}{BOLD}{text}{RESET}"

def _pill(text, color=MINT):
    return f"{color}[{text}]{RESET}"

def _icon_line(icon, label, value, label_color=GREY, value_color=WHITE):
    return f"  {icon}  {label_color}{label:<18}{RESET}{value_color}{value}{RESET}"

def _section(title, color=PURPLE):
    bar = f"{color}{'━' * _W}{RESET}"
    heading = f"  {color}{BOLD}{title}{RESET}"
    return f"\n{bar}\n{heading}\n{bar}"


def print_banner():
    PENGUIN = r"""
                                                                                
                                  ▄▄▄▄▄                                         
                          ▀▀███▄█████████▄                                      
                              ▀████████████▄                                    
                                ▀███████████▄                                   
                                  ███████████▄                                  
                                  ████████████▄                                 
                                  ▀    ▀███████▄                                
                                         ▀██████▄▄▄                             
                                ▄          ██████████▄                          
                               █▀           ████████████▄                       
                              ██             █████████████▄                     
                             ███▄             ████▀████████▄                    
                             ████             ▀███   ▀▀██████                   
                            ▄████              ███        ▀▀██                  
                            █████▄              ███                             
                            ██████              ███                             
                             █▀▀ ██             ███                             
                                 ██▄            ███                             
                                 ▄██            ███                             
                              ▄██████▄       ▄▄▄███                             
                              ▀▀▀▀▀▀▀▀       ▀███▀▀                             
                                                                                              
    """

    # Print penguin in purple
    for line in PENGUIN.splitlines():
        if line.strip():
            print(f"{PURPLE}{line}{RESET}")

    print(f"  {PURPLE}{BOLD}NeuroTidy{RESET}  {DIM}v{__version__}{RESET}  {GREY}·  AI-Powered Python & Deep Learning Code Analyzer{RESET}")
    print(f"  {DIM}github.com/ADI7HYANSNAIR/ERROR-108-AWS-HACKATHON{RESET}")
    print()
    print(_rule('─', GREY))
    print()

    cmds = [
        ("explain",  "✦", "Explain code in plain language"),
        ("analyze",  "✦", "Static quality & style analysis"),
        ("optimize", "✦", "DL performance optimization tips"),
        ("debug",    "✦", "Diagnose errors & suggest fixes"),
        ("review",   "✦", "AI PR code review bot"),
    ]
    print(f"  {GREY}{BOLD}COMMANDS{RESET}")
    print()
    for name, icon, desc in cmds:
        print(f"  {MINT}{icon}{RESET}  {WHITE}{BOLD}{name:<10}{RESET}  {GREY}{desc}{RESET}")

    print()
    print(_rule('─', GREY))
    print()


# ─── Pretty Printers ──────────────────────────────────────────────────────────
def print_explanation(result: dict):
    print(_section("📖  CODE EXPLANATION", PURPLE))
    explanation = result.get('explanation', '')
    
    # Handle structured JSON explanation vs raw text
    if isinstance(explanation, dict):
        for key, val in explanation.items():
            title = key.replace('_', ' ').title()
            print(f"\n  {_h(title, CYAN)}")
            if isinstance(val, list):
                for item in val:
                    print(f"    - {item}")
            else:
                print(textwrap.indent(textwrap.fill(str(val), width=_W-4), "    "))
    else:
        # Improved markdown parsing for terminal
        for line in str(explanation).splitlines():
            line = line.strip()
            if not line:
                print()
                continue
            
            if line.startswith('### '):
                print(f"\n  {_h(line[4:], GREEN)}")
            elif line.startswith('## '):
                print(f"\n  {_h(line[3:], GREEN)}")
            elif line.startswith('# '):
                print(f"\n  {_h(line[2:], YELLOW)}")
            elif line.startswith('**') and line.endswith('**'):
                print(f"\n  {_h(line.strip('*'), CYAN)}")
            elif line.startswith('* ') or line.startswith('- '):
                content = line[2:]
                wrapped = textwrap.fill(content, width=_W-6)
                print(textwrap.indent(wrapped, "    - "))
            elif line[0].isdigit() and (line[1:3] == '. ' or line[1:2] == '.'):
                print(f"  {BOLD}{line}{RESET}")
            else:
                print(textwrap.indent(textwrap.fill(line, width=_W-4), "  "))
    print()


def print_analysis(result: dict):
    print(_section("🔍  STATIC ANALYSIS REPORT", PURPLE))
    print()
    summary = result.get('summary', '')
    if summary:
        print(f"  {GREY}{summary}{RESET}")
        print()

    violations = result.get('violations', [])
    if violations:
        print(f"  {AMBER}{BOLD}{'VIOLATIONS':}{RESET}  {GREY}({len(violations)} found){RESET}")
        print()
        for v in violations:
            severity = v.get('severity', '')
            if severity in ('CRITICAL', 'HIGH'):
                sc, tag = ROSE, f"● {severity}"
            elif severity == 'MEDIUM':
                sc, tag = AMBER, f"◆ {severity}"
            else:
                sc, tag = GREY, f"○ {severity}"

            line_ref = f"  line {v.get('line_number')}" if v.get('line_number') else ''
            rule_id  = v.get('rule_id', '')
            desc     = v.get('description', '')

            print(f"  {sc}{BOLD}{tag:<14}{RESET}  {DIM}{rule_id}{line_ref}{RESET}")
            print(f"  {'':14}  {WHITE}{desc}{RESET}")
            print()

    ai = result.get('ai_insights', {})
    if ai and not ai.get('error'):
        print(f"  {SKY}{BOLD}AI INSIGHTS{RESET}")
        print()
        if 'readability_score' in ai:
            rb = ai['readability_score']
            mb = ai.get('maintainability_score', '?')
            print(f"  {_label('Readability',     GREY)}    {_score_bar(rb)}  {WHITE}{rb}/100{RESET}")
            print(f"  {_label('Maintainability', GREY)}    {_score_bar(mb)}  {WHITE}{mb}/100{RESET}")
            print()
        if 'top_recommendation' in ai:
            print(f"  {MINT}💡{RESET}  {WHITE}{ai['top_recommendation']}{RESET}")
            print()

    print(_rule('─', GREY))
    print()


def _score_bar(score, width=16):
    try:
        n = int(score)
        filled = round(n / 100 * width)
        color = MINT if n >= 70 else AMBER if n >= 40 else ROSE
        bar = f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"
        return bar
    except Exception:
        return f"{DIM}{'░' * width}{RESET}"


def print_optimization(result: dict):
    print(_section("⚡  DL OPTIMIZATION REPORT", PURPLE))
    print()
    summary = result.get('summary', '')
    if summary:
        print(f"  {GREY}{summary}{RESET}")
        print()

    violations = result.get('violations', [])
    if violations:
        print(f"  {AMBER}{BOLD}ISSUES FOUND{RESET}  {GREY}({len(violations)}){RESET}")
        print()
        for v in violations:
            severity = v.get('severity', '')
            sc = ROSE if severity == 'HIGH' else AMBER if severity == 'MEDIUM' else GREY
            print(f"  {sc}{BOLD}{'▸ ' + severity:<12}{RESET}  {DIM}[{v.get('rule_id')}]{RESET}")
            print(f"  {'':12}  {WHITE}{v.get('description', '')}{RESET}")
            fix = v.get('suggested_fix', '')
            if fix:
                print(f"  {'':12}  {MINT}→  {fix}{RESET}")
            print()

    ai = result.get('ai_insights', {})
    if ai and not ai.get('error'):
        print(f"  {SKY}{BOLD}AI PERFORMANCE INSIGHTS{RESET}")
        print()
        if 'performance_score' in ai:
            ps = ai['performance_score']
            print(f"  {_label('Performance', GREY)}       {_score_bar(ps)}  {WHITE}{ps}/100{RESET}")
        if 'estimated_speedup' in ai:
            print(f"  {MINT}⚡{RESET}  Estimated speedup:  {WHITE}{ai['estimated_speedup']}{RESET}")
        for tip in ai.get('quick_wins', []):
            print(f"  {MINT}✓{RESET}  {WHITE}{tip}{RESET}")
        print()

    print(_rule('─', GREY))
    print()


def print_debug(result: dict):
    print(_section("🐛  BUG ANALYSIS REPORT", PURPLE))
    print()
    print(f"  {_label('Error Type',  GREY)}    {ROSE}{BOLD}{result.get('error_type', 'Unknown')}{RESET}")
    print(f"  {_label('Root Cause',  GREY)}    {WHITE}{result.get('root_cause', '')}{RESET}")

    faulty_lines = result.get('faulty_lines', [])
    if faulty_lines:
        lns = ', '.join(f"L{l}" for l in faulty_lines)
        print(f"  {_label('Faulty Lines', GREY)}   {AMBER}{lns}{RESET}")

    tips = result.get('learning_tips', [])
    if tips:
        print()
        print(f"  {SKY}{BOLD}LEARNING TIPS{RESET}")
        print()
        for tip in tips:
            print(f"  {GREY}•{RESET}  {WHITE}{tip}{RESET}")

    fixes = result.get('suggested_fixes', [])
    if fixes:
        print()
        print(f"  {MINT}{BOLD}SUGGESTED FIXES{RESET}")
        print()
        for fix in fixes:
            print(f"  {MINT}→{RESET}  {WHITE}{fix}{RESET}")

    ai = result.get('explanation', {})
    if isinstance(ai, dict) and 'simple_explanation' in ai:
        print()
        print(f"  {SKY}{BOLD}AI EXPLANATION{RESET}")
        print()
        print(f"  {WHITE}{ai['simple_explanation']}{RESET}")
        steps = ai.get('step_by_step_fix', [])
        if steps:
            print()
            print(f"  {MINT}{BOLD}STEP-BY-STEP FIX{RESET}")
            print()
            for i, s in enumerate(steps, 1):
                print(f"  {PURPLE}{BOLD}{i}.{RESET}  {WHITE}{s}{RESET}")

    print()
    print(_rule('─', GREY))
    print()


def print_review(result: dict):
    status = result.get('status', '')
    status_color = MINT if status.lower() in ('success', 'posted', 'ok') else AMBER

    print(_section("🔍  PR REVIEW REPORT", PURPLE))
    print()
    print(_icon_line('●', 'Status',         f"{status_color}{BOLD}{status}{RESET}",         GREY, ''))
    print(_icon_line('·', 'PR Number',       f"#{result.get('pr_number', '?')}",             GREY, WHITE))
    print(_icon_line('·', 'Repository',      result.get('repo', ''),                         GREY, WHITE))
    print(_icon_line('·', 'Files Reviewed',  str(result.get('files_reviewed', 0)),            GREY, WHITE))
    print(_icon_line('·', 'Comments Posted', str(result.get('comments_posted', 0)),           GREY, WHITE))

    reason = result.get('reason', '')
    if reason:
        print()
        print(f"  {AMBER}ℹ  {reason}{RESET}")

    print()
    print(_rule('─', GREY))
    print()


def _h(text: str, color: str = CYAN) -> str:
    return f"{color}{BOLD}{text}{RESET}"


# ─── Commands ─────────────────────────────────────────────────────────────────
def cmd_explain(args, endpoint: str):
    code = ""
    if args.file:
        try:
            code = Path(args.file).read_text(encoding='utf-8')
        except Exception as e:
            print(f"{RED}Error reading file {args.file}: {e}{RESET}")
            sys.exit(1)
    else:
        code = args.code

    if not code:
        print(f"{RED}Error: provide a file or --code{RESET}")
        sys.exit(1)
        
    print(f"  {GREY}Explaining code  {DIM}({args.mode} mode){RESET}  …\n")
    result = call_api(endpoint, 'explain', {'code': code, 'mode': args.mode})
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_explanation(result)


def cmd_analyze(args, endpoint: str):
    code = ""
    if args.file:
        try:
            code = Path(args.file).read_text(encoding='utf-8')
        except Exception as e:
            print(f"{RED}Error reading file {args.file}: {e}{RESET}")
            sys.exit(1)
    else:
        code = args.code

    if not code:
        print(f"{RED}Error: provide a file or --code{RESET}")
        sys.exit(1)
        
    print(f"  {GREY}Analyzing code quality …{RESET}\n")
    result = call_api(endpoint, 'analyze', {'code': code, 'use_ai': not args.no_ai})
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_analysis(result)


def cmd_optimize(args, endpoint: str):
    code = ""
    if args.file:
        try:
            code = Path(args.file).read_text(encoding='utf-8')
        except Exception as e:
            print(f"{RED}Error reading file {args.file}: {e}{RESET}")
            sys.exit(1)
    else:
        code = args.code

    if not code:
        print(f"{RED}Error: provide a file or --code{RESET}")
        sys.exit(1)
        
    print(f"  {GREY}Checking for DL optimizations …{RESET}\n")
    result = call_api(endpoint, 'optimize', {'code': code, 'use_ai': not args.no_ai})
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_optimization(result)


def cmd_debug(args, endpoint: str):
    code = ""
    if args.file:
        try:
            code = Path(args.file).read_text(encoding='utf-8')
        except Exception as e:
            print(f"{RED}Error reading file {args.file}: {e}{RESET}")
            sys.exit(1)
            
    print(f"  {GREY}Debugging error …{RESET}\n")
    result = call_api(endpoint, 'debug', {
        'error': args.error or '',
        'stack_trace': args.stack_trace or '',
        'code': code,
    })
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_debug(result)


def cmd_review(args, endpoint: str):
    diff_text = ""
    repo_name = "local/file"
    pr_num = 0

    if hasattr(args, 'file') and args.file:
        try:
            file_path = Path(args.file)
            code = file_path.read_text(encoding='utf-8')
            # Create a "pseudo-diff" that looks like a new file addition
            # This allows the review endpoint to process a single file as a change
            line_count = len(code.splitlines())
            diff_text = f"--- /dev/null\n+++ {file_path.name}\n@@ -0,0 +1,{line_count} @@\n"
            for line in code.splitlines():
                diff_text += f"+{line}\n"
            repo_name = f"local/{file_path.name}"
        except Exception as e:
            print(f"{RED}Error reading file {args.file}: {e}{RESET}")
            sys.exit(1)
            
    elif hasattr(args, 'diff') and args.diff:
        try:
            diff_text = Path(args.diff).read_text(encoding='utf-8')
            repo_name = getattr(args, 'repo', '') or 'local/local'
            pr_num = getattr(args, 'pr', 0) or 0
        except Exception as e:
            print(f"{RED}Error reading diff file {args.diff}: {e}{RESET}")
            sys.exit(1)
            
    elif hasattr(args, 'repo') and args.repo and hasattr(args, 'pr') and args.pr:
        # Leave diff_text empty, the backend will fetch it from GitHub
        payload = {
            'action': 'opened',
            'pull_request': {
                'number': args.pr,
                'head': {'sha': 'api-triggered'},
                'diff_url': f'https://github.com/{args.repo}/pull/{args.pr}.diff',
            },
            'repository': {'full_name': args.repo},
        }
        _submit_review(endpoint, payload)
        return
    else:
        print(f"{RED}Error: provide a file, --diff <file>, or --repo owner/repo --pr N{RESET}")
        sys.exit(1)

    # Submit the local diff (either from a file or a .diff file)
    payload = {
        'action': 'opened',
        'pull_request': {
            'number': pr_num,
            'head': {'sha': 'local-analysis'},
            'diff_url': '',
            '_local_diff': diff_text,
        },
        'repository': {'full_name': repo_name},
    }
    _submit_review(endpoint, payload)


def _submit_review(endpoint: str, payload: dict):
    print(f"  {GREY}Submitting code for AI review …{RESET}\n")
    result = call_api(endpoint, 'review', payload)
    if 'error' in result:
        print(f"{RED}Error: {result['error']}{RESET}")
        sys.exit(1)
    print_review(result)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog='neurotidy',
        description='NeuroTidy — AI-powered Python & DL Code Analyzer + GitHub PR Review Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          neurotidy explain train.py --mode beginner
          neurotidy analyze model.py
          neurotidy optimize train.py
          neurotidy debug --error "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
          neurotidy review sample_dl.py
          neurotidy review --diff changes.diff
          neurotidy review --repo myorg/myrepo --pr 42
        """)
    )
    parser.add_argument('--endpoint', help='API endpoint URL (overrides configuration)')
    parser.add_argument('--version', action='version', version=f'neurotidy {__version__}')
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

    # review
    p_review = subparsers.add_parser('review', help='Submit a PR for AI code review (GitHub PR bot)')
    p_review.add_argument('file', nargs='?', help='Python file to review directly')
    p_review_src = p_review.add_mutually_exclusive_group()
    p_review_src.add_argument('--diff', help='Path to a .diff file to review locally')
    p_review.add_argument('--repo', help='GitHub repo (owner/repo) for live PR review')
    p_review.add_argument('--pr', type=int, help='Pull Request number for live review')

    args = parser.parse_args()

    print_banner()

    endpoint = args.endpoint or load_config()
    if not endpoint:
        print(f"{ROSE}{BOLD}  ✗  No API endpoint configured{RESET}")
        print()
        print(f"  {GREY}Set {WHITE}NEUROTIDY_API_ENDPOINT{GREY} in config.eng or export it:{RESET}")
        print()
        print(f"  {DIM}export NEUROTIDY_API_ENDPOINT=https://your-api.execute-api.us-east-1.amazonaws.com/prod{RESET}")
        print()
        sys.exit(1)

    dispatch = {
        'explain':  cmd_explain,
        'analyze':  cmd_analyze,
        'optimize': cmd_optimize,
        'debug':    cmd_debug,
        'review':   cmd_review,
    }
    dispatch[args.command](args, endpoint)


if __name__ == '__main__':
    main()
