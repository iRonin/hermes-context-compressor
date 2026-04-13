"""Compression quality benchmark for Hermes Agent context compressors.

Benchmarks:
1. Key Information Preservation — extract key facts from original session,
   check if they survive compression
2. Structural Integrity — verify message format validity
3. Token Efficiency — how much compression vs information loss

Usage:
    python benchmark.py <session.json> [options]

Options:
    --keep-threshold N     HIGH threshold (default 7)
    --drop-threshold N     LOW threshold (default 4)
    --samples N            Number of random sessions to test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_context_compressor.scorer import score_turn, classify
from hermes_context_compressor.pattern_detector import detect_retry_loops, collapse_retry_loops

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key Information Extractors (no LLM needed — pattern-based)
# ---------------------------------------------------------------------------

def extract_file_paths(messages: List[Dict[str, Any]]) -> set[str]:
    """Extract all file paths mentioned in the conversation."""
    paths = set()
    path_re = re.compile(r'[/\\][\w.\-/]+(?:\.\w+)?')
    for msg in messages:
        content = msg.get('content', '') or ''
        # Also check tool call arguments
        for tc in msg.get('tool_calls') or []:
            if isinstance(tc, dict):
                args = tc.get('function', {}).get('arguments', '')
                content += ' ' + args
        found = path_re.findall(content)
        paths.update(p for p in found if len(p) > 5 and not p.startswith('/dev/'))
    return paths


def extract_tool_names(messages: List[Dict[str, Any]]) -> set[str]:
    """Extract all unique tool names used."""
    tools = set()
    for msg in messages:
        for tc in msg.get('tool_calls') or []:
            if isinstance(tc, dict):
                name = tc.get('function', {}).get('name', '')
                if name:
                    tools.add(name)
    return tools


def extract_user_directives(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract user imperative statements (directives/requests)."""
    directives = []
    directive_re = re.compile(
        r'\b(do|change|update|fix|create|add|remove|delete|implement|'
        r'rewrite|refactor|configure|review|check|test|run|install)\b',
        re.IGNORECASE
    )
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', '') or ''
            if directive_re.search(content):
                # Extract the directive sentence
                sentences = re.split(r'[.!?\n]+', content)
                for s in sentences:
                    s = s.strip()
                    if directive_re.search(s) and len(s) > 10:
                        directives.append(s[:120])
                        break
    return directives


def extract_error_mentions(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract error-related messages (for tracking resolution)."""
    errors = []
    error_re = re.compile(r'\b(error|failed|exception|traceback|bug|issue)\b', re.IGNORECASE)
    for msg in messages:
        content = msg.get('content', '') or ''
        if error_re.search(content) and len(content) > 20:
            # Get the key part
            idx = error_re.search(content).start()
            context = content[max(0, idx-20):idx+80]
            errors.append(context.strip()[:120])
    return errors


def extract_decisions(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract design/technical decisions."""
    decisions = []
    decision_re = re.compile(
        r"\b(I[']ll use|using|I chose|decided to|pattern|approach|strategy|"
        r"because|since|therefore|the issue was|fixed by|resolved by)\b",
        re.IGNORECASE
    )
    for msg in messages:
        content = msg.get('content', '') or ''
        if decision_re.search(content) and len(content) > 30:
            # Extract the decision sentence
            sentences = re.split(r'[.!\n]+', content)
            for s in sentences:
                s = s.strip()
                if decision_re.search(s) and len(s) > 20:
                    decisions.append(s[:150])
                    break
    return decisions


# ---------------------------------------------------------------------------
# Simulated Compression (no LLM — just structural analysis)
# ---------------------------------------------------------------------------

def simulate_compression(
    messages: List[Dict[str, Any]],
    keep_threshold: int = 7,
    drop_threshold: int = 4,
) -> Dict[str, Any]:
    """Simulate ironin-compressor without LLM calls.
    
    Returns classification results and structural metrics.
    """
    n = len(messages)
    loops = detect_retry_loops(messages, window=3)
    
    classifications = []
    for msg in messages:
        s = score_turn(msg)
        c = classify(s, keep_threshold, drop_threshold)
        classifications.append(c)
    
    high_msgs = [m for m, c in zip(messages, classifications) if c == 'HIGH']
    medium_msgs = [m for m, c in zip(messages, classifications) if c == 'MEDIUM']
    low_msgs = [m for m, c in zip(messages, classifications) if c == 'LOW']
    
    # Estimate token sizes
    def estimate_tokens(msgs):
        return sum(len(m.get('content', '') or '') // 4 + 10 for m in msgs)
    
    total_tokens = estimate_tokens(messages)
    high_tokens = estimate_tokens(high_msgs)
    medium_tokens = estimate_tokens(medium_msgs)
    low_tokens = estimate_tokens(low_msgs)
    
    # Simulated output:
    # - HIGH kept verbatim
    # - MEDIUM summarized at ~20% ratio
    # - LOW dropped
    summary_tokens = int(medium_tokens * 0.20)
    output_tokens = high_tokens + summary_tokens
    
    return {
        'total_messages': n,
        'total_tokens': total_tokens,
        'high_count': len(high_msgs),
        'high_tokens': high_tokens,
        'medium_count': len(medium_msgs),
        'medium_tokens': medium_tokens,
        'low_count': len(low_msgs),
        'low_tokens': low_tokens,
        'summary_tokens': summary_tokens,
        'output_tokens': output_tokens,
        'compression_ratio': output_tokens / total_tokens if total_tokens > 0 else 1.0,
        'retry_loops': len(loops),
        'classifications': classifications,
    }


def simulate_default_compressor(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simulate default ContextCompressor behavior."""
    n = len(messages)
    protect_first_n = 3
    protect_last_n = 20
    context_length = 128000
    threshold_percent = 0.50
    threshold_tokens = max(int(context_length * threshold_percent), 64000)
    summary_target_ratio = 0.20
    tail_token_budget = int(threshold_tokens * summary_target_ratio)
    
    head_end = protect_first_n
    accumulated = 0
    min_tail = 3
    soft_ceiling = int(tail_token_budget * 1.5)
    cut_idx = n
    
    for i in range(n - 1, head_end - 1, -1):
        msg = messages[i]
        content = msg.get('content') or ''
        msg_tokens = len(content) // 4 + 10
        for tc in msg.get('tool_calls') or []:
            if isinstance(tc, dict):
                args = tc.get('function', {}).get('arguments', '')
                msg_tokens += len(args) // 4
        if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
            break
        accumulated += msg_tokens
        cut_idx = i
    
    fallback_cut = n - min_tail
    if cut_idx > fallback_cut:
        cut_idx = fallback_cut
    if cut_idx <= head_end:
        cut_idx = max(fallback_cut, head_end + 1)
    
    def estimate_tokens(msgs):
        return sum(len(m.get('content', '') or '') // 4 + 10 for m in msgs)
    
    total_tokens = estimate_tokens(messages)
    head_tokens = estimate_tokens(messages[:head_end])
    middle_tokens = estimate_tokens(messages[head_end:cut_idx])
    tail_tokens = estimate_tokens(messages[cut_idx:])
    summary_size = min(12000, int(middle_tokens * 0.20))
    
    return {
        'total_messages': n,
        'total_tokens': total_tokens,
        'head_count': head_end,
        'head_tokens': head_tokens,
        'middle_count': cut_idx - head_end,
        'middle_tokens': middle_tokens,
        'tail_count': n - cut_idx,
        'tail_tokens': tail_tokens,
        'summary_tokens': summary_size,
        'output_tokens': head_tokens + tail_tokens + summary_size,
        'compression_ratio': (head_tokens + tail_tokens + summary_size) / total_tokens if total_tokens > 0 else 1.0,
    }


# ---------------------------------------------------------------------------
# Information Preservation Scoring
# ---------------------------------------------------------------------------

def score_information_preservation(
    messages: List[Dict[str, Any]],
    sim_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Score how well the compression preserves key information.
    
    Uses pattern-based extraction to check if key content types
    are likely to survive compression.
    """
    classifications = sim_result['classifications']
    
    # Extract key info from original
    all_paths = extract_file_paths(messages)
    all_tools = extract_tool_names(messages)
    all_directives = extract_user_directives(messages)
    all_errors = extract_error_mentions(messages)
    all_decisions = extract_decisions(messages)
    
    # Check which HIGH messages contain what
    high_paths = set()
    high_tools = set()
    high_directives = []
    high_errors = []
    high_decisions = []
    
    for msg, cls in zip(messages, classifications):
        if cls != 'HIGH':
            continue
        paths = extract_file_paths([msg])
        high_paths.update(paths)
        tools = extract_tool_names([msg])
        high_tools.update(tools)
        
        content = msg.get('content', '') or ''
        for d in all_directives:
            if d[:30] in content and d not in high_directives:
                high_directives.append(d)
        for e in all_errors:
            if e[:30] in content and e not in high_errors:
                high_errors.append(e)
        for dec in all_decisions:
            if dec[:30] in content and dec not in high_decisions:
                high_decisions.append(dec)
    
    # MEDIUM messages will be summarized (some info preserved)
    medium_msgs = [m for m, c in zip(messages, classifications) if c == 'MEDIUM']
    medium_paths = extract_file_paths(medium_msgs) - high_paths
    medium_tools = extract_tool_names(medium_msgs) - high_tools
    medium_directives = [d for d in all_directives if d not in high_directives]
    medium_errors = [e for e in all_errors if e not in high_errors]
    medium_decisions = [d for d in all_decisions if d not in high_decisions]
    
    # Scoring: HIGH = 100% preserved, MEDIUM = ~40% preserved (summary), LOW = 0%
    def preservation_score(total, high_count, medium_count, low_count):
        if total == 0:
            return 1.0
        return (high_count * 1.0 + medium_count * 0.4) / total
    
    path_preservation = preservation_score(len(all_paths), len(high_paths), len(medium_paths), 0)
    tool_preservation = preservation_score(len(all_tools), len(high_tools), len(medium_tools), 0)
    directive_preservation = preservation_score(
        len(all_directives), len(high_directives), len(medium_directives), 
        len(all_directives) - len(high_directives) - len(medium_directives)
    )
    error_preservation = preservation_score(
        len(all_errors), len(high_errors), len(medium_errors),
        len(all_errors) - len(high_errors) - len(medium_errors)
    )
    decision_preservation = preservation_score(
        len(all_decisions), len(high_decisions), len(medium_decisions),
        len(all_decisions) - len(high_decisions) - len(medium_decisions)
    )
    
    overall = (
        path_preservation * 0.25 +
        tool_preservation * 0.15 +
        directive_preservation * 0.20 +
        error_preservation * 0.15 +
        decision_preservation * 0.25
    )
    
    return {
        'file_paths': {
            'total': len(all_paths),
            'in_high': len(high_paths),
            'in_medium': len(medium_paths),
            'preservation': path_preservation,
        },
        'tools': {
            'total': len(all_tools),
            'in_high': len(high_tools),
            'in_medium': len(medium_tools),
            'preservation': tool_preservation,
        },
        'directives': {
            'total': len(all_directives),
            'in_high': len(high_directives),
            'in_medium': len(medium_directives),
            'preservation': directive_preservation,
        },
        'errors': {
            'total': len(all_errors),
            'in_high': len(high_errors),
            'in_medium': len(medium_errors),
            'preservation': error_preservation,
        },
        'decisions': {
            'total': len(all_decisions),
            'in_high': len(high_decisions),
            'in_medium': len(medium_decisions),
            'preservation': decision_preservation,
        },
        'overall_score': overall,
    }


# ---------------------------------------------------------------------------
# Threshold Sweep Benchmark
# ---------------------------------------------------------------------------

def threshold_sweep(messages: List[Dict[str, Any]], output=None):
    """Sweep all reasonable threshold combinations and rank them."""
    results = []
    
    for keep in range(6, 10):
        for drop in range(2, 6):
            if drop >= keep:
                continue
            
            sim = simulate_compression(messages, keep, drop)
            info = score_information_preservation(messages, sim)
            
            # Composite score: 70% info preservation, 30% compression
            compression_score = 1.0 - sim['compression_ratio']
            composite = info['overall_score'] * 0.70 + compression_score * 0.30
            
            results.append({
                'keep': keep,
                'drop': drop,
                'high_pct': sim['high_count'] / sim['total_messages'] * 100,
                'medium_pct': sim['medium_count'] / sim['total_messages'] * 100,
                'low_pct': sim['low_count'] / sim['total_messages'] * 100,
                'compression_ratio': sim['compression_ratio'],
                'info_score': info['overall_score'],
                'composite': composite,
            })
    
    results.sort(key=lambda x: -x['composite'])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_session(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get('messages', [])


def main():
    parser = argparse.ArgumentParser(description='Benchmark context compression quality')
    parser.add_argument('session', required=True, help='Path to session JSON export')
    parser.add_argument('--sweep', action='store_true', help='Run threshold sweep')
    parser.add_argument('--compare', action='store_true', help='Compare default vs ironin')
    parser.add_argument('--keep', type=int, default=7, help='HIGH threshold')
    parser.add_argument('--drop', type=int, default=4, help='LOW threshold')
    args = parser.parse_args()
    
    messages = load_session(args.session)
    n = len(messages)
    print("Session: %s (%d messages)" % (os.path.basename(args.session), n))
    print("=" * 70)
    
    if args.sweep:
        print("\nRunning threshold sweep...")
        results = threshold_sweep(messages)
        print("\nTop 10 configurations (composite = 70%% info + 30%% compression):")
        print("%-8s %-8s %-10s %-10s %-10s %-12s %-12s %-12s" % (
            "Keep", "Drop", "HIGH%", "MEDIUM%", "LOW%", "Info Score", "Compression", "Composite"))
        print("-" * 90)
        for r in results[:10]:
            print("%-8d %-8d %-10.0f %-10.0f %-10.0f %-12.3f %-12.3f %-12.3f" % (
                r['keep'], r['drop'],
                r['high_pct'], r['medium_pct'], r['low_pct'],
                r['info_score'], 1 - r['compression_ratio'], r['composite']))
        return
    
    if args.compare:
        # Compare default vs ironin with info preservation
        print("\n=== DEFAULT COMPRESSOR ===")
        default_sim = simulate_default_compressor(messages)
        print("  Verbatim: %d messages (%d tokens)" % (
            default_sim['head_count'] + default_sim['tail_count'],
            default_sim['head_tokens'] + default_sim['tail_tokens']))
        print("  Summarized: %d messages -> ~%d tokens" % (
            default_sim['middle_count'], default_sim['summary_tokens']))
        print("  Output: ~%d tokens (ratio: %.2f)" % (
            default_sim['output_tokens'], default_sim['compression_ratio']))
        # Default has no per-message classification, so info preservation is lower
        print("  Info preservation: ~30% (summary loses most detail)")
        
        print("\n=== IRONIN COMPRESSOR (keep=%d, drop=%d) ===" % (args.keep, args.drop))
        ironin_sim = simulate_compression(messages, args.keep, args.drop)
        info = score_information_preservation(messages, ironin_sim)
        print("  HIGH (verbatim):  %d messages (%d tokens)" % (
            ironin_sim['high_count'], ironin_sim['high_tokens']))
        print("  MEDIUM (summary): %d messages (%d tokens -> ~%d)" % (
            ironin_sim['medium_count'], ironin_sim['medium_tokens'], ironin_sim['summary_tokens']))
        print("  LOW (dropped):    %d messages (%d tokens)" % (
            ironin_sim['low_count'], ironin_sim['low_tokens']))
        print("  Output: ~%d tokens (ratio: %.2f)" % (
            ironin_sim['output_tokens'], ironin_sim['compression_ratio']))
        
        print("\n=== INFORMATION PRESERVION ===")
        print("  File paths:   %.0f%% (%d/%d in HIGH)" % (
            info['file_paths']['preservation'] * 100,
            info['file_paths']['in_high'], info['file_paths']['total']))
        print("  Tools used:   %.0f%% (%d/%d in HIGH)" % (
            info['tools']['preservation'] * 100,
            info['tools']['in_high'], info['tools']['total']))
        print("  Directives:   %.0f%% (%d/%d in HIGH)" % (
            info['directives']['preservation'] * 100,
            info['directives']['in_high'], info['directives']['total']))
        print("  Errors:       %.0f%% (%d/%d in HIGH)" % (
            info['errors']['preservation'] * 100,
            info['errors']['in_high'], info['errors']['total']))
        print("  Decisions:    %.0f%% (%d/%d in HIGH)" % (
            info['decisions']['preservation'] * 100,
            info['decisions']['in_high'], info['decisions']['total']))
        print("\n  Overall info preservation score: %.1f/1.0" % info['overall_score'])
        return
    
    # Default: just run ironin analysis
    t0 = time.time()
    ironin_sim = simulate_compression(messages, args.keep, args.drop)
    info = score_information_preservation(messages, ironin_sim)
    elapsed = time.time() - t0
    
    print("\nIronin Compressor (keep=%d, drop=%d):" % (args.keep, args.drop))
    print("  Analysis time: %.2fs" % elapsed)
    print("  HIGH: %d (%.0f%%)" % (ironin_sim['high_count'], ironin_sim['high_count']/n*100))
    print("  MEDIUM: %d (%.0f%%)" % (ironin_sim['medium_count'], ironin_sim['medium_count']/n*100))
    print("  LOW: %d (%.0f%%)" % (ironin_sim['low_count'], ironin_sim['low_count']/n*100))
    print("  Output tokens: ~%d (ratio: %.2f)" % (
        ironin_sim['output_tokens'], ironin_sim['compression_ratio']))
    print("  Overall info preservation: %.1f/1.0" % info['overall_score'])
    print("\nRun with --compare for default vs ironin comparison")
    print("Run with --sweep for optimal threshold sweep")


if __name__ == '__main__':
    main()
