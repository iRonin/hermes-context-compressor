# hermes-context-compressor

Smart context compression plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Replaces the built-in blanket summarization with heuristic turn scoring that distinguishes between high-value content (design decisions, fixes, key outputs) and low-value noise (retry loops, acknowledgments, iterative edits).

## How It Works

Each message turn is scored 0-10 based on content signals:

| Score | Classification | Treatment |
|-------|---------------|-----------|
| >= keep_threshold | HIGH | Kept verbatim in compressed output |
| drop_threshold to keep_threshold-1 | MEDIUM | Sent to LLM for structured summary |
| < drop_threshold | LOW | Dropped with a marker note |

### Scoring Signals

**HIGH signals (+points):**
- User directives: "fix", "change", "create", "implement" (+3)
- Design decisions: "I'll use", "pattern", "because", "strategy" (+3)
- Error diagnosis: "issue was", "fixed by", "root cause" (+3)
- File writes/patches: write_file, patch tool calls (+2)
- File success: "file written", "created", "saved" (+2)
- Test passes: "12 passed", "all tests passed" (+2)
- Specific values: file paths, line numbers, error codes (+2)

**LOW signals (-points):**
- Short text < 50 chars (-1)
- Retry language: "oops", "let me try", "actually" (-2)
- Tool error with same-tool retry (-2)
- Acknowledgments: "ok", "got it", "sure", "will do" (-2)
- Empty content without tool calls (-3)

### Retry Loop Detection

When the same tool operates on the same file within 3 consecutive turns (write → error → write), the loop is detected and collapsed into a single entry.

## Installation

### Option 1: Symlink into Hermes plugins

```bash
cd ~/Work/Hermes/hermes-agent/plugins/context_engine
ln -s ~/Work/Hermes/hermes-context-compressor/plugins/context_engine/ironin_compressor ironin_compressor
```

### Option 2: Install as package

```bash
pip install -e ~/Work/Hermes/hermes-context-compressor
```

## Configuration

Activate in `~/.hermes/config.yaml`:

```yaml
context:
  engine: "ironin-compressor"

smart_compressor:
  keep_threshold: 7           # score >= this -> keep verbatim (4-10)
  drop_threshold: 4           # score < this -> drop (0-6)
  max_retries_window: 3       # same tool+file within N turns -> collapse (2-10)
  summarize_low: false        # include LOW turns in LLM summary?
  summarize_high: false       # ALSO summarize HIGH turns (for iterative compression safety)
  target_tokens: 0            # adaptive target: compress to this many tokens (0 = disabled)
  skip_scoring_under_msgs: 6  # don't score tiny conversations (4-20)
```

### Adaptive Compression with `target_tokens`

When `target_tokens` is set (non-zero), the compressor dynamically adjusts the keep/drop thresholds to hit your target output token budget. This is useful for:

- **Staying in optimal pricing tiers**: e.g., keep sessions under 200K tokens to avoid higher per-token costs above 256K
- **Maximum context preservation**: compress to the highest quality level that fits your budget
- **Iterative compression**: re-run `/compress` with a lower target to reduce further

Example configs:

```yaml
# Stay in optimal pricing tier (<200K)
smart_compressor:
  target_tokens: 200000

# Aggressive: compress to 150K even from 500K session
smart_compressor:
  target_tokens: 150000

# Maximum compression: fit into 100K
smart_compressor:
  target_tokens: 100000
```

The algorithm scores all messages first, then searches for the keep/drop threshold combination that gets closest to the target without exceeding it. Higher thresholds mean more messages get summarized/dropped, lower thresholds mean more kept verbatim.

### `summarize_high` — Iterative Compression Safety

When `summarize_high: true`, the compressor generates a structured summary of HIGH-value turns **in addition** to keeping them verbatim. This is useful for sessions that may compress multiple times:

1. **First compression**: HIGH turns kept verbatim + summary generated
2. **Second compression**: If the verbatim HIGH turns are re-scored lower (due to changed context), the summary preserves the key information

The summary is injected before the verbatim HIGH messages with a clear marker so the agent knows both are present.

## Benchmarking

Run the built-in benchmark to compare compression strategies on your sessions:

```bash
cd ~/Work/Hermes/hermes-context-compressor

# Compare default vs ironin on your session
python -m hermes_context_compressor.benchmark /path/to/session.json --compare

# Find optimal thresholds for your session
python -m hermes_context_compressor.benchmark /path/to/session.json --sweep

# Quick analysis with custom thresholds
python -m hermes_context_compressor.benchmark /path/to/session.json --keep 8 --drop 5
```

### Benchmark Metrics

| Metric | What it measures |
|--------|-----------------|
| File path preservation | % of file paths kept in HIGH vs summarized in MEDIUM |
| Tool preservation | % of unique tool calls kept verbatim |
| Directive preservation | % of user requests/directives preserved |
| Error preservation | % of error mentions kept verbatim |
| Decision preservation | % of design/technical decisions preserved |
| Overall score | Weighted composite of all above |

### Example Benchmark Output

```
=== COMPARISON ===
Metric                                        Default       Ironin
----------------------------------------------------------------
Messages kept verbatim                             65          362
Messages summarized                               570          268
Messages dropped                                    0            5
Retry loop detection                               No Yes (35 loops)

=== INFORMATION PRESERVATION ===
  File paths:   92% (537/617 in HIGH)
  Tools used:   53% (2/9 in HIGH)
  Directives:   86% (13/17 in HIGH)
  Errors:       48% (17/40 in HIGH)
  Decisions:    56% (60/111 in HIGH)

  Overall info preservation score: 0.7/1.0
```

### Adaptive Compression Results (635-message session)

| Target | Keep | Drop | HIGH% | MEDIUM% | Output | Savings |
|--------|------|------|-------|---------|--------|---------|
| 200K (optimal) | 6 | 2 | 81% | 19% | 194K | 10% |
| 150K (aggressive) | 8 | 2 | 22% | 78% | 83K | 62% |
| 100K (maximum) | 8 | 2 | 22% | 78% | 83K | 62% |

The adaptive system automatically finds the best thresholds for your target. For a 215K session, 200K target barely changes anything (81% verbatim), but 150K target compresses aggressively to 83K (62% savings) while still keeping the highest-value 22% of messages verbatim.

## Results (tested on 635-message session)

| Metric | Built-in Compressor | ironin-compressor (keep=7) | ironin-compressor (target=150K) |
|--------|-------------------|---------------------------|-------------------------------|
| HIGH kept | N/A (all summarized) | 362 (57%) | 137 (22%) |
| MEDIUM summarized | N/A | 268 (42%) | 498 (78%) |
| LOW dropped | N/A | 5 (1%) | 0 (0%) |
| Retry loops | N/A | 35 detected | 35 detected |
| Output tokens | ~30K | ~193K | ~83K |
| Info preservation | ~30% | ~70% | ~57% |

## Architecture

```
hermes_context_compressor/
├── __init__.py           # Public API
├── config.py             # Config parsing with clamps
├── scorer.py             # Heuristic turn scorer
├── pattern_detector.py   # Retry loop detection
├── compressor.py         # IroninCompressor (extends ContextCompressor)
└── benchmark.py          # Compression quality benchmark

plugins/context_engine/ironin_compressor/
├── __init__.py           # Plugin entry point (register)
└── plugin.yaml           # Metadata
```

## License

MIT
