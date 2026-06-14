# GOAL: Causal Hidden-State Injection on 4B+ Model (autonomy-6zuj)

## Context
We have validated that tool-steering vectors have directional structure at 0.5B (cosine similarity test: 100% switch rate). Now we need the **Tier-2 validation**: causal injection during `generate()` on a 4B+ instruction-tuned model, measuring actual tool-switch accuracy.

## Prior Art
- `validate_fast.py` — proxy validation at 0.5B using cosine similarity
- `REPORT.md` — full analysis, verdict: ADOPT (with caution), next step is causal validation
- Paper: arXiv:2605.07990 (*Tool Calling is Linearly Readable and Steerable*)

## Task for Codex
Implement `validate_causal.py` that:

1. **Loads a 4B+ instruction-tuned model** with 4-bit quantization if needed
   - Primary: `Qwen/Qwen2.5-7B-Instruct` (7B params, ~3.5GB INT4)
   - Fallback: `google/gemma-3-4b-it` (4B params, ~2GB INT4)
   - Use `bitsandbytes` 4-bit loading: `load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16`
   - If model doesn't fit in 8GB RAM, document and fall back to smaller

2. **Computes steering vectors** (reuse logic from `validate_fast.py`)
   - Same 5 tools: weather, calculator, translator, search, email
   - 3 training queries per tool, 2 held-out per tool
   - Mean penultimate-layer hidden state per tool
   - Steering vector = target_mean - source_mean

3. **Implements causal injection via `register_forward_hook`**
   - Hook into the penultimate transformer layer (layer index = num_layers - 2)
   - During `generate()`, add scaled steering vector to hidden states
   - Scale factor: start with 1.0, make configurable
   - The hook should only activate when a flag is set (so we can do before/after comparison)

4. **Measures actual tool-switch accuracy**
   - Baseline: generate WITHOUT steering → record predicted tool
   - Steered: generate WITH steering → record predicted tool
   - For each held-out query, test all source→target pairs (20 queries × 4 targets = 80 tests)
   - Accuracy = % of times steered generation predicts the target tool
   - Target: ≥80% for ADOPT verdict

5. **Produces JSON results artifact**
   - before/after accuracy per tool pair
   - overall accuracy
   - latency (ms per generation)
   - peak memory usage (MB)
   - scale factor used

6. **Produces VERDICT.md**
   - ADOPT if accuracy ≥80%
   - REJECT if <80%, with analysis of why
   - Compare to proxy results from 0.5B model

## Acceptance Criteria
- [ ] `validate_causal.py` runs end-to-end without errors
- [ ] Model loads successfully (or fallback documented)
- [ ] Steering hook activates during generation and changes output
- [ ] Tool-switch accuracy measured on held-out queries
- [ ] `results_causal.json` produced with all metrics
- [ ] `VERDICT.md` produced with ADOPT/REJECT and evidence
- [ ] All files committed and pushed to GitHub remote

## Blocker Handling
- If 4B model doesn't fit: try Gemma-3-4B-IT, then document and close with REJECT (resource limit)
- If bitsandbytes install fails: use `torch_dtype=torch.float16` with `device_map="auto"`
- If accuracy <80%: document, analyze failure mode, REJECT

## Constraints
- No OpenViking/Polymarket work
- Must run on CPU (no GPU available)
- Must complete in reasonable time (<30 min for full run)
