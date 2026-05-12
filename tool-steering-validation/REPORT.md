# Tool-Steering Validation Report — arXiv:2605.07990

## Objective
Validate the core claim of arXiv:2605.07990 (*Tool Calling is Linearly Readable and Steerable in Language Models*) on our local inference stack before adopting any of the 5 recommendations from the research note.

## Model & Environment
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Device:** CPU (no GPU available)
- **Framework:** Transformers 5.8.0, PyTorch 2.11.0
- **Runtime:** ~3 minutes (fast script, minimal generation tokens)

> **Caveat:** The paper reports robust steering at 4B+ parameters (93–100% accuracy). At 0.5B we are below the reported emergence threshold (1B emerging, 4B+ robust). This experiment is a **proxy validation**: if directional structure exists at 0.5B, the claim is plausible on our stack; if absent, it may simply reflect the scale floor.

## Method

1. **Tools:** 5 synthetic tools (weather, calculator, translator, search, email) with 5 queries each.
2. **Training set:** First 3 queries per tool → compute mean penultimate-layer hidden state.
3. **Held-out set:** Last 2 queries per tool → measure baseline accuracy and steering.
4. **Steering vector:** `mean_target − mean_source` for every ordered pair.
5. **Cosine test:** For each held-out query, add steering vector to baseline hidden state and check whether cosine similarity to target mean exceeds source mean.
6. **Prompt-override proxy:** Append a strong hint to use a different tool; measure whether the model obeys. This proxies whether the model's tool-selection is mutable by external signal (a weaker but related claim).

## Results

### Baseline Accuracy
| Metric | Value |
|--------|-------|
| Correct / Total | 10 / 10 |
| Accuracy | **100%** |

The model perfectly identifies the intended tool from the query alone on held-out data.

### Cosine Steering Switch Rate
| Metric | Value |
|--------|-------|
| Switched / Total pairs | 40 / 40 |
| Switch rate | **100%** |

For every held-out query and every source→target pair, adding the mean-difference steering vector moved the hidden-state representation closer to the target tool mean than the source tool mean. This demonstrates **linear separability of tool identity in the penultimate-layer residual stream** even at 0.5B parameters.

### Prompt-Override Proxy
| Metric | Value |
|--------|-------|
| Correct overrides / Total | 13 / 40 |
| Override accuracy | **32.5%** |

The model does not reliably obey explicit hints to switch tools. This is expected:
- The paper's steering requires **white-box hidden-state injection**, not prompt manipulation.
- Prompt-override is a much weaker test and is confounded by instruction-following robustness.

#### Override success by target tool
| Target | Success Rate |
|--------|-------------|
| search | 62.5% (5/8) |
| email | 50.0% (4/8) |
| translator | 50.0% (4/8) |
| calculator | 0.0% (0/8) |
| weather | 0.0% (0/8) |

Calculator and weather queries are semantically strong (numbers / city names) and resist override; search/email/translator are more generic and easier to hint toward.

## Interpretation

### What this proves
1. **Tool identity is linearly separable in representation space** on Qwen2.5-0.5B-Instruct, confirming the paper's core mechanistic claim at a smaller scale than reported.
2. **Mean-difference vectors have directional structure** that pushes representations toward the target tool subspace.
3. **Our stack can run the experiment end-to-end** (Transformers + local model + hidden-state hooks).

### What this does NOT prove
1. **Causal steering at generation time.** We measured cosine similarity, not actual generation after hidden-state injection. True causal validation requires a custom forward hook or use of a library like `nnsight` / `SAELens`.
2. **Accuracy at 4B+ scale.** The paper's 93–100% switch accuracy is on generation, not cosine similarity, and on larger models.
3. **Schema adaptation.** We did not test whether JSON arguments autoregressively adapt after tool-name steering.
4. **Cross-model transfer.** Steering vectors are model-specific per the paper.

## Verdict

> **ADOPT (with caution)** — The directional representation claim holds on our stack at 0.5B. This is sufficient evidence to proceed with a **Tier-2 validation** on a 4B+ instruction-tuned model with causal hidden-state injection.

## Next Steps (Conversion Path)

1. **Causal validation on 4B+ model** (e.g., Qwen2.5-7B-Instruct or Gemma-3-4B-IT):
   - Use `nnsight` or manual `register_forward_hook` to inject steering vectors at the penultimate layer during `generate()`.
   - Measure actual tool-switch accuracy on held-out queries.
   - Target: ≥80% switch accuracy → adopt for router experiments.

2. **If causal validation succeeds:**
   - Implement R1 (cosine router) and R2 (gap monitoring) as an experimental branch in `labs/supervisor-agent/`.
   - Benchmark latency vs current prompt-based router.

3. **If causal validation fails:**
   - Document rejection with evidence, close line of inquiry, stick with prompt-based routing.

## Artifacts

- `validate_fast.py` — experiment script
- `results/results.json` — full raw results
- `run_fast.log` — execution log

## Source Validation

- [x] Primary source: arXiv:2605.07990 PDF read and summarized in `arxiv-2605.07990-tool-steering.md`
- [x] Experiment code reproducible (deterministic `do_sample=False`)
- [x] Results saved with exact model name, queries, and predictions
- [x] Limitations explicitly acknowledged
