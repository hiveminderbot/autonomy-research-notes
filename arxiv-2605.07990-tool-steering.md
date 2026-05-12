# arXiv:2605.07990 — Tool Calling is Linearly Readable and Steerable in Language Models

**Authors:** Zekun Wu, Ze Wang, Seonglae Cho, Yufei Yang, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz  
**Institutions:** UCL, HolisticAI, Imperial College London  
**Date:** 2026-05-08  
**URL:** https://arxiv.org/abs/2605.07990  
**PDF:** https://arxiv.org/pdf/2605.07990  
**Bead:** autonomy-nle2

---

## Core Claim

Tool identity is **linearly readable and steerable** inside the residual stream of instruction-tuned LLMs across Gemma 3, Qwen 3/2.5, and Llama 3.1 (270M–27B). Adding a mean-difference vector between two tools’ average activations switches tool selection at 77–100% accuracy (93–100% at 4B+), and the JSON arguments autoregressively adapt to the new tool’s schema.

---

## Key Findings

### 1. Small subspace for tool identity
- 15 tools fit in ~10 principal components (91% variance) across all 12 IT models.
- At K=200 real APIs, only 36 dimensions needed (18% of theoretical max).
- Base models show comparable PCA spectra (89–93%), but steering fails without IT.

### 2. Three-stage circuit (Gemma 3 4B)
- **Early layers (L0–3):** 38 tool-specific features light up.
- **Mid layers (L16–30):** Attention heads pull in tool-name/entity context (e.g., L24 H1 attends to "Tokyo" at weight 0.43).
- **Late layers (L30–33):** JSON formatting features (shared with non-tool code generation).

### 3. Causal localization via activation patching
- On Gemma 3 4B, **L17 H0 and H1** dominate: swapping them drops confidence by 6.5 and 3.7 points.
- Attention heads carry 80% (Gemma) / 88% (Qwen) of total importance; MLPs the rest.
- Per-component top-k comparison: top-3 heads vs top-3 MLPs = 2.7× gap on Gemma; tied on Qwen.

### 4. Steering works and is direction-specific
- Random Gaussian vectors at matched norm: **0%** switch rate.
- Unit vector along target tool’s unembedding row at matched magnitude: **93–100%**.
- Parallel component of steering vector does all the work; orthogonal component leaves choice untouched.

### 5. Schema adaptation is autoregressive, not steering-carried
- Steering changes the tool name; the rest of the JSON follows from autoregressive generation conditioned on that name.
- Prefill experiment (force target name in prompt) matches steering schema-correctness exactly on Gemma/Qwen.

### 6. Error prediction before execution
- Cosine gap between top-1 and top-2 tool means predicts mistakes:
  - Gemma 3 12B: smallest-gap quartile → 14% error rate; largest-gap → 0% (14× concentration).
  - Gemma 3 27B: 17% vs 1% (21× concentration).

### 7. Base vs instruction-tuned gap
- Base models encode tool identity (cosine readout 69–82% on BFCL) but generate correctly only 2–10%.
- IT models close the output gap but cosine readout falls below generation (e.g., Gemma 3 4B: 92% gen vs 74% readout).
- Interpretation: pretraining forms the representation; instruction tuning wires it to output.

### 8. Scale emergence
- 270M: no steerable structure (27%, chance level).
- 1B: emerging (43% at 15 tools, 100% at 5 tools).
- 4B+: robust (≥83% at 15 tools, ≥93% at 4B+).

---

## Limitations (from paper)

1. **Single-turn, fixed-menu only.** Multi-turn τ-bench airline shows mixed results (−30 to +10 pp vs fresh baseline).
2. **Prompt length ceiling.** At K=750 with descriptions, Gemma 3 4B representations blur (k₉₀ drops to 27). Without descriptions, handles 1,500+ tools.
3. **Near-synonym tools fail.** 10 similar Sports APIs → 0% steering.
4. **Argument steering fails.** Can switch tool name but not argument values (e.g., "Tokyo" → "Paris" = 0/30).
5. **Cross-family transfer = 0%.** Each model learns its own subspace.
6. **Fabricated arguments.** Steered outputs sometimes invent values (e.g., "PizzaPalace") — schema-correct but not API-valid.
7. **Dense transformers only.** No MoE or state-space model tests.

---

## Actionable Recommendations for Our Stack

### R1. Add per-tool mean-activation profiling to Hermes tool router
**Why:** The paper shows tool identity is linearly readable from the penultimate-layer residual. We can compute per-tool mean activations from a small calibration set (2–3 queries per tool) and use cosine similarity as a fast, training-free routing signal.
**Implementation:** Hook the penultimate-layer hidden state in our inference backend (vLLM/llama.cpp), compute cosine to each tool mean, and use the gap between top-1 and top-2 as a confidence score.
**Expected benefit:** Sub-millisecond routing decision; flags low-confidence calls for human review or subagent escalation.
**Risk:** Only validated on dense IT models; our router may use different architectures. Needs calibration on our actual tool set.

### R2. Use activation-gap monitoring as a real-time hallucination guard
**Why:** The paper shows small top-1/top-2 gaps predict wrong-tool calls at 14–21× higher rates.
**Implementation:** Log the cosine gap for every tool call. If gap < median threshold, trigger a verification subagent or pause for user confirmation.
**Expected benefit:** Catches ~92% of tool-selection errors on Gemma 12B/27B before execution.
**Risk:** Threshold is model-specific; needs per-model calibration. Adds latency if verification is expensive.

### R3. Replace complex prompt-engineering with linear steering for tool selection
**Why:** Steering via mean-difference vectors achieves 93–100% at 4B+ with no prompt changes, outperforming description-heavy prompts for small models.
**Implementation:** Instead of embedding tool descriptions in the system prompt, pre-compute steering vectors for the most common tool switches and inject them at inference time.
**Expected benefit:** Shorter prompts → lower latency, lower cost, larger effective context. At K=1500 tools without descriptions, Gemma 3 4B handles smoothly; with descriptions, K=750 causes collapse.
**Risk:** Requires white-box access to model weights (not available for API-only models). Best for self-hosted models.

### R4. Enforce context-window limits per subagent to preserve cooperation
**Why:** This is cross-referenced with arXiv:2605.08060 (Memory Curse paper, bead autonomy-n7b6). Longer context windows degrade multi-agent cooperation. Tool-calling steering is single-turn only; multi-turn transfer is fragile.
**Implementation:** Cap per-subagent context to ~4K tokens for tool-routing tasks. Use a separate summarizer subagent to compress history before routing decisions.
**Expected benefit:** Preserves both tool-steering reliability and inter-subagent cooperation.

### R5. Validate on our actual model before production deployment
**Why:** The paper’s results are on Gemma 3, Qwen 3, Llama 3.1. Our stack may use different models (e.g., Kimi, Claude, GPT-4o).
**Implementation:** Run a 5-tool steering experiment on our primary inference model. Collect 2–3 queries per tool, compute mean-difference vectors, and measure switch accuracy on held-out queries.
**Acceptance:** ≥80% switch accuracy on 5 tools → adopt steering for internal router experiments. <80% → reject and stick with prompt-based routing.

---

## Conversion Path to Tier 2

1. **Immediate (this week):** Implement R5 — 5-tool steering validation on our primary model. Write a 100-line Python script using Transformers + SAELens or manual hooking. Target: 2–3 hours.
2. **Short-term (next 2 weeks):** If R5 succeeds, implement R1 (cosine router) and R2 (gap monitoring) as an experimental branch in the supervisor-agent repo. Benchmark vs current prompt-based router on latency and accuracy.
3. **Medium-term (next month):** If benchmark shows ≥10% accuracy or latency improvement, open a PR and deploy to staging. If not, document rejection with evidence and close the line of inquiry.

---

## Source Validation

- [x] Primary source downloaded and read: arXiv:2605.07990 PDF (36 pages)
- [x] All key claims traced to specific tables/figures in the paper
- [x] Limitations section explicitly acknowledged
- [x] Recommendations tied to our stack (Hermes supervisor-agent, vLLM/llama.cpp inference)
- [x] Concrete next experiment defined (R5 validation)

---

## Status

**Research lead → Tier 1.5** (high-signal, provenance-backed, clear conversion path to Tier 2 via R5 validation experiment).
