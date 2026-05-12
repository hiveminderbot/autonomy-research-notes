# arXiv:2605.08060 — The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents

**Authors:** Jiayuan Liu, Tianqin Li, Shiyi Du, Xin Luo, Haoxuan Zeng, Emanuel Tewolde, Tai Sing Lee, Tonghan Wang, Carl Kingsford, Vincent Conitzer  
**Institutions:** CMU, FOCAL, University of Michigan, Harvard  
**Date:** 2026-05-08  
**URL:** https://arxiv.org/abs/2605.08060  
**PDF:** https://arxiv.org/pdf/2605.08060  
**Bead:** autonomy-n7b6

---

## Core Claim

Expanding accessible interaction history in multi-agent LLM social dilemmas **systematically degrades cooperation** in 18 of 28 model–game settings (the "memory curse"). The mechanism is not paranoia but **eroding forward-looking intent**: longer history makes agents more history-following and risk-minimizing, trapping them in retaliatory cycles.

---

## Key Findings

### 1. Memory curse is widespread
- 7 LLMs × 4 games × 9 history lengths × 3 seeds = 500-round interactions.
- 18/28 model–game settings show cooperation decay as history length (HL) increases.
- Example: Gemma-3-12B in Trust Game drops from 51.2% cooperation (HL=2) to 9.5% (HL=80).
- Variance explodes at long HL: Llama-4-Scout-17B in Public Goods goes from tight bounds at HL=2 to ±24.0% at HL=80.

### 2. Two behavioral regimes
- **Memory Immune (10/28):** Models maintain ≥95% cooperation at all HL. Driven by intrinsic cooperation tendency + forward-looking reasoning about long-run reciprocity.
- **Memory Cursed (18/28):** Cooperation peaks at short HL (typically ≤5) then declines. Agents become "historically overfitted" — occasional noisy defection triggers retaliation that persists in the long context window.

### 3. Cognitive basis: forward-looking vs history-following reasoning
- Lexical analysis of 378,000 CoT traces:
  - Memory Immune models retain forward-looking ratio ~0.504 at HL=80.
  - Memory Cursed models drop to ~0.340.
  - Forward-looking keywords: "long-term", "future", "sustain", "reciprocate".
  - History-following keywords: "defected", "risk", "protect", "grudge".

### 4. Causal validation via targeted fine-tuning
- LoRA adapter on Mistral-7B (universally Memory Cursed) trained exclusively on forward-looking PG reasoning traces.
- Result: cooperation surges +14.7 to +79.3 pp at HL=80 across all four games.
- Zero-shot transfer to untrained games (TD, PD) confirms the adapter changes a general reasoning tendency, not just action labels.
- General task ability preserved (GSM8K, TriviaQA, HumanEval, MBPP unaffected).

### 5. Memory content > context length
- **Sanitization experiment:** At HL=80, replace 78 of 80 history rounds with synthetic mutual-cooperation records, leaving only 2 real rounds.
- Cooperation restores substantially for all models (Figure 5b).
- Conclusion: the curse is driven by accumulated negative content, not prompt length itself.

### 6. Explicit reasoning amplifies the curse
- **No-CoT ablation:** Removing explicit chain-of-thought generally mitigates memory curse.
- Extreme example: Llama-3.3-70B in its hardest game cooperates 100% without CoT at HL=80, but collapses to 6.9% with CoT (-93.1 pp penalty for deliberation).
- CoT traces devote additional space to enumerating past defections, making retaliatory decisions easier to justify.

### 7. Asymmetric memory contagion
- In 3-player Public Goods, replacing HL=2 agents with HL=80 grudge-holders drags group welfare down.
- Even a single HL=2 forgiver surrounded by two HL=80 grudge-holders maintains significantly higher cooperation (+33 pp gap).
- Bounded recall serves as a functional forgiveness mechanism.

---

## Limitations (from paper)

1. **Static memory only.** No dynamic curation (selective forgetting, RAG, summarization) tested.
2. **Fixed game structures.** PD, TD, PG, TG are well-defined; open-ended N-player societies not studied.
3. **Homogeneous models.** All agents use same backbone; heterogeneous architectures unexplored.
4. **Single-turn reasoning per round.** No multi-turn deliberation within a round.
5. **Temperature 0.7 fixed.** No sweep across sampling strategies.
6. **Cloudflare Workers AI only.** Different inference backends may yield different variance.

---

## Actionable Recommendations for Our Stack

### R1. Cap per-subagent context window for multi-agent cooperation tasks
**Why:** The paper shows cooperation peaks at HL≤5 and degrades sharply beyond. Our supervisor-agent dispatches to multiple subagents; each should receive only the most recent 3–5 turns of relevant history.
**Implementation:** Add a `max_history_turns=5` parameter to subagent dispatch. Use a summarizer subagent to compress older history into a 1-paragraph context summary.
**Expected benefit:** Preserves cooperation and trust repair between subagents. Prevents historical overfitting to old errors.
**Risk:** May lose long-range dependencies. Mitigate via explicit state tracking (key-value store of decisions) rather than verbatim history.

### R2. Add forward-looking reasoning prompt injection for multi-agent tasks
**Why:** The LoRA experiment proves forward-looking reasoning causally sustains cooperation. We can achieve a similar effect via prompt engineering without fine-tuning.
**Implementation:** Prepend a system prompt fragment to all multi-agent subagents:
> "You are optimizing long-term cumulative reward. Focus on future reciprocity and sustained cooperation. Do not over-index on isolated past defections. Trust can be repaired."
**Expected benefit:** Increases forward-looking ratio in CoT traces, reducing memory curse severity.
**Risk:** May be ignored by weaker models. Test on our target model before deployment.

### R3. Implement memory sanitization / synthetic cooperation history for trust repair
**Why:** Sanitization experiment shows replacing negative history with synthetic cooperative records restores cooperation.
**Implementation:** When a subagent has experienced a sequence of failures or conflicts, reset its visible history to a synthetic "successful cooperation" template before the next task. Keep only the most recent 1–2 real rounds.
**Expected benefit:** Breaks retaliatory cycles. Useful for long-running agent loops that occasionally hit errors.
**Risk:** Could hide real failure modes. Use only for transient noise, not persistent bugs.

### R4. Make CoT optional for cooperation-critical subagents
**Why:** No-CoT ablation shows explicit reasoning can amplify the memory curse by 60–90 pp in some settings.
**Implementation:** For subagents whose primary goal is coordination (not complex reasoning), allow a `reasoning: minimal` mode that outputs decisions without enumerating historical grievances.
**Expected benefit:** Reduces path-dependent retaliation in coordination tasks.
**Risk:** Loses interpretability. Log reasoning internally but do not expose it to the agent's own context window.

### R5. Validate on our actual multi-agent setup before changing defaults
**Why:** Results are on 2–3 player social dilemmas (PD, TD, PG, TG). Our supervisor-agent architecture is not a game-theoretic dilemma.
**Implementation:** Run a controlled experiment: two subagents collaborate on a 20-turn document-editing task. Vary history length (2 vs 10 vs 20 turns). Measure:
- Task completion rate
- Cooperation keywords in logs
- Frequency of "redo" / "reject" cycles (proxy for defection)
**Acceptance:** If HL=10 shows ≥20% more reject cycles than HL=2, adopt R1 cap. If not, document negative result and keep current defaults.

---

## Conversion Path to Tier 2

1. **Immediate (this week):** Implement R5 — controlled history-length experiment on our supervisor-agent repo. Use two subagents on a real task (e.g., code review loop). Measure reject-cycle frequency at HL=2, 5, 10. Target: 2–3 hours.
2. **Short-term (next 2 weeks):** If R5 shows degradation, implement R1 (history cap) + R2 (forward-looking prompt) as configurable options. Benchmark on 10 real tasks vs baseline.
3. **Medium-term (next month):** If benchmark shows ≥15% improvement in task completion or cooperation metrics, open a PR. If not, document rejection with evidence and close the line of inquiry.

---

## Cross-Reference with arXiv:2605.07990 (Tool Steering)

- Tool-steering paper (autonomy-nle2) recommends **short prompts** for tool routing (K=1500 tools without descriptions vs K=750 collapse with descriptions).
- Memory-curse paper recommends **short history** for multi-agent cooperation (HL≤5 optimal).
- **Combined implication:** Our supervisor-agent should use **compressed context** for both tool selection and subagent coordination. Verbatim history is a liability in both domains.
- **Unified design:** A context-compression layer that summarizes history + tool descriptions into a fixed-size embedding or summary, rather than passing full transcripts.

---

## Source Validation

- [x] Primary source downloaded and read: arXiv:2605.08060 PDF (15 pages)
- [x] All key claims traced to specific tables/figures in the paper
- [x] Limitations section explicitly acknowledged
- [x] Recommendations tied to our stack (Hermes supervisor-agent, multi-agent dispatch)
- [x] Concrete next experiment defined (R5 validation)
- [x] Cross-referenced with related paper (arXiv:2605.07990)

---

## Status

**Research lead → Tier 1.5** (high-signal, provenance-backed, clear conversion path to Tier 2 via R5 validation experiment).
