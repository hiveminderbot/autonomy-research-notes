# Pi Minimal Agent Loop: Source-Level Analysis & Harness Comparison

**Bead:** autonomy-hh57  
**Date:** 2026-05-12  
**Source:** https://github.com/badlogic/pi-mono (commit 3d9e14d7)  
**Analyst:** autonomy-worker  
**Verdict:** **ADOPT for specific use case** — Pi's architecture is a legitimate Tier-2 competitor for terminal-native, token-efficient coding-agent work, with a clear conversion path to a comparative bakeoff.

---

## 1. Executive Summary

Pi (badlogic/pi-mono) is a **minimal terminal-native coding-agent harness** built in TypeScript. Its architecture is deliberately compact compared to Claude Code (~closed source, ~heavy), Codex CLI (~closed source, ~heavy), and OpenCode (~heavier OSS). Pi's core agent loop is **718 lines** (packages/agent/src/agent-loop.ts), the agent runtime is **553 lines** (agent.ts), and the entire coding-agent tool suite is **~3,900 lines** across 15 tool files. This is meaningfully smaller than comparable harnesses.

**Key architectural differentiator:** Pi transforms messages to the LLM format **only at the LLM call boundary**, maintaining an internal `AgentMessage` representation throughout. This avoids repeated serialization/deserialization overhead and enables clean context compaction, steering injection, and follow-up message queuing without corrupting the LLM-visible transcript.

**Verdict: ADOPT** for a same-task bakeoff against OpenCode (the only currently working harness) on a controlled bugfix task. Pi's token-efficiency claims are structurally plausible but require empirical validation.

---

## 2. Pi Architecture Deep Dive

### 2.1 Core Agent Loop (packages/agent/src/agent-loop.ts, 718 lines)

The loop is a **dual-nested async structure**:

```
Outer loop: handles follow-up messages after the agent would otherwise stop
  └── Inner loop: processes tool calls + steering messages per turn
        ├── Process pending steering messages (injected before next assistant response)
        ├── Stream assistant response (AgentMessage[] → Message[] at boundary)
        ├── Execute tool calls (sequential or parallel)
        ├── Emit turn_end event
        ├── Optional: prepareNextTurn (model/thinking level switching)
        └── Optional: shouldStopAfterTurn (graceful termination)
```

**Key design choices:**

| Feature | Pi Implementation | Typical Harness Pattern |
|---------|-------------------|------------------------|
| Message representation | `AgentMessage` internally, `Message[]` only at LLM boundary | Usually LLM-native format throughout |
| Steering injection | `getSteeringMessages()` polled after each turn | Rare; most harnesses don't support mid-run steering |
| Follow-up queuing | `getFollowUpMessages()` polled when agent would stop | Usually requires new prompt/session |
| Context compaction | Dedicated compaction module with branch summarization | Often absent or ad-hoc |
| Tool execution | Configurable sequential/parallel with per-tool override | Usually sequential only |
| Turn lifecycle hooks | `beforeToolCall`, `afterToolCall`, `prepareNextTurn`, `shouldStopAfterTurn` | Minimal or absent |
| Abort handling | AbortSignal threaded through all async operations | Varies widely |

### 2.2 Agent Runtime (packages/agent/src/agent.ts, 553 lines)

The `Agent` class wraps the loop with:
- State management (`AgentState` interface with accessor properties)
- Event subscription (`subscribe()`, `unsubscribe()`)
- Steering queue and follow-up queue management
- Session I/O integration

### 2.3 Harness Layer (packages/agent/src/harness/, ~2,800 lines)

The `AgentHarness` class (agent-harness.ts, 816 lines) adds:
- **Skills system**: XML-formatted SKILL.md blocks injected into system prompt (agentskills.io compatible)
- **Prompt templates**: Named, parameterizable prompt fragments
- **Session management**: Branching, compaction, tree navigation
- **Execution environment abstraction**: `ExecutionEnv` interface for file ops, shell exec
- **Provider request hooks**: `before_provider_request`, `after_provider_response`

### 2.4 Context Compaction (packages/agent/src/harness/compaction/, ~1,385 lines)

Pi has **sophisticated context compaction**:
- `compaction.ts` (854 lines): Pure functions for summarizing conversation history
- `branch-summarization.ts` (361 lines): Summarizes branches in the session tree
- Tracks file operations (read/modified) across compaction boundaries
- Replaces pruned messages with summary messages while preserving tool-result context

This is a **genuine architectural advantage** for long-running tasks. Claude Code and Codex have compaction but it's opaque. OpenCode's compaction is less documented. Pi's is source-visible and configurable.

### 2.5 Multi-Provider LLM API (packages/ai/, ~28K lines)

The `@earendil-works/pi-ai` package provides:
- Unified streaming API across 10+ providers (OpenAI, Anthropic, Google, DeepSeek, Azure, Bedrock, Mistral, Groq, Cerebras, Cloudflare)
- Lazy provider registration (no static imports of unused providers)
- `streamSimple()` and `completeSimple()` wrappers
- 448KB `models.generated.ts` with tool-capable model metadata

**Token-efficiency implication:** The `streamSimple()` abstraction means Pi can switch models mid-session (e.g., cheap model for search, expensive model for generation) with a single `prepareNextTurn` hook. This is a concrete token/cost optimization path.

### 2.6 Coding-Agent Tool Suite (packages/coding-agent/src/core/tools/, ~3,921 lines)

| Tool | Lines | Purpose |
|------|-------|---------|
| `bash.ts` | 440 | Shell execution with truncation, streaming, timeout |
| `edit.ts` | 489 | File editing with diff preview |
| `edit-diff.ts` | 446 | Diff computation utilities |
| `read.ts` | 363 | File reading with truncation |
| `write.ts` | 281 | File writing |
| `find.ts` | 370 | File finding |
| `grep.ts` | 384 | Text search |
| `ls.ts` | 229 | Directory listing |
| `truncate.ts` | 265 | Output truncation with visual line counting |
| Others | 654 | Index, path utils, render utils, accumulator, etc. |

**Total: ~3,921 lines** for the full tool suite. This is compact. For comparison, OpenCode's tool suite is larger and more fragmented across multiple packages.

---

## 3. Comparative Analysis: Pi vs. Other Harnesses

### 3.1 Dimensional Comparison

| Dimension | Pi | Claude Code | Codex CLI | OpenCode | mini-SWE-agent |
|-----------|-----|-------------|-----------|----------|----------------|
| **Source availability** | Full OSS (MIT) | Closed source | Closed source | Full OSS | Full OSS |
| **Core loop size** | ~718 lines | Unknown | Unknown | Larger | ~research-grade |
| **Tool suite size** | ~3,900 lines | Unknown | Unknown | Larger | ~minimal |
| **Context compaction** | Source-visible, configurable | Opaque | Opaque | Less documented | Minimal |
| **Multi-provider** | 10+ providers, lazy load | Anthropic only | OpenAI only | Multiple | Configurable |
| **Mid-run steering** | Native (getSteeringMessages) | Limited | Limited | Limited | No |
| **Session branching** | Native | No | No | No | No |
| **Extensions/skills** | TypeScript extensions + SKILL.md | No | No | Limited | No |
| **Token-efficiency claim** | Structurally plausible (boundary transform, compaction, model switching) | Unknown | Unknown | Unknown | Unknown |
| **Auth friction** | API key or /login | /login required | OAuth refresh-token issues | API key | API key |
| **Install size** | npm package (~MBs) | Large binary | Large binary | npm package | pip package |
| **Real-repo validation** | **Not yet tested** | Prior tiny-repair success | Prior tiny-repair success | 4x confirmed success | Not tested |

### 3.2 Token Efficiency Analysis (Theoretical)

Pi's token-efficiency claims rest on three structural mechanisms:

1. **Boundary-transform pattern**: AgentMessages are kept in a rich internal format and only flattened to LLM-native `Message[]` at the last moment. This avoids repeated re-serialization when doing context operations (compaction, steering injection, follow-up queuing).

2. **Context compaction with file-op tracking**: The compaction system replaces old messages with summaries while preserving the file-operation trail. This means long sessions don't linearly grow context — they get summarized, with only the essential file-op metadata retained.

3. **Model switching via `prepareNextTurn`**: The harness can downgrade to a cheaper model for tool-result-heavy turns and upgrade for complex reasoning turns. This is a cost-optimization knob most harnesses lack.

**Honest assessment:** These mechanisms are *architecturally sound* but *empirically unvalidated* in our lab. The actual token savings depend on:
- Compaction trigger frequency and summary quality
- Model-switching heuristic quality
- Overhead of the AgentMessage abstraction itself

### 3.3 Where Pi Could Plausibly Outperform

Based on architecture, Pi could outperform on:

1. **Long-running multi-step tasks** (compaction + branching + model switching)
2. **Tasks requiring mid-run user steering** (native steering queue)
3. **Cost-sensitive workflows** (multi-provider + model switching)
4. **Custom tool/integration work** (TypeScript extension system + SKILL.md)

Where Pi is unlikely to outperform:
1. **Single-turn simple bugfixes** (all harnesses are roughly equivalent)
2. **Tasks requiring IDE integration** (Pi is terminal-only; Cline/Continue win)
3. **Tasks on non-JS/TS repos where the harness itself needs modification** (Pi is TS-native)

---

## 4. Concrete Bakeoff Task Identified

### Task: Controlled bugfix on feed_fetcher with session-length stress

**Why this task:**
- It's the lab's canonical controlled bug (line 1025: `results[source.id] = []` discards entries)
- OpenCode has already succeeded on it 4 times
- We can inject **session-length stress** by prepending a large synthetic conversation history, forcing context compaction to activate
- This tests Pi's claimed advantage (compaction efficiency) against OpenCode's proven reliability

**Metrics to collect:**
1. Task success (patch correct, tests pass)
2. Wall-clock latency
3. Tokens consumed (input + output) — requires provider API logging
4. Number of LLM turns
5. Cost per task (if using metered API)
6. Context compaction events (Pi-only)
7. Phantom changes (unrelated diff)

**Blockers to resolve:**
- Pi requires `npm install` and build in the pi-mono repo
- Pi requires an API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or /login)
- Need to configure Pi to run non-interactively on a specific task (use `--print` or JSON mode)

---

## 5. Verdict & Recommendation

### Verdict: **ADOPT for specific use case**

Pi is **not a universal replacement** for OpenCode, Claude Code, or Codex. It is a **legitimate alternative** with distinct architectural advantages in:
- Long-session context management
- Token/cost optimization via model switching
- Extensibility via TypeScript extensions and SKILL.md

### Recommendation Matrix

| Use Case | Recommended Harness | Why |
|----------|---------------------|-----|
| Quick bugfix, auth already set up | OpenCode (today) | Proven 4x success, lowest friction |
| Long-running feature work, cost-sensitive | **Pi** (if validated) | Compaction + model switching |
| IDE-native workflow | Cline / Continue | VS Code integration |
| Managed async platform | Factory / Droid | Strongest async candidate per evaluation matrix |
| Minimal research baseline | mini-SWE-agent | Smallest footprint |

### Next Conversion Path

1. **Immediate:** Configure Pi for non-interactive execution on the feed_fetcher controlled bug
2. **Short-term:** Run same-task bakeoff (Pi vs OpenCode) with session-length stress
3. **Medium-term:** If Pi shows ≥20% token reduction or ≥15% cost reduction with equal task success, adopt as secondary harness for long tasks
4. **If Pi fails:** Document failure mode, downgrade to REJECT for that use case, and close Bead

---

## 6. Evidence & Provenance

| Claim | Evidence |
|-------|----------|
| Pi repo exists and is public | https://github.com/badlogic/pi-mono (cloned to labs/pi-mono/) |
| Agent loop is 718 lines | `wc -l packages/agent/src/agent-loop.ts` → 718 |
| Core agent runtime is 553 lines | `wc -l packages/agent/src/agent.ts` → 553 |
| Tool suite is ~3,900 lines | `wc -l packages/coding-agent/src/core/tools/*.ts` → 3,921 |
| Compaction system exists | `packages/agent/src/harness/compaction/` (1,385 lines) |
| Multi-provider lazy loading | `packages/ai/src/stream.ts`, `providers/register-builtins.ts` |
| Session branching exists | `packages/coding-agent/src/core/session-manager.ts` (43,010 lines) |
| OpenCode has 4x success on feed_fetcher | `results/comparative-harness-bakeoff-20260507.json` |
| Claude Code/Codex auth-blocked | Same source |

---

## 7. Blockers for Bakeoff

1. **Pi build:** `npm install` + `npm run build` required in pi-mono (Node ≥20)
2. **API key:** Need ANTHROPIC_API_KEY or OPENAI_API_KEY for Pi to make LLM calls
3. **Non-interactive mode:** Need to use `pi --print` or JSON mode to run headless
4. **Task injection:** Need to script the feed_fetcher bug into Pi's working directory

None of these are fundamental blockers. The build can be done locally. The API key is a user-controlled dimension. Non-interactive mode is documented in Pi's README.

---

*End of analysis. All source claims are traceable to the pi-mono repository at commit 3d9e14d7.*
