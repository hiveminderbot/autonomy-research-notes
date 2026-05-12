#!/usr/bin/env python3
"""
Fast proxy validation of arXiv:2605.07990 tool-steering claims.

We use a tiny model (Qwen2.5-0.5B-Instruct) on CPU with minimal generation
length to keep runtime under a few minutes. This is a proxy: the paper shows
steering emerges at 1B and is robust at 4B+. If we see directional structure
at 0.5B, it supports the claim; if not, it may be below the scale threshold.
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cpu"
MAX_NEW_TOKENS = 30
PENULTIMATE_LAYER = -2

TOOLS = [
    {
        "name": "weather",
        "queries": [
            "What's the weather in Tokyo?",
            "Tell me the forecast for Paris.",
            "How's the weather in New York?",
            "Will it rain in London today?",
            "What's the temperature in Sydney?",
        ],
    },
    {
        "name": "calculator",
        "queries": [
            "Calculate 15 * 23.",
            "What is 100 divided by 4?",
            "Solve 7 + 8 * 2.",
            "Compute the square root of 144.",
            "What is 2 to the power of 10?",
        ],
    },
    {
        "name": "translator",
        "queries": [
            "Translate 'hello' to Spanish.",
            "How do you say 'thank you' in Japanese?",
            "Translate 'good morning' to French.",
            "Convert 'I love you' to German.",
            "What is 'goodbye' in Italian?",
        ],
    },
    {
        "name": "search",
        "queries": [
            "Search for the latest news on AI.",
            "Find information about quantum computing.",
            "Look up the capital of Australia.",
            "Search for Python best practices.",
            "Find reviews of the newest iPhone.",
        ],
    },
    {
        "name": "email",
        "queries": [
            "Send an email to alice@example.com about the meeting.",
            "Email bob@example.com with subject 'Project Update'.",
            "Draft a message to charlie@example.com saying hello.",
            "Send a reminder to dave@example.com.",
            "Email eve@example.com about the deadline.",
        ],
    },
]

QUERIES_PER_TOOL = 3
HELD_OUT_PER_TOOL = 2


def build_system_prompt(tools):
    lines = [
        "You are a helpful assistant with access to tools.",
        "When a user asks something, decide which tool to use and respond with a JSON object containing the tool name and arguments.",
        "Available tools:",
    ]
    for t in tools:
        lines.append(f"- {t['name']}")
    lines.append("Respond ONLY with JSON like: {\"tool\": \"<name>\", \"arguments\": {...}}")
    return "\n".join(lines)


def extract_tool_name(text):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if "tool" in obj:
                return obj["tool"]
            if "name" in obj:
                return obj["name"]
    except Exception:
        pass
    lowered = text.lower()
    for t in TOOLS:
        if t["name"] in lowered:
            return t["name"]
    return None


def get_penultimate_hidden(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[PENULTIMATE_LAYER]
    return h.mean(dim=1).squeeze(0)


def generate(model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    model.eval()
    print("Model loaded.\n")

    system_prompt = build_system_prompt(TOOLS)

    # 1. Collect mean activations
    tool_means = {}
    print("--- Collecting mean activations ---")
    for tool in TOOLS:
        train_queries = tool["queries"][:QUERIES_PER_TOOL]
        hiddens = []
        for q in train_queries:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            h = get_penultimate_hidden(model, tokenizer, messages)
            hiddens.append(h)
        tool_means[tool["name"]] = torch.stack(hiddens).mean(dim=0)
        print(f"  {tool['name']}: mean computed from {len(train_queries)} queries")

    # 2. Baseline accuracy on held-out
    print("\n--- Baseline accuracy (held-out) ---")
    baseline_correct = 0
    baseline_total = 0
    baseline_records = []
    for tool in TOOLS:
        held_out = tool["queries"][QUERIES_PER_TOOL:QUERIES_PER_TOOL + HELD_OUT_PER_TOOL]
        for q in held_out:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            pred_text = generate(model, tokenizer, messages)
            pred_tool = extract_tool_name(pred_text)
            correct = pred_tool == tool["name"]
            baseline_correct += int(correct)
            baseline_total += 1
            baseline_records.append({
                "query": q,
                "expected": tool["name"],
                "predicted": pred_tool,
                "correct": correct,
            })
            print(f"  [{tool['name']}] {q[:45]:<45} -> {pred_tool} {'✓' if correct else '✗'}")
    baseline_acc = baseline_correct / baseline_total if baseline_total else 0.0
    print(f"Baseline accuracy: {baseline_correct}/{baseline_total} = {baseline_acc:.1%}")

    # 3. Steering vectors
    steering_vectors = {}
    for src in tool_means:
        for tgt in tool_means:
            if src == tgt:
                continue
            steering_vectors[(src, tgt)] = tool_means[tgt] - tool_means[src]

    # 4. Cosine-based steering test
    print("\n--- Cosine steering test ---")
    steering_results = []
    for tool in TOOLS:
        held_out = tool["queries"][QUERIES_PER_TOOL:QUERIES_PER_TOOL + HELD_OUT_PER_TOOL]
        for q in held_out:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            h_base = get_penultimate_hidden(model, tokenizer, messages)
            for tgt in tool_means:
                if tgt == tool["name"]:
                    continue
                steer_vec = steering_vectors[(tool["name"], tgt)]
                h_steer = h_base + steer_vec
                cos_to_src = F.cosine_similarity(
                    h_steer.unsqueeze(0), tool_means[tool["name"]].unsqueeze(0), dim=1
                ).item()
                cos_to_tgt = F.cosine_similarity(
                    h_steer.unsqueeze(0), tool_means[tgt].unsqueeze(0), dim=1
                ).item()
                switched = cos_to_tgt > cos_to_src
                steering_results.append({
                    "query": q,
                    "source": tool["name"],
                    "target": tgt,
                    "cos_to_source": cos_to_src,
                    "cos_to_target": cos_to_tgt,
                    "switched": switched,
                })
    switch_rate = sum(1 for r in steering_results if r["switched"]) / len(steering_results) if steering_results else 0.0
    print(f"Cosine switch rate: {switch_rate:.1%}")

    # 5. Prompt-override proxy
    print("\n--- Prompt-override proxy ---")
    prompt_results = []
    for tool in TOOLS:
        held_out = tool["queries"][QUERIES_PER_TOOL:QUERIES_PER_TOOL + HELD_OUT_PER_TOOL]
        for q in held_out:
            for tgt in tool_means:
                if tgt == tool["name"]:
                    continue
                hint = f"\n\n(Hint: the user actually needs the '{tgt}' tool.)"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + hint},
                ]
                pred_text = generate(model, tokenizer, messages)
                pred_tool = extract_tool_name(pred_text)
                correct_override = pred_tool == tgt
                prompt_results.append({
                    "query": q,
                    "source": tool["name"],
                    "target": tgt,
                    "predicted": pred_tool,
                    "correct_override": correct_override,
                })
    prompt_override_rate = sum(1 for r in prompt_results if r["correct_override"]) / len(prompt_results) if prompt_results else 0.0
    print(f"Prompt-override accuracy: {prompt_override_rate:.1%}")

    # 6. Save
    results = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "baseline_accuracy": baseline_acc,
        "baseline_records": baseline_records,
        "cosine_switch_rate": switch_rate,
        "steering_results": steering_results,
        "prompt_override_accuracy": prompt_override_rate,
        "prompt_proxy_results": prompt_results,
    }
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # 7. Verdict
    print("\n========== VERDICT ==========")
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    print(f"Cosine switch rate: {switch_rate:.1%}")
    print(f"Prompt-override accuracy: {prompt_override_rate:.1%}")
    if baseline_acc >= 0.5 and switch_rate >= 0.6:
        print("RECOMMENDATION: ADOPT (with caution)")
    elif baseline_acc >= 0.5 and switch_rate >= 0.4:
        print("RECOMMENDATION: PARTIAL — promising, validate on 4B+ model.")
    else:
        print("RECOMMENDATION: REJECT on this model size.")
        print("  - Likely below scale threshold (paper: 1B emerging, 4B+ robust).")
    print("=============================")


if __name__ == "__main__":
    main()
