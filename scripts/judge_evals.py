#!/usr/bin/env python3
"""LLM-as-judge: compare expected vs RustyRAG answers using Cerebras Qwen."""

import asyncio
import json
import os
import sys
import time

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "qwen-3-235b-a22b-instruct-2507"
MAX_RETRIES = 3
RETRY_DELAY = 1.0
CONCURRENCY = 10

JUDGE_PROMPT = """\
You are an impartial judge evaluating whether an AI assistant's answer is correct.

Compare the expected answer with the AI's actual answer. The AI answer does NOT need to be word-for-word identical. It passes if it conveys the same core facts, meaning, and conclusions. Minor differences in wording, extra detail, or slightly different phrasing are acceptable as long as the key information is correct.

It FAILS if:
- The core facts are wrong or contradicted
- Key information from the expected answer is missing
- The answer is about a completely different topic

Question: {question}

Expected answer: {expected}

AI answer: {actual}

Respond with ONLY "True" if the answer is correct, or "False" if it is not. No explanation."""


def judge(session, api_key: str, question: str, expected: str, actual: str) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content.lower().startswith("true")
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                print(f"    FAILED after {MAX_RETRIES} attempts: {e}", flush=True)
                return False


def judge_one(session, api_key, idx, total, r):
    """Judge a single result. Returns (idx, judgment_dict)."""
    if r["rustyrag_answer"].startswith("ERROR:"):
        print(f"  [{idx+1}/{total}] SKIP (error)", flush=True)
        return idx, {"question": r["question"], "verdict": "SKIP"}, None, None

    verdict = judge(session, api_key, r["question"], r["expected_answer"], r["rustyrag_answer"])
    symbol = "PASS" if verdict else "FAIL"
    print(f"  [{idx+1}/{total}] {symbol}", flush=True)

    return idx, {
        "question": r["question"],
        "expected_answer": r["expected_answer"],
        "rustyrag_answer": r["rustyrag_answer"],
        "verdict": verdict,
        "ttft_ms": r["ttft_ms"],
        "total_ms": r["total_ms"],
    }, r["ttft_ms"], r["total_ms"]


def main():
    eval_path = sys.argv[1] if len(sys.argv) > 1 else "docs/eval_results.json"

    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if line.startswith("CEREBRAS_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        print("Error: CEREBRAS_API_KEY not set in environment or .env")
        sys.exit(1)

    with open(eval_path) as f:
        data = json.load(f)

    results = data["results"]
    total = len(results)
    print(f"Judging {total} eval results with {MODEL} ({CONCURRENCY} concurrent)...\n", flush=True)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=CONCURRENCY, pool_maxsize=CONCURRENCY)
    session.mount("https://", adapter)

    judgments = [None] * total
    passed = 0
    failed = 0
    ttft_values = []
    total_time_values = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(judge_one, session, api_key, i, total, r): i
            for i, r in enumerate(results)
        }

        for future in as_completed(futures):
            idx, judgment, ttft, total_ms = future.result()
            judgments[idx] = judgment

            if judgment.get("verdict") == "SKIP":
                continue
            elif judgment["verdict"]:
                passed += 1
            else:
                failed += 1

            if ttft is not None:
                ttft_values.append(ttft)
                total_time_values.append(total_ms)

    # Stats
    judged = passed + failed
    pass_pct = (passed / judged * 100) if judged > 0 else 0
    fail_pct = (failed / judged * 100) if judged > 0 else 0
    avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0
    avg_total = sum(total_time_values) / len(total_time_values) if total_time_values else 0

    print(f"\n{'='*50}", flush=True)
    print(f"Results Summary", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Total questions:  {total}", flush=True)
    print(f"Judged:           {judged}", flush=True)
    print(f"Passed:           {passed}/{judged} ({pass_pct:.1f}%)", flush=True)
    print(f"Failed:           {failed}/{judged} ({fail_pct:.1f}%)", flush=True)
    print(f"Avg TTFT:         {avg_ttft:.0f}ms", flush=True)
    print(f"Avg Total Time:   {avg_total:.0f}ms", flush=True)
    print(f"{'='*50}", flush=True)

    # Save detailed results
    output = {
        "model": MODEL,
        "total_questions": total,
        "judged": judged,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_pct, 1),
        "fail_rate": round(fail_pct, 1),
        "avg_ttft_ms": round(avg_ttft),
        "avg_total_ms": round(avg_total),
        "judgments": judgments,
    }

    out_path = eval_path.replace("eval_results", "eval_judged")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
