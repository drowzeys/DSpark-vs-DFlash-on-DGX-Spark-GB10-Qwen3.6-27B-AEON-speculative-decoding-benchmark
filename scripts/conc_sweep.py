#!/usr/bin/env python3
"""Concurrency sweep C=1..18 for DFlash(.4) vs DSpark(.3), twin GB10.

At each concurrency level C, fire C simultaneous requests (200 tok, temp=1.0),
measure aggregate tok/s = sum(completion_tokens)/wall_time and spec-decode
acceptance from /metrics delta. Both arms swept in parallel (separate nodes).
"""
import os, json, time, urllib.request, threading, argparse
from concurrent.futures import ThreadPoolExecutor

# Override endpoints via env: DFLASH_URL / DSPARK_URL (each arm on its own host/port).
ARMS = {"DFlash": os.environ.get("DFLASH_URL", "http://localhost:8010"),
        "DSpark": os.environ.get("DSPARK_URL", "http://localhost:8011")}

POOL = [
    "Write a Python function that merges two sorted lists into one sorted list.",
    "A train travels 60 km in 45 minutes. What is its speed in km/h? Show your work.",
    "Explain how a hash map works to a beginner, with an analogy.",
    "Implement binary search in Python and explain its time complexity.",
    "Find all real solutions of x^2 - 5x + 6 = 0 and explain.",
    "What are the trade-offs between TCP and UDP?",
    "Write a Python decorator that memoizes a function's results.",
    "Compute the derivative of f(x) = x^3 - 2x^2 + 7 and evaluate at x=2.",
    "Describe the difference between processes and threads.",
    "Implement a LRU cache class in Python using OrderedDict.",
]


def post(base, prompt, temp, max_tokens, seed, timeout=300):
    body = {"model": "aeon27b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "seed": seed}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["usage"]["completion_tokens"], time.perf_counter() - t0


def metrics(base):
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
            txt = r.read().decode()
    except Exception:
        return (0.0, 0.0)
    dft = acc = 0.0
    for ln in txt.splitlines():
        if ln.startswith("vllm:spec_decode_num_draft_tokens_total"): dft = float(ln.split()[-1])
        elif ln.startswith("vllm:spec_decode_num_accepted_tokens_total"): acc = float(ln.split()[-1])
    return (dft, acc)


def sweep_arm(name, base, levels, temp, max_tokens, out):
    # warmup
    try: post(base, "hi", temp, 8, 0)
    except Exception as e: print(f"[{name}] warmup fail {e}", flush=True)
    for C in levels:
        d0, a0 = metrics(base)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=C) as ex:
            futs = [ex.submit(post, base, POOL[i % len(POOL)], temp, max_tokens, 1000 + C * 100 + i)
                    for i in range(C)]
            per = [f.result() for f in futs]
        wall = time.perf_counter() - t0
        d1, a1 = metrics(base)
        toks = sum(p[0] for p in per)
        dd, da = d1 - d0, a1 - a0
        agg = round(toks / wall, 1)
        per_req = round(sum(p[0] / p[1] for p in per) / len(per), 1)
        acc = round(da / dd, 4) if dd else 0.0
        out[C] = {"agg_tok_s": agg, "per_req_tok_s": per_req, "accept": acc, "toks": toks, "wall": round(wall, 2)}
        print(f"[{name}] C={C:2d}  agg={agg:6.1f} tok/s  per_req={per_req:5.1f}  accept={acc:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-c", type=int, default=18)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    levels = list(range(1, a.max_c + 1))
    results = {arm: {} for arm in ARMS}
    threads = []
    for arm, base in ARMS.items():
        th = threading.Thread(target=sweep_arm, args=(arm, base, levels, a.temp, a.max_tokens, results[arm]))
        th.start(); threads.append(th)
    for th in threads: th.join()
    json.dump(results, open("conc_sweep.json", "w"), indent=1)
    # table
    print("\n" + "=" * 74)
    print(f"{'C':>3} | {'DFlash agg':>10} {'acc':>7} | {'DSpark agg':>10} {'acc':>7} | {'Δ agg':>7}")
    print("-" * 74)
    for C in levels:
        df = results["DFlash"].get(C, {}); ds = results["DSpark"].get(C, {})
        dfa, dsa = df.get("agg_tok_s", 0), ds.get("agg_tok_s", 0)
        delta = f"{(dsa/dfa-1)*100:+.1f}%" if dfa else "-"
        print(f"{C:>3} | {dfa:>10.1f} {df.get('accept',0):>7.3f} | {dsa:>10.1f} {ds.get('accept',0):>7.3f} | {delta:>7}")
    print("[done] wrote conc_sweep.json")


if __name__ == "__main__":
    main()
