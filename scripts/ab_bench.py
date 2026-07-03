#!/usr/bin/env python3
"""2-node side-by-side A/B: DFlash (.4) vs DSpark (.3) on the same base NVFP4 target.

Mirrors AEON's published methodology (single-stream, K=8, temp=1.0, per-domain),
but the ARMS run on twin GB10 nodes concurrently. Reports per-domain tok/s and
spec-decode acceptance (accepted/draft from /metrics delta) for each arm, plus
the DSpark-vs-DFlash delta. Absolute tok/s is GB10 bandwidth-bound (~273 GB/s),
so the meaningful, hardware-independent comparisons are ACCEPT RATE and the
RELATIVE delta.
"""
import os, json, time, urllib.request, threading, argparse
from collections import defaultdict

# Endpoints for the two arms. Override via env for your own hosts, e.g.
#   DFLASH_URL=http://nodeA:8010 DSPARK_URL=http://nodeB:8010 python3 ab_bench.py
ARMS = {
    "DFlash": os.environ.get("DFLASH_URL", "http://localhost:8010"),
    "DSpark": os.environ.get("DSPARK_URL", "http://localhost:8011"),
}

PROMPTS = {
    "code": [
        "Write a Python function that merges two sorted lists into one sorted list.",
        "Implement binary search in Python and explain its time complexity.",
        "Write a Python decorator that memoizes a function's results.",
        "Implement a LRU cache class in Python using OrderedDict.",
        "Write a function to detect a cycle in a singly linked list.",
        "Parse a CSV string into a list of dicts in Python without the csv module.",
    ],
    "math": [
        "A train travels 60 km in 45 minutes. What is its speed in km/h? Show your work.",
        "Find all real solutions of x^2 - 5x + 6 = 0 and explain.",
        "What is the sum of the first 50 positive even numbers? Show the reasoning.",
        "If 3 workers build a wall in 12 hours, how long for 4 workers? Explain.",
        "Compute the derivative of f(x) = x^3 - 2x^2 + 7 and evaluate at x=2.",
        "A bag has 4 red and 6 blue balls. Probability of drawing two red without replacement?",
    ],
    "chat": [
        "Explain how a hash map works to a beginner, with an analogy.",
        "What are the trade-offs between TCP and UDP?",
        "Describe the difference between processes and threads.",
        "Explain what a database index is and when to use one.",
        "Summarize how HTTPS keeps a connection secure.",
        "What is the CAP theorem and why does it matter for distributed systems?",
    ],
}


def post(base, body, timeout=180):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.perf_counter() - t0


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


def run_arm_domain(base, prompts, temp, max_tokens, seed, result, key):
    """Single-stream over the domain's prompts; bracket accept metrics."""
    d0, a0 = metrics(base)
    toks = 0; wall = 0.0; n = 0
    for i, p in enumerate(prompts):
        body = {"model": "aeon27b", "messages": [{"role": "user", "content": p}],
                "max_tokens": max_tokens, "temperature": temp, "seed": seed + i}
        try:
            d, dt = post(base, body)
            toks += d["usage"]["completion_tokens"]; wall += dt; n += 1
        except Exception as e:
            result[key] = {"error": str(e)}; return
    d1, a1 = metrics(base)
    dd, da = d1 - d0, a1 - a0
    result[key] = {"tok_s": round(toks / wall, 2) if wall else 0.0,
                   "accept_rate": round(da / dd, 4) if dd else 0.0,
                   "mean_accept_len": round((da / dd) * 8 if dd else 0.0, 2),
                   "toks": toks, "wall": round(wall, 2), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--warmup", action="store_true", default=True)
    a = ap.parse_args()

    # warmup both arms
    for base in ARMS.values():
        try: post(base, {"model": "aeon27b", "messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 8, "temperature": a.temp, "seed": 0})
        except Exception as e: print(f"[warn] warmup failed {base}: {e}")

    agg = defaultdict(lambda: defaultdict(list))  # domain -> arm -> [round dicts]
    for rnd in range(1, a.rounds + 1):
        for dom, prompts in PROMPTS.items():
            res = {}
            threads = []
            for arm, base in ARMS.items():
                th = threading.Thread(target=run_arm_domain,
                                      args=(base, prompts, a.temp, a.max_tokens, a.seed + rnd * 1000, res, arm))
                th.start(); threads.append(th)
            for th in threads: th.join()
            for arm in ARMS:
                agg[dom][arm].append(res.get(arm, {}))
            line = f"[r{rnd}] {dom:6s} "
            for arm in ARMS:
                r = res.get(arm, {})
                line += f"| {arm}: {r.get('tok_s','ERR')} tok/s acc={r.get('accept_rate','-')} "
            print(line, flush=True)

    # aggregate: mean across rounds
    def mean(xs): xs = [x for x in xs if x]; return sum(xs) / len(xs) if xs else 0.0
    print("\n" + "=" * 78)
    print(f"{'domain':7s} {'arm':7s} {'tok/s':>8s} {'accept':>8s} {'acc_len':>8s}   vs DFlash")
    print("-" * 78)
    summary = {}
    for dom in PROMPTS:
        base_ts = mean([r.get("tok_s", 0) for r in agg[dom]["DFlash"]])
        for arm in ARMS:
            ts = mean([r.get("tok_s", 0) for r in agg[dom][arm]])
            acc = mean([r.get("accept_rate", 0) for r in agg[dom][arm]])
            al = mean([r.get("mean_accept_len", 0) for r in agg[dom][arm]])
            delta = f"{(ts/base_ts-1)*100:+.1f}%" if base_ts and arm != "DFlash" else ("baseline" if arm == "DFlash" else "-")
            summary[f"{dom}/{arm}"] = {"tok_s": round(ts, 2), "accept": round(acc, 4), "acc_len": round(al, 2)}
            print(f"{dom:7s} {arm:7s} {ts:8.2f} {acc:8.4f} {al:8.2f}   {delta}")
        print("-" * 78)
    # overall
    print("OVERALL (mean of domain tok/s):")
    for arm in ARMS:
        overall_ts = mean([mean([r.get("tok_s", 0) for r in agg[dom][arm]]) for dom in PROMPTS])
        overall_acc = mean([mean([r.get("accept_rate", 0) for r in agg[dom][arm]]) for dom in PROMPTS])
        print(f"  {arm:7s} tok/s={overall_ts:.2f}  accept={overall_acc:.4f}")
    json.dump(summary, open("ab_results.json", "w"), indent=1)
    print("\n[done] wrote ab_results.json")


if __name__ == "__main__":
    main()
