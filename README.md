# DSpark vs DFlash on DGX Spark (GB10) — Qwen3.6-27B-AEON speculative-decoding benchmark

A reproducible, twin-**GB10** side-by-side benchmark of **DSpark** (a VanillaMarkov
semi-autoregressive draft head) against bare **DFlash** block-diffusion drafting, on the
**AEON vLLM Ultimate 0.24.0** stack, serving the Qwen3.6-27B-AEON NVFP4 target.

> **TL;DR** — On bandwidth-bound GB10, DSpark's Markov head beats bare DFlash by
> **+13.9% aggregate single-stream tok/s** and higher acceptance in every domain, by
> **repairing DFlash's draft-suffix acceptance decay**. Absolute tok/s is GB10
> memory-bandwidth-bound (~273 GB/s) and *not* comparable to the high-bandwidth Blackwell
> numbers in the upstream repos — but **acceptance rate and the relative DSpark-vs-DFlash
> delta are hardware-independent**, and those reproduce (and slightly exceed) the upstream
> +10.9% headline.

---

## Credits & upstream

This work stands entirely on two upstream projects — all model weights, the drafting
method, and the serving image are theirs:

- **hiyak** — *DSpark VanillaMarkov semi-AR draft head + training/serving method.*
  - Repo: <https://github.com/hikarioyama/dspark-aeon-27b>
  - Draft weights: <https://huggingface.co/Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft>
- **Aeon Forge** — *AEON vLLM Ultimate 0.24.0 (sm_121a/GB10) image + the Qwen3.6-27B-AEON NVFP4 target.*
  - Repo: <https://github.com/AEON-7/vllm-ultimate-dgx-spark>
  - Container: `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0`
  - Models: <https://huggingface.co/AEON-7>
- **DFlash baseline draft** — z-lab: <https://huggingface.co/z-lab/Qwen3.6-27B-DFlash>

This repository is only the **GB10 port + benchmark harness + measurements**.

---

## Hardware & setup

| | |
|---|---|
| Nodes | 2× NVIDIA DGX Spark — **GB10**, sm_121a Blackwell, 128 GB unified LPDDR5X (~273 GB/s), 200G RoCE fabric |
| Topology | Twin-node side-by-side: **DFlash** on node A, **DSpark** on node B, identical config, swept concurrently |
| Image | AEON vLLM Ultimate — `vllm 0.24.0+aeon.sm121a.dflash` |
| Target | `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4` — base NVFP4, **SSM kept BF16** (26 GB; `Qwen3_5ForConditionalGeneration`, 64-layer hybrid 3×linear-attn + 1×full-attn, thinking model) |
| DFlash draft | `z-lab/Qwen3.6-27B-DFlash` (3.46 GB) |
| DSpark draft | `Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft` (3.7 GB, `markov_rank=256`, vanilla) |
| Serve flags | `K=8`, `draft_sample_method=probabilistic`, `--attention-backend flash_attn`, `--mamba-cache-dtype float32`, `--quantization compressed-tensors`, `--gpu-memory-utilization 0.85`, `temp=1.0` |

> The target is the **base** NVFP4 (SSM in BF16). The upstream serve script's
> `…-NVFP4-uniform` (SSM also FP4, its 161 tok/s regime) is an unpublished local requant;
> vLLM 0.24 prefers the BF16 SSM, so the base target is used as-is.

---

## The port: vLLM 0.23 → 0.24

DSpark ships drop-in patches targeting vLLM **0.23**; the AEON image is **0.24** and
DFlash-only (no Markov path). The port (`patches/do_port.py`, reproducible) **additively**
splices into the 0.24 stock files:

- `_VanillaMarkov` / `_GatedMarkovHead` low-rank transition-bias heads,
- the `markov_rank>0` head build in `DFlashQwen3Model.__init__`,
- `sample_draft_block_semiar` (greedy) and `sample_draft_block_semiar_sample` (temp>0, lossless) on `DFlashQwen3ForCausalLM`,
- `_compute_conf_prefix_lengths` (no-op here; this checkpoint has no confidence head).

The two patched files are **bind-mounted** into the container. Because the patch is
byte-identical / no-op for a draft without `markov_rank`, **both arms run the exact same
binary** — only the draft weights differ.

### Bug fixed along the way

The stock probabilistic draft path (`compute_probs_and_sample_next_token`, present in
**both 0.23 and 0.24**) does `logits.div_(temperature.view(-1, 1))` where the draft logits
are block-expanded to `[num_reqs × num_spec, V]` while `temperature` is per-request
`[num_reqs]`. This broadcasts only at `num_reqs == 1`; with **≥2 concurrent requests it
shape-mismatches and kills the engine**. The port block-expands temperature
(`repeat_interleave`) to the draft-block rows — required for the concurrency sweep below.

---

## Results

### Table 1 — Single-stream A/B (mean of 3 rounds, temp=1.0, K=8, 200 tok/req)

| Domain | DFlash tok/s | DFlash accept | DFlash acc-len | DSpark tok/s | DSpark accept | DSpark acc-len | **Δ tok/s** |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 22.82 | 0.337 | 2.69 | 25.87 | 0.405 | 3.24 | **+13.4%** |
| math | 28.97 | 0.473 | 3.78 | 33.04 | 0.569 | 4.55 | **+14.0%** |
| chat | 20.66 | 0.289 | 2.31 | 23.58 | 0.351 | 2.81 | **+14.1%** |
| **overall** | **24.15** | **0.366** | — | **27.50** | **0.442** | — | **+13.9%** |

*acc-len = mean accepted tokens per K=8 draft block.*

### Table 2 — Per-position draft acceptance (normalized to position 0)

The mechanism. DFlash's acceptance collapses down the block; DSpark's Markov head holds the
**suffix** — ~2× the acceptance at position 7 — which is where the extra accepted length
(and throughput) comes from.

| draft position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DFlash** | 1.00 | 0.73 | 0.53 | 0.38 | 0.28 | 0.21 | 0.15 | 0.11 |
| **DSpark** | 1.00 | 0.77 | 0.60 | 0.49 | 0.40 | 0.32 | 0.26 | **0.21** |

### Table 3 — Concurrency sweep, C = 1 … 18 (temp=1.0, K=8, 200 tok/req)

Aggregate throughput (all in-flight requests) and spec-decode acceptance per arm.

| C | DFlash agg tok/s | DFlash accept | DSpark agg tok/s | DSpark accept | Δ agg | Δ accept |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.5 | 0.351 | 23.9 | 0.365 | -2.4% | +0.014 |
| 2 | 31.9 | 0.459 | 36.6 | 0.549 | +14.7% | +0.090 |
| 3 | 51.8 | 0.343 | 59.8 | 0.438 | +15.4% | +0.095 |
| 4 | 67.2 | 0.395 | 65.3 | 0.466 | -2.8% | +0.071 |
| 5 | 72.7 | 0.386 | 76.5 | 0.464 | +5.2% | +0.078 |
| 6 | 100.8 | 0.410 | 91.2 | 0.430 | -9.5% | +0.020 |
| 7 | 108.4 | 0.386 | 106.6 | 0.432 | -1.7% | +0.046 |
| 8 | 112.4 | 0.378 | 119.0 | 0.478 | +5.9% | +0.101 |
| 9 | 124.3 | 0.372 | 133.5 | 0.475 | +7.4% | +0.103 |
| 10 | 120.0 | 0.372 | 129.6 | 0.434 | +8.0% | +0.063 |
| 11 | 146.2 | 0.407 | 153.5 | 0.449 | +5.0% | +0.042 |
| 12 | 147.4 | 0.384 | 157.8 | 0.478 | +7.1% | +0.094 |
| 13 | 160.2 | 0.384 | 173.0 | 0.467 | +8.0% | +0.083 |
| 14 | 167.9 | 0.406 | 164.6 | 0.428 | -2.0% | +0.022 |
| 15 | 168.6 | 0.395 | 163.2 | 0.439 | -3.2% | +0.045 |
| 16 | 179.1 | 0.393 | 168.6 | 0.434 | -5.9% | +0.041 |
| 17 | 180.2 | 0.387 | 174.0 | 0.459 | -3.4% | +0.072 |
| 18 | 173.5 | 0.373 | 178.1 | 0.451 | +2.7% | +0.078 |

*Peak aggregate: DFlash **180 tok/s**, DSpark **178 tok/s** (both near C=16–18). Mean acceptance across the sweep: DFlash **0.388**, DSpark **0.452** (+0.064). Single-round per-level, so the aggregate-tok/s Δ carries ~±10% run-to-run noise; the **acceptance gap is the robust, consistent signal** — DSpark leads at every level.*

---

## Key findings

1. **GB10 is bandwidth-bound.** Single-stream is ~19–34 tok/s: 26 GB of weights ÷ ~273 GB/s
   ≈ 10 tok/s dense, × ~2 from spec-decode ≈ 20 tok/s. The upstream **133 / 161 tok/s**
   figures were measured on a far-higher-bandwidth Blackwell — **absolute tok/s does not
   transfer across that hardware gap**. Acceptance and the relative delta do.
2. **DSpark wins on GB10: +13.9% aggregate**, matching/exceeding the upstream README's
   **+10.9%** headline — and *contradicting* the upstream serve-script note that "bare
   DFlash wins" on wall-clock. Hypothesis: on bandwidth-bound GB10 the Markov head's
   per-position GEMV overhead is **negligible** against the 26 GB verify pass, so its
   acceptance edge converts almost fully to throughput; on a compute/bandwidth-fast
   Blackwell that overhead is relatively larger and can eat the gain. **The hardware flips
   the conclusion.**
3. **The win is a suffix-decay repair** (Table 2), exactly the DSpark design intent — a
   sliver of position-0 acceptance traded for much stronger positions 3–7.
4. **Under concurrency (Table 3), DSpark's acceptance edge is robust** — it leads at
   *every* level C=1…18 (mean 0.452 vs 0.388). Aggregate throughput, however, **converges**
   as concurrency rises (both peak ~180 tok/s near C=16): at high batch the verify pass is
   amortized across many sequences, so draft *quality* matters less for aggregate tok/s and
   the acceptance advantage stops translating to throughput. DSpark's throughput lead is
   clearest at **low-to-mid concurrency (C≈8–13, +5–8%)**, i.e. exactly the latency-sensitive
   regime where speculative decoding matters most.

---

## Reproduction

- `scripts/serve-docker.sh <name> <draft_dir>` — serve the target with a chosen draft in the AEON 0.24 container (bind-mounts the ported files + persistent compile cache).
- `scripts/ab_bench.py` — single-stream 2-node A/B (Table 1).
- `scripts/conc_sweep.py` — concurrency sweep 1→18 (Table 3).
- `patches/do_port.py` — regenerates the 0.23→0.24 patched vLLM files; `patches/{qwen3_dflash.py,llm_base_proposer.py}` are the generated results.
- `results/*.json` — raw measurements.

## License / disclaimer

Benchmark harness and port script: MIT. Upstream DSpark code © hiyak (MIT); AEON vLLM
Ultimate © Aeon Forge (Apache-2.0); model weights under their respective upstream licenses.
This repo redistributes no model weights. Numbers are from a specific 2×GB10 setup and are
provided as-is.
