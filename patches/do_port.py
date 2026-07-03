#!/usr/bin/env python3
"""Port DSpark (vLLM 0.23) Markov semi-AR draft patches onto the AEON 0.24 image.

Splices self-contained DSpark blocks out of the 0.23 patched files and into the
0.24 stock files at verified anchors. Additive only; DFlash path untouched.
Outputs to v024_patched/. Re-runnable.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
V23 = os.path.join(HERE, "v023_dspark")
V24 = os.path.join(HERE, "v024_stock")
OUT = os.path.join(HERE, "v024_patched")
os.makedirs(OUT, exist_ok=True)


def rd(p):
    with open(p) as f:
        return f.read()


def lines(p):
    with open(p) as f:
        return f.readlines()


def slice_lines(fl, a, b):
    """1-indexed inclusive line range -> text."""
    return "".join(fl[a - 1:b])


def splice(text, anchor, insert_after=True, replacement=None, label=""):
    if replacement is not None:
        if anchor not in text:
            sys.exit(f"[FAIL] replace anchor not found: {label}")
        return text.replace(anchor, replacement, 1)
    if anchor not in text:
        sys.exit(f"[FAIL] insert anchor not found: {label}")
    idx = text.index(anchor) + (len(anchor) if insert_after else 0)
    return text  # unused branch


# ---------------- qwen3_dflash.py ----------------
q23 = lines(os.path.join(V23, "qwen3_dflash.py"))
q24 = rd(os.path.join(V24, "qwen3_dflash.py"))

CLASSES = slice_lines(q23, 51, 250)      # _VanillaMarkov ... _markov_semiar_sample_block
INIT_BLOCK = slice_lines(q23, 488, 529)  # markov_head + confidence_head build
METHODS = slice_lines(q23, 803, 1073)    # sample_draft_block_semiar[_sample] + predict_confidence_step

# 1. import os
if "\nimport os\n" not in q24:
    q24 = q24.replace(
        "from collections.abc import Iterable, Mapping\n",
        "import os\nfrom collections.abc import Iterable, Mapping\n", 1)

# 2. DSpark standalone classes/helpers after the logger line
anchor = "logger = init_logger(__name__)\n"
q24 = q24.replace(anchor, anchor + "\n\n" + CLASSES.rstrip("\n") + "\n", 1)

# 3. markov_head + confidence_head into DFlashQwen3Model.__init__ (after self.norm)
norm_anchor = (
    "        self.norm = RMSNorm(\n"
    "            self.config.hidden_size,\n"
    "            eps=self.config.rms_norm_eps,\n"
    "        )\n"
)
q24 = splice(q24, norm_anchor, replacement=norm_anchor + INIT_BLOCK, label="Model.__init__ norm")

# 4. sample_draft_block_semiar* methods into DFlashQwen3ForCausalLM (after compute_logits)
ret_anchor = "        return logits_new\n"
q24 = splice(q24, ret_anchor, replacement=ret_anchor + "\n" + METHODS, label="ForCausalLM compute_logits return")

with open(os.path.join(OUT, "qwen3_dflash.py"), "w") as f:
    f.write(q24)
print("[ok] qwen3_dflash.py written")


# ---------------- llm_base_proposer.py ----------------
p23 = lines(os.path.join(V23, "llm_base_proposer.py"))
p24 = rd(os.path.join(V24, "llm_base_proposer.py"))

CONF_METHOD = slice_lines(p23, 456, 537)   # _compute_conf_prefix_lengths
PROPOSE_BLOCK = slice_lines(p23, 629, 710) # DSpark early-exit block (markov branches + fallback)

# 1. import os
if "\nimport os\n" not in p24:
    p24 = p24.replace(
        "from vllm.distributed.parallel_state import get_pp_group\n",
        "import os\nfrom vllm.distributed.parallel_state import get_pp_group\n", 1)

# 2. conf-threshold init after self._last_draft_probs init in __init__
lastdp_anchor = "        self._last_draft_probs: torch.Tensor | None = None\n"
CONF_INIT = (
    "        try:\n"
    "            self._conf_threshold: float = float(\n"
    "                os.environ.get(\"DSPARK_CONF_THRESHOLD\", \"0.0\") or 0.0\n"
    "            )\n"
    "        except (TypeError, ValueError):\n"
    "            self._conf_threshold = 0.0\n"
    "        self._last_draft_prefix_lengths = None\n"
)
p24 = splice(p24, lastdp_anchor, replacement=lastdp_anchor + CONF_INIT, label="proposer __init__ last_draft_probs")

# 3. _compute_conf_prefix_lengths method before def propose(
propose_anchor = "    def propose(\n"
p24 = splice(p24, propose_anchor, replacement=CONF_METHOD + "\n" + propose_anchor, label="proposer def propose")

# 4. replace 0.24 early-exit block with DSpark version
old_block = (
    "        # Early exit if there is only one draft token to be generated.\n"
    "        if self.num_speculative_tokens == 1 or self.parallel_drafting:\n"
    "            draft_token_ids, draft_probs = self._sample_draft_tokens(\n"
    "                sample_hidden_states, sampling_metadata\n"
    "            )\n"
    "            if draft_probs is not None:\n"
    "                self._last_draft_probs = draft_probs.view(\n"
    "                    -1, self.num_speculative_tokens, draft_probs.shape[-1]\n"
    "                ).contiguous()\n"
    "            return draft_token_ids.view(-1, self.num_speculative_tokens)\n"
)
p24 = splice(p24, old_block, replacement=PROPOSE_BLOCK, label="proposer early-exit block")

# 5. BLOCK-EXPAND FIX in compute_probs_and_sample_next_token: parallel/DFlash
# draft logits are [num_reqs*num_spec, V] while temperature is per-request
# [num_reqs]. div_ broadcasts only at num_reqs==1; >=2 concurrent requests
# shape-mismatch and kill the engine. Expand temperature to match the
# block-expanded rows (request-major / position-minor layout). Present in stock
# 0.23 and 0.24 alike; a strict robustness fix, no behavior change at num_reqs==1.
temp_anchor = (
    "    temperature = sampling_metadata.temperature\n"
    "    # Avoid division by zero if there are greedy requests.\n"
)
temp_fix = (
    "    temperature = sampling_metadata.temperature\n"
    "    # BLOCK-EXPAND FIX: draft logits are [num_reqs*num_spec, V] while\n"
    "    # temperature is per-request [num_reqs]; broadcast only works at\n"
    "    # num_reqs==1. Expand to block rows (request-major/position-minor).\n"
    "    if (\n"
    "        temperature.shape[0] != logits.shape[0]\n"
    "        and logits.shape[0] % temperature.shape[0] == 0\n"
    "    ):\n"
    "        temperature = temperature.repeat_interleave(\n"
    "            logits.shape[0] // temperature.shape[0]\n"
    "        )\n"
    "    # Avoid division by zero if there are greedy requests.\n"
)
p24 = splice(p24, temp_anchor, replacement=temp_fix, label="compute_probs block-expand fix")

with open(os.path.join(OUT, "llm_base_proposer.py"), "w") as f:
    f.write(p24)
print("[ok] llm_base_proposer.py written")
print("[done] patched files in", OUT)
