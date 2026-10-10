# SmolLM2-360M + LoRA smoke test: plan

2026-10-10. Plan only; nothing here has been run. Context: `fl_fwd_ft_aishwwarya_session_2026-10-01.md` §9–§10.

**Goal:** show the model ports cleanly and read the controller's numbers (`p`, trips/commit, commits/h, `cos·Φ`).
Accuracy is out of scope (the roberta smoke `145932` sat at chance and still counted as a clean port).

The porting ladder is buildplan §5.8; this plan follows it from free checks to a 15-min run.

## Stage 0: code changes (the only code work)

1. **Register the model.** Add a `"llama"` entry to `MODEL_CLASSES["classification"]` in `expts/initializer.py`
   (`LlamaConfig`, `LlamaForSequenceClassification`, `AutoTokenizer`).
   - Set `tokenizer.pad_token = eos_token` and `config.pad_token_id`. Llama has no pad token and
     `data_preprocessing/text_classification_preprocessor.py:161` reads `tokenizer.pad_token`, so tokenization
     crashes without it.
2. **LoRA path.** The `lora` branch already uses `LoRAConfig(r=8, alpha=16)` via `adapters`.
   - Check `adapters.init` accepts the Llama class and the default target is q,v.
   - Check the classification head (`score`) is trainable; it adds `hidden × num_labels` to `p` (~4k).
3. **`pre_classifier` needs no guard.** The drop in `trainer/forward_training/tc_transformer_trainer_distribute.py:216`
   is gated on `model_type == "distilbert"`, so Llama is untouched (confirm when run).

## Stage 1: free checks (no data, no run)

4. **Port init probe:** `expt_scripts/probe_port_init.py --model-type llama --model-name <smollm2-id> --rf 16`.
   - Expect `p` ≈ 0.8M. Millions means the head or backbone isn't frozen.
   - Read the chord with `FWDLLM_FD_SCALE_INVARIANT` ON and OFF. Scaling the roberta row's ON value to this `p`
     gives ~0.38 vs the 0.50 baseline (estimate). `run_node4_p.sh:27` exports the flag as 1, so decide it here.
5. **One-batch sanity check:** loss finite; JVP difference not zero or noise in bf16. If it is, compute the JVP
   loss in fp32.

## Stage 2: data

6. **Pre-tokenize:** `expt_scripts/pretokenize_dataset.py --model-type llama --model-name <smollm2-id>`. The cache is
   keyed by model; without it every trainer tokenizes cold.
7. Dataset: agnews, as for the other baselines.

## Stage 3: 15-min smoke

8. **Launch** via `expt_scripts/nodes/run_node_p4.sh` with `P4_MODEL_TYPE=llama P4_MODEL_NAME=<smollm2-id>`.
   - Few trainers, with a matching partition (`P4_PARTITION`, `P4_NUM_TRAINERS`). N < 100 without a matching
     partition reads only the first N shards.
   - `s` = 2.9 reaches Φ ≈ 2.9 ~3.7× sooner but under-trains: controller check only.
   - Retune lr (roberta needed 3e-4).
9. **Read:**

| Reading | Pass condition |
|---|---|
| `[ProbeDim] p` | ≈ 0.8M |
| `cos·Φ` | ≈ 1.0000 (roberta's "ported clean" signal) |
| Trips/commit | 10–20 |
| Commits/h | wall ≈ 400 ÷ commits/h |
| Deaths / OOMs | 0 |

## Stage 4: decide

10. `p` and `cos·Φ` clean and ≤ ~10 h projected: launch the full run to its own stop. Φ* at a new `p` (N5c) is the
    open question, so don't cut it early.
11. Chord drifting: flip `FWDLLM_FD_SCALE_INVARIANT` and rerun the smoke.
12. Slower than expected: re-profile `sim_charge_profiles`, or sim timing won't match real.

## Risks

- bf16 precision in the JVP (two near-equal losses).
- Wrong `p` from an unfrozen head.
- Padding or pooling mistakes (Llama pools the last non-pad token).
- Stale sim charge profile.

## Expected full-run cost (extrapolated, not measured)

~5–10 h, central ~9 h: 50 h (roberta) ÷ 5.3 (`p` 4.23M → ~0.8M) at ~1× forward cost. Per-commit fixed overhead
(aggregation, probe, sim charging) doesn't shrink with `p`, so expect above the low end.
