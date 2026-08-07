# FluxTune — probe plan: what we can test, in parallel, without touching the critical path

**Companion to `FLUXTUNE_DIVERGENCE_HANDOFF.md`.** That document says *what is wrong*; this one says
*what to build to check it*, in what order, and what each thing costs.

**Governing rule for everything here:** no probe lands in `trainer/` or `aggregator/` until two
independent probes agree on the number it is meant to produce. Probes are standalone scripts that
**import the production code** — the same discipline as `scripts/profile_jvp_opt.py`, which reuses
`expts.initializer.create_model` and `fwdgrad_utils.calculate_jvp` so a validated result transfers
into the real trainer as a config flag, not a rewrite.

---

## 0. The short version

| workstream | what it needs | runs where | parallel with |
|---|---|---|---|
| **A — log replay** | nothing; runs already on disk | laptop / CPU, minutes | B, C |
| **B — measurement rig** | 1 GPU, real model + real JVP math, no FL stack | `test_fwdllm` env, minutes–1 h per probe | A, C |
| **C — trajectory replica** | 1 GPU, single process, real model + real aggregation arithmetic | hours per trajectory | A, B |
| **D — sim runs** (real only to confirm a winner) | 8 GPUs, ≥4 h **vclock** ≈ 1.2 h wall | the existing launcher | **fix arms gated on B + C; falsification arms are not (§4b)** |

**A, B and C are fully independent and can be built and run simultaneously.** D is the only thing
that must wait, and it is also the only thing that costs a day per data point.

**Nine of the ten open questions in the handoff (§9, §13) are answerable in A, B or C.** The single
exception is **H-F** (is the instability cross-baseline?), which needs the 4 h `fwdllm` runs — though
C can answer a strong version of it first (§4.4).

---

## 1. What the existing code already gives us for free

| handle | where | what it buys |
|---|---|---|
| `expts.initializer.create_model` | `expts/initializer.py:61` | the exact production DistilBERT + AdapterHub model, backbone frozen |
| `build_model(num_labels, seq)` | `scripts/profile_jvp_opt.py:64` | that call already wrapped with the real `ClassificationArgs` (adapter PEFT, fp16, seq 192, batch 8) |
| `calculate_jvp`, `functional_get_loss` | `trainer/forward_training/fwdgrad_utils.py:105,66` | the *real* central-FD JVP math, importable standalone |
| `stage1_vmap_fd` | `scripts/profile_jvp_opt.py:143` | all `P` probes in one batched pass — **verified bit-identical** to the production loop, and the reason a replica is affordable |
| `calculate_var`, `calculate_snr`, `calculate_cos_sim` | `fwdgrad_utils.py:186,243,349` | the real commit-gate statistics |
| `TextClassificationDataManager.load_federated_data` | `data_manager/text_classification_data_manager.py` | the real agnews H5 partitions at `niid_label_clients=100_alpha=1` — same data the runs saw |
| `_server_update_step` / `_apply_weighted_update` | `aggregator/FedSgdAggregator.py:238,311` | the exact server arithmetic (~15 lines) to mirror in a replica |
| `server_update` telemetry (`_emit_server_update`) | `FedSgdAggregator.py:338` | `‖Δθ‖`, `‖W‖`, `η` per commit — already logged under `server_update_audit` |
| `characterize_variance_curve.py`, `audit_weight_redundancy.py` | `expt_scripts/` | streaming-telemetry idiom to copy, incl. the `data_id`-cycling workaround |

**The costly parts of a fluxtune experiment — MQTT, the selector, 8-GPU orchestration, the sim
clock — are all irrelevant to the divergence question.** The divergence is a property of the
estimator and the step rule. That is what makes workstreams B and C possible at all.

---

## 2. Workstream A — log replay (no GPU, runs today)

All four are **read-only over runs already on disk**, mutually independent, minutes each. They turn
the handoff's one-off analyses into re-runnable scripts so every future run is scored automatically.

### A1 · `expt_scripts/replay_step_geometry.py` — the ρ / random-walk scorecard
Reproduces §4 Legs 1–2 from `server_update` telemetry alone: the orthogonality ratio
`Σ Δ(‖W‖²) / Σ‖Δθ‖²` (expect 1.032), the reconstructed `‖θ_tr‖` trajectory, per-commit `ρ`, the
geometric-walk prediction, and the norm-doubling time.
- **Input:** `$RUN/telemetry/aggregator_*.jsonl`. **No GPU. No new telemetry.**
- **Settles:** makes Legs 1–2 auditable and re-runnable; becomes the pass/fail gate for every run in
  workstreams C and D. Baseline-agnostic, so it scores `fwdllm` runs identically → the **method** half
  of H-F, ready before the data exists.
- **Reuse:** streaming pattern from `characterize_variance_curve.py:36`.

### A2 · `expt_scripts/analyze_jvp_selection.py` — selection gain, Gaussianity, k-sweep
Parses the `All JVPs sorted by magnitude` trainer lines (34,447 of them per run) and emits
`E[v∥²]` for every rule (random / top-1 / coin-flip top-2 / top-k average, k = 1..P), the synthetic
iid-Gaussian control, skew/kurtosis, and `rms|d|` vs commit index.
- **Settles:** §8.2, §8.3 and the `|d|` growth half of Leg 2 — currently one-off shell work.
- **Gotchas to encode:** `model version` in that log line is the **round**; in runs before 2026-08-07
  `chosen jvp` is the argmax, **not** the coin-flip winner (L2 fixed it, and added `max jvp` /
  `chosen idx`), so the parser must handle both formats. Normalise within each event by that event's
  own 10-sample rms before pooling.

### A3 · extend `expt_scripts/characterize_variance_curve.py` — gate-death census
It already reconstructs per-bin `(iteration → var)` curves. Add: commit-reason breakdown
(natural / plateau / cap), the achievable variance floor per 20-bin block, and the drift ratio vs
`‖θ_tr‖²` from A1.
- **Settles:** Leg 3 becomes a standing report — and the `36× = (6.0×)²` identity gets checked
  automatically rather than by hand.
- **Extend, don't fork.** The bin reconstruction and the `data_id`-cycling handling are already right.

### A4 · collapse-signature check (small addition to `plot_run.py`)
Overlay `‖θ_tr‖` (from A1) on the accuracy/loss curve, and flag the `loss > ln(num_classes)` crossing.
- **Settles:** H-D end to end for any run from 2026-08-07 — L4 now emits `logit_norm`, `pred_entropy`
  and `top_class_share` on every `agg_eval`. For older runs only the `‖θ_tr‖`-vs-loss half is
  available; B3 recovers the rest offline.

---

## 3. Workstream B — the measurement rig (1 GPU, no FL stack)

**One new script, `scripts/probe_gradient_quality.py`, then five measurements on it.** It is the
`profile_jvp_opt.py` pattern pointed at *statistics* instead of latency: build the real model, load a
real agnews bin, and compare the forward-gradient machinery against a backprop ground truth that the
production system can never see.

**Why this is the highest-value thing to build.** `cos(G,g) ≈ 0.0231` is the only load-bearing number
in the handoff still resting on an assumption (isotropy), and **every sizing in §12 is proportional to
`1/cos²`.** Handoff §13 proposes measuring it server-side; the rig measures the same thing offline,
with no protocol change, no run, and no risk — and then reuses the identical harness for four more
open questions.

### The rig (~200 lines, all reuse)
```
build_model(4, 192)                      # scripts/profile_jvp_opt.py:64 — the production model
load one real bin (batch 8)              # TextClassificationDataManager, agnews niid alpha=1
g_true  = autograd backprop on trainable params        # the ground truth, backprop-only
u_i     = the real forward-grad upload for probe i     # stage1_vmap_fd -> d_i, then d_i * v_i
G_N     = (1/N) * sum of N such uploads                # mirrors _apply_weighted_update
report: cos(G_N, g_true), ||G_N||/||g_true||, rho, and all of it as a function of N
```

### B1 · `cos(G,g)` ground truth — **do this first**
Sweep `N` = 1, 10, 50, 185, 500, 1000 and check `cos ∝ √N` and `‖G‖/‖g‖ ∝ √(p/N)`.
- **Settles:** the isotropy assumption itself. If the `√(N/p)` law holds, every number in handoff §6
  and §12 becomes MEASURED. If it fails, **§6 must be revisited before anything is built** — which is
  exactly why this comes before all construction.
- **Decision rule:** `cos ≈ 0.023` at `N = 185` ⇒ §12 stands as written. Materially higher ⇒
  requirements relax as `1/cos²`; recompute the sizings first.

### B2 · H-B — is `|d| ∝ ‖θ_tr‖` genuine gradient growth or an FD artifact?
Same rig, scale the trainable weights by `s ∈ {1, 2, 4, 6}` (or load real checkpoints along the
trajectory) and plot `‖g_backprop‖` and `rms|d|` together.
- **Settles:** H-B. Genuine ⇒ weight decay (S-G) is *curative*, not cosmetic, and the bar S-A has to
  clear goes up. Artifact ⇒ the `v`-normalisation / relative-`h` thread reopens.
- **Free** once B1 exists — same model, same probes, one extra backprop.

### B3 · H-D — the collapse endpoint
At each scaled `‖θ_tr‖`, log logit norm, prediction entropy, and per-class prediction share.
- **Settles:** whether inflation → logit saturation → single-class collapse is the actual causal
  chain, and whether `acc = 0.250 / mcc = 0.000` is reproducible from norm alone.
- **Answers A4's blocked half without adding eval telemetry.**

### B4 · H-G / S-I — does `p` actually buy `cos`?
Re-run B1 with `pre_classifier` frozen (`p`: 1,040,932 → 450,340) and across adapter ranks.
- **Settles:** the `cos ∝ 1/√p` law empirically, so the claimed 2.31× is measured before a 4 h run is
  spent on it. **It does not settle whether accuracy survives** — that is D2.
- This is the cheapest confirmation of the handoff's strongest new claim (§8.5): for forward-gradient
  FL, PEFT rank is a gradient-quality parameter, not an efficiency detail.

### B5 · H-H — is the discarded curvature usable?
Add the one `L(θ)` pass the central difference is missing, recover `vᵀHv` per candidate, and correlate
it with the realised loss decrease at the step scale S-A would actually take.
- **Settles:** H-H, i.e. whether the only *non*-stability-neutral selection metric is real (§8.6-B).
- **Also gives Family B's split-half SNR for free** — a forward pass yields per-sample losses.

> **B1 is the root; B2–B5 are one extra measurement each on the same rig.** Build B1 alone, verify it,
> then B2–B5 in any order or in parallel.

---

## 4. Workstream C — the single-process trajectory replica (1 GPU)

**`scripts/replica_fluxtune_loop.py`** — the whole optimization loop in one process: draw probes,
compute JVPs, pool `K` uploads, run the real variance gate, commit with the real server step, repeat.
No MQTT, no selector, no flame runtime, no sim clock.

**Why this exists.** B measures the estimator at a point in time. Only a trajectory shows the
*interaction* — the ρ feedback loop, the gate drifting out from under itself, the rise-peak-collapse
shape. And it is the only way to A/B a proposed fix without spending 4 h × 8 GPUs per arm.

```
what it reuses verbatim        build_model / stage1_vmap_fd / calculate_jvp / calculate_var
                               / TextClassificationDataManager / the _apply_weighted_update arithmetic
what it replaces               the transport, the selector, and the 8-GPU orchestration
what it must reproduce first   rho ~ 0.115 flat, orthogonality ratio ~ 1.03,
                               norm doubling ~110 commits, rise-peak-collapse
```

### C0 · the validation gate — non-negotiable
Run the replica at the shipped config and score it with **A1**. If it does not reproduce the four
constants above, the replica is wrong and nothing downstream of it counts. Only after that does any
A/B mean anything.

### C1 · A/B every proposed fix, one flag each
Each is a few lines *inside the replica*, mirroring the production flag that would eventually carry it:
`probe_combine: {select | mean}` (S-H) · `server_step_rule: {raw_sgd | trust_ratio}` (S-A) ·
`ρ*` annealing (S-B) · `trainable_scope` (S-I) · split-half-cosine gate (S-E) · weight decay (S-G).
- **Predicted, therefore falsifiable:** S-H should drop `ρ` by `√(E[v∥²]·P) = 5.5×` **on the first
  commit**. That is a one-commit check, not a trajectory — run it first as a smoke test.
- The value is the **ranking and the interaction**, not the absolute accuracy. Anything that wins here
  earns a real run; anything that loses never costs one.

### C2 · the ablations that are structurally impossible today
`K`, `I`, `P`, `p` and `η` are all free variables in the replica. That makes runnable, in an afternoon,
the sweeps the production system cannot do cleanly: **the K-sweep and P-sweep under a *pinned* `ρ`**,
which handoff §10.3b shows are uninterpretable under raw SGD because both `ρ` and `cos` move at once.
This is also where the `ρ ≤ cos` criterion gets tested as a *prediction* — configurations placed
deliberately on either side of the boundary should diverge or not, as predicted.

### C3 · H-E staleness — **not here**
The replica has no real concurrency, so it cannot produce genuine staleness. H-E stays in D.

### C4 · a strong pre-answer to H-F
`_server_update_step` is shared code, so the replica can run the **sync (fwdllm) pooling arithmetic**
with the same estimator and score it with A1. If sync diverges too at a matched horizon, the
instability is not fluxtune-specific and S-A/S-B are cross-baseline hygiene rather than a
contribution — which changes the paper's claim structure. **This does not replace D3** (the replica
does not reproduce fwdllm's real timing, which is why its runs stall at `data_id=38`), but it gets the
answer weeks earlier and tells us whether D3 is urgent.

### Cost, stated honestly
Per upload the production trainer does `2P = 20` forward passes (batch 8, seq 192); one commit pools
`N ≈ 185` uploads ⇒ **≈ 3,700 passes/commit, ≈ 700k for a 189-commit trajectory.** Two things make
this tractable: `stage1_vmap_fd` batches all `P` probes into ~2 effective passes (already verified
bit-identical), and there is no orchestration overhead — the reference run spent 75 s/commit across
8 GPUs. Read the real per-pass number off `profile_jvp_opt.py` before committing to a schedule.
If it is still too slow, shorten `seq` and/or reduce `N`, and accept the trade explicitly: **the
replica's constants will shift, its mechanism and its ranking of fixes will not.** C0 on one
full-scale trajectory is what licenses that trade.

---

## 4b. Tonight — a 3-node sim portfolio that needs zero code

The gating in §5 is about *fix* arms. **Falsification arms are not gated** — they test whether the
handoff's model is right, and every one is a launcher flag: `--agg-goal`, `--c`, `--learning-rate`,
`--perturbation-count`, `--max-iter-per-data-id`, `--var-threshold`, `--max-runtime-s`. All arms run
`--mode sim` (see §5 for why, and for the two caveats) with `--server-update-audit
--pool-split-half-audit` so **A1** can score them.

**Launcher gotchas, all of which cost a wasted run if missed:**

- The baseline flag is **`--only`**, not `--baselines` (unknown args abort).
- **`--yes`** — otherwise each invocation stops at an interactive `[y/N]` prompt.
- **`--clean`** — back-to-back runs otherwise `DIRTY_ABORT` on a prior run's stray workers.
- **`--force`** — the sim-charge-profile pre-flight blocks because the 2026-08-04/05 reals are newer
  than the profile. They are audit-on runs, so **re-profiling from them would bake diagnostic
  overhead into the vclock model** — force is correct here. Confirm with `--dry-run` that it is the
  only `✗` first, since `--force` overrides every check.
- **`--num-trainers 100`** pins `minInitialTrainers`, so varying `--c` does not move the warmup
  threshold underneath a sweep.
- The baseline → yaml map is **hardcoded** in `ALL_RUNS`; there is no custom-yaml flag, and
  `fwdllm_plus` has no entry. Use the config flags above, never a hand-edited yaml.

**Each arm makes a numeric prediction that is checkable at commit ~10, minutes into the run.**
That is the point: if a prediction misses, §6 is wrong and no fix built on it is worth landing.
| node | arm | prediction (from §6.3) | what a miss would mean |
|---|---|---|---|
| **1** | **`K`-sweep: agg_goal 10 → 20 → 30 → 60** (c = 2K), 4 h vclock each, plus a **baseline replicate at K = 10** | `ρ ∝ 1/√N`: **0.115 → 0.081 → 0.066 → 0.047**, `cos ∝ √N`, `ρ/cos` improves ∝ K, norm doubling stretches ∝ K, **`ρ·cos` flat across the whole sweep** | a *curve* falsifies far harder than a point: if `ρ` does not track `1/√K`, the pooling identity — the most load-bearing claim in the document — is wrong and §6.3's ranking collapses |
| **2** | **`η`-sweep: 0.01 → 0.002 → 0.0005**, 8 h vclock each | `ρ ∝ η` **on commit 1, exactly**: 0.115 → 0.023 → 0.0058. η = 0.002 lands *at* the boundary (no collapse in 4 h, ~5× slower climb); η = 0.0005 goes under it but **still random-walks** — divergence merely deferred, which is S-B's whole argument | if `ρ` does not scale 1:1 with `η`, the step model is wrong. If it does but collapse timing does not move as predicted, the boundary is mis-calibrated and every §12 sizing needs redoing |
| **3** | **`fwdllm` 8 h (**H-F**), then `fwdllm_plus`, then a fluxtune `P = 30` arm** — this node is nearly free (`sim_rate` 12.6) | H-F: same random-walk signature ⇒ **shared defect, S-A/S-B are hygiene**; different ⇒ **fluxtune-specific, they are a contribution**. `P = 30`: §8.1 predicts `ρ/cos` **unchanged** and `E[v∥²]` up ⇒ *slightly faster* divergence | H-F has no "miss" — both outcomes are informative and both change the paper's claim structure. The `P` arm is the counterintuitive prediction: more probes must **not** help while selection discards them |

```bash
cd lib/python/examples/fwdllm/expt_scripts
A="--only fluxtune --mode sim --yes --clean --force --server-update-audit \
   --pool-split-half-audit --num-trainers 100 --num-gpus 8"

# node 1 -- K-sweep (c must stay >= K and <= 100); first leg repeated as the anchor
for KC in 10:30 10:30 20:40 30:60 50:100; do
  ./run_sequential.sh $A --agg-goal ${KC%%:*} --c ${KC##*:} --max-runtime-s 14400
done

# node 2 -- eta-sweep
for LR in 0.01 0.002 0.0005; do
  ./run_sequential.sh $A --learning-rate $LR --max-runtime-s 28800
done

# node 3 -- H-F, then the P arm (sim_rate 12.6 makes the fwdllm leg nearly free)
./run_sequential.sh ${A/--only fluxtune/--only fwdllm} --max-runtime-s 28800
./run_sequential.sh $A --perturbation-count 30 --max-runtime-s 14400
```

**Why sweeps, not single points.** At sim's wall cost a sweep is nearly the same price as one arm, and
a predicted *slope* (`ρ ∝ 1/√K`, `ρ ∝ η`) is a far stronger test than a predicted value — it cannot be
rescued by a fudge factor. Nodes 1 and 2 also move in opposite directions on the two objectives: node 1
predicts stability gain at **no** progress cost, node 2 predicts it at **exactly 5×** progress cost.
Together they test the `ρ/cos` vs `ρ·cos` split the whole solution ranking rests on.

**Still excluded:**
- **`max_iter_per_data_id` 20 → 40.** Same `N` gain as node 1 but bought with serial `I`, so commits
  halve and the run ends short of the horizon. Node 1 dominates it (§12 S-D).
- **`var_threshold` retuning.** Predicted dead: 0 of 186 bins ever reached the threshold (Leg 3).

**Standing rule applies, on the vclock:** ≥4 h — the 2 h runs terminate at 0.85 norm doublings, exactly
at the accuracy peak. In sim that costs ~1.2 h wall.

### Telemetry worth adding *before* launching (≈1 h of work, all audit-gated, no behaviour change)

Each follows the existing `server_update_audit` idiom: off by default, emit-only, wrapped so it can
never fault training. These are not critical-path changes.

| # | add | why it is worth delaying launch for |
|---|---|---|
| **L1** | **split-half cosine of the pooled uploads, per commit** | **the single highest-value new number.** A gradient-free proxy for `cos(G,g)` that needs no backprop, no probe batch, and no protocol change — so all three arms above report their *actual* stability margin, not one inferred from theory. It is also S-E's gate statistic and S-C's setpoint, measured before either is built |
| **L2** | **fix the `chosen jvp` label** (`tc_transformer_trainer_distribute.py:485` logged the argmax, not the coin-flip winner) | a one-line correctness fix. Without it every selection analysis silently reads the wrong probe, and tonight's logs inherit the bug |
| **L3** | **`‖θ_tr‖` logged directly** beside `‖W‖` | removes the reconstruction step in A1 (which currently anchors on a separately measured init constant), so `ρ` is measured rather than inferred |
| **L4** | **logit norm + prediction entropy at `agg_eval`** | settles H-D from the runs themselves instead of from B3, and gives the production monitor the handoff asks for |

**All four are implemented** (2026-08-07). L1 is behind `--pool-split-half-audit` (its own flag: it
adds a pass over params × uploads, so it must not change `--server-update-audit`'s cost profile);
L2/L3/L4 are correctness or free, and are on by default.

> **How to read L1 — do not average per-commit cosines.** The record carries the **raw** components
> (`split_half_dot`, `split_half_norm_a/b`, `pool_size`), because one commit's cosine sits under the
> `1/√p ≈ 1e-3` sampling floor while today's signal is `≈ 3e-4`. Pool it:
> `cos(G,g) ≈ √(2 · Σdot / Σ‖a‖‖b‖)` over ~100+ commits. Simulation at the real `p`, `N` confirms
> both this identity and `cos = √(N/p)` to 0.1%. The same arithmetic is why **S-E cannot be the first
> fix landed** — see the constraint box in handoff §12 S-E.

---

## 5. Workstream D — sim runs (real only to confirm a winner)

Only these need the launcher. **Standing rule: ≥4 h of vclock or the result is uninformative** —
2 h runs terminate at 0.85 norm doublings, exactly at the accuracy peak, and will report a win for a
configuration that diverges.

| # | run | mode | why it cannot be replaced by A/B/C |
|---|---|---|---|
| **D1** | the winning fix stack, 4 h, flags on | sim to pick, **real to confirm** | end-to-end confirmation on the real system |
| **D2** | `freeze pre_classifier`, 4 h (H-G) | sim | B4 proves `cos` improves; only a full run shows whether **accuracy survives** |
| **D3** | `fwdllm` / `fwdllm_plus`, 8 h (H-F) | sim | needs the baselines' own dynamics; C4 pre-answers, does not replace |
| **D4** | K/C sweep with the staleness histogram (H-E, C3-freshness) | sim | genuine staleness needs genuine concurrency |

**Run these in SIM.** Real/sim parity is established (`simulate_fwdllm.md`), and sim is dramatically
cheaper in wall clock — measured `sim_rate` (virtual-s per wall-s), stable across runs:

```
fluxtune  sim_rate = 3.33     4 h vclock -> ~1.2 h wall     8 h -> ~2.4 h
fwdllm    sim_rate = 12.6     8 h vclock -> ~0.6 h wall
```

Trainers still do real forward-grad compute (gradient values are mode-invariant); what sim removes is
the **waiting** — arrival jitter, transport, dispatch settle — which is where the async real path
spends most of its wall clock. That is the whole `sim_rate`.

**Consequences.** The ≥4 h floor is a floor on **vclock** (it is really a floor on *commits*, i.e. norm
doublings), so `--max-runtime-s 14400 --mode sim` costs ~1.2 h wall, not 4 h. Every arm below is
therefore affordable overnight, and the portfolio should be a **sweep** rather than a single point.

**Two caveats that survive.** (a) Compare **sim to sim** — include a sim baseline replicate to anchor,
since the reference constants (`ρ = 0.115`, doubling ≈ 110 commits) were measured on real runs.
(b) Ordering differs between modes, so pool composition differs commit-to-commit; that is harmless for
`ρ`, `cos` and norm growth (aggregates over 100+ commits) and not safe for a claimed A/B win of a few
percent. Confirm the *winner* on real (D1); discover in sim.

**One telemetry fix before any selection A/B reads that field:**
`tc_transformer_trainer_distribute.py:485` logs the argmax but labels it `chosen jvp`, while the
actual pick is the coin-flip result (`:481-483`). A2 must work around it; D must not inherit it.

---

## 6. Coverage — every open question, and where it gets answered

| open item | A | B | C | D |
|---|:-:|:-:|:-:|:-:|
| `cos(G,g)` measured (handoff §13) | | **B1** | | |
| isotropy / `√(p/N)` noise model | | **B1** | | |
| H-B `|d| ∝ ‖θ_tr‖` genuine vs artifact | A2 | **B2** | | |
| H-C FD faithfulness over the run | | **B2** | | |
| H-D collapse = logit saturation | A4 | **B3** | | |
| H-G / S-I `p` → `cos` | | **B4** | C1 | **D2** (accuracy) |
| H-H curvature signal | | **B5** | | |
| S-H average-all-P | A2 (offline) | | **C1** | D1 |
| S-A trust-ratio step | A1 (sensor) | | **C1** | D1 |
| S-B ρ annealing | | | **C1** | D1 |
| S-E split-half gate | A3 | B5 | **C1** | D1 |
| criterion `ρ ≤ cos` as a prediction | | | **C2** | |
| clean K / P ablation under pinned ρ | | | **C2** | D4 |
| H-E staleness at high K/C | | | | **D4** |
| H-F cross-baseline instability | A1 (method) | | C4 (strong) | **D3** |

---

## 7. Build order

1. **A1** — half a day, no GPU, and it becomes the scorecard every later run is graded by.
2. **B1** — the rig. Settles the one assumption everything else is sized off. **Nothing should be
   built until B1 has either confirmed `cos ≈ 0.023` or forced §6 to be rewritten.**
3. **A2, A3** and **B2–B5** — fully parallel, one measurement each.
4. **C0** — the replica plus its validation gate. Start it in parallel with step 2; it does not depend
   on B1's answer, only on the same imports.
5. **C1, C2** — the A/B matrix. The S-H one-commit `5.5× ρ drop` check is the cheapest possible
   falsification and should be the very first thing run on a validated replica.
6. **D** — only for the arms C ranked first, at ≥4 h, plus D3 in parallel since it changes the paper's
   claim structure and should not be left until the end.

**Nothing in steps 1–5 modifies `trainer/`, `aggregator/`, or any yaml on the critical path.** The
production changes are the flags in handoff §12, and they land only with a replica A/B behind them.
