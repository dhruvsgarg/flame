# HANDOFF — H11

**TEMPORARY. Delete when H11 closes**; findings move to `simulate_fwdllm.md` §A/§E/§G. This file exists so
one hypothesis's validate/invalidate state survives a session boundary without re-deriving it.
**H13 closed and has been purged from here** — probe C confirmed it and `jvp_eval_mode` removes it; the
record lives in `simulate_fwdllm.md` §B/§G.

---

## H11 — sim over-dispatches at the round boundary

**Status: root-caused, FIXED, unit-tested — LIVE VALIDATION PENDING (the one run still owed).**

**Defect.** `_release_sim_slots_at_agg_goal`'s legacy path clears `_sim_inflight_expected` AND
`_sim_pending_commit`, then calls `_sim_hold_busy_slots`, which rebuilds `outstanding` from those now-empty
sets and deletes everything not in it from `all_selected`. Both re-pick guards hit zero at one instant, so the
boundary's successive top-up `select()` calls re-pick ends that are still training.

**Evidence (production telemetry, `felix_round` sim `run_20260801_163513`).** End `…0395` picked 5x and
`…0409` 2x at a frozen vclock, `in_pending_commit: false` across every repeat → 35 picks against c=30. Real
issued 30 unique. `felix_it` is a clean NEGATIVE (0 over-dispatch both modes, all 24652 re-picks are true
churn), which localizes the defect to the round-boundary batch path, NOT AsyncOort's `select()`.

**Fix.** Fold `_trainer_inflight_dispatch_version` (dispatched-not-yet-returned, maintained in BOTH modes,
and the only in-flight record the boundary does not clear) into sim's `outstanding` — the same half real's
`_PendingCommitUnion` already carries, which is why real never had the bug.
`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`, 3 tests in
`tests/mode/test_fwdllm_sim_grad_residence.py::TestBoundaryDropKeepsDispatchedGuard` (2 fail without it).

**What the unit tests do NOT cover.** They drive `_sim_hold_busy_slots` with a fake channel, so they pin the
mechanism. They do not confirm that in a live run `_trainer_inflight_dispatch_version` is populated at the
right moment relative to the boundary. That is exactly what the pending run tests.

### PENDING LAUNCH — `felix_round` real+sim, **7200s** (now §B node A, with `jvp_eval_mode` ON)

```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --mode both --max-runtime-s 7200 --only felix_round --yes
python trace_boundary_repicks.py ../experiments/<new felix_round sim dir>
python run_parity.py --yes --baselines felix_round
```

**⚠ MUST be 7200s, NOT 3600s** — an earlier plan said 3600s and it would have been a wasted run. The defect
only fires at the round-1→2 boundary, and that boundary arrives late:

| leg | round-2 re-draw first fires at |
|---|---|
| real `run_20260802_104547` | wall **4823s** |
| real `run_20260801_084040` | wall **4270s** |
| sim `run_20260801_163513` | **vclock 4441s** (wall 1390s) |

`--max-runtime-s` caps WALL in real mode and VCLOCK in sim mode, so 3600s stops both legs before the boundary
exists and the rung grades a run in which the defect cannot occur. (§C's "one MECHANISM rung → 1800s" is a
general rule that does not hold for a BOUNDARY-gated defect — duration must reach the boundary.)

**Exit criteria.**
- `trace_boundary_repicks.py` on the new sim dir reads **OVER-DISPATCH=0**.
- `selection_detail` goes green on `felix_round`.
- Boundary picks read **30 unique on both sides** (real already does).
- No regression in `concurrency_cap` / `r1_inflight_overlap` / `slot_utilization`.

If over-dispatch persists, the guard is being cleared somewhere else as well — re-run
`trace_boundary_repicks.py` and check `in_all_selected` / `in_pending_commit` in the `selection` event's
`per_trainer` block at the repeats, which is how the original was localized.

---

## Node status

| node | job | state |
|---|---|---|
| 1 | `felix_round` real+sim **7200s** — H11 live validation | **STILL OWED** — now folded into node A of `simulate_fwdllm.md` §B, which runs it with `jvp_eval_mode` ON |
| 2 | `fluxtune` + `fwdllm` real replicates (floors) | **done** — `run_20260802_172341`, `run_20260802_192556`; unread (§B T0.1) |
| 3 | `fwdllm_it_unaware` + `fwdllm_it_oracular` 7200s pairs | **done** — `run_20260802_172249`/`_192444`, `run_20260802_193900`/`_214059`; ungraded (§B T0.1) |
| bench | H13 probe C | **done** — CONFIRMED, §B |
