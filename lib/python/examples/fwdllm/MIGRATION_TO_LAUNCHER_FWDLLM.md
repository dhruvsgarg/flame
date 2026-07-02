# fwdllm: pending follow-ups (post-migration)

The launcher migration itself, and every durable lesson (positive and
negative) from doing it and hardening it since, are folded into
[`../MIGRATING_TO_LAUNCHER.md`](../MIGRATING_TO_LAUNCHER.md) — the common
patterns are in its core sections (§2 aggregator gotchas, §5 telemetry, §8
launcher/spawner gotchas), and fwdllm's own specifics are in §9. That doc is
the one to read for context on how fwdllm works and what was already fixed.
For the full investigation history (evidence trails, exact log lines,
commit-by-commit narrative) behind everything below, see
`git log -- lib/python/examples/fwdllm/MIGRATION_TO_LAUNCHER_FWDLLM.md` —
this file itself only used to carry that narrative and has been trimmed to
just what's still open.

**Nothing below is blocking the `launcher-script-fwdllm` PR.** All are
either genuine follow-up work or verification steps worth doing at some
point, not merge blockers.

---

## Pending correctness/design follow-ups

1. **fwdllm's round-cached reselection has no mechanism to replace a
   trainer that's stuck but not formally departed.** Confirmed via a real
   n=100 run (`run_20260701_182242_fwdllm_n100_smoke`): the aggregator's
   live candidate pool stayed at exactly 30 of 100 trainers for the entire
   1.5h run, "hasn't received weights" fired ~9,700 times, and the run
   stalled completely for the last 47 of 90 minutes — ending at only ~46%
   accuracy / `data_id` 29 of 150, vs. `fwdllm_plus`/`fluxtune` reaching
   80%+ accuracy in the same window. Cause: `--min-initial-trainers 95`
   gates startup at high trainer counts, and once fwdllm's per-round
   selection cache fills, only *explicit* departure (disconnect/`UN_AVL`)
   triggers a replacement — a trainer that's merely stuck (e.g. never
   finished receiving its initial weights) occupies a cache slot
   indefinitely. `fwdllm_plus`/`fluxtune` don't share this because they
   reselect continuously rather than caching per round.
   **Fix direction** (not started): either teach the round-cache to also
   replace a member that's gone N minutes without a real response (not just
   formally-departed members), or default `--min-initial-trainers` more
   conservatively at high trainer counts.
2. **`channel.await_join()` race** (shared `flame` channel/trainer code,
   not fwdllm-specific): only catches peers already joined at broadcast
   time, no timeout of its own. Currently mitigated (launcher's
   concurrent-polling force-kill bounds the cost) but not fixed. Proper fix
   is a timeout on `await_join()` itself, or in the trainer's
   `_fetch_weights`/`_send_grads` — separate PR, touches shared code beyond
   this example.

## Pending telemetry/analysis improvements

3. **No first-class metric for the class of issue in #1.** Root-causing it
   took an hour of hand-grepping raw aggregator logs (`Total ends: N`,
   `hasn't received weights`, staleness rejections). A plot/metric for
   "live channel-end pool size vs. configured `--num-trainers` over time"
   and "trainers stuck at `model_version=-1` past N minutes" would turn
   that into a 30-second `analyze_run.py` check, and would generalize to
   catching the same class of issue in future examples.
4. **fwdllm's own `selection`/`selection/why` telemetry stays structurally
   thin.** `RandomSelector` carries no `believed_I`/`system_util`/`temporal`
   factors (no "why" beyond uniform chance), and its real selector only
   fires ~once per round under the cache-reuse design (`fwdllm_plus` is
   richer, since it reselects every iteration). Worth a plot/metric that's
   actually informative for this selection pattern instead of reusing
   oort-family-shaped plots that don't fit it.
5. `training_budget_s`/`overran`/`remaining_time_s` stay unpopulated for
   fwdllm by design (no faithful budget concept given its flat-delay
   model) — revisit only if fwdllm's trainer ever grows a real budget model.

## Next sanity checks

- **Re-run a fresh experiment (any scale) with the current code live
  end-to-end** and confirm `sim_round_duration_s`/`utility_belief`/
  `agg_observed_s`-dependent plots populate against *real* GPU telemetry —
  they were verified against synthetic + partial real data, but the last
  completed n=100 run predates all of it, so no single real run has
  exercised the full telemetry surface at once yet.
- **Read the resulting accuracy/loss curves for an actual learning-progress
  verdict** (as opposed to the deadlock/throttle/plot-coverage verdicts
  this investigation focused on) — `plots/performance/accuracy_over_rounds.pdf`
  is the artifact to open.
- Once merged: the deletion PR per
  [`DELETION_CANDIDATES.md`](DELETION_CANDIDATES.md).
