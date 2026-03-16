# AsyncOortSelector Underperformance — Debug Checklist

**Setup**: `async_n100_c30_k10`, `distilbert` / `agnews`, `lr=0.01`, 10 clients numerical run. Comparing `async_oort` vs `async_random`.

**Log evidence**: `test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_14_03_03_23.log`

---

## Hypotheses

### [x] H1 — Speed info missing & not influencing selection
**Status**: **FIXED**.

`speed_last_{50,100,200}` tracks the **wall-clock duration** of each trainer's last training round (`PROP_ROUND_DURATION`, a `timedelta`) in a rolling window. It's the diagnostic view of the speed distribution the selector is working with, and feeds into [global_system_utility](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#605-641) which penalizes slow trainers.

**From the log**: all speed windows are `None`:
```
'speed_last_50': {'min': None, 'max': None, ...}
```

**How speed is supposed to flow — the full chain:**

1. **Aggregator sets `PROP_ROUND_DURATION` when weights arrive** ([asyncfl/top_aggregator.py:L428-434](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/mode/horizontal/asyncfl/top_aggregator.py#L428-434)):
   ```python
   channel.set_end_property(end, PROP_ROUND_DURATION, recv_wts_ts - sent_wts_ts)
   ```
   This happens inside the `else` branch guarded by a `SEND_TIMEOUT_WAIT_S` (90s) check at [L355](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/mode/horizontal/asyncfl/top_aggregator.py#L355). If no valid `sent_wts_ts` is found in `_track_trainer_version_duration_s`, the branch is **skipped entirely** and `PROP_ROUND_DURATION` is never set.

2. **Selector reads it after each selection** ([async_oort.py:L328-340](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#L328-340)):
   ```python
   end_speed = ends[selected_end_id].get_property(PROP_ROUND_DURATION)
   if end_speed is not None:   # <-- guard gates the append
       self._selector_stats[...]["speed_last_{window}"].append(end_speed.total_seconds())
   ```
   Because `PROP_ROUND_DURATION` is `None`, the append never fires → `speed_last_*` stays `None`.

3. **Speed penalty is silently disabled** ([async_oort.py:L613-640](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#L629-630)):
   ```python
   if end_round_duration is None:
       return 1   # no penalty at all for slow trainers
   ```

**Most likely cause**: `_track_trainer_version_duration_s` is not being populated for [end](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#1249-1773) before the weight receive path runs, so the `else` branch at [L402-434](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/mode/horizontal/asyncfl/top_aggregator.py#L402-434) is never reached and `set_end_property` for `PROP_ROUND_DURATION` is never called. Check if this dict is being initialized for each trainer when weights are first **sent** to them in [_distribute_weights](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py#1328-1335).

---

### [ ] H2 — `stat_utility` range too narrow (5–8) for meaningful differentiation
**Status**: Stat_util diagnostic tracking fixed (uncommented). Range narrowness still needs verification in next run.

Log shows utility values like `7.50`, `5.94` — a ~2-point spread. With such a narrow range, [sample_by_util()](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#389-426)'s probability weights are nearly uniform (e.g. 7.5/total ≈ 5.94/total), so exploitation degrades to near-random.

Additionally, the `stat_util` tracking is **commented out** ([L335-337](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#L335-337)):
```python
# if end_stat_util is not None:
#     self._selector_stats[...]["util_last_{window}"].append(end_stat_util)
```
So even if the range were wide, stats diagnostics stay `None`.

---

### [ ] H3 — Exploration factor too high (0.9 → mostly random)
**Status**: Plausible, pending review.

`exploration_factor = 0.9` means 90% of candidates come from [sample_by_speed()](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#427-435) which is just `np.random.choice` ([L434](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#L434)). Only 10% come from actual utility-based exploitation. Oort barely differs from random in this configuration.

Decays at `0.98x` per selection toward `min=0.2`, so it takes many rounds to shift meaningful weight to exploitation.

---

### [x] H4 — O(N) timeout scan causes large `time_per_selection` overhead
**Status**: **REJECTED**. Both `async_oort._handle_send_state` ([L1322-1376](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#L1322-1376)) and `async_random._handle_send_state` ([L483-532](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_random.py#L483-532)) have the identical timeout scan loop. The overhead is shared — this cannot explain any performance difference between the two selectors.

---

### [x] H5 — `stat_utility` tracking missing causes scoring to degrade to random only
**Status**: **PARTIALLY REJECTED** as sole cause. The commented-out `stat_util` tracking only affects the [_selector_stats](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#220-222) **diagnostic summary** logged every 5 selections. The actual `PROP_STAT_UTILITY` on each `End` object is read directly in [fetch_statistical_utility()](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#490-519) ([L510](file:///Users/gaurav/Projects/flame_clone/lib/python/flame/selector/async_oort.py#L510)) — that path is fine. So Oort *is* reading stat_utility for selection; the issue is the values are too narrow (H2) and speed is missing (H1).

---

## Open Questions
- Where is `PROP_ROUND_DURATION` supposed to be set on `End` objects? Is it set in the trainer/aggregator pipeline at all?
- What is the configured `selectType`? If it's not `"default"`, the exploration/exploitation path is bypassed entirely.
