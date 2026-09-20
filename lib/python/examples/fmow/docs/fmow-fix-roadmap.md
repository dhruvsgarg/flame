# Proposed Flame and FMoW fixes

These proposals address runtime issues found during FMoW review. They are not
implemented by the satellite setup changes. Historical evidence and source
findings are in the [reliability audit](../../../../../docs/fmow-flame-reliability-audit.md);
shared interface proposals are in the [contract audit](../../../../../docs/flame-contract-audit.md).

## Priority order

| Priority | Problem | Proposed change |
| --- | --- | --- |
| 1 | Sync can consume 200 results, use 175, and leave discarded trainers marked busy. No future availability event can then be mistaken for end of simulation. | Track each assignment and resolve it on acceptance, discard, or failure. Check pending work before stopping. |
| 2 | Sync can wait for all selected trainers even when fewer results are needed. A first result can have no timeout; incomplete chunk transfers can mix with later messages. | Add task failure reporting, bounded waits, and identified transfers with integrity checks and retries. |
| 3 | The aggregator retains accepted deltas on its GPU until reduction. | Accumulate incrementally where the algorithm allows; bound retained results and reserve aggregation/evaluation memory. |
| 4 | Connected trainers may still be loading data or may fail during startup. | Report readiness after initialization and supervise process failures throughout the run. |
| 5 | Some waits ignore the stop flag. | Make waits and training loops respond to cancellation. |
| 6 | FMoW clears the evaluation flag without signaling the shared completion event, allowing a five-minute wait. | Release both signals in `finally`. |
| 7 | Cleanup matches broad script names and can kill another run's processes. | Track and stop only processes owned by the run. |

Saved logs confirm an early stop with 200 connected trainers, 17 marked busy,
and 183 eligible against a goal of 185. They do not establish why all 17 remained
busy. A separate run confirms an aggregation OOM at round 139. Other mechanisms
above come from source inspection or focused reproductions, not full GPU tests.

## Shared task tracking

Channels, selectors, and aggregators currently keep overlapping records of work.
A shared coordinator should own assignment identity, dispatch, receipt, failure,
and completion. Selectors choose participants; aggregation policies decide whether
to accept, retain, or discard results. Every decision must resolve the assignment.

Use sync and async as initial consumers of this interface while preserving their
different aggregation rules. See the contract audit for the full repair order.

## Fewer GPU processes

The current 200-satellite, eight-GPU setup creates 25 trainer processes per GPU,
each with its own model and CUDA context. A worker pool could serve logical
satellites while preserving each satellite's data and state. Start with one worker
per GPU, then measure whether additional workers improve throughput within memory limits.

Queued execution must preserve the assigned model, data cutoff, and seed.
If modeled durations are 10 and 20 seconds from time zero, results must become
available at those simulated times even when a GPU computes them sequentially.
This requires separating modeled duration from host execution time. The current
timing rule mixes the two. Worker pooling is a proposal, not current behavior.
