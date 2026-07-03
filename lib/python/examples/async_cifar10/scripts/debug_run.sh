#!/bin/bash
# Configurable debug runner for targeted baseline comparison.
#
# Generates a single filtered+patched YAML on the fly from the parity template
# (felix_oort_refl_feddance_alpha0.1_parity.yaml — each baseline as a sim+real
# pair), keeping only the requested baselines and overriding runtime. Node- and
# duration-agnostic: the same invocation works on any machine — pick baselines,
# mode, duration.
#
# Usage (run anywhere):
#   debug_run.sh --baselines oort [--runtime-s 3600] [--mode sim|real|both]
#   debug_run.sh --baselines 'refl feddance' --runtime-s 1800
#   debug_run.sh smoke        # 48 trainers, 4 rounds, all baselines
#
# --baselines is matched against the 'baseline:' field in the parity config, so
# any baseline runs regardless of machine (felix/oort/refl/feddance). An unknown
# baseline is a no-op (not an error).
#
# Runtime:
#   --runtime-s sets max_experiment_runtime_s for BOTH real and sim variants.
#   Real mode:  wall-clock seconds (passes directly).
#   Sim mode:   virtual-clock seconds (vclock fix ensures sim stops at this
#               many virtual seconds, which completes in far less wall-clock
#               time for sync baselines like Refl).
set -u

# repo example dir (portable across nodes)
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"       # .../examples/async_cifar10
REPO_ROOT="$(cd "$EX/../../../.." && pwd)"                  # flame/

# shared harness: conda activation, launch+ticker, log asserts, preflight bridge.
# The conda activation, PYTHONPATH pin, and launch/progress loop that used to be
# inlined here now live in examples/scripts/expt_runner.sh (shared with fwdllm).
# shellcheck source=../../scripts/expt_runner.sh
source "$REPO_ROOT/lib/python/examples/scripts/expt_runner.sh"

expt_activate_conda dg_flame           # default env dg_flame (FLAME_CONDA_ENV overrides)
expt_pin_pythonpath "$REPO_ROOT"
cd "$EX" || exit 1
SCR=expt_scripts_2026
LOGDIR="${FLAME_LOGDIR:-/tmp/debug_run_logs}"; mkdir -p "$LOGDIR"
export FLAME_BATCH_CONTINUE_ON_ERROR=1

# defaults
RUNTIME_S=10800
BASELINES="felix refl"
SIM_WALL_CEILING_S=""  # empty = max_experiment_runtime_s (1×, tight guard; sim should be faster than real)
MODE="both"            # sim | real | both — which time_mode variant(s) of each baseline to run
NUM_TRAINERS=""        # empty = use whatever's in the parity config (300); non-smoke override only
ALPHA=""               # empty = use the parity config's dirichlet_alpha (0.1); e.g. 100 for homogeneous
DRY_RUN=0              # --dry-run: show the pre-flight table + checks, generate cfg, DON'T launch
SHOW_ALL=0             # --show-all: expand tier ③ + list passing checks
STRICT=0               # --strict: a BLOCKING pre-flight check aborts (default: warn + continue, so
                       # smoke_suite.sh's non-interactive timeout-wrapped runs never hang/abort)
AFTER=""               # --after: comma list of post-launch hooks (parity,plot) -- see after_* below

usage() {
  echo "usage: $0 [--baselines 'felix refl'] [--runtime-s 3600] [--mode sim|real|both] [--sim-wall-ceiling-s 2700] [--trace syn_20]"
  echo "       $0 smoke [--baselines ...] [--mode sim|real|both] [--trace syn_20]"
  echo ""
  echo "  --baselines           which baselines to run (any of felix oort oort_star refl feddance fedbuff);"
  echo "                        filtered from the parity config, node-agnostic."
  echo "  --mode                which time_mode variant(s) to run for each baseline:"
  echo "                        'sim' (only the simulated run), 'real' (only the real run),"
  echo "                        or 'both' (default, runs both sequentially). Lets you split"
  echo "                        e.g. felix-sim on one machine and felix-real on another."
  echo "  --sim-wall-ceiling-s  wall-clock ceiling for sim mode (default: = runtime_s)."
  echo "                        A well-behaved sim finishes in <= real-mode wall time."
  echo "                        Fires [SIM_WALL_CEILING] warning + stops when exceeded."
  echo "  --trace               availability trace name(s) to substitute, space-separated for"
  echo "                        multiple (e.g. 'syn_20 syn_50' queues both, one experiment set each)."
  echo "                        Default: use whatever is in the parity config (syn_0)."
  echo "  --num-trainers        non-smoke only: shrink the cohort below the parity config's 300,"
  echo "                        scaling min_trainers_to_start down with it (gap of 8, same ratio as"
  echo "                        smoke). Use this instead of 'smoke' when you need a real --runtime-s"
  echo "                        budget (e.g. a vclock floor for an availability trace) that smoke's"
  echo "                        hardcoded rounds=4/runtime=240 would cut short."
  echo "  --alpha               Dirichlet alpha override (default: parity config's 0.1). Supported"
  echo "                        values have an n300 split: 0.1 / 1.0 / 10.0 / 100.0 (100=homogeneous)."
  echo "                        When set, the split lookup uses the n300 partition for that alpha."
  echo "  --dry-run             show the pre-flight hyperparameter table + feasibility checks and the"
  echo "                        generated cfg, then exit WITHOUT launching."
  echo "  --show-all            expand tier ③ (config-baked rows) + list the passing checks too."
  echo "  --strict              abort if a pre-flight check is BLOCKING (default: warn + continue, so"
  echo "                        smoke_suite.sh's non-interactive runs never hang/abort)."
  echo "  --after HOOKS          comma-separated post-launch hooks: parity (scripts.parity.cli --batch"
  echo "                        sim-vs-real per baseline) and/or plot (analyze_run cross-baseline"
  echo "                        streaming figs). Absorbs the old compare_overnight.sh. e.g. --after parity,plot"
  exit 2
}

# parse args
TRACE=""  # empty = use whatever is in the parity config (syn_0)
if [ "${1:-}" = "smoke" ]; then
  SMOKE=1; shift
  BASELINES="felix oort oort_star refl feddance fedbuff"   # smoke default: validate all
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --baselines) BASELINES="$2"; shift 2 ;;
      --mode)      MODE="$2"; shift 2 ;;
      --trace)     TRACE="$2"; shift 2 ;;
      --alpha)     ALPHA="$2"; shift 2 ;;
      --dry-run)   DRY_RUN=1; shift ;;
      --show-all)  SHOW_ALL=1; shift ;;
      --strict)    STRICT=1; shift ;;
      --after)     AFTER="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
else
  SMOKE=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --baselines)           BASELINES="$2"; shift 2 ;;
      --runtime-s)           RUNTIME_S="$2"; shift 2 ;;
      --mode)                MODE="$2"; shift 2 ;;
      --sim-wall-ceiling-s)  SIM_WALL_CEILING_S="$2"; shift 2 ;;
      --wall-runtime-s)      SIM_WALL_CEILING_S="$2"; shift 2 ;;  # backward compat alias
      --trace)               TRACE="$2"; shift 2 ;;
      --num-trainers)        NUM_TRAINERS="$2"; shift 2 ;;
      --alpha)               ALPHA="$2"; shift 2 ;;
      --dry-run)             DRY_RUN=1; shift ;;
      --show-all)            SHOW_ALL=1; shift ;;
      --strict)              STRICT=1; shift ;;
      --after)               AFTER="$2"; shift 2 ;;
      # --node is DEPRECATED (node1/node2 split removed): baselines are filtered
      # from a single node-agnostic parity config, so the node is irrelevant.
      # Accept+ignore so existing wrappers don't hard-error.
      --node)                echo "WARNING: --node '$2' is deprecated and ignored (runner is now node-agnostic)." >&2; shift 2 ;;
      *) usage ;;
    esac
  done
fi
case "$MODE" in sim|real|both) ;; *) echo "ERROR: --mode must be sim|real|both (got '$MODE')" >&2; exit 2 ;; esac

# Generate a single filtered+patched YAML from the parity source config.
# $1 = baselines (space-separated), $2 = runtime_s, $3 = output path,
# [$4 = smoke: 1|0], [$5 = sim_wall_ceiling_s: int or ""], [$6 = mode: sim|real|both],
# [$7 = trace: trace name or ""], [$8 = num_trainers override: int or "", non-smoke only],
# [$9 = alpha override: float or ""]
make_debug_yaml() {
  python - "$SCR" "$1" "$2" "$3" "${4:-0}" "${5:-}" "${6:-both}" "${7:-}" "${8:-}" "${9:-}" <<'PY'
import yaml, sys, copy, os
scr, baselines_str, runtime_s, outpath, smoke, ceil_arg = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1",
    sys.argv[6] if len(sys.argv) > 6 else ""
)
mode = (sys.argv[7] if len(sys.argv) > 7 else "both").lower()
# Space-separated list of trace names (e.g. "syn_20 syn_50"); "" -> [""] (no substitution).
trace_overrides = sys.argv[8].strip().split() if len(sys.argv) > 8 and sys.argv[8].strip() else [""]
num_trainers_override = int(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9].strip() else None
alpha_override = float(sys.argv[10]) if len(sys.argv) > 10 and sys.argv[10].strip() else None
requested = set(baselines_str.lower().split())
# Deterministic selection seed (same for real+sim). Default 1234; SEED=none disables.
_seed_env = os.environ.get("SEED", "1234").strip()
seed_val = None if _seed_env.lower() in ("none", "") else int(_seed_env)


def exp_mode(e):
    """sim | real for an experiment, from its time_mode field (preferred) or its
    name suffix (_sim / _real)."""
    tm = str((e.get("trainer") or {}).get("time_mode") or "").lower()
    if tm.startswith("sim"):
        return "sim"
    if tm == "real":
        return "real"
    n = e.get("name", "").lower()
    if n.endswith("_real"):
        return "real"
    if n.endswith("_sim"):
        return "sim"
    return "unknown"


# Single node-agnostic parity config holding every baseline (felix, oort, refl,
# feddance) × {sim, real}; filter it to the requested baselines/mode.
src = f"{scr}/felix_oort_refl_feddance_alpha0.1_parity.yaml"
try:
    # encoding="utf-8" explicit: the config has non-ASCII chars (e.g. "->" arrows
    # in comments/descriptions); without this, open() falls back to the node's
    # locale-preferred encoding, which mis-decodes them on non-UTF-8 locales
    # (e.g. C/POSIX) and yaml.safe_load then rejects the resulting control chars.
    cfg = yaml.safe_load(open(src, encoding="utf-8"))
except FileNotFoundError:
    print(f"ERROR: parity config not found: {src}", flush=True)
    sys.exit(1)

kept = []
for e_src in cfg.get("experiments", []):
    bl = e_src.get("baseline", "").lower()
    if bl not in requested:
        continue
    if mode != "both" and exp_mode(e_src) != mode:
        continue
    # One experiment per requested trace (trace_overrides has 1 entry, "", when
    # --trace wasn't given, so this loop is a no-op pass-through by default).
    for trace_override in trace_overrides:
        e = copy.deepcopy(e_src)
        h = e["aggregator"]["config_overrides"]["hyperparameters"]
        h["max_experiment_runtime_s"] = runtime_s
        # Deterministic seed: the SAME value for every experiment so the real and sim
        # variants of each baseline make identical selection draws (dedicated per-
        # selector RNG, PARITY "Determinism / seeding"). Without this, real vs sim are
        # two independent stochastic paths and participation/utility can never match.
        # Override per-invocation with SEED=<n>; SEED=none disables (legacy unseeded).
        if seed_val is not None:
            h["seed"] = seed_val
        # sim_wall_ceiling_s: tight wall guard -- sim must finish in <= this many
        # wall-seconds (default = max_experiment_runtime_s = 1x; a healthy sim is faster).
        h["sim_wall_ceiling_s"] = int(ceil_arg) if ceil_arg else runtime_s
        if smoke:
            e["trainer"]["num_trainers"] = 48
            h["rounds"] = 4
            h["min_trainers_to_start"] = 40
            h["min_trainers_join_timeout_s"] = 120
            e["name"] = "dbg_smoke_" + e["name"]
        else:
            # High round cap so the wall/vclock budget (max_experiment_runtime_s) is the
            # binding stop condition, not an early round-count termination.
            h["rounds"] = 20000
            if num_trainers_override:
                # Shrink the cohort but keep runtime_s as the real budget (unlike
                # smoke, which hardcodes rounds=4/runtime=240 -- too short for a
                # trace-driven vclock floor like syn_20's first UN_AVL at t=600s).
                # Same join-barrier slack ratio as smoke (gap of 8 below the count).
                # Preserve the config's native partition size as split_num_trainers
                # so the shrunk cohort reads the existing n<orig> split (e.g. n300)
                # instead of demanding a dedicated n<override> split file that may
                # not exist (there is no cifar10_alpha0.1_n10 split, only n48/50/300).
                # spawn_all spawns num_trainers trainers but keys the split lookup on
                # split_num_trainers -- the two are independent by design.
                orig_n = e["trainer"].get("num_trainers", 300)
                e["trainer"]["num_trainers"] = num_trainers_override
                e["trainer"]["split_num_trainers"] = orig_n
                h["min_trainers_to_start"] = max(1, num_trainers_override - 8)
            e["name"] = f"dbg_{e['name']}"
        # --alpha override: repoint dirichlet_alpha and the split lookup. Only n300
        # splits exist for every alpha (0.1/1.0/10.0/100.0=homogeneous); n48/n50
        # exist for alpha0.1 only. So read the n300 partition for the chosen alpha
        # (the cohort stays num_trainers, spawned as the first num_trainers of the
        # 300-way split via the split_num_trainers decoupling). Name gets an
        # alpha<..> tag so run dirs are distinguishable across alphas.
        if alpha_override is not None:
            e["trainer"].setdefault("dataset", {})["dirichlet_alpha"] = alpha_override
            e["trainer"]["split_num_trainers"] = 300
            e["name"] = f"{e['name']}_alpha{str(alpha_override).replace('.', 'p')}"
        # --trace override: substitute availability trace in trainer + aggregator config.
        if trace_override:
            # trainer.availability.mode is NOT read by anything (main.py/config.py never
            # touch config.availability) -- vestigial from an earlier design, kept
            # write-only here so as not to silently drop a field some other consumer may
            # still expect. The trainer's ACTUAL trace selection comes from
            # hyperparameters.client_notify.trace (see main.py's state_avl_event_ts
            # assignment), which lives under trainer.config_overrides.hyperparameters,
            # not trainer.hyperparameters (that block is base-model HP only: batchSize/
            # learningRate/etc, merged from configs/trainer_base.yaml's own client_notify
            # default of trace=syn_0). Before this fix, ONLY the aggregator's own trace
            # read (via `h` below) was ever overridden -- every debug_run.sh-launched
            # trainer, real and sim, ran with client_notify.trace stuck at the
            # trainer_base.yaml default (syn_0, always-available) regardless of the
            # requested --trace, silently no-op'ing the trainer-side avl_state machinery
            # (and hence the real-mode send-gate and all avail_change telemetry) for
            # every trace-driven run this project has ever launched. Root-caused Jul 1
            # via UNAVAILABILITY_DESIGN.md Batch 3 T3.1.
            avail = e["trainer"].setdefault("availability", {})
            old_trace = avail.get("mode", "syn_0")
            avail["mode"] = trace_override
            t_co_hp = e["trainer"].setdefault("config_overrides", {}).setdefault("hyperparameters", {})
            t_co_hp.setdefault("client_notify", {})["trace"] = trace_override
            if "trackTrainerAvail" in h:
                h["trackTrainerAvail"]["trace"] = trace_override
                # For baselines NOT on the ORACULAR legacy path (felix, feddance,
                # oracle, fedbuff): activate the new sim_unavailability gate so
                # _init_availability picks up the trace (Sec 7 felix master-gate).
                # ORACULAR baselines (oort, refl) already activate via the legacy path.
                if h["trackTrainerAvail"].get("type", "").upper() != "ORACULAR":
                    h["simUnavailability"] = True
                    # proactive_inflight_evict is set directly in each experiment's
                    # config_overrides HP (T1 two-axis split); no auto-detection needed
                    # here. The client_notify.enabled check below is always False
                    # (Stage H is future), so proactiveInflightEvict is never set by
                    # this branch -- the explicit YAML value is authoritative.
                    t_hp = e.get("trainer", {}).get("hyperparameters", {})
                    if str(t_hp.get("client_notify", {}).get("enabled", "False")).lower() == "true":
                        h["proactiveInflightEvict"] = True
            elif "client_notify" in h and isinstance(h["client_notify"], dict):
                h["client_notify"]["trace"] = trace_override
                h["simUnavailability"] = True
            elif e["aggregator"].get("tracking_mode", "oracular").lower() != "oracular":
                # Non-oracular baseline with no HP-level tracking block (e.g. feddance
                # in v1, which has no client_notify in HP and no trackTrainerAvail).
                # Inject trace via availability_trace so _init_availability finds it.
                h["availability_trace"] = trace_override
                h["simUnavailability"] = True
            # Rewrite syn_<digits> or syn<digits> in the name so run dirs are identifiable.
            import re
            e["name"] = re.sub(r"syn_?[0-9]+", trace_override, e["name"])
        e["aggregator"]["config_overrides"]["job"]["id"] = e["name"]
        kept.append(e)

if not kept:
    print(f"WARNING: no experiments matched baselines={baselines_str} mode={mode}",
          flush=True)
    sys.exit(0)

cfg["experiments"] = kept
yaml.safe_dump(cfg, open(outpath, "w", encoding="utf-8"), sort_keys=False)
print(f"Generated {outpath} with {len(kept)} experiment(s): "
      f"{[e['name'] for e in kept]}", flush=True)
PY
}

# Count experiments in a generated YAML (used to estimate budget and track progress).
_count_exps() {
  python3 - "$1" <<'PY'
import yaml, sys
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(len(d.get('experiments', [])))
PY
}

# Thin wrapper over the shared harness's expt_launch (identical mechanics:
# 30s progress ticker + run_* dir counting + the START/DONE log lines). Kept as
# a named function so the two call sites below are unchanged.
run_node() {
  local label="$1" cfg="$2" budget_s="${3:-0}" n_exps="${4:-1}"
  expt_launch "$label" "$cfg" "$EX" "$budget_s" "$n_exps" "$LOGDIR"
}

# cifar_preflight <cfg> -- render the tiered hyperparameter table + feasibility
# checks (shared examples/scripts/expt_runner.py) for the just-generated combined
# cfg. Returns 2 if a check is BLOCKING. Callers decide what to do with that:
# by default a block only WARNs and continues (so smoke_suite.sh's non-interactive
# timeout-wrapped invocations never hang or abort); --strict makes it fatal.
GPUS_VISIBLE="$( (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) || echo 0)"
cifar_preflight() {
  local cfg="$1"
  EXPT_RUNNER_DIR="$EXPT_RUNNER_DIR" CFG="$cfg" \
  BASELINES="$BASELINES" MODE="$MODE" RUNTIME_S="$RUNTIME_S" TRACE="$TRACE" \
  ALPHA="$ALPHA" NUM_TRAINERS="$NUM_TRAINERS" SIM_WALL_CEILING_S="$SIM_WALL_CEILING_S" \
  GPUS_VISIBLE="$GPUS_VISIBLE" DRY_RUN="$DRY_RUN" SHOW_ALL="$SHOW_ALL" \
  EX="$EX" LOGDIR="$LOGDIR" \
  python - <<'PY'
import os, sys, yaml
sys.path.insert(0, os.environ["EXPT_RUNNER_DIR"])
import expt_runner
env = os.environ.get
cfg = yaml.safe_load(open(env("CFG"), encoding="utf-8"))
exps = cfg.get("experiments", [])
gpus_vis = int(env("GPUS_VISIBLE") or "0")

rows2, checks = [], []
for e in exps:
    h = e["aggregator"]["config_overrides"]["hyperparameters"]
    n = e.get("trainer", {}).get("num_trainers")
    ng = e.get("execution", {}).get("num_gpus")
    mts = h.get("min_trainers_to_start")
    a = (e.get("trainer", {}).get("dataset", {}) or {}).get("dirichlet_alpha")
    rows2.append({"label": e.get("name", "?")[:26],
                  "value": f"n_trainers={n}  n_gpus={ng}  min_start={mts}  rounds={h.get('rounds')}  alpha={a}"})
    if isinstance(n, int) and isinstance(mts, int) and n < mts:
        checks.append({"name": f"num_trainers >= min_trainers_to_start ({e.get('name')})",
                       "level": "error", "detail": f"{n} < {mts} — join barrier never clears"})
    if isinstance(ng, int) and gpus_vis and ng > gpus_vis:
        checks.append({"name": f"num_gpus <= gpus_visible ({e.get('name')})",
                       "level": "error", "detail": f"num_gpus={ng} > visible={gpus_vis}"})

tiers = [
    {"name": "① REVIEW EVERY RUN", "rows": [
        {"label": "baselines", "value": env("BASELINES")},
        {"label": "mode", "value": env("MODE"),
         **({"level": "warn", "note": "single-sided: parity needs both"} if env("MODE") != "both" else {})},
        {"label": "runtime_s", "value": env("RUNTIME_S")},
        {"label": "trace", "value": env("TRACE") or "<parity config default: syn_0>",
         **({"level": "warn", "note": "not syn_0"} if (env("TRACE") and env("TRACE") != "syn_0") else {})},
        {"label": "sim_wall_ceiling", "value": env("SIM_WALL_CEILING_S") or "= runtime_s (auto)"},
        {"label": "alpha", "value": env("ALPHA") or "<parity config default: 0.1>"},
    ]},
    {"name": "② PER-EXPERIMENT (moderate)", "rows": rows2},
    {"name": "③ RARELY CHANGED", "collapsed": True, "rows": [
        {"label": "env", "value": os.environ.get("CONDA_DEFAULT_ENV", "?")},
        {"label": "gpus_visible", "value": str(gpus_vis)},
        {"label": "example_dir", "value": env("EX")},
        {"label": "logdir", "value": env("LOGDIR")},
    ]},
]
checks.append({"name": "run names carry _real/_sim tags for parity glob", "level": "ok",
               "detail": "make_debug_yaml keeps the parity config's _sim/_real suffixes"})
spec = {"title": "CIFAR DEBUG RUN", "subtitle": f"{len(exps)} experiment(s)",
        "dry_run": env("DRY_RUN") == "1", "tiers": tiers, "checks": checks}
sys.exit(expt_runner.render_and_gate(spec, show_all=(env("SHOW_ALL") == "1")))
PY
}

# gate_or_continue <preflight_rc> -- shared post-preflight decision for both paths.
gate_or_continue() {
  local rc="$1"
  if [ "$DRY_RUN" = "1" ]; then
    echo "--dry-run: generated cfg in $LOGDIR. Nothing launched."; exit 0
  fi
  if [ "$rc" -eq 2 ]; then
    if [ "$STRICT" = "1" ]; then
      echo "Pre-flight BLOCKED (exit 2) and --strict set. Nothing launched." >&2; exit 2
    fi
    echo "WARNING: pre-flight flagged a BLOCKING check (continuing; pass --strict to abort)." >&2
  fi
}

# ---- post-launch hooks (--after ...), dispatched by expt_dispatch_after ----
# after_parity / after_plot absorb what compare_overnight.sh used to do (its
# per-baseline sim-vs-real parity + cross-baseline streaming plots), but off the
# maintained scripts.parity.cli engine (compare_overnight used the legacy
# scripts/parity_check.py).
after_parity() {
  python -m scripts.parity.cli --batch --experiments-dir experiments \
    --baselines $BASELINES --json-out "$LOGDIR/parity_<baseline>.json"
}
after_plot() {
  local ar="$REPO_ROOT/scripts/analysis/analyze_run.py" mode b d
  [ -f "$ar" ] || { echo "  [after:plot] $ar not found — skipping" >&2; return 0; }
  for mode in sim real; do
    local dirs=() labels=()
    for b in $BASELINES; do
      d=$(ls -dt experiments/run_*dbg_*"${b}"*_"${mode}"* 2>/dev/null | head -1)
      [ -n "$d" ] && [ -d "$d/telemetry" ] && { dirs+=("$d/telemetry"); labels+=("$b"); }
    done
    if [ "${#dirs[@]}" -ge 2 ]; then
      echo "  [after:plot] $mode cross-baseline: ${labels[*]}"
      python "$ar" --compare-streaming "${dirs[@]}" --labels "${labels[@]}" \
        --out "$LOGDIR/${mode}_cross" || true
    fi
  done
}

# ---- smoke mode ----
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE DEBUG: 48 trainers, 4 rounds, baselines=${BASELINES} ==="
  cfg="$LOGDIR/dbg_smoke.yaml"
  # Clear any stale config from a previous invocation so a no-match run is
  # skipped (not silently re-running a leftover config).
  rm -f "$cfg"
  make_debug_yaml "$BASELINES" 240 "$cfg" 1 "$SIM_WALL_CEILING_S" "$MODE" "$TRACE" "" "$ALPHA"
  if [ -f "$cfg" ]; then
    _n=$(_count_exps "$cfg")
    cifar_preflight "$cfg"; gate_or_continue $?
    run_node "dbg_smoke" "$cfg" $(( _n * 240 )) "$_n"
    expt_assert_run "$EX" "$EXPT_LAST_MARKER" "dbg_smoke"
    [ -n "$AFTER" ] && expt_dispatch_after "$AFTER"
  elif [ "$DRY_RUN" = "1" ]; then
    echo "--dry-run: no experiments matched baselines='$BASELINES' mode=$MODE. Nothing to show."; exit 0
  fi
  echo "=== SMOKE RESULTS ==="
  for dd in experiments/run_*dbg_smoke_*; do
    [ -d "$dd" ] || continue
    t=$(ls "$dd"/telemetry/aggregator_*.jsonl 2>/dev/null | head -1)
    rounds=$(grep -c "agg_round" "$t" 2>/dev/null || echo 0)
    printf "  %-60s agg_round_events=%s\n" "$(basename "$dd")" "$rounds"
  done
  exit 0
fi

# ---- normal run mode ----
echo "=== DEBUG RUN: baselines='$BASELINES' mode=$MODE runtime_s=$RUNTIME_S sim_wall_ceiling_s=${SIM_WALL_CEILING_S:-auto(=runtime_s)} num_trainers=${NUM_TRAINERS:-300(default)} alpha=${ALPHA:-0.1(default)} ==="
cfg="$LOGDIR/debug_run.yaml"
# Clear any stale config so a no-match run is skipped (not silently re-running
# a previous baseline's leftover config).
rm -f "$cfg"
make_debug_yaml "$BASELINES" "$RUNTIME_S" "$cfg" 0 "$SIM_WALL_CEILING_S" "$MODE" "$TRACE" "$NUM_TRAINERS" "$ALPHA"

if [ ! -f "$cfg" ]; then
  echo "No experiments matched for baselines='$BASELINES'. Nothing to run."
  exit 0
fi

_n_exps=$(_count_exps "$cfg")
_budget=$(( _n_exps * RUNTIME_S ))
echo "  queued: $_n_exps exp(s), estimated budget ~${_budget}s (sim finishes faster than real)"
cifar_preflight "$cfg"; gate_or_continue $?
run_node "debug_run" "$cfg" "$_budget" "$_n_exps"
expt_assert_run "$EX" "$EXPT_LAST_MARKER" "debug_run"
[ -n "$AFTER" ] && expt_dispatch_after "$AFTER"
echo "Logs: $LOGDIR/debug_run.out"
echo "Run dirs: experiments/run_*dbg_*"
