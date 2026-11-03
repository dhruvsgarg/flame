#!/usr/bin/env python3
"""FX-N129 (V5): in-process bench leg. The harness's own aggregator and trainer classes run in one leg, MQTT replaced by an
in-memory channel: a dispatch hands the payload to the trainer's fetch -> train / evaluate -> upload, and the reply lands in
the end's queue. `--workers 0` runs each task inline (every reply present at the next drain); `--workers W` runs trainers in W
worker processes, and the aggregator's own sim gate (expected completion = dispatch + known delay, cold-start hold) waits for
any reply that can precede the next commit, as in the harness. Configs come from the launcher (the leg's `debug_run.yaml`),
telemetry keeps the harness layout, so accuracy_table / felix_plots / predictor_audit / config_audit read a bench leg unchanged.

  fl_bench.py LEG.yaml --out DIR [--gpu 0] [--workers W] [--seed S] [--runtime-s T]
"""
import argparse
import importlib.util
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EX = HERE.parent / "async_cifar10"
sys.path[:0] = [str(HERE.parents[1]), str(EX), str(EX / "trainer" / "pytorch"), str(EX / "aggregator" / "pytorch")]


def _fast_yaml():
    """The launcher's metadata (registry, splits, traces) parsed once per file version: pickle cache keyed by path, mtime, size."""
    import hashlib
    import pickle
    import yaml
    if getattr(yaml, "_bench_fast", False):
        return
    orig, loader = yaml.safe_load, getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    import fl_data
    cache = Path(os.environ.get("FLAME_BENCH_CACHE") or fl_data.cache_dir("flame_bench")) / "yaml"

    def safe_load(stream):
        name = getattr(stream, "name", None)
        if not isinstance(name, str) or not os.path.isfile(name):
            return yaml.load(stream, Loader=loader) if loader is not yaml.SafeLoader else orig(stream)
        st = os.stat(name)
        key = hashlib.sha256(f"{os.path.realpath(name)}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:24]
        f = cache / f"{key}.pkl"
        if f.exists():
            return pickle.loads(f.read_bytes())
        data = yaml.load(stream, Loader=loader)
        cache.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(tmp, f)
        return data
    yaml.safe_load, yaml._bench_fast = safe_load, True


def resolve(yaml_path):
    """(experiment, aggregator config, trainer configs, aggregator main, trainer main): the launcher's own merge (runner)."""
    _fast_yaml()
    from flame.launch.baselines import load_baselines
    from flame.launch.experiment_config import ExperimentBatch
    from flame.launch.runner import ExperimentRunner
    from flame.launch.spawner import ConfigGenerator, MetadataLoader

    exp = ExperimentBatch.from_yaml(Path(yaml_path)).experiments[0]
    r = ExperimentRunner.__new__(ExperimentRunner)  # no signal handlers, no spawners
    r.example_dir, r.metadata_dir_default = EX, EX / "metadata"
    paths = r._resolve_example_paths(exp)
    entry = r._resolve_baseline(exp, load_baselines(paths["metadata_dir"]))
    tmpl = paths["example_dir"] / exp.aggregator.config_template if exp.aggregator.config_template else None
    agg, _ = r._build_aggregator_config(exp, tmpl, entry)
    r._validate_selector_label(exp, agg)
    agg.setdefault("hyperparameters", {})["time_mode"] = exp.trainer.time_mode
    gen = ConfigGenerator(MetadataLoader(paths["metadata_dir"]), paths["trainer_base"])
    gen.set_baseline_overrides(r._build_trainer_baseline_overrides(exp, entry)[0])
    over = {"job.id": agg["job"]["id"], "job.name": agg["job"]["name"],
            "hyperparameters.training_delay_enabled": str(exp.trainer.enable_training_delays)}
    over.update({f"hyperparameters.{k}": v for k, v in (exp.trainer.hyperparameters or {}).items()})
    ids = range(exp.trainer.start_id, exp.trainer.start_id + exp.trainer.num_trainers)
    trainers = [gen.generate_trainer_config(
        tid, exp.trainer.dataset.dirichlet_alpha, exp.trainer.availability.mode, dataset_name=exp.trainer.dataset.name,
        num_trainers=exp.trainer.split_num_trainers or exp.trainer.num_trainers, skip_index_splits=exp.trainer.dataset.path_style,
        **over) for tid in ids]
    return exp, agg, trainers, paths["example_dir"] / r._resolve_aggregator_main(exp, entry), paths["trainer_main"]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config(d):
    from flame.config import Config
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(d, f)
    try:
        return Config(f.name)
    finally:
        os.unlink(f.name)


def _quiet_logging(path):
    """flame installs INFO stdout handlers at import; a bench leg keeps WARNING+ in one file."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    root.setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        lg = logging.getLogger(name)
        lg.setLevel(logging.NOTSET)
        lg.handlers, lg.propagate = [], True


class RoutingWriter:
    """Process-global telemetry writer: a trainer task's events go to its own trainer_<id>.jsonl, the rest to `default`."""

    def __init__(self, default, run_dir):
        self.default, self.run_dir, self.trainers, self._tls = default, run_dir, {}, threading.local()
        self.path = default.path if default is not None else os.path.join(run_dir, "_none")

    def writer_for(self, tid):
        from flame.telemetry import TelemetryWriter
        w = self.trainers.get(tid)
        if w is None:
            w = self.trainers[tid] = TelemetryWriter(os.path.join(self.run_dir, f"trainer_{tid}.jsonl"), "trainer", tid)
        return w

    def emit(self, event, **fields):
        w = getattr(self._tls, "w", None) or self.default
        if w is not None:
            w.emit(event, **fields)

    def close(self):
        for w in [self.default, *self.trainers.values()]:
            if w is not None:
                w.close()


class _CM:
    """ChannelManager stand-in (both roles): get_by_tag over the config's func tags."""

    def __init__(self, config, channels):
        self._config, self._channels = config, channels

    def __call__(self, config):  # ChannelManager()(config): channels already built
        self._config = config

    def get_by_tag(self, tag):
        name = (self._config.func_tag_map or {}).get(tag)
        return self._channels.get(name)

    def get(self, name):
        return self._channels.get(name)

    def join_all(self):
        pass

    def leave(self, name):
        pass

    def cleanup(self):
        pass


class TrainerChannel:
    """A trainer's side: its one aggregator end, the task just delivered, and the reply path."""

    def __init__(self, tid, reply):
        self.tid, self.reply, self.inbox, self._name = tid, reply, deque(), "param-channel"
        self._selector = type("Sel", (), {"ordered_updates_recv_ends": [], "_cleanup_send_ends": lambda s: None})()
        self.departed_eot = None

    def await_join(self, timeout=None):
        return False

    def all_ends(self):
        return ["aggregator"]

    def one_end(self, state=None):
        return "aggregator"

    def recv(self, end_id):
        from flame.channel import decode_message
        if not self.inbox:
            return None, None
        payload, ts = self.inbox.popleft()
        return decode_message(payload), ts

    def send(self, end_id, message):
        from flame.channel import encode_message
        self.reply(self.tid, encode_message(message))

    def cleanup_recvd_ends(self):
        self._selector.ordered_updates_recv_ends = []

    def update_trainer_state(self, state, timestamp):
        pass  # sim notify is the aggregator's trace replay (F42a)

    def leave(self):
        pass

    def join(self):
        pass


class _GC:
    """gc for the trainer module: load_data's per-trainer gc.collect is a separate process's hygiene, not a model term."""

    def __getattr__(self, k):
        import gc
        return getattr(gc, k)

    @staticmethod
    def collect(*_a, **_k):
        return 0


class TrainerHost:
    """Builds trainers from their launcher configs and runs one delivered task at a time (in the leg or in a worker)."""

    def __init__(self, cfgs, battery, time_mode, trainer_main, tel_dir, reply):
        _fast_yaml()
        import dataclasses
        import functools
        import fl_data
        from flame import telemetry
        import flame.mode.horizontal.syncfl.trainer as sync_trainer

        for k, s in list(fl_data.SPECS.items()):  # one train-set object per process; trainers take Subsets of it
            fl_data.SPECS[k] = dataclasses.replace(s, train=functools.lru_cache(maxsize=1)(s.train))
        self.writer = telemetry._writer if isinstance(telemetry._writer, RoutingWriter) else None
        if self.writer is None and tel_dir:
            self.writer = telemetry._writer = RoutingWriter(None, tel_dir)
        mod = _load_module(trainer_main, "bench_trainer_main")
        mod.gc = _GC()
        self.trainers, self.tch, self.tasks = {}, {}, 0
        for d in cfgs:
            cfg = _config(d)
            t = mod.PyTorchCifar10Trainer(cfg, battery, time_mode)
            tid = str(t.trainer_id)
            ch = TrainerChannel(tid, reply)
            cm = _CM(cfg, {n: ch for n in cfg.channels})
            with self._as(tid):
                t.initialize()
                sync_trainer.ChannelManager = lambda cm=cm: cm  # internal_init: self.cm = ChannelManager(); self.cm(config)
                t.internal_init()
                t.init_oort_variables()
                t.load_data()
            self.trainers[tid], self.tch[tid] = t, ch

    def _as(self, tid):
        host = self

        class _Ctx:
            def __enter__(self):
                if host.writer is not None:
                    host.writer._tls.w = host.writer.writer_for(tid)

            def __exit__(self, *exc):
                if host.writer is not None:
                    host.writer._tls.w = None
        return _Ctx()

    def run(self, tid, payload):
        from flame.mode.horizontal.syncfl.trainer import TAG_FETCH, TAG_UPLOAD
        t = self.trainers[tid]
        self.tch[tid].inbox.append((payload, datetime.now()))
        with self._as(tid):
            t._fetch_weights(TAG_FETCH)
            if not t.fetch_success or t._work_done:
                return
            t.check_and_sleep()
            t.train()
            t.evaluate()
            t.put(TAG_UPLOAD)
            self.tasks += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()


def _worker_main(cfgs, battery, time_mode, trainer_main, tel_dir, log_path, env, tasks, results):
    """Worker process: host a share of the trainers; run each (tid, payload) task as it arrives."""
    os.environ.update(env)
    _quiet_logging(log_path)
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    parent = os.getppid()

    def _orphan_watch():  # an orphaned worker exits with its leg
        while os.getppid() == parent:
            time.sleep(5)
        os._exit(0)
    threading.Thread(target=_orphan_watch, daemon=True).start()
    pending = []
    host = TrainerHost(cfgs, battery, time_mode, trainer_main, tel_dir, lambda tid, p: pending.append((tid, p)))
    results.put(("ready", None))
    last = None
    while True:
        item = tasks.get()
        if item is None:
            break
        tid, payload = item
        if payload is None:  # same payload as this worker's previous task (one version's model, sent once)
            payload = last
        last = payload
        host.run(tid, payload)
        for r in pending:
            results.put(r)
        pending.clear()
    host.close()
    results.put(("done", host.tasks))


class InlineRouter:
    """Workers 0: a dispatch runs its trainer's whole task before send_payload returns."""

    def __init__(self, host_args):
        self.host = TrainerHost(*host_args, reply=lambda tid, p: self.channel.put_reply(tid, p))
        self.channel = None

    @property
    def ids(self):
        return list(self.host.trainers)

    def deliver(self, tid, payload):
        self.host.run(tid, payload)

    def tasks(self):
        return self.host.tasks

    def close(self):
        self.host.close()


class PoolRouter:
    """Workers W: trainers split over W processes (trainer i -> worker i mod W); replies stream back on one queue."""

    def __init__(self, cfgs, battery, time_mode, trainer_main, tel_dir, log_path, workers):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.results, self.channel, self._tasks, self._sent = ctx.Queue(), None, 0, {}
        env = {k: v for k, v in os.environ.items() if k.startswith(("FLAME_", "CUDA_")) or k.endswith("_NUM_THREADS")}
        self.queues, self.procs, self.of = [], [], {}
        gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
        for w in range(workers):
            share = cfgs[w::workers]
            q = ctx.Queue()
            wenv = dict(env, CUDA_VISIBLE_DEVICES=gpus[w % len(gpus)]) if len(gpus) > 1 else env  # --gpu A,B: workers round-robin
            p = ctx.Process(target=_worker_main, args=(share, battery, time_mode, str(trainer_main), tel_dir, log_path, wenv,
                                                        q, self.results), daemon=True)
            p.start()
            self.queues.append(q)
            self.procs.append(p)
            for d in share:
                self.of[str(d["taskid"])] = w
        ready = 0
        while ready < workers:
            try:
                kind, _ = self.results.get(timeout=5)
            except queue.Empty:  # a worker that died building its trainers never reports ready
                self._check_workers()
                continue
            ready += kind == "ready"
        self.ids = [str(d["taskid"]) for d in cfgs]
        self._rx = threading.Thread(target=self._pump, daemon=True)
        self._rx.start()
        self._closing = False
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        """A dead worker strands its trainers' replies: fail the leg now, not at the pool's wall timeout."""
        while not self._closing:
            self._check_workers()
            time.sleep(5)

    def _check_workers(self):
        for w, p in enumerate(self.procs):
            if p.exitcode not in (None, 0):
                print(f"[fl_bench] worker {w} died rc={p.exitcode}: failing the leg", flush=True)
                os._exit(70)

    def _pump(self):
        while True:
            tid, payload = self.results.get()
            if tid == "done":
                self._tasks += payload
                continue
            if tid == "stop":
                return
            self.channel.put_reply(tid, payload)

    def deliver(self, tid, payload):
        w = self.of[tid]
        same = self._sent.get(w) is payload  # the aggregator reuses one payload object per (version, task, stamp)
        self._sent[w] = payload
        self.queues[w].put((tid, None if same else payload))

    def tasks(self):
        return self._tasks

    def close(self):
        self._closing = True
        for q in self.queues:
            q.put(None)
        for p in self.procs:
            p.join(timeout=120)
        self.results.put(("stop", None))


def make_agg_channel(router, selector, job_id, name, me, other, groupby):
    """The aggregator's Channel with the transport swapped; replies wait in a per-end queue the drains read."""
    from flame.channel import Channel, END_LAST_AVAIL_TS, KEY_CH_STATE, KEY_CH_SELECT_REQUESTER, encode_message, _emit_wire
    from flame.config import TrainerAvailState
    from flame.end import End
    from flame.mode.role import Role
    from flame.selector.properties import PROP_AVL_STATE

    class BenchEnd(End):
        """An End whose receive side is the channel's reply queue (thread-safe, filled by the router)."""

        def __init__(self, end_id, ch):
            super().__init__(end_id)
            self._ch = ch

        def get_ready_nowait(self):
            return self._ch._pop(self.end_id)

        def is_rxq_empty(self):
            return not self._ch._replies.get(self.end_id)

    class InprocAggChannel(Channel):
        def __init__(self):
            self._backend, self._selector, self._job_id, self._name = _NullBackend("aggregator"), selector, job_id, name
            self._my_role, self._other_role, self._groupby = me, other, groupby
            self.properties, self.departed_eot, self.mc, self.trainer_unavail_list = {}, None, Role.mc, None
            self._ends, self._end_state_info, self._active_recv_fifo_tasks = {}, {}, set()
            self._arrival_seq, self._arrival_ev, self._arrival_cv = 0, None, threading.Condition()
            self._replies = {}

        def add_end(self, end_id):
            self._ends[end_id] = BenchEnd(end_id, self)
            self._replies[end_id] = deque()
            self.set_end_property(end_id, END_LAST_AVAIL_TS, datetime.now())
            self.set_end_property(end_id, PROP_AVL_STATE, TrainerAvailState.AVL_TRAIN)
            self._end_state_info[end_id] = {END_LAST_AVAIL_TS: datetime.now()}

        def ends(self, state=None, task_to_perform="train", agg_version_key=None, trainer_version_keys=None, data_id=None,
                 allow_recv_bootstrap=False, num_to_select=None):
            if state is not None:
                self.properties[KEY_CH_STATE] = state
            self.properties[KEY_CH_SELECT_REQUESTER] = self.get_backend_id()
            selected = self._selector.select(
                ends=self._ends, channel_props=self.properties, trainer_unavail_list=self.trainer_unavail_list or [],
                task_to_perform=task_to_perform, agg_version_key=agg_version_key, trainer_version_keys=trainer_version_keys,
                data_id=data_id, allow_recv_bootstrap=allow_recv_bootstrap,
                **({"num_to_select": num_to_select} if num_to_select else {}))
            for end_id, kv in (selected or {}).items():
                if kv:
                    self._ends[end_id].set_property(*kv)
            return list(selected or {})

        def send_payload(self, end_id, payload):
            if not self.has(end_id):
                return False
            self.mc.accumulate("bytes", "send", len(payload))
            _emit_wire("send", end_id, len(payload))
            router.deliver(end_id, payload)
            return True

        def send(self, end_id, message):
            return self.send_payload(end_id, encode_message(message))

        def broadcast(self, message):
            self.mc.accumulate("bytes", "broadcast", len(encode_message(message)))

        def put_reply(self, end_id, payload):
            with self._arrival_cv:
                if end_id in self._replies:
                    self._replies[end_id].append(payload)
                    self._arrival_seq += 1
                    self._arrival_cv.notify_all()

        def _pop(self, end_id):
            """Next reply, stamped with its arrival as the aggregator reads it (after the dispatch record)."""
            with self._arrival_cv:
                q = self._replies.get(end_id)
                return (q.popleft(), datetime.now()) if q else None

        def _wait_any(self, end_ids, timeout):
            if not timeout:
                return
            with self._arrival_cv:
                self._arrival_cv.wait_for(lambda: any(self._replies.get(e) for e in end_ids), timeout)

        def recv(self, end_id):
            payload = self._pop(end_id) if self.has(end_id) else None
            if payload is None:
                return None, None
            msg, md = self._apply_recv_payload(end_id, payload)
            return msg, md[1]

        def recv_fifo(self, end_ids, first_k=0, timeout=None, deadline=None):
            first_k = min(first_k, len(end_ids)) if first_k > 0 else len(end_ids)
            left, got = [e for e in end_ids if self.has(e)], 0
            while got < first_k:
                ready = [e for e in left if self._replies.get(e)]
                if not ready:
                    wait = timeout if deadline is None else max(0.01, min(timeout or 1e9, deadline - time.time()))
                    self._wait_any(left, wait)
                    ready = [e for e in left if self._replies.get(e)]
                    if not ready:
                        yield None, ("", datetime.now())
                        return
                for e in ready[: first_k - got]:
                    left.remove(e)
                    got += 1
                    yield self._apply_recv_payload(e, self._pop(e))

        def drain_ready(self, end_ids, timeout=None):
            live = [e for e in end_ids if self.has(e)]
            if not any(self._replies.get(e) for e in live):
                self._wait_any(live, timeout)
            out = []
            for end_id in live:
                while (payload := self._pop(end_id)) is not None:
                    out.append(self._apply_recv_payload(end_id, payload))
            return out

        def ends_with_pending_rx(self):
            return {e for e, q in self._replies.items() if q}

        def await_join(self, timeout=None):
            return False

        def wait_arrival(self, after_seq, timeout):
            with self._arrival_cv:
                return self._arrival_cv.wait_for(lambda: self._arrival_seq > after_seq, timeout)

        def join(self):
            pass

        def leave(self):
            pass

        def flush_tx(self, timeout=60.0):
            return True

        def set_checkin_handler(self, fn):
            self._checkin_handler = fn

    return InprocAggChannel()


class _NullBackend:
    def __init__(self, uid):
        self._uid = uid

    def uid(self):
        return self._uid

    def set_cleanup_ready(self, end_id):
        pass


def run_leg(yaml_path, out, label=None, workers=0):
    _fast_yaml()
    import yaml
    for e in yaml.safe_load(open(yaml_path))["experiments"][:1]:
        off = ((e.get("aggregator") or {}).get("config_overrides") or {}).get("hyperparameters", {}).get("trace_offset_s")
        if off is not None:  # before resolve: the spawner shifts trainer traces by it (FX-N151)
            os.environ["FLAME_TRACE_OFFSET_S"] = str(off)
    exp, agg_cfg, tcfgs, agg_main, trainer_main = resolve(yaml_path)
    hp = agg_cfg["hyperparameters"]
    job = agg_cfg["job"]["id"]
    run_dir = Path(out) / f"run_{datetime.now():%Y%m%d_%H%M%S}_{label + '_' if label else ''}{job}"
    tel = run_dir / "telemetry"
    tel.mkdir(parents=True, exist_ok=True)
    (run_dir / "aggregator_config.json").write_text(json.dumps(agg_cfg, indent=4))
    (run_dir / "debug_run.yaml").write_text(Path(yaml_path).read_text())

    from flame import telemetry
    from flame.channel_manager import _build_selector_kwargs
    from flame.selectors import selector_provider
    import flame.mode.horizontal.syncfl.top_aggregator as sync_top

    _quiet_logging(run_dir / "bench.log")
    agg_w = telemetry.configure(role="aggregator", end_id=job, run_dir=str(tel))
    if agg_w is not None:
        telemetry._writer = RoutingWriter(agg_w, str(tel))
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    t_setup = time.time()
    host_args = (tcfgs, exp.trainer.battery_threshold, exp.trainer.time_mode, trainer_main, str(tel))
    router = InlineRouter(host_args) if workers <= 0 else PoolRouter(*host_args, str(run_dir / "bench.log"), workers)

    class AggCM(_CM):
        def __call__(self, config):
            self._config, self._channels = config, {}
            for name, cc in config.channels.items():
                me, other = (cc.pair[0], cc.pair[1]) if config.role == cc.pair[0] else (cc.pair[1], cc.pair[0])
                sel = selector_provider.get(config.selector.sort, _seed=getattr(config.hyperparameters, "seed", None),
                                            **_build_selector_kwargs(config.hyperparameters, config.selector.kwargs))
                self._channels[name] = make_agg_channel(router, sel, config.job.job_id, name, me, other,
                                                        cc.group_by.groupable_value(config.group_association.get(name)))
            router.channel = next(iter(self._channels.values()))

    agg_mod = _load_module(agg_main, "bench_agg_main")
    sync_top.ChannelManager = lambda: AggCM(None, {})
    a = agg_mod.PyTorchCifar10Aggregator(_config(agg_cfg), False)
    orig_init = a.internal_init

    def internal_init():  # trainers join once the channel exists
        orig_init()
        for tid in router.ids:
            router.channel.add_end(tid)
    a.internal_init = internal_init
    a.compose()
    t0 = time.time()
    a.run()
    wall = time.time() - t0
    router.close()
    if isinstance(telemetry._writer, RoutingWriter):
        telemetry._writer.close()
        telemetry._writer = None
    summary = dict(run_dir=str(run_dir), wall_s=wall, setup_s=t0 - t_setup, tasks=router.tasks(), workers=workers,
                   rounds=getattr(a, "_round", None), vclock=a._vclock.now)
    (run_dir / "bench_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--workers", type=int, default=0, help="trainer worker processes (0 = inline)")
    ap.add_argument("--runtime-s", type=float, default=None, help="override max_experiment_runtime_s (smokes)")
    ap.add_argument("--seed", type=int, default=None, help="aggregator seed (harness --seeds i: 1233 + i)")
    ap.add_argument("--threads", type=int, default=1, help="math threads per process (legs pack many to a node; R18)")
    a = ap.parse_args(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.threads)  # workers inherit it
    y = a.yaml
    if a.runtime_s or a.seed is not None:
        _fast_yaml()
        import yaml
        d = yaml.safe_load(open(y))
        for e in d["experiments"]:
            h = e["aggregator"]["config_overrides"]["hyperparameters"]
            if a.runtime_s:
                h["max_experiment_runtime_s"] = a.runtime_s
            if a.seed is not None:
                h["seed"] = a.seed
        os.makedirs(a.out, exist_ok=True)
        y = os.path.join(a.out, f"_bench_input_{os.getpid()}.yaml")
        yaml.safe_dump(d, open(y, "w"), sort_keys=False)
    print(json.dumps(run_leg(y, a.out, a.label, a.workers)), flush=True)
    from flame.launch.cli import exit_after_run
    exit_after_run()  # PL24: skip native teardown
    return 0


if __name__ == "__main__":
    sys.exit(main())
