# MQTT Network-Emulation PoC — Implementation Plan

**Status:** Planning complete, implementation not started. This is the pickup
doc for the next session — follow §7 in order.

**Relationship to `NETWORK_EMULATION_DESIGN.md`:** that doc designs the
*durable, integrated* capability (launcher-managed, per-trainer shaping,
wired into `runner.py`). This doc is the **hello-world PoC** that comes
first: a standalone client/server pair that talks MQTT in the same style as
`flame/backend/mqtt.py`, shaped by Toxiproxy, with **zero changes to
trainer/aggregator/channel_manager code**. It proves the mechanism end-to-end
before any integration work starts. Toxiproxy was already decided as the
shaping mechanism in that doc's §3 (no-sudo constraint) — this plan reuses
that decision verbatim.

**Environment split (why this is a plan-only doc for now):** this dev node
has no MQTT broker and hasn't been confirmed to have Go. The actual run
happens on a different node that has MQTT (mosquitto) available. So the
task right now is to get all code written and correct on this node, then
push it over and run it there — nothing here can be executed/validated
locally. Write carefully, don't assume a local test loop will catch mistakes.

---

## 1. What "mimic the real style" means concretely

Reuse the real wire format for the part being measured (data-plane chunked
transfer); simplify everything that's pure orchestration and not what we're
testing (JOIN/LEAVE/presence). Concretely, import and reuse directly from
the installed `flame` package (read-only imports, no modification):

- `flame.backend.chunk_store.ChunkStore` — same 1 MB chunking (`get_chunk`
  for sending, `assemble` for receiving) that the real MQTT backend uses.
- `flame.proto.backend_msg_pb2.Data` — same protobuf message
  (`end_id`, `channel_name`, `payload`, `seqno`, `eom`), wrapped in
  `google.protobuf.any_pb2.Any`, same as `flame/backend/mqtt.py:send_chunk`.
- `flame.backend.mqtt.MqttQoS` — same QoS enum (use `EXACTLY_ONCE`, matching
  real data-plane sends).
- `flame.common.constants.MQTT_TOPIC_PREFIX` (`/flame`) — reuse the prefix,
  add a `netpoc` sub-namespace so PoC traffic can never collide with a real
  job's topics.
- `paho.mqtt.client`, `MQTTv5`, `CallbackAPIVersion.VERSION1` — same client
  library/version/callback API as `flame/backend/mqtt.py:configure`.

Deliberately **not** reused: `Notify` proto / JOIN-LEAVE-STATE_UPDATE
machinery, `ChannelManager`, `AbstractBackend`, the asyncio
`background_thread_loop` + `AsyncioHelper` plumbing. Those are orchestration
concerns belonging to the real backend, not to what this PoC measures. The
PoC's control plane (see §4) is a deliberately minimal ad hoc handshake over
plain-text MQTT messages — call this out in `chunked_mqtt.py`'s module
docstring so nobody mistakes it for a reproduction of the real Notify
protocol.

Threading model: use paho's built-in `client.loop_start()` (background
thread), not the real backend's asyncio integration — irrelevant to wire
format fidelity, and much less code for a throwaway script.

---

## 2. Directory layout (all new, nothing existing touched)

```
lib/python/examples/async_cifar10/network_poc/
  PLAN.md                 <- this file
  README.md               <- prereqs + how to run, written last
  setup_toxiproxy.sh       <- no-sudo conda+go install of toxiproxy-server/cli
  toxiproxy_ctl.py         <- HTTP API wrapper: create/shape/teardown proxies
  chunked_mqtt.py          <- shared sender/receiver using real chunk+proto format
  peer.py                  <- client/server role process (the actual PoC test)
  profiles.yaml            <- network profiles x payload sizes to sweep
  run_sweep.py             <- orchestrator: the "script" the PoC deliverable is
  results/                 <- CSV + logs land here (gitignore'd)
```

---

## 3. Toxiproxy install — no sudo, via conda + `go install`

Confirmed by reading the upstream repo directly (`go.mod` and `Makefile` at
`github.com/Shopify/toxiproxy`, checked 2026-07-21):

- Module path: `github.com/Shopify/toxiproxy/v2`
- Two `cmd/` packages: `cmd/server` and `cmd/cli` (**not** named
  `toxiproxy-server`/`toxiproxy-cli`  — that renaming only happens in the
  project's own `Makefile` via `go build -o toxiproxy-server ./cmd/server`).
  Plain `go install .../cmd/server@latest` will therefore produce a binary
  literally called `server`, and `.../cmd/cli@latest` one called `cli` — both
  are far too generic to drop onto `PATH`. The install script must rename
  them after install.

No conda-forge package for toxiproxy itself exists (checked via web search,
2026-07-21) — only `go` does (`conda-forge::go`, confirmed available). So the
plan is: install `go` via conda (ties into the existing conda-managed env,
satisfying "tied into setup.py" in spirit — conda is the vehicle, `go` is a
conda dependency, the resulting binaries land inside that same env's `bin/`
so they're on `PATH` automatically whenever the env is active), then use `go
install` to build the two toxiproxy binaries straight into `$CONDA_PREFIX/bin`.

**`setup_toxiproxy.sh` plan (per-node, run once per node/env):**

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Require an active conda env (so binaries land somewhere durable + on PATH).
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "activate the flame conda env first (conda activate my_flame_env)" >&2
  exit 1
fi

# 2. Ensure go is present; install via conda if missing (no sudo needed).
if ! command -v go >/dev/null 2>&1; then
  echo "installing go via conda-forge..."
  conda install -y -c conda-forge go
fi

# 3. Build toxiproxy-server / toxiproxy-cli into the conda env's bin/.
#    GOBIN controls go install's output dir; pin a version tag instead of
#    @latest once we've picked one, for reproducibility across nodes.
export GOBIN="$CONDA_PREFIX/bin"
TOXIPROXY_VERSION="v2.9.0"   # confirm latest tag before first real run
go install "github.com/Shopify/toxiproxy/v2/cmd/server@${TOXIPROXY_VERSION}"
go install "github.com/Shopify/toxiproxy/v2/cmd/cli@${TOXIPROXY_VERSION}"
mv "$GOBIN/server" "$GOBIN/toxiproxy-server"
mv "$GOBIN/cli" "$GOBIN/toxiproxy-cli"

# 4. Sanity check.
toxiproxy-server --version
toxiproxy-cli --version
echo "toxiproxy installed into $GOBIN"
```

Notes for the pickup session:
- Confirm the current latest toxiproxy release tag before hardcoding it
  (was unverified as of this planning pass — `v2.9.0` above is a
  placeholder, check https://github.com/Shopify/toxiproxy/releases).
- This needs outbound internet on the target node (Go module proxy fetch) —
  same assumption `NETWORK_EMULATION_DESIGN.md` §3.2 already makes for the
  prebuilt-binary download path, so no new constraint.
- Must be re-run once per conda env (not once per repo checkout) since the
  binaries live in `$CONDA_PREFIX/bin`.

---

## 4. Data flow / state machine (`peer.py`)

One `peer.py` process, `--role server|client`. A test = one server process +
one client process, run concurrently, talking over three topics scoped by a
shared `--run-id`:

- `ctrl` topic — plain-text handshake messages (not protobuf — see §1).
- `data/down` topic — server→client chunked payload (mimics aggregator
  broadcasting weights to a trainer).
- `data/up` topic — client→server chunked payload (mimics a trainer
  uploading a model update to the aggregator).

Sequence (clock-skew-free by construction: every phase's start/end timestamp
is always taken by the *same* process, never compared across processes):

1. Both connect + subscribe (`ctrl`, plus their own inbound data topic).
   **Client connects through the Toxiproxy proxy** (`127.0.0.1:<proxy_port>`);
   **server connects directly to the real broker** — this mirrors
   `NETWORK_EMULATION_DESIGN.md` §1's decision that only the trainer side is
   namespaced/proxied, the aggregator+broker stay unshaped.
2. Client publishes `ctrl: HELLO`.
3. Server receives `HELLO` → publishes `ctrl: READY`.
4. Client receives `READY` → records `t0 = now()` → publishes
   `ctrl: START_DOWN`.
5. Server receives `START_DOWN` → sends `down_bytes` of random payload,
   chunked via `ChunkStore`, on `data/down`.
6. Client's receiver assembles chunks; on `eom` → records `t1 = now()` →
   computes `elapsed_down = t1 - t0`, `throughput_down_mbps` → publishes
   `ctrl: DOWN_ACK`.
7. Server receives `DOWN_ACK` → records `t0 = now()` → publishes
   `ctrl: START_UP`.
8. Client receives `START_UP` → sends `up_bytes` of random payload, chunked,
   on `data/up`.
9. Server's receiver assembles chunks; on `eom` → records `t1 = now()` →
   computes `elapsed_up`, `throughput_up_mbps` → publishes `ctrl: DONE`.
10. Client receives `DONE`. Both sides write their own half of the result to
    `--result-file` as one JSON line, disconnect, exit 0.
    - Client's file: `{role, run_id, elapsed_down_s, bytes_down, throughput_down_mbps}`
    - Server's file: `{role, run_id, elapsed_up_s, bytes_up, throughput_up_mbps}`
11. `run_sweep.py` reads both files after both processes exit and merges into
    one CSV row (§6).

Safety: both roles take `--timeout` (default e.g. 120s); if a phase doesn't
complete in time, exit non-zero without writing a result file, so the
orchestrator can mark that combo `FAILED` and move on rather than hanging
the whole sweep on one bad config.

Payload correctness check (cheap, worth keeping): after assembly, assert
`len(received) == expected_bytes` before computing throughput — catches
chunker bugs immediately instead of producing silently-wrong timing numbers.

---

## 5. `toxiproxy_ctl.py` plan

Thin wrapper over the Toxiproxy HTTP API (`127.0.0.1:8474` by default),
using stdlib `urllib.request` only (no new dependency) — mirrors the
`NetworkShaper` interface shape from `NETWORK_EMULATION_DESIGN.md` §2/§3.3
(`create_link` / `apply_profile` / `teardown_link`), scoped down to what the
PoC actually needs:

```python
@dataclass
class NetworkProfile:
    latency_ms: float
    jitter_ms: float
    bandwidth_mbps: float

class ToxiproxyClient:
    def __init__(self, api_host="127.0.0.1", api_port=8474): ...
    def create_proxy(self, name, listen, upstream) -> None: ...   # POST /proxies
    def delete_proxy(self, name) -> None: ...                     # DELETE /proxies/<name>
    def add_toxic(self, proxy, name, type_, stream, attributes, toxicity=1.0): ...  # POST /proxies/<name>/toxics
    def apply_profile(self, proxy, profile: NetworkProfile, stream: str) -> None:
        # stream is "upstream" (client->server, i.e. "up") or
        # "downstream" (server->client, i.e. "down")
        self.add_toxic(proxy, f"latency_{stream}", "latency", stream,
                        {"latency": profile.latency_ms, "jitter": profile.jitter_ms})
        # toxiproxy's bandwidth toxic "rate" attribute is KB/s, not kbps —
        # convert from Mbps: 1 Mbps = 1e6 bit/s = 125,000 B/s = 125 KB/s
        self.add_toxic(proxy, f"bandwidth_{stream}", "bandwidth", stream,
                        {"rate": profile.bandwidth_mbps * 125})
```

Double-check the exact toxiproxy HTTP API JSON shapes (`stream` field name,
whether it's `"upstream"`/`"downstream"` literal strings or something else)
against the live `toxiproxy-server` once it's installed on the target node —
the design doc's CLI-form commands (§3.2) used `-u`/`-d` flags, so confirm
the HTTP API's field name/values map the same way before trusting this
sketch verbatim.

Optional: a tiny `if __name__ == "__main__":` argparse CLI for manual poking
during bring-up (`create`/`toxic`/`delete` subcommands) — nice for debugging
Checkpoint-A-style by hand before trusting the orchestrator, not required
for `run_sweep.py` itself.

---

## 6. `profiles.yaml` + `run_sweep.py` plan

**`profiles.yaml`** — asymmetric-capable (down defaults mirrored to up if
`up:` omitted, since real last-mile links are usually asymmetric):

```yaml
profiles:
  - name: lan
    down: {latency_ms: 1, jitter_ms: 0, bandwidth_mbps: 1000}
  - name: wifi
    down: {latency_ms: 10, jitter_ms: 2, bandwidth_mbps: 100}
  - name: lte
    down: {latency_ms: 40, jitter_ms: 8, bandwidth_mbps: 30}
    up:   {latency_ms: 40, jitter_ms: 8, bandwidth_mbps: 8}
  - name: congested_3g
    down: {latency_ms: 150, jitter_ms: 30, bandwidth_mbps: 2}
    up:   {latency_ms: 150, jitter_ms: 30, bandwidth_mbps: 0.5}
  - name: satellite
    down: {latency_ms: 600, jitter_ms: 50, bandwidth_mbps: 5}

payload_sizes_bytes:
  - {label: control_msg,      bytes: 4096}
  - {label: model_update_1mb, bytes: 1048576}
  - {label: model_update_10mb, bytes: 10485760}
```

Start with this small matrix (5 profiles x 3 sizes = 15 combos) for the
first real run — `congested_3g`/`satellite` x 10 MB will legitimately take
~1-3 minutes each, which is fine but don't default to a much bigger matrix
without a `--profiles`/`--sizes` filter flag on `run_sweep.py` for smoke
testing.

**`run_sweep.py` plan:**

1. Best-effort broker liveness check at start (same check pattern as the
   example's README: `systemctl is-active mosquitto || pgrep mosquitto`),
   warn (don't hard-fail) if not detected.
2. Spawn `toxiproxy-server` once as a sidecar subprocess (`subprocess.Popen`
   + log capture), same lifecycle pattern as `AggregatorSpawner`
   (`flame/launch/aggregator_spawner.py`) that `NETWORK_EMULATION_DESIGN.md`
   §3.1 already points at: SIGTERM, short grace period, then SIGKILL on
   teardown. Wrap the whole sweep in `try/finally` so this always gets torn
   down, even on an aborted run.
3. For each `(profile, payload_size)` combo:
   a. Pick a proxy name + fixed local listen port (reused across iterations
      since proxies are torn down between iterations — no need for a port
      pool at this scale).
   b. `toxiproxy_ctl.create_proxy(...)`, then `apply_profile(..., "downstream")`
      and `apply_profile(..., "upstream")` using the resolved (possibly
      asymmetric) profile.
   c. Generate a fresh `run_id` (e.g. `f"{profile}_{size_label}_{int(time.time())}"`).
   d. `subprocess.Popen` the server `peer.py`, then the client `peer.py`
      (client pointed at the proxy port, server at the real broker).
   e. `wait()` both with `--timeout`; on timeout, kill both, record a
      `FAILED` row, continue to next combo (don't abort the whole sweep).
   f. On success, read both `--result-file` JSONs, merge, and also compute a
      *theoretical* expected time (`latency_s + bytes*8 / (bandwidth_mbps *
      1e6)`) per direction — this is the sanity-check column that lets a
      human eyeball "did the shaping actually take effect" without staring
      at raw toxiproxy config, mirroring the validation ask in
      `NETWORK_EMULATION_DESIGN.md`'s Checkpoint A ("measured RTT/throughput
      match configured values within noise").
   g. Append one row to a CSV (`results/netpoc_sweep_<timestamp>.csv`) with
      columns: `profile, payload_label, down_bytes, up_bytes, latency_ms,
      jitter_ms, bw_down_mbps, bw_up_mbps, elapsed_down_s,
      expected_down_s, throughput_down_mbps, elapsed_up_s, expected_up_s,
      throughput_up_mbps, status`.
   h. `toxiproxy_ctl.delete_proxy(...)` — teardown before the next combo.
4. After all combos: kill the `toxiproxy-server` sidecar, print a summary
   table (measured vs. expected per row) to stdout, print the CSV path.

CLI flags: `--broker-host`, `--broker-port` (defaults `127.0.0.1`/`1883`),
`--profiles-file` (default `profiles.yaml`), `--profiles`/`--sizes` (optional
name filters for a quick smoke subset), `--out-dir` (default `results/`).

---

## 7. Pickup checklist — build in this order, validate after each step

Don't write everything then test once — same "validate after each group"
discipline as `NETWORK_EMULATION_DESIGN.md` §5.

1. **`setup_toxiproxy.sh`** — write + run on the target node. Validate:
   `toxiproxy-server --version` and `toxiproxy-cli --version` both work,
   confirm the HTTP API responds (`curl 127.0.0.1:8474/proxies` while the
   server is running).
2. **`toxiproxy_ctl.py`** — write, manually create/shape/delete one proxy by
   hand (small script or the optional CLI from §5) in front of some dummy
   TCP echo server (not even MQTT yet). Validate: measured latency/bandwidth
   through the proxy roughly match the configured profile (`nc`/`curl`
   timing, or a tiny throwaway Python timing script) — this is
   `NETWORK_EMULATION_DESIGN.md`'s Checkpoint A, just via Toxiproxy instead
   of `tc`/netns per §3.3's note that the checkpoint structure carries over.
3. **`chunked_mqtt.py`** — write the send/receive helpers. Validate:
   two local `python -c` snippets on the *unshaped* real broker (no
   toxiproxy in the loop yet) can round-trip a payload and the received
   bytes match exactly — isolates chunker/proto correctness from the
   network-shaping mechanism.
4. **`peer.py`** — write the full state machine from §4. Validate: one
   client + one server, unshaped broker, complete a full down+up cycle and
   produce correct JSON result files.
5. **`profiles.yaml` + `run_sweep.py`** — write the orchestrator. Validate:
   run the 15-combo matrix end-to-end, confirm the CSV has 15 rows, confirm
   measured times move in the expected direction as latency/bandwidth
   worsen across profiles, confirm `toxiproxy-cli list`/`ps aux | grep
   toxiproxy` show nothing left running after the sweep finishes (clean
   teardown, no stray proxies — same "no residue" bar as
   `NETWORK_EMULATION_DESIGN.md` Checkpoint D).
6. **`README.md`** — write last, once the above is proven, so it documents
   what actually works rather than what was planned.

## 8. Explicitly deferred (Phase 2, not part of this PoC)

Per the user's "go step by step" — do not build this until §7 is fully
working and validated:

- **Dynamic mid-run profile changes**: updating a live proxy's toxics
  (`toxic update`, not `toxic add`) *while* a transfer is in flight, to
  emulate a link degrading/recovering mid-round rather than being static
  for the whole test. This is `NETWORK_EMULATION_DESIGN.md` §5 Checkpoint
  E's item 13 (dynamic profile updates) — same deferral, same reasoning
  (needs its own control-channel/timing design once the static case is
  solid).

## 9. Open items to resolve when resuming

- Confirm target node has (or can get) both MQTT (mosquitto) and outbound
  internet access for the `go install` step — the design doc already
  assumes internet for the prebuilt-binary path, so this isn't a new
  requirement, just needs a two-line confirmation before starting §7.1.
- Pin an actual toxiproxy release tag (placeholder `v2.9.0` in §3 — verify
  against https://github.com/Shopify/toxiproxy/releases before running
  `setup_toxiproxy.sh` for real).
- Confirm the Toxiproxy HTTP API's toxic `stream` field values
  (`"upstream"`/`"downstream"`) directly against a running instance — §5
  flagged this as unverified against live docs, only inferred from the CLI
  flag semantics in `NETWORK_EMULATION_DESIGN.md` §3.2.
