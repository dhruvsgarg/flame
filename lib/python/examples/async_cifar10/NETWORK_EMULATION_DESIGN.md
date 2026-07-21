# Network Emulation — Design Doc

**Status:** This is a first pass to unblock a follow-up implementation session, not a
final spec — architecture and step list are both expected to change on review. Scope: a durable,
reusable capability for shaping the **network characteristics of each aggregator↔trainer link**
(latency, jitter, bandwidth — loss later) on a **single physical node**, configurable per
experiment/baseline, independent of which FLAME example is running.

**Starter target: async_cifar10.** Same rationale as `UNAVAILABILITY_DESIGN.md` (also a
cross-cutting substrate first built here, later inherited by fwdllm): async_cifar10 is simpler,
isolated from the active fwdllm parity work, and a clean place to prove the mechanism before
generalizing. fwdllm integration is an explicit later step, not part of this init doc's build.

---

## 1. Key architectural insight (read this first)

The obvious worry is: all trainers + the aggregator run as sibling processes on one host, talking
MQTT through a single local broker (`127.0.0.1:1883`) — every trainer's traffic shares the same
destination, so `tc` classification by IP/port looks fragile (ephemeral, unstable source ports).

**This worry goes away if each trainer gets its own network namespace + veth pair.** Once a
trainer's traffic must physically pass through *its own dedicated virtual interface* to reach the
rest of the host, `tc` needs zero classification logic — one `netem`/`HTB` qdisc per veth end,
full stop. This is true **regardless of backend (MQTT or P2P)** — the isolation happens at the
interface layer, below anything FLAME's channel code knows about. Consequence: **this capability
needs no changes to `channel_manager.py`, backend code, or any example's trainer/aggregator
logic.** It is purely a `flame/launch/` (process-lifecycle) concern, which is exactly what makes
it durable and reusable — every example that goes through the launcher gets it for free.

Each veth pair has two ends, so upload (trainer→aggregator) and download (aggregator→trainer) can
be shaped independently (netem on the namespace-side end vs. the host-side end) — no `ifb`
ingress-redirect trickery needed.

The aggregator and the MQTT broker stay in the root namespace for v1 (only trainers are
namespaced); the broker must listen on the bridge IP, not only `127.0.0.1`, so namespaced trainers
can reach it — a one-line Mosquitto config change, not a code change.

---

## 2. Layered abstraction

Four layers, each independently testable and swappable:

1. **`NetworkProfile`** — plain data (latency_ms, jitter_ms, bandwidth_mbps, [later: loss_pct]).
   No knowledge of enforcement mechanism. This is the unit that config files produce.
2. **Link resolver** — expands a baseline's `network:` config block (rules matched by
   `trainer_id`, with a fallback default) into a concrete `{trainer_id: NetworkProfile}` map,
   using the same static `trainer_registry.yaml` identity the launcher already reads at spawn
   time (`spawner.py:45-48`).
3. **`NetworkShaper` (abstract) → `TcNetnsShaper` (concrete)** — the enforcement interface:
   `create_link(trainer_id) / apply_profile(trainer_id, profile) / update_profile(...) /
   teardown_link(trainer_id)`. Mirrors the existing `BackendType` pattern (MQTT/P2P/LOCAL/SHM) —
   same idea, different axis (network quality, not transport). A later alternative backend
   (containers, cloud-native shaping, or a no-op for pure software sim parity) implements the same
   interface without touching callers.
4. **Launcher lifecycle hook** — `runner.py`'s per-trainer spawn loop calls the resolver + shaper
   *before* `spawner.py` execs the trainer process, so the namespace exists before the trainer's
   first packet.

## 3. Privilege model — DECIDED: Toxiproxy (userspace, no sudo)

`ip netns add`, `ip link add veth...`, and `tc qdisc/class add` all need `CAP_NET_ADMIN`. The
target host has **no sudo access**, and the usual no-root escape hatch (unprivileged user
namespaces via `unshare --user --net`) was tested and is a dead end there:
`kernel.apparmor_restrict_unprivileged_userns = 1`, and no `bwrap`/`podman`/rootless-docker or
`newuidmap`/`newgidmap` are installed to work around it. `setcap` is also out — applying a
capability to a binary is itself a one-time root action. So every netns/veth/`tc` option from the
original draft is unavailable on this host; the shaping mechanism must live entirely in userspace.

**Decision: [Toxiproxy](https://github.com/Shopify/toxiproxy)** — a small TCP proxy (Shopify, MIT
license) that sits between client and server and injects latency/jitter/bandwidth-limit/timeout on
the connection. It ships as a static Go binary with no dependencies and runs as a normal
unprivileged process — nothing to install as root, nothing to `setcap`.

**How it replaces netns/tc in this design:** each trainer gets its own Toxiproxy proxy instance
(one per trainer, same cardinality as the old veth-pair-per-trainer plan) listening on a local
port and forwarding to the real broker. The trainer's outbound connection is pointed at
`127.0.0.1:<proxy_port>` instead of the broker directly — a config value, not a code change, so
§1's "no `channel_manager.py`/backend changes" property still holds. Toxics can be attached
separately to the **upstream** (trainer→broker, "upload") and **downstream** (broker→trainer,
"download") direction of each proxy, which is the same independent-shaping property the veth
two-ends trick gave us in the original draft.

### 3.1 Runtime model — how it runs, and how our code talks to it

**On the node:** `toxiproxy-server` is *not* a systemd/root-registered service — it's a plain
process, alive for the duration of one experiment run, then killed. This is different from
mosquitto, which this repo always treats as externally pre-started (the README's `systemctl
is-active mosquitto || pgrep mosquitto` check, never spawned by our code). `toxiproxy-server`
instead should be **launcher-managed**: started right before trainers spawn, torn down with them.
There's an existing precedent to mirror exactly — `AggregatorSpawner`
(`flame/launch/aggregator_spawner.py`) already wraps a sidecar process with `subprocess.Popen` +
log capture + `terminate()` (SIGTERM, 2s grace, then `kill()`), and `runner.py`'s signal
handler/`_cleanup()` (`ExperimentRunner.__init__` around line 117, `_cleanup()` around line 763,
also called from the `finally` in `run_experiment` around line 429) already starts/stops the
aggregator that way on `SIGINT`/normal exit. A `ToxiproxyServerSpawner` is the same class, same
lifecycle hook, nothing new to invent.

**With our code — two separate touchpoints, only one of them new:**
- **Control plane (launcher only).** `runner.py`'s per-trainer spawn loop calls the
  `ToxiproxyShaper` (§3.2 below) to create/configure/teardown that trainer's proxy — shelling out
  to `toxiproxy-cli`, or hitting the HTTP API on `127.0.0.1:8474`. This is the *only* code in the
  repo that ever talks to toxiproxy.
- **Data plane (trainer/aggregator — unchanged).** Trainer processes never call toxiproxy and
  never see its control API. They just get handed a broker address that happens to be
  `127.0.0.1:<proxy_port>` instead of the real broker; the proxy forwards traffic transparently,
  injecting the configured delay/bandwidth cap in between. This is what preserves §1's
  "no `channel_manager.py` changes" property, just via a proxied address instead of a bridge IP.

So at runtime: one sidecar process per experiment (launcher-spawned/torn-down, not an OS service),
one proxy socket per trainer, and trainer code stays completely unaware any of it exists.

### 3.2 No-sudo install steps

1. Download the prebuilt binaries from the [Toxiproxy releases page](
   https://github.com/Shopify/toxiproxy/releases) — grab `toxiproxy-server-linux-amd64` and
   `toxiproxy-cli-linux-amd64` (or the matching arch). No package manager involved.
2. `chmod +x` both, drop them somewhere on `PATH` that doesn't need root — a repo-local
   `bin/` dir or `~/.local/bin` works fine.
3. Start the control process once per experiment run (it does not need root, and does not itself
   proxy anything until you tell it to): `toxiproxy-server &` — by default it listens for control
   commands on `127.0.0.1:8474`.
4. For each trainer, create its dedicated proxy via the CLI (or the HTTP API, same thing):
   ```
   toxiproxy-cli create trainer_<id> \
     --listen 127.0.0.1:<per-trainer-port> \
     --upstream <broker_host>:<broker_port>
   ```
5. Attach toxics to shape the link — this is the direct analog of `apply_profile(trainer_id,
   profile)` in §2's `NetworkShaper` interface:
   ```
   toxiproxy-cli toxic add trainer_<id> -t latency  -a latency=<latency_ms> -a jitter=<jitter_ms> -u
   toxiproxy-cli toxic add trainer_<id> -t bandwidth -a rate=<bandwidth_kbps> -u
   ```
   (`-u` = upstream/upload direction; drop it, or use `-d`, for downstream/download — add both
   commands once per direction to shape upload and download independently.)
6. Point the trainer's broker connection string at `127.0.0.1:<per-trainer-port>` instead of the
   real broker address (a launcher/config change, same spot that would have pointed it at the
   bridge IP in the netns design).
7. Teardown: `toxiproxy-cli delete trainer_<id>` per trainer, then kill the `toxiproxy-server`
   process — no namespaces or interfaces to clean up, so there's no `ip netns list` residue class
   of bug to worry about.

### 3.3 Consequence for the rest of this doc

§2's layer 3 becomes `ToxiproxyShaper` implementing the same `NetworkShaper` interface
(`create_link` = `toxiproxy-cli create`, `apply_profile` = `toxic add`/`toxic update`,
`teardown_link` = `toxiproxy-cli delete`) — callers in §2/§4 don't change. §5's Checkpoint A–D
steps that reference `ip netns`/veth/`tc` should be re-read as "create a Toxiproxy proxy" /
"add a toxic" instead; the checkpoint *structure* (prove mechanism by hand → wrap in code →
wire into one trainer → generalize to N trainers) still applies unchanged, only the concrete
commands in each step swap out. Revise §5's step text before starting implementation.

## 4. Config schema (sketch — lives in the per-experiment YAML, sibling to `trainer:`/
`aggregator:`/`execution:`, **not** in `baselines.yaml`, which is an algorithm/hyperparameter
catalog, not infra)

```yaml
network:
  enabled: true
  default: {latency_ms: 10, jitter_ms: 2, bandwidth_mbps: 100}
  links:
    - match: {trainer_id: [1, 2, 3]}
      latency_ms: 80
      bandwidth_mbps: 15
    - match: {trainer_id: "*"}        # falls back to `default` if omitted
```

---

## 5. Stepwise plan — grouped into checkpoints, validate after each group

Do not build all of §5 before testing anything — each group below ends with an explicit
validation action. Don't proceed to the next group until that validation passes.

### Checkpoint A — prove the OS mechanism, no FLAME code touched
1. By hand (shell), create two network namespaces + one veth pair connecting them.
2. Apply `netem delay` + an `HTB` bandwidth cap to one veth end.
3. Run `ping`/`iperf3` between the namespaces.
- **Validate**: measured RTT and throughput match the configured values (within noise). If this
  doesn't hold with plain shell commands, nothing built on top of it will either — stop here and
  fix the mechanics first.

### Checkpoint B — wrap the mechanism in code, still standalone
4. Write `flame/launch/netshape.py`: `NetworkProfile` dataclass + `TcNetnsShaper` class
   implementing create/apply/teardown, using list-form `subprocess.run(["ip", ...], check=True)`
   (matches existing shell-out style in `runner.py`/`execution_config_generator.py`).
5. Write a throwaway script that calls `TcNetnsShaper` directly against two *fake* namespaces (no
   trainer process involved yet).
6. Unit-test the resolver (`network:` YAML → `{trainer_id: NetworkProfile}` map) with plain dict
   fixtures — no namespaces, no subprocess.
- **Validate**: the standalone script reproduces Checkpoint A's ping/iperf numbers programmatically;
  resolver unit tests pass on synthetic configs (including the fallback-default and multi-rule
  cases).

### Checkpoint C — wire into async_cifar10, one trainer, static profile
7. Add the `network:` block parsing to `experiment_config.py`.
8. Hook `runner.py`'s per-trainer spawn loop: resolve this trainer's profile, call
   `TcNetnsShaper.create_link` + `apply_profile` *before* `spawner.py` execs it; point the
   trainer's outbound MQTT connection at the broker's bridge IP (Mosquitto config change per §1).
9. Run one real (tiny) async_cifar10 experiment: 1 trainer (namespaced) + 1 aggregator (root ns).
- **Validate**: the run completes correctly end-to-end (no functional regression), and the
  trainer's round-trip telemetry visibly reflects the configured added latency (compare against
  an otherwise-identical run with `network.enabled: false`).

### Checkpoint D — heterogeneous, multi-trainer
10. Extend to N trainers with *different* per-trainer profiles from the `links:` match rules.
11. Add `teardown_link` cleanup on experiment exit (and on crash — check `runner.py`'s existing
    cleanup/signal-handling path).
12. Add the sudoers-scoped helper invocation from §3 in place of ad hoc `sudo`.
- **Validate**: telemetry shows each trainer's round timing tracking its *own* configured profile
  (not the average / not another trainer's), and repeated runs leave no stray namespaces/veths
  behind (`ip netns list` clean after teardown).

### Checkpoint E — stretch goals (only after A–D are solid)
13. Dynamic mid-run profile updates (`tc class change`) triggered by an aggregator-issued control
    signal — needs its own control-channel design, deliberately deferred out of this init doc.
14. Generalize to fwdllm: since §1's insight means no backend/channel code changes, this should be
    almost entirely reusing `netshape.py` + the `runner.py` hook — but confirm, don't assume.
15. Aggregator-side shaping (simulating a remote/cloud aggregator uplink) — optional, symmetric
    across trainers, lower priority than trainer heterogeneity.

---

## 6. Open questions for the next session
- Confirm the privilege model (§3) with whoever operates these hosts before writing `netshape.py`.
- Does the bridge need its own subnet/DHCP-like allocation, or are static per-veth `/30`s simpler
  at the expected trainer counts (~10–100)?
- Cleanup-on-crash: what does `runner.py` already do on `SIGINT`/abnormal exit, and does namespace
  teardown need to hook the same path?
- Is `loss_pct` (packet loss) needed for v1, or purely a later addition to `NetworkProfile`?
