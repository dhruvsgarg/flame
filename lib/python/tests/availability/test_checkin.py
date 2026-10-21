"""F42-T1: Bonawitz check-in with pace steering (flame.availability.checkin)."""
import math

from flame.availability.checkin import CheckinLedger

# client "a": AVL_TRAIN [0, 100), UN_AVL [100, 300), AVL_TRAIN [300, inf); "b": always AVL_TRAIN
ON = {"a": [(0, 100), (300, math.inf)], "b": [(0, math.inf)]}


def is_train(e, t):
    return any(a <= t < b for a, b in ON[e])


def train_at(e, t):
    return t if is_train(e, t) else min((a for a, _ in ON[e] if a > t), default=math.inf)


def test_sync_window_pools_then_rejects_into_next_window():
    led = CheckinLedger(sync=True, initial_gap_s=200.0)
    close = led.open_window(0.0)
    assert close == 20.0  # W = 10% of the round-time estimate
    led.advance(close, ["a", "b"], train_at, is_train)
    assert set(led.pool) == {"a", "b"} and led.stats["checkins"] == 2
    told = dict(led.after_select(close, ["b"], slots_left=False))
    assert set(told) == {"a"} and 200.0 <= told["a"] <= 220.0  # inside the next window [0 + gap, + W]
    assert led.pool == {} and led.window is None


def test_unavailable_at_reconnect_is_a_missed_opportunity():
    led = CheckinLedger(sync=True, initial_gap_s=200.0)
    led.open_window(0.0)
    led.advance(20.0, ["a"], train_at, is_train)
    (_, r), = led.after_select(20.0, [], slots_left=False)
    assert 200 <= r <= 220 and not is_train("a", r)
    assert led.next_checkin(150.0, ["a"], train_at, is_train) == r  # the reconnect is a wake-up
    led.advance(299.0, ["a"], train_at, is_train)
    assert led.stats["missed"] == 1 and "a" not in led.pool
    assert led.next_checkin(299.0, ["a"], train_at, is_train) == 300  # back at its next flip
    led.advance(310.0, ["a"], train_at, is_train)  # checks in at 300, no window open: rejected again
    assert led.stats["rejects"] == 2 and led.hold["a"] > 300


def test_checkin_between_windows_is_rejected_and_comes_back_in_the_next():
    led = CheckinLedger(sync=True, initial_gap_s=100.0)
    led.open_window(0.0)
    led.after_select(10.0, [], slots_left=False)
    led.release("b", 50.0)  # b's task ends mid-round: checks in at 50, rejected
    led.advance(60.0, ["b"], train_at, is_train)
    assert led.stats["rejects"] == 1 and 100.0 <= led.hold["b"] <= 110.0
    close = led.open_window(100.0)
    assert led.gap == 100.0  # EWMA of round starts: 0.75 * 100 + 0.25 * 100
    led.advance(close, ["b"], train_at, is_train)
    assert "b" in led.pool


def test_departure_is_never_learned_from_the_pool():
    led = CheckinLedger(sync=True, initial_gap_s=1000.0)
    led.release("a", 50.0)
    led.open_window(50.0)  # window [50, 150]: a checked in at 50 and left at 100 -- still pooled
    led.advance(150.0, ["a"], train_at, is_train)
    assert "a" in led.pool


def test_async_pools_only_while_slots_are_free():
    led = CheckinLedger(sync=False, initial_gap_s=10.0)
    led.advance(0.0, ["a", "b"], train_at, is_train)  # every slot is free at the start
    assert set(led.pool) == {"a", "b"}
    assert led.after_select(0.0, ["a", "b"], slots_left=False) == []
    led.release("a", 30.0)
    led.release("b", 30.0)  # back at 30 while slots are full: both rejected
    led.advance(30.0, ["a", "b"], train_at, is_train)
    assert led.stats["rejects"] == 2 and 35.0 <= led.hold["a"] <= 45.0 and 40.0 <= led.hold["b"] <= 60.0  # m x gap x U
    led.after_select(31.0, [], slots_left=True)  # a slot is free from 31
    led.advance(70.0, ["a", "b"], train_at, is_train)
    assert set(led.pool) == {"a", "b"}
    told = led.after_select(70.0, ["a"], slots_left=False)  # slots full again: the rest is told to reconnect
    assert [e for e, _ in told] == ["b"] and led.open_since is None


def test_real_checkin_message_path():
    led = CheckinLedger(sync=False, initial_gap_s=10.0)
    led.after_select(0.0, [], slots_left=False)
    assert led.on_checkin("a", 0.0) is not None  # slots full: rejected with a reconnect time
    led.after_select(1.0, [], slots_left=True)
    assert led.on_checkin("a", 2.0) is None and "a" in led.pool


def _population(n=75, hours=8.0, seed=1):
    """Random on/off clients (mean on 2 h, off 1 h): ON[e] intervals, as the module's is_train / train_at read them."""
    import random
    rng, out = random.Random(seed), {}
    for i in range(n):
        t, on, iv = 0.0, rng.random() < 0.67, []
        while t < hours * 3600:
            d = rng.expovariate(1 / (7200 if on else 3600))
            if on:
                iv.append((t, t + d))
            t, on = t + d, not on
        out[f"c{i}"] = iv
    return out


def _drive(sync, hours=8.0, task_s=120.0, k=10, c=15):
    """A minimal engine: sync rounds pick k from each closed window and last task_s; async keeps c slots busy for task_s."""
    ON.clear()
    ON.update(_population(hours=hours))
    led = CheckinLedger(sync=sync, initial_gap_s=300.0 if sync else 300.0 / c)
    busy, now, picks = {}, 0.0, 0
    while now < hours * 3600:
        for e in [e for e, t in busy.items() if t <= now]:
            del busy[e]
            led.release(e, now)
        idle = [e for e in ON if e not in busy]
        if sync:
            if busy:
                now = min(busy.values())
                continue
            close = led.open_window(now, round_id=picks)
            led.advance(close, idle, train_at, is_train)
            now = close
            got = sorted(led.pool)[:k]
            if len(got) < k:  # starved window keeps collecting
                nxt = led.next_checkin(now, [e for e in idle if e not in led.pool], train_at, is_train)
                now = nxt if nxt is not None else now + 60
                continue
            led.after_select(now, got, slots_left=False)
        else:
            led.advance(now, idle, train_at, is_train)
            got = sorted(led.pool)[: c - len(busy)]
            led.after_select(now, got, slots_left=len(busy) + len(got) < c)
            nxt = [t for t in busy.values()] + [now + task_s]
            if len(busy) + len(got) < c:
                w = led.next_checkin(now, [e for e in idle if e not in led.pool and e not in got], train_at, is_train)
                nxt += [w] if w is not None else []
        for e in got:
            busy[e] = now + task_s
        picks += len(got)
        if not sync:
            now = min(nxt)
    return led, picks


def test_sync_pace_steering_does_not_churn():
    led, picks = _drive(sync=True)
    rounds = picks / 10
    assert rounds >= 20
    assert led.stats["rejects"] / (75 * rounds) < 1.0  # at most one return per idle client per round, never a poll loop
    assert 1.0 < led.gap / 120.0 < 10  # the round estimate tracks round starts, not top-up windows


def test_async_pace_steering_matches_the_dispatch_rate():
    led, picks = _drive(sync=False)
    assert picks >= 0.5 * 15 * 8 * 3600 / 120  # slots stay mostly busy
    assert led.stats["rejects"] / max(1, picks) < 5  # held clients come back at about the dispatch rate
