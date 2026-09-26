"""A scripted day in the life, for watching the pet without training anything.

`symbio-pet --demo` plays it on a loop; `--snapshot DIR` renders its key
frames. Every state the live feed can produce appears once: asleep, waking,
idle, thinking, a headmaster run the golden gate keeps, a skill run it rolls
back, and a resumed run that dies on a garbage loss.
"""

from __future__ import annotations

import math
import random
import time

from symbio_pet.feed import Snapshot, status_line

LOOP_S = 100.0

# (name, role, start, seconds training, steps, prior steps, loss from, loss to)
RUNS = (
    ("keep", None, 19.0, 24.0, 240, 0, 2.6, 0.34),
    ("spit", "sql_helper", 56.0, 14.0, 120, 0, 1.9, 0.95),
    ("fail", None, 82.0, 6.0, 60, 240, 0.5, 0.42),
)
JUDGE_S = {"keep": 5.0, "spit": 4.0}


def _curves(name: str, steps: int, total: int, start: float, end: float):
    """Loss every 10 steps and validation every 60, falling toward `end`."""
    noise = random.Random(name)
    train, val = [], []
    for step in range(10, steps + 1, 10):
        progress = step / total
        loss = end + (start - end) * math.exp(-4.0 * progress)
        train.append([step, round(loss * (1 + noise.uniform(-0.06, 0.06)), 4)])
        if step % 60 == 0:
            val.append([step, round(loss * (1.04 + noise.uniform(0.0, 0.05)), 4)])
    return train, [[1, round(start * 1.02, 4)]] + val


class DemoFeed:
    interval = 0.25

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.fired: set[tuple[int, str]] = set()

    def poll(self) -> Snapshot:
        loop, t = divmod(self.clock() - self.started, LOOP_S)
        snap = self.at(t, int(loop))
        snap.status = status_line(snap)
        return snap

    def _once(self, loop: int, name: str) -> bool:
        if (loop, name) in self.fired:
            return False
        self.fired.add((loop, name))
        return True

    def at(self, t: float, loop: int = 0) -> Snapshot:
        s = Snapshot()
        s.presence = "asleep" if t < 5 else "waking" if t < 8 else "awake"
        s.busy = 14 <= t < 19
        steps_on_disk = 240 if (loop or t >= 48) else 0
        s.has_adapter, s.adapter_iters = steps_on_disk > 0, steps_on_disk

        for name, role, start, seconds, steps, prior, high, low in RUNS:
            end = start + seconds
            judged = end + JUDGE_S.get(name, 0.0)
            if not start <= t < judged + 8:
                continue
            s.run, s.skill, s.iters, s.prior_iters = f"demo-{loop}-{name}", role, steps, prior
            done = min(steps, int((t - start) / seconds * steps) // 10 * 10)
            s.iter = done
            s.train, s.val = _curves(name, done, steps, high, low)
            if t < end:
                s.activity = "training"
            elif name == "fail":
                if t < end + 8:
                    s.activity = "failed"
                    s.reason = "implausible validation loss nan at iter 60"
            elif t < judged:
                s.activity = "judging"
                s.busy = True           # the golden set runs on the model
                s.total_iters = prior + steps
            else:
                s.total_iters = prior + steps
                verdict = "kept" if name == "keep" else "rolled_back"
                if t < judged + 3 and self._once(loop, name):
                    s.verdict = verdict
                    s.verdict_skill = role
            break
        return s
