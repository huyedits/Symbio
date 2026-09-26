"""Where the cat goes on its own: down to the floor, and for a stroll now and then.

Pure Python over plain numbers, so it can be stepped in a test: the window
supplies where the pet sits on screen and where the floor and the screen's
edges are, and gets back where to put it. The floor is the bottom of the
screen's visible frame, the top of the Dock.

It only strolls when the cat has nothing better to do: not asleep, not
thinking, not training, not being judged, not while someone is holding it or
has the pointer on it.
"""

from __future__ import annotations

import math
import random

from symbio_pet.cat import WALK_SPEED, Cat

FALL = -1400.0      # pt/s², screen space
HARD_LANDING = -250.0


class Roamer:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.enabled = True
        self.vy = 0.0
        self.target: float | None = None
        self.next_stroll = self.rng.uniform(6.0, 15.0)   # in cat time
        self.falling = False
        self.arrived = False     # true for the one step a stroll ends on

    def _rest(self, cat: Cat, low: float, high: float) -> None:
        self.target = None
        self.next_stroll = cat.t + self.rng.uniform(low, high)

    def step(self, cat: Cat, dt: float, x: float, y: float, floor: float,
             left: float, right: float) -> tuple[float, float]:
        """One frame. (x, y) is the window's origin; returns where it goes."""
        self.arrived = False
        if cat.held or not self.enabled:
            # Wherever it is put, it stays.
            self.vy = 0.0
            self.falling = False
            self.target = None
            cat.walking = False
            return x, y

        if y > floor + 0.5 or self.vy > 0:
            self.falling = True
            cat.walking = False
            self.vy += FALL * dt
            y += self.vy * dt
            if y <= floor:
                y = floor
                if self.vy < HARD_LANDING:
                    cat.land()
                self.vy = 0.0
                self.falling = False
            return x, y
        self.falling = False
        y = floor     # also lifts it back up when the Dock grows under it
        x = min(max(x, left), right)

        if cat.mood != "idle" or cat.hovered:
            if self.target is not None:
                self._rest(cat, 8.0, 20.0)
            cat.walking = False
            return x, y

        if self.target is None:
            if cat.t < self.next_stroll:
                cat.walking = False
                return x, y
            span = self.rng.uniform(120.0, 420.0) * self.rng.choice((-1.0, 1.0))
            goal = min(max(x + span, left), right)
            if abs(goal - x) < 60.0:
                goal = min(max(x - span, left), right)
            if abs(goal - x) < 60.0:
                self._rest(cat, 10.0, 20.0)
                cat.walking = False
                return x, y
            self.target = goal

        distance = self.target - x
        cat.facing_goal = 1.0 if distance > 0 else -1.0
        cat.walking = True
        # Up on its feet and turned round first; the stride ramps in.
        if cat.pose != "stand" or cat.pose_t < 1.0 or abs(cat.facing - cat.facing_goal) > 0.2:
            return x, y
        move = WALK_SPEED * dt * cat.walk_w
        if abs(distance) <= move:
            x = self.target
            self.arrived = True
            cat.walking = False
            self._rest(cat, 20.0, 60.0)
        else:
            x += math.copysign(move, distance)
        return x, y
