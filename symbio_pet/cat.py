"""The cat: pose, face, collar charm and everything in the air, frame by frame.

Pure Python, no AppKit, so it can be stepped in a test and drawn by anything;
view.py turns it into paths. The rig lives in the cat's own space — x toward
the way it faces, y up from the ground under it — and `world` maps a rig
point into the pet window (origin bottom left, y up).

It is drawn as a tilcayo (Leopardus tilcayo), the newest cat species: a tiny
spotted cat from Bolivia's Yungas cloud forests, described in Current Biology
on 17 September 2026 — muted light-brown coat, extra-large irregular dark
rosettes, a small scrunched-up face, short round ears and long whiskers.

What each part means is the point of the thing:

* the cat is the frozen base model: the same cat before and after any run;
* the charm on its collar is the LoRA adapter, a small thing hung on the cat
  rather than a change to it. A fresh run starts it from nothing (LoRA's B
  matrices start at zero, so a new adapter is an exact no-op) and it grows
  with every step trained into it;
* each fish treat it catches is a couple of training steps, thrown at the
  pace the trainer reports them;
* the tail is the loss: lashing while the loss is high against where the run
  started, settling as the run converges;
* at the golden gate the charm is kept (it locks on with a sparkle) or rolled
  back (the cat swats it off the collar).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from symbio_pet.feed import Snapshot, run_label, run_steps

WIDTH, HEIGHT = 240.0, 250.0
CX = WIDTH / 2
GROUND = 24.0          # the window row the cat stands on
GRAVITY = -900.0       # pt/s², for everything thrown
CHARM_MAX = 8.0
SWAT_AT = 0.35         # seconds into a rollback when the charm leaves
POSE_S = 0.4           # seconds to move from one pose to the next
WALK_SPEED = 42.0      # pt/s along the floor
WALK_HZ = 1.8          # stride cycles per second
PANEL = (CX - 72.0, 192.0, 144.0, 50.0)   # x, y, w, h of the loss sparkline

FAR_HIND, FAR_FRONT, NEAR_HIND, NEAR_FRONT = range(4)

# Each pose facing right, relative to the ground point under the cat.
# body/haunch/chest: (x, y, rx, ry)   head: (x, y, rx, ry, tilt degrees)
# legs, in drawing order far hind, far front, near hind, near front:
#   (hip x, hip y, foot x, foot y, width, opacity, drawn in front of the body)
# tail: base, control 1, control 2, tip   collar: (x, y, w, h, tilt)
POSES: dict[str, dict] = {
    "sit": {
        "body": (-4.0, 31.0, 23.0, 28.0),
        "haunch": (-15.0, 16.0, 21.0, 16.0),
        "chest": (8.0, 35.0, 10.0, 15.0),
        "head": (6.0, 70.0, 25.0, 21.0, 0.0),
        "legs": ((-20.0, 10.0, -12.0, 3.0, 8.0, 0.0, 0.0),
                 (4.0, 30.0, 4.0, 3.0, 8.0, 1.0, 1.0),
                 (-10.0, 10.0, -1.0, 3.0, 9.0, 1.0, 1.0),
                 (13.0, 30.0, 14.0, 3.0, 9.0, 1.0, 1.0)),
        "tail": (-34.0, 8.0, -56.0, 12.0, -60.0, 44.0, -46.0, 56.0),
        "tail_front": 0.0,
        "collar": (5.0, 50.0, 30.0, 8.0, 0.0),
        "charm": (9.0, 43.0),
    },
    "stand": {
        "body": (-4.0, 30.0, 34.0, 16.0),
        "haunch": (-26.0, 28.0, 15.0, 15.0),
        "chest": (22.0, 28.0, 9.0, 10.0),
        "head": (34.0, 48.0, 22.0, 19.0, 0.0),
        "legs": ((-30.0, 20.0, -30.0, 3.0, 8.0, 1.0, 0.0),
                 (16.0, 20.0, 16.0, 3.0, 8.0, 1.0, 0.0),
                 (-24.0, 20.0, -24.0, 3.0, 9.0, 1.0, 1.0),
                 (22.0, 20.0, 22.0, 3.0, 9.0, 1.0, 1.0)),
        "tail": (-38.0, 34.0, -56.0, 40.0, -62.0, 62.0, -50.0, 72.0),
        "tail_front": 0.0,
        "collar": (26.0, 36.0, 24.0, 7.0, -20.0),
        "charm": (30.0, 29.0),
    },
    "loaf": {
        "body": (-2.0, 16.0, 38.0, 16.0),
        "haunch": (-24.0, 15.0, 16.0, 14.0),
        "chest": (22.0, 14.0, 8.0, 8.0),
        "head": (26.0, 30.0, 22.0, 18.0, -6.0),
        "legs": ((-28.0, 6.0, -20.0, 3.0, 7.0, 0.0, 0.0),
                 (18.0, 8.0, 26.0, 3.0, 7.0, 0.0, 0.0),
                 (-22.0, 6.0, -14.0, 3.0, 7.0, 0.0, 1.0),
                 (24.0, 7.0, 32.0, 3.0, 7.0, 1.0, 1.0)),
        "tail": (-38.0, 8.0, -52.0, 1.0, -12.0, 1.0, 16.0, 4.0),
        "tail_front": 1.0,
        "collar": (16.0, 18.0, 22.0, 6.0, -10.0),
        "charm": (20.0, 12.0),
    },
    "held": {
        "body": (0.0, 44.0, 19.0, 31.0),
        "haunch": (0.0, 18.0, 16.0, 12.0),
        "chest": (4.0, 50.0, 9.0, 15.0),
        "head": (2.0, 92.0, 24.0, 20.0, 0.0),
        "legs": ((-6.0, 16.0, -8.0, -6.0, 8.0, 1.0, 0.0),
                 (-6.0, 60.0, -8.0, 36.0, 7.5, 1.0, 0.0),
                 (8.0, 16.0, 10.0, -6.0, 9.0, 1.0, 1.0),
                 (8.0, 60.0, 10.0, 36.0, 8.5, 1.0, 1.0)),
        "tail": (-2.0, 12.0, -6.0, -4.0, 4.0, -10.0, 0.0, -20.0),
        "tail_front": 0.0,
        "collar": (2.0, 74.0, 26.0, 7.0, 0.0),
        "charm": (4.0, 67.0),
    },
}


def charm_radius(steps: int) -> float:
    """The charm's size for an adapter trained `steps` steps.

    Nothing at all for no adapter, then logarithmic, so a 1,000-step adapter
    still hangs neatly from the collar and a 20-step one is still visible.
    """
    if steps <= 0:
        return 0.0
    return min(CHARM_MAX, 2.5 + 1.0 * math.log2(1.0 + steps / 10.0))


def approach(value: float, target: float, rate: float, dt: float) -> float:
    """Exponential approach: closes ~63% of the gap every 1/rate seconds."""
    return target + (value - target) * math.exp(-rate * dt)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _lerp(a, b, t):
    if isinstance(a, (int, float)):
        return a + (b - a) * t
    return tuple(_lerp(x, y, t) for x, y in zip(a, b))


def blend(a: dict, b: dict, t: float) -> dict:
    return {key: _lerp(a[key], b[key], t) for key in a}


def _rotate(point, center, angle):
    dx, dy = point[0] - center[0], point[1] - center[1]
    c, s = math.cos(angle), math.sin(angle)
    return (center[0] + dx * c - dy * s, center[1] + dx * s + dy * c)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    ttl: float
    r: float
    kind: str          # crumb | sparkle | heart | dust | z
    life: float = 0.0
    gravity: float = 0.0

    @property
    def fade(self) -> float:
        return max(0.0, 1.0 - self.life / self.ttl)


@dataclass
class Treat:
    """A fish treat in flight: a couple of training steps."""
    x: float
    y: float
    vx: float
    vy: float
    flight: float      # seconds until it reaches the mouth it was thrown at
    age: float = 0.0


@dataclass
class Ejecta:
    """The charm, swatted off."""
    x: float
    y: float
    vx: float
    vy: float
    r: float
    cyan: bool
    age: float = 0.0
    bounces: int = 0

    @property
    def alpha(self) -> float:
        return max(0.0, min(1.0, (2.4 - self.age) / 0.6))


class Cat:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.t = 0.0
        self.snap = Snapshot()
        # Squash spring: q > 0 is wider and shorter, q < 0 taller and thinner.
        self.q = 0.0
        self.qv = 0.0
        self.hop_t: float | None = None
        self.hop_h = 12.0
        self.hop_y = 0.0
        self.pose = "loaf"
        self.pose_t = 1.0
        self._from = dict(POSES["loaf"])
        self.facing = 1.0
        self.facing_goal = 1.0
        self.walking = False
        self.walk_w = 0.0
        self.walk_phase = 0.0
        self.tail_phase = 0.0
        self.tail_amp = 0.04
        self.tail_hz = 0.2
        self.perk = 0.0
        self.droop = 0.0
        self.twitch_t: float | None = None
        self.next_twitch = 6.0
        self.sleepy = 1.0          # 0 awake .. 1 asleep; starts asleep, no flash
        self.thought = 0.0
        self.charm_r = 0.0
        self.charm_cyan = 0.0
        self.charm_pulse = 0.0
        self.flash = 0.0
        self.ring_age: float | None = None
        self.ring_from = 0.0
        self.treats: list[Treat] = []
        self.queue = 0
        self.release_every = 0.25
        self.release_clock = 0.0
        self.particles: list[Particle] = []
        self.ejecta: Ejecta | None = None
        self.anim: tuple[str, float] | None = None
        self.anim_cyan = False
        self._anim_step = 0
        self.run: str | None = None
        self.seen_iter = 0
        self.report_t = 0.0
        self.blink = 0.0
        self.blink_t: float | None = None
        self.next_blink = 2.0
        self.look = (0.0, 0.0)
        self.look_goal = (0.0, 0.0)
        self.next_look = 3.0
        self.pointer: tuple[float, float] | None = None
        self.hovered = False
        self.held = False
        self.happy_until = -1.0
        self.chew_until = -1.0
        self.groom = 0.0
        self.groom_until = -1.0
        self.next_groom = 25.0
        self.yawn_t: float | None = None
        self.next_yawn = 1.5
        self.z_clock = 0.0
        self.show_graph = False
        self._rig = self._build_rig()

    # ── what the cat is doing ────────────────────────────────────────

    @property
    def mood(self) -> str:
        if self.anim:
            return self.anim[0]                     # kept | rolled_back
        if self.held:
            return "held"
        s = self.snap
        if s.activity != "idle":
            return s.activity                       # training | judging | failed | fainted
        if s.presence == "asleep":
            return "asleep"
        if s.presence == "waking":
            return "waking"
        return "thinking" if s.busy else "idle"

    def update(self, dt: float, snap: Snapshot | None = None) -> None:
        dt = max(0.0, min(dt, 0.1))
        self.t += dt
        if snap is not None:
            self._take(snap)
        self._animate()
        self._settle(dt)
        self._move(dt)
        self._hop(dt)
        self._feed(dt)
        self._fly(dt)
        self._face(dt)
        self._ambient(dt)
        self._rig = self._build_rig()

    def _take(self, snap: Snapshot) -> None:
        if snap.run != self.run:
            # First sight of a run, possibly mid-way: nothing is owed for the
            # steps it took before the cat was watching.
            self.run = snap.run
            self.seen_iter = snap.iter
            self.report_t = self.t
        elif snap.activity == "training" and snap.iter > self.seen_iter:
            steps = snap.iter - self.seen_iter
            self.seen_iter = snap.iter
            since = max(0.1, self.t - self.report_t)
            self.report_t = self.t
            self.queue = min(12, self.queue + min(8, max(1, round(steps / 2))))
            # Spread over about as long as the trainer took to report them,
            # so the cat eats at the pace the run is going.
            self.release_every = min(3.0, max(0.12, since / self.queue))
        self.snap = snap
        if snap.verdict:
            self._start(snap.verdict, snap.verdict_skill is not None)

    # ── the golden gate ──────────────────────────────────────────────

    def _start(self, kind: str, cyan: bool) -> None:
        self.anim = (kind, self.t)
        self.anim_cyan = cyan
        self._anim_step = 0
        if kind == "kept":
            self.flash = 1.0
            self.ring_age = 0.0
            self.ring_from = max(4.0, self.charm_r)
            self.happy_until = self.t + 2.4
            x, y = self.charm_pos()
            for _ in range(7):
                angle = self.rng.uniform(0.0, 2 * math.pi)
                reach = self.rng.uniform(10.0, 26.0)
                self.particles.append(Particle(
                    x + math.cos(angle) * reach, y + math.sin(angle) * reach,
                    self.rng.uniform(-10, 10), self.rng.uniform(14, 40),
                    self.rng.uniform(0.9, 1.4), self.rng.uniform(2.5, 4.5), "sparkle"))
            hx, hy = self.head_top()
            for i in range(2):
                self.particles.append(Particle(
                    hx + (i - 0.5) * 22, hy, self.rng.uniform(-6, 6), 26.0,
                    1.6, 5.0, "heart"))

    def _animate(self) -> None:
        if not self.anim:
            return
        kind, started = self.anim
        age = self.t - started
        if kind == "kept":
            if age >= 0.5 and self._anim_step == 0:
                self._anim_step = 1
                self.hop(12.0)
            if age > 2.4:
                self.anim = None
        else:
            if age >= SWAT_AT and self._anim_step == 0:
                self._anim_step = 1
                self._knock_off()
            if age > 2.9:
                self.anim = None

    def _knock_off(self) -> None:
        x, y = self.charm_pos()
        self.ejecta = Ejecta(x, y, self.facing * 165.0, 230.0,
                             max(2.5, self.charm_r), self.charm_cyan > 0.5)
        self.charm_r = 0.0
        self.qv -= 0.9

    def _swat(self) -> tuple[float, float]:
        """How far the near front paw is raised toward the charm (0..1), and
        how far past it the swipe carries."""
        if not self.anim or self.anim[0] != "rolled_back":
            return 0.0, 0.0
        age = self.t - self.anim[1]
        raised = (smoothstep(age / 0.3) if age < 0.3 else 1.0 if age < 0.45
                  else 1.0 - smoothstep((age - 0.45) / 0.45))
        swipe = 1.0 if SWAT_AT - 0.05 <= age <= 0.5 else 0.0
        return raised, swipe

    # ── springs and smoothing ────────────────────────────────────────

    def _charm_target(self) -> float:
        s = self.snap
        kind = self.anim[0] if self.anim else None
        if kind == "rolled_back":
            if self._anim_step == 0:
                return self.charm_r
            # What hangs there again is whatever sits on disk, shown once the
            # swatted one is well clear.
            if self.t - self.anim[1] < 0.9:
                return 0.0
            return charm_radius(s.adapter_iters) if s.has_adapter else 0.0
        if s.activity != "idle" or kind == "kept":
            return charm_radius(run_steps(s))
        return charm_radius(s.adapter_iters) if s.has_adapter else 0.0

    def _settle(self, dt: float) -> None:
        mood = self.mood
        # A stiff, lightly damped spring: ~2 Hz, gone in about a second.
        self.qv += (-170.0 * self.q - 9.0 * self.qv) * dt
        self.q = max(-0.3, min(0.35, self.q + self.qv * dt))

        s = self.snap
        if mood == "training":
            ratio = s.train[-1][1] / max(1e-6, s.train[0][1]) if s.train else 1.0
            amp, hz = 0.12 + 0.5 * max(0.0, min(1.5, ratio)), 1.6
        elif self.walk_w > 0.5:
            amp, hz = 0.16, 1.1
        else:
            amp, hz = {"thinking": (0.08, 2.6), "judging": (0.05, 0.5),
                       "asleep": (0.03, 0.18), "failed": (0.02, 0.3),
                       "fainted": (0.0, 0.2), "held": (0.3, 1.2),
                       "waking": (0.1, 0.4), "kept": (0.25, 2.0),
                       "rolled_back": (0.15, 1.0)}.get(mood, (0.14, 0.35))
        self.tail_amp = approach(self.tail_amp, amp, 2.0, dt)
        self.tail_hz = approach(self.tail_hz, hz, 2.0, dt)
        self.tail_phase += 2 * math.pi * self.tail_hz * dt

        perk = (1.0 if mood in ("thinking", "held") else
                0.5 if mood == "training" or self.hovered else 0.0)
        droop = (1.0 if mood in ("failed", "fainted") else
                 0.35 * self.sleepy)
        self.perk = approach(self.perk, perk, 6.0, dt)
        self.droop = approach(self.droop, droop, 4.0, dt)
        self.sleepy = approach(self.sleepy, 1.0 if mood == "asleep" else 0.0, 1.2, dt)
        self.thought = approach(self.thought, 1.0 if mood == "thinking" else 0.0, 4.0, dt)
        self.flash = max(0.0, self.flash - dt / 1.4)
        self.charm_pulse = max(0.0, self.charm_pulse - dt * 2.5)
        self.charm_r = approach(self.charm_r, self._charm_target(), 2.2, dt)
        if self.anim and not (self.anim[0] == "rolled_back" and self._anim_step):
            cyan = self.anim_cyan
        else:
            # A swatted charm is gone: what hangs there again is the one on
            # disk, which the idle cat always shows as the headmaster's.
            cyan = bool(s.skill) and s.activity != "idle"
        self.charm_cyan = approach(self.charm_cyan, 1.0 if cyan else 0.0, 3.0, dt)
        if self.ring_age is not None:
            self.ring_age += dt
            if self.ring_age > 1.0:
                self.ring_age = None
        groom = 1.0 if self.t < self.groom_until and mood == "idle" and not self.walking else 0.0
        self.groom = approach(self.groom, groom, 7.0, dt)

    def _pose_target(self) -> str:
        mood = self.mood
        if mood == "held":
            return "held"
        if mood in ("asleep", "fainted"):
            return "loaf"
        if self.walking:
            return "stand"
        return "sit"

    def _blended(self) -> dict:
        return blend(self._from, POSES[self.pose], smoothstep(self.pose_t))

    def _move(self, dt: float) -> None:
        target = self._pose_target()
        if target != self.pose:
            self._from = self._blended()      # from wherever it is right now
            self.pose = target
            self.pose_t = 0.0
        self.pose_t = min(1.0, self.pose_t + dt / POSE_S)
        on_feet = self.walking and self.pose == "stand" and self.pose_t >= 1.0
        self.walk_w = approach(self.walk_w, 1.0 if on_feet else 0.0, 6.0, dt)
        if self.walk_w > 0.01:
            self.walk_phase += 2 * math.pi * WALK_HZ * dt * self.walk_w
        # Turning round: the facing passes through zero, the classic flip.
        self.facing = approach(self.facing, self.facing_goal, 14.0, dt)

    def hop(self, height: float = 10.0) -> None:
        if self.hop_t is None and not self.held:
            self.hop_t = 0.0
            self.hop_h = height
            self.qv += 1.1       # crouch

    def _hop(self, dt: float) -> None:
        if self.hop_t is None:
            self.hop_y = 0.0
            return
        crouch, air = 0.1, 0.38
        before = self.hop_t
        self.hop_t += dt
        if before < crouch <= self.hop_t:
            self.qv -= 2.2       # take off, stretched
        if self.hop_t < crouch:
            self.hop_y = 0.0
        elif self.hop_t < crouch + air:
            p = (self.hop_t - crouch) / air
            self.hop_y = 4.0 * self.hop_h * p * (1.0 - p)
        else:
            self.hop_y = 0.0
            self.hop_t = None
            self.qv += 2.0       # land, squashed

    # ── eating ───────────────────────────────────────────────────────

    def _feed(self, dt: float) -> None:
        if self.snap.activity != "training":
            self.queue = 0
        elif self.queue > 0:
            self.release_clock += dt
            if self.release_clock >= self.release_every:
                self.release_clock = 0.0
                self.queue -= 1
                self._toss()
        for treat in list(self.treats):
            treat.age += dt
            treat.vy += GRAVITY * dt
            treat.x += treat.vx * dt
            treat.y += treat.vy * dt
            if treat.age >= treat.flight:
                self.treats.remove(treat)
                self._eat(treat)
            elif treat.y < -20:
                self.treats.remove(treat)

    def _toss(self) -> None:
        """A fish treat thrown in from one side, on an arc to the mouth."""
        side = self.rng.choice((-1.0, 1.0))
        x0 = CX + side * self.rng.uniform(100.0, 112.0)
        y0 = self.rng.uniform(110.0, 150.0)
        tx, ty = self.mouth_pos()
        flight = self.rng.uniform(0.55, 0.75)
        vx = (tx - x0) / flight
        vy = (ty - y0 - 0.5 * GRAVITY * flight * flight) / flight
        self.treats.append(Treat(x0, y0, vx, vy, flight))

    def _eat(self, treat: Treat) -> None:
        self.chew_until = self.t + 0.3
        self.charm_pulse = 1.0
        self.qv += 0.35
        for _ in range(3):
            self.particles.append(Particle(
                treat.x, treat.y, self.rng.uniform(-40, 40), self.rng.uniform(20, 70),
                0.4, self.rng.uniform(1.0, 1.6), "crumb", gravity=GRAVITY))

    # ── everything else that moves ───────────────────────────────────

    def _fly(self, dt: float) -> None:
        for p in list(self.particles):
            p.life += dt
            if p.life >= p.ttl:
                self.particles.remove(p)
                continue
            p.vy += p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            if p.kind == "dust":
                p.r += 10.0 * dt
            elif p.kind == "z":
                p.r += 2.5 * dt
        e = self.ejecta
        if e is not None:
            e.age += dt
            e.vy += GRAVITY * dt
            e.x += e.vx * dt
            e.y += e.vy * dt
            if e.y - e.r <= GROUND and e.vy < 0:
                e.y = GROUND + e.r
                if e.bounces < 2:
                    e.vy = -e.vy * 0.4
                    e.vx *= 0.6
                    e.bounces += 1
                    self.particles.append(Particle(e.x, GROUND + 2, 0, 6, 0.45, 3.0, "dust"))
                else:
                    e.vy = 0.0
                    e.vx *= 0.9
            if e.age > 2.4 or not -40 < e.x < WIDTH + 40:
                self.ejecta = None

    def _face(self, dt: float) -> None:
        mood = self.mood
        if self.blink_t is None and self.t >= self.next_blink and mood != "asleep":
            self.blink_t = 0.0
        if self.blink_t is not None:
            self.blink_t += dt
            half = 0.08
            self.blink = (self.blink_t / half if self.blink_t < half
                          else max(0.0, 2.0 - self.blink_t / half))
            if self.blink_t >= 2 * half:
                self.blink_t = None
                self.blink = 0.0
                self.next_blink = self.t + (self.rng.uniform(1.2, 2.4) if mood == "waking"
                                            else self.rng.uniform(2.5, 6.0))

        goal = (0.0, 0.0)
        age = self.t - self.anim[1] if self.anim else 0.0
        if mood == "thinking":
            goal = (0.5, 0.8)
        elif mood == "judging":
            goal = (0.15, -0.9)
        elif mood == "rolled_back":
            # Watch the paw, then look anywhere but at what just happened.
            goal = (0.25, -0.8) if age < SWAT_AT + 0.2 else (-0.7, 0.35)
        elif mood == "training" and self.treats:
            ex, ey = self.eye_pos()
            near = min(self.treats, key=lambda k: (k.x - ex) ** 2 + (k.y - ey) ** 2)
            goal = self._toward(near.x, near.y)
        elif self.walk_w > 0.5:
            goal = (0.8, 0.0)
        elif self.pointer is not None and mood in ("idle", "waking", "training", "held"):
            px, py = self.pointer
            ex, ey = self.eye_pos()
            goal = (self._toward(px, py) if math.hypot(px - ex, py - ey) < 320
                    else self._wander())
        elif mood == "idle":
            goal = self._wander()
        self.look = (approach(self.look[0], goal[0], 9.0, dt),
                     approach(self.look[1], goal[1], 9.0, dt))

    def _wander(self) -> tuple[float, float]:
        if self.t >= self.next_look:
            self.next_look = self.t + self.rng.uniform(3.0, 8.0)
            self.look_goal = (self.rng.uniform(-0.8, 0.8), self.rng.uniform(-0.4, 0.5))
        return self.look_goal

    def _toward(self, x: float, y: float) -> tuple[float, float]:
        """A window point as a gaze, in the cat's own space (x toward facing)."""
        ex, ey = self.eye_pos()
        dx, dy = (x - ex) * (1.0 if self.facing >= 0 else -1.0), y - ey
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return (0.0, 0.0)
        reach = min(1.0, dist / 110.0)
        return (dx / dist * reach, dy / dist * reach)

    def _ambient(self, dt: float) -> None:
        mood = self.mood
        if self.sleepy > 0.6:
            self.z_clock += dt
            if self.z_clock > 1.8:
                self.z_clock = 0.0
                hx, hy = self.head_top()
                self.particles.append(Particle(
                    hx - self.facing * 8, hy - 4, -self.facing * self.rng.uniform(4, 9),
                    self.rng.uniform(13, 17), 3.2, 8.0, "z"))
        if mood == "idle" and not self.walking:
            if self.t >= self.next_groom:
                self.next_groom = self.t + self.rng.uniform(25.0, 60.0)
                self.groom_until = self.t + 2.2
        elif self.t >= self.next_groom:
            self.next_groom = self.t + 15.0
        if mood == "waking":
            if self.yawn_t is None and self.t >= self.next_yawn:
                self.yawn_t = 0.0
            if self.yawn_t is not None:
                self.yawn_t += dt
                if self.yawn_t > 0.9:
                    self.yawn_t = None
                    self.next_yawn = self.t + self.rng.uniform(2.0, 3.5)
        else:
            self.yawn_t = None
        if self.twitch_t is not None:
            self.twitch_t += dt
            if self.twitch_t > 0.25:
                self.twitch_t = None
        elif mood in ("idle", "thinking") and self.t >= self.next_twitch:
            self.twitch_t = 0.0
            self.next_twitch = self.t + self.rng.uniform(5.0, 12.0)

    # ── being handled ────────────────────────────────────────────────

    def poke(self) -> None:
        self.happy_until = self.t + 0.9
        self.hop(9.0)

    def grab(self) -> None:
        self.held = True
        self.walking = False
        self.hop_t = None
        self.hop_y = 0.0

    def drop(self) -> None:
        self.held = False
        self.qv += 2.4

    def land(self) -> None:
        """Back on the floor after a fall."""
        self.qv += 3.0
        for side in (-1.0, 1.0):
            self.particles.append(Particle(CX + side * 26, GROUND + 2, side * 30, 8,
                                           0.5, 3.0, "dust"))

    # ── geometry ─────────────────────────────────────────────────────

    def squash(self) -> tuple[float, float]:
        return 1.0 + 0.5 * self.q, 1.0 - self.q

    def world(self, x: float, y: float) -> tuple[float, float]:
        """A point in the cat's own space, in window coordinates."""
        sx, sy = self.squash()
        return CX + self.facing * x * sx, GROUND + self.hop_y + y * sy

    def rig(self) -> dict:
        return self._rig

    def _build_rig(self) -> dict:
        p = self._blended()
        mood = self.mood
        t = self.t

        bx, by, brx, bry = p["body"]
        amp = 0.012 + 0.02 * self.sleepy
        period = 3.0 + 1.6 * self.sleepy
        breath = amp * math.sin(2 * math.pi * t / period)
        bry *= 1 + breath
        brx *= 1 + 0.4 * breath
        hx, hy, hrx, hry, tilt = p["head"]
        hy += bry * breath * 0.6
        legs = [list(leg) for leg in p["legs"]]

        w = self.walk_w
        if w > 0.01:
            phase = self.walk_phase
            bob = 1.3 * abs(math.sin(phase)) * w
            by += bob
            hy += 1.0 * abs(math.sin(phase + 0.5)) * w
            for i, offset in ((NEAR_FRONT, 0.0), (FAR_HIND, 0.0),
                              (FAR_FRONT, math.pi), (NEAR_HIND, math.pi)):
                s, c = math.sin(phase + offset), math.cos(phase + offset)
                legs[i][0] += 1.5 * s * w
                legs[i][1] += bob
                legs[i][2] += 6.0 * s * w
                legs[i][3] += 3.5 * max(0.0, c) * w
        if mood == "held":
            for i, leg in enumerate(legs):
                leg[2] += 2.0 * math.sin(t * 3.0 + i)
        if t < self.chew_until:
            hy -= 1.2

        tilt += {"judging": -7.0, "thinking": 6.0}.get(mood, 0.0)
        tilt -= 9.0 * self.groom
        if self.yawn_t is not None:
            tilt += 7.0 * math.sin(math.pi * min(1.0, self.yawn_t / 0.9))

        charm = p["charm"]
        mouth = (hx + 3.0, hy - 9.0)
        paw = legs[NEAR_FRONT]
        if self.groom > 0.01:
            paw[2] += (mouth[0] - 1.0 - paw[2]) * self.groom
            paw[3] += (mouth[1] - 5.0 - paw[3]) * self.groom
        raised, swipe = self._swat()
        if raised > 0.0:
            paw[2] += (charm[0] + 7.0 + 8.0 * swipe - paw[2]) * raised
            paw[3] += (charm[1] + 1.0 - paw[3]) * raised

        tb = p["tail"]
        theta = self.tail_amp * math.sin(self.tail_phase)
        base = (tb[0], tb[1])
        c1 = _rotate((tb[2], tb[3]), base, 0.4 * theta)
        c2 = _rotate((tb[4], tb[5]), base, 0.8 * theta)
        tip = _rotate((tb[6], tb[7]), base, theta)

        return {
            "body": (bx, by, brx, bry),
            "haunch": p["haunch"],
            "chest": p["chest"],
            "head": (hx, hy, hrx, hry, tilt),
            "legs": [tuple(leg) for leg in legs],
            "tail": (base[0], base[1], c1[0], c1[1], c2[0], c2[1], tip[0], tip[1]),
            "tail_front": p["tail_front"] > 0.5,
            "collar": p["collar"],
            "charm": charm,
            "ears": (self._ear(-1.0, hrx, hry), self._ear(1.0, hrx, hry)),
            "face": self._face_parts(self.world(*mouth)),
        }

    def _ear(self, side: float, hrx: float, hry: float):
        """One ear in head space: (base, tip, base). side -1 is the far ear."""
        center = math.radians(90.0 - side * 40.0)
        spread = math.radians(25.0)
        a1, a2 = center - spread, center + spread
        base1 = (hrx * 0.95 * math.cos(a1), hry * 0.95 * math.sin(a1))
        base2 = (hrx * 0.95 * math.cos(a2), hry * 0.95 * math.sin(a2))
        # Short and round: the tilcayo's, not a house cat's tall points.
        length = 12.5 + 2.5 * self.perk - 3.0 * self.droop
        lean = 70.0 + 6.0 * self.perk - 38.0 * self.droop
        if self.twitch_t is not None and side > 0:
            lean += 14.0 * math.sin(math.pi * self.twitch_t / 0.25)
        direction = math.radians(lean if side > 0 else 180.0 - lean)
        root = (hrx * 0.8 * math.cos(center), hry * 0.8 * math.sin(center))
        tip = (root[0] + length * math.cos(direction), root[1] + length * math.sin(direction))
        return (base1, tip, base2)

    def _face_parts(self, mouth_at: tuple[float, float]) -> dict:
        mood = self.mood
        age = self.t - self.anim[1] if self.anim else 0.0
        eyes = "open"
        if mood == "asleep" or (self.sleepy > 0.5 and mood != "held"):
            eyes = "closed"
        elif mood in ("failed", "fainted"):
            eyes = "x"
        elif mood == "rolled_back" and age < SWAT_AT + 0.1:
            eyes = "half"
        elif mood == "kept" or self.t < self.happy_until or self.groom > 0.5:
            eyes = "happy"
        elif self.yawn_t is not None:
            eyes = "squeeze"
        elif mood == "held":
            eyes = "wide"
        elif mood == "waking":
            eyes = "half"

        mouth = "w"
        if mood == "asleep" or self.sleepy > 0.5:
            mouth = "sleep"
        elif self.yawn_t is not None:
            mouth = "yawn"
        elif self.groom > 0.5:
            mouth = "tongue"
        elif mood == "held":
            mouth = "open"
        elif mood == "kept" or self.t < self.happy_until:
            mouth = "grin"
        elif mood == "rolled_back":
            mouth = "flat" if age < SWAT_AT else "w"
        elif mood in ("failed", "fainted"):
            mouth = "wavy"
        elif mood in ("judging", "thinking"):
            mouth = "flat"
        elif mood == "training":
            mx, my = mouth_at
            near = any(math.hypot(k.x - mx, k.y - my) < 45 for k in self.treats)
            mouth = "open" if near or self.t < self.chew_until else "w"
        return {"eyes": eyes, "mouth": mouth, "blink": self.blink,
                "look": self.look, "blush": self.hovered or eyes == "happy"}

    # ── positions in the window, for everything drawn outside the rig ─

    def mouth_pos(self) -> tuple[float, float]:
        hx, hy = self._rig["head"][:2]
        return self.world(hx + 3.0, hy - 9.0)

    def eye_pos(self) -> tuple[float, float]:
        hx, hy = self._rig["head"][:2]
        return self.world(hx + 3.0, hy + 1.5)

    def head_top(self) -> tuple[float, float]:
        hx, hy, _, hry, _ = self._rig["head"]
        return self.world(hx, hy + hry + 10.0)

    def charm_pos(self) -> tuple[float, float]:
        return self.world(*self._rig["charm"])

    def bounds(self) -> tuple[float, float]:
        """The cat's horizontal extent in the window, tail to nose."""
        xs = [self.world(x, 0)[0] for x in (-62.0, 58.0)]
        return min(xs), max(xs)

    def shadow(self) -> dict:
        bx, _, brx, _ = self._rig["body"]
        lift = min(1.0, self.hop_y / 20.0)
        x, _ = self.world(bx, 0.0)
        return {"x": x, "y": GROUND - 1.0, "w": (brx * 2.0 + 26.0) * (1 - 0.3 * lift),
                "h": 8.0 * (1 - 0.3 * lift), "alpha": 0.2 * (1 - 0.5 * lift)}

    def ring(self) -> dict | None:
        if self.ring_age is None:
            return None
        x, y = self.charm_pos()
        p = self.ring_age / 1.0
        return {"x": x, "y": y, "r": self.ring_from + 40.0 * p, "alpha": 0.8 * (1 - p)}

    def thought_bubble(self) -> dict | None:
        if self.thought < 0.03:
            return None
        hx, hy = self.head_top()
        side = 1.0 if self.facing >= 0 else -1.0
        return {"x": hx + side * 14.0, "y": hy - 2.0, "side": side,
                "alpha": self.thought, "t": self.t}

    def panel(self) -> dict | None:
        s = self.snap
        kind = self.anim[0] if self.anim else None
        shown = (s.activity in ("training", "judging", "failed", "fainted")
                 or kind is not None or (self.show_graph and bool(s.train)))
        if not shown:
            return None
        label = run_label(s.skill)
        if len(label) > 16:
            label = label[:15] + "…"
        if kind == "kept":
            title, detail = "kept", f"{run_steps(s)} steps"
        elif kind == "rolled_back":
            title, detail = "rolled back", "swatted off"
        elif s.activity == "judging":
            title, detail = "golden gate", "judging…"
        elif s.activity in ("failed", "fainted"):
            title = "run failed" if s.activity == "failed" else "trainer died"
            detail = label
        else:
            budget = f"/{s.iters}" if s.iters else ""
            title, detail = label, f"{s.iter}{budget}"
        return {"title": title, "detail": detail,
                "loss": s.train[-1][1] if s.train else None,
                "train": s.train, "val": s.val,
                "progress": (min(1.0, s.iter / s.iters) if s.iters else None),
                "cyan": self.charm_cyan > 0.5}
