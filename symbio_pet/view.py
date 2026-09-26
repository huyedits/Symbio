"""The window and the drawing: AppKit, through PyObjC.

A borderless, transparent, floating panel with one view in it. The view is
redrawn, at up to 30 fps, from whatever cat.Cat says; the Cat is fed a
Snapshot by a polling thread, so a slow `ps` never stalls a frame, and the
Roamer moves the whole panel when the cat goes for a stroll.

A non-activating panel, not a window: clicking or dragging the pet must not
take focus away from whatever the user was typing into. Fully transparent
pixels pass clicks through to the windows underneath, which is the window
server's default for a non-opaque window, so only the cat itself is solid.
"""

from __future__ import annotations

import json
import math
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import objc
from AppKit import (
    NSAffineTransform,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSAttributedString,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSBitmapImageFileTypePNG,
    NSBitmapImageRep,
    NSColor,
    NSDeviceRGBColorSpace,
    NSEvent,
    NSFloatingWindowLevel,
    NSFont,
    NSFontAttributeName,
    NSFontWeightBold,
    NSFontWeightRegular,
    NSFontWeightSemibold,
    NSForegroundColorAttributeName,
    NSGradient,
    NSGraphicsContext,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSScreen,
    NSShadow,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSRunLoop, NSRunLoopCommonModes, NSTimer

from symbio_pet.cat import (
    CX, FAR_FRONT, FAR_HIND, GROUND, HEIGHT, PANEL, WIDTH, Cat,
)
from symbio_pet.roam import Roamer

# Frames per second by how much is moving. A frame costs 3-4 ms of CPU in a
# live window (measured; under 1 ms into an offscreen bitmap), and a pet sits
# on the desktop all day, so it only runs fast while something is in flight:
# ~6% of a core sitting at 15 fps is why sitting is 10 and sleeping 6.
FPS_MOVING, FPS_CALM, FPS_ASLEEP = 24.0, 10.0, 6.0

# sRGB, 0-1. A tilcayo: muted light-brown coat, large irregular dark rosettes.
# The descriptions give the coat, the rosettes, the scrunched face, the short
# round ears and the long whiskers; the pale underside, eye rings, cheek and
# forehead streaks, ringed tail and amber eyes are the tiger-cat pattern it
# belongs to, not details reported for this species.
FUR = (0.79, 0.66, 0.49)
FUR_FAR = (0.70, 0.58, 0.42)
ROSETTE = (0.19, 0.13, 0.09)
ROSETTE_FILL = (0.63, 0.50, 0.35)
CREAM = (0.96, 0.92, 0.84)
LINE = (0.24, 0.17, 0.12)
EAR_INNER = (0.95, 0.86, 0.80)
IRIS = (0.80, 0.62, 0.25)
NOSE = (0.72, 0.46, 0.42)
MOUTH = (0.40, 0.12, 0.14)
TONGUE = (0.98, 0.55, 0.58)
BLUSH = (1.0, 0.52, 0.58)
COLLAR = (0.13, 0.56, 0.50)
GOLD_IN, GOLD_OUT, GOLD_GLOW = (1.0, 0.98, 0.80), (1.0, 0.68, 0.16), (1.0, 0.82, 0.32)
CYAN_IN, CYAN_OUT, CYAN_GLOW = (0.90, 1.0, 1.0), (0.16, 0.76, 0.95), (0.40, 0.90, 1.0)
TREAT = (0.86, 0.62, 0.33)
INK = (0.10, 0.10, 0.12)
WHITE = (1.0, 1.0, 1.0)


# Rosettes in each part's own unit ellipse, so they move and stretch with it:
# (u, v, radius, stretch, turn in degrees). Fixed, not random per frame.
BODY_ROSETTES = ((-0.55, 0.45, 5.5, 1.25, 20.0), (-0.1, 0.62, 5.0, 1.1, -15.0),
                 (0.3, 0.42, 4.6, 1.3, 40.0), (-0.72, -0.05, 5.0, 1.2, 70.0),
                 (-0.28, 0.1, 5.8, 1.15, 5.0), (0.18, -0.2, 4.8, 1.25, -30.0),
                 (-0.48, -0.5, 4.4, 1.3, 50.0), (0.02, -0.58, 4.2, 1.2, 10.0))
HAUNCH_ROSETTES = ((-0.35, 0.35, 4.8, 1.2, 30.0), (0.3, 0.22, 4.2, 1.3, -20.0),
                   (-0.1, -0.4, 4.0, 1.25, 60.0))
# Broken rims: a rosette is dark arcs around a darker centre, not a ring.
ROSETTE_ARCS = ((10.0, 70.0), (100.0, 60.0), (185.0, 75.0), (280.0, 55.0))


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def bezier(p0, p1, p2, p3, t):
    """A point on a cubic Bezier, and the unit normal there."""
    u = 1.0 - t
    point = tuple(u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d
                  for a, b, c, d in zip(p0, p1, p2, p3))
    tangent = tuple(3 * u * u * (b - a) + 6 * u * t * (c - b) + 3 * t * t * (d - c)
                    for a, b, c, d in zip(p0, p1, p2, p3))
    length = math.hypot(*tangent) or 1.0
    return point, (-tangent[1] / length, tangent[0] / length)


def oval(cx, cy, w, h, degrees=0.0):
    path = NSBezierPath.bezierPathWithOvalInRect_(((-w / 2, -h / 2), (w, h)))
    transform = NSAffineTransform.transform()
    transform.translateXBy_yBy_(cx, cy)
    if degrees:
        transform.rotateByDegrees_(degrees)
    path.transformUsingAffineTransform_(transform)
    return path


def circle(cx, cy, r):
    return NSBezierPath.bezierPathWithOvalInRect_(((cx - r, cy - r), (2 * r, 2 * r)))


def line(*points):
    path = NSBezierPath.bezierPath()
    path.moveToPoint_(points[0])
    for point in points[1:]:
        path.lineToPoint_(point)
    return path


class Painter:
    """Turns a Cat into paths, in the current graphics context."""

    def __init__(self):
        self._colors: dict = {}
        self.title_font = NSFont.systemFontOfSize_weight_(10.5, NSFontWeightSemibold)
        self.small_font = NSFont.monospacedDigitSystemFontOfSize_weight_(10.0, NSFontWeightRegular)
        self.tiny_font = NSFont.monospacedDigitSystemFontOfSize_weight_(9.0, NSFontWeightRegular)
        self._z_fonts: dict = {}

    def z_font(self, size):
        size = round(size * 2) / 2
        font = self._z_fonts.get(size)
        if font is None:
            font = self._z_fonts[size] = NSFont.systemFontOfSize_weight_(size, NSFontWeightBold)
        return font

    def color(self, rgb, alpha=1.0):
        key = (round(rgb[0], 3), round(rgb[1], 3), round(rgb[2], 3),
               round(max(0.0, min(1.0, alpha)), 3))
        found = self._colors.get(key)
        if found is None:
            if len(self._colors) > 4096:
                self._colors.clear()
            found = self._colors[key] = NSColor.colorWithSRGBRed_green_blue_alpha_(*key)
        return found

    def fill(self, path, rgb, alpha=1.0):
        self.color(rgb, alpha).setFill()
        path.fill()

    def stroke(self, path, rgb, alpha=1.0, width=2.0):
        path.setLineWidth_(width)
        path.setLineCapStyle_(1)     # round
        path.setLineJoinStyle_(1)
        self.color(rgb, alpha).setStroke()
        path.stroke()

    def shape(self, path, rgb, alpha=1.0, width=1.8):
        """Filled in `rgb` and outlined, the way every part of the cat is."""
        self.fill(path, rgb, alpha)
        self.stroke(path, LINE, alpha, width)

    def text(self, string, x, y, font, rgb, alpha=1.0, right=False, shadow=False):
        """Draw `string` with its box's bottom left at (x, y), or ending at x
        when `right`. Returns the width drawn, so a caller can fit things
        beside it."""
        attributed = NSAttributedString.alloc().initWithString_attributes_(
            string, {NSFontAttributeName: font,
                     NSForegroundColorAttributeName: self.color(rgb, alpha)})
        if right:
            x -= attributed.size().width
        if shadow:
            NSGraphicsContext.saveGraphicsState()
            glow = NSShadow.alloc().init()
            glow.setShadowColor_(self.color((0, 0, 0), 0.55 * alpha))
            glow.setShadowBlurRadius_(2.5)
            glow.setShadowOffset_((0, -0.5))
            glow.set()
        attributed.drawAtPoint_((x, y))
        if shadow:
            NSGraphicsContext.restoreGraphicsState()
        return attributed.size().width

    # ── the whole frame ──────────────────────────────────────────────

    def draw(self, cat: Cat) -> None:
        rig = cat.rig()
        sh = cat.shadow()
        self.fill(oval(sh["x"], sh["y"], sh["w"], sh["h"]), (0, 0, 0), sh["alpha"])

        # The cat itself, in its own space: facing right, feet on y = 0.
        NSGraphicsContext.saveGraphicsState()
        sx, sy = cat.squash()
        facing = cat.facing if abs(cat.facing) > 0.05 else math.copysign(0.05, cat.facing or 1)
        place = NSAffineTransform.transform()
        place.translateXBy_yBy_(CX, GROUND + cat.hop_y)
        place.scaleXBy_yBy_(facing * sx, sy)
        place.concat()
        self._legs(rig, front=False)
        if not rig["tail_front"]:
            self._tail(rig)
        self._torso(rig)
        self._legs(rig, front=True)
        self._collar(rig, cat)
        self._head(rig)
        if rig["tail_front"]:
            self._tail(rig)
        NSGraphicsContext.restoreGraphicsState()

        ring = cat.ring()
        if ring:
            tint = mix(GOLD_GLOW, CYAN_GLOW, cat.charm_cyan)
            self.stroke(circle(ring["x"], ring["y"], ring["r"]), tint, ring["alpha"], 2.5)
        self._particles(cat)
        for treat in cat.treats:
            self._treat(treat)
        if cat.ejecta is not None:
            e = cat.ejecta
            self._charm(e.x, e.y, e.r, 1.0 if e.cyan else 0.0, e.alpha, 1.0)
        self._thought(cat)
        self._panel(cat)

    # ── the body ─────────────────────────────────────────────────────

    def _legs(self, rig, front):
        for i, (hx, hy, fx, fy, width, alpha, in_front) in enumerate(rig["legs"]):
            if (in_front > 0.5) != front or alpha < 0.05:
                continue
            far = i in (FAR_HIND, FAR_FRONT)
            # Outlined from a quarter of the way down: a cap drawn at the hip
            # would put a ring on the body where the leg joins it.
            start = (hx + (fx - hx) * 0.25, hy + (fy - hy) * 0.25)
            self.stroke(line(start, (fx, fy)), LINE, alpha, width + 3.4)
            self.stroke(line((hx, hy), (fx, fy)), FUR_FAR if far else FUR, alpha, width)
            if not far and alpha > 0.5:
                spots = NSBezierPath.bezierPath()
                for along, across in ((0.45, -0.22), (0.68, 0.2)):
                    spots.appendBezierPath_(circle(hx + (fx - hx) * along + across * width,
                                                   hy + (fy - hy) * along, 1.35))
                self.fill(spots, ROSETTE, 0.85 * alpha)

    def _tail(self, rig):
        bx, by, c1x, c1y, c2x, c2y, tx, ty = rig["tail"]
        points = ((bx, by), (c1x, c1y), (c2x, c2y), (tx, ty))
        path = NSBezierPath.bezierPath()
        path.moveToPoint_((bx, by))
        path.curveToPoint_controlPoint1_controlPoint2_((tx, ty), (c1x, c1y), (c2x, c2y))
        self.stroke(path, LINE, 1.0, 10.8)
        self.stroke(path, FUR, 1.0, 7.4)
        # Dark bands, then a dark tip.
        bands = NSBezierPath.bezierPath()
        for t in (0.42, 0.58, 0.74):
            (x, y), (nx, ny) = bezier(*points, t)
            bands.moveToPoint_((x - nx * 3.6, y - ny * 3.6))
            bands.lineToPoint_((x + nx * 3.6, y + ny * 3.6))
        self.stroke(bands, ROSETTE, 0.9, 2.6)
        self.stroke(line(*[bezier(*points, t)[0] for t in (0.88, 0.94, 1.0)]),
                    ROSETTE, 1.0, 7.2)

    def _rosettes(self, cx, cy, rx, ry, spots):
        # One fill and one stroke for all of them: in a live window each draw
        # call, not each pixel, is what a frame costs.
        centers = NSBezierPath.bezierPath()
        rims = NSBezierPath.bezierPath()
        for u, v, size, stretch, turn in spots:
            place = NSAffineTransform.transform()
            place.translateXBy_yBy_(cx + u * rx, cy + v * ry)
            place.rotateByDegrees_(turn)
            place.scaleXBy_yBy_(stretch, 1.0)
            center = circle(0.0, 0.0, size * 0.62)
            rim = NSBezierPath.bezierPath()
            for start, span in ROSETTE_ARCS:
                a = math.radians(start)
                rim.moveToPoint_((size * math.cos(a), size * math.sin(a)))
                rim.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_(
                    (0.0, 0.0), size, start, start + span)
            center.transformUsingAffineTransform_(place)
            rim.transformUsingAffineTransform_(place)
            centers.appendBezierPath_(center)
            rims.appendBezierPath_(rim)
        self.fill(centers, ROSETTE_FILL, 0.9)
        self.stroke(rims, ROSETTE, 0.92, 2.1)

    def _torso(self, rig):
        bx, by, brx, bry = rig["body"]
        body = oval(bx, by, 2 * brx, 2 * bry)
        self.fill(body, FUR)
        # The rosettes, kept inside the body's outline; the pale chest over them.
        NSGraphicsContext.saveGraphicsState()
        body.addClip()
        self._rosettes(bx, by, brx, bry, BODY_ROSETTES)
        cx, cy, crx, cry = rig["chest"]
        self.fill(oval(cx, cy, 2 * crx, 2 * cry), CREAM)
        NSGraphicsContext.restoreGraphicsState()
        self.stroke(body, LINE, 1.0, 1.8)
        hx, hy, hrx, hry = rig["haunch"]
        haunch = oval(hx, hy, 2 * hrx, 2 * hry)
        self.fill(haunch, FUR)
        NSGraphicsContext.saveGraphicsState()
        haunch.addClip()
        self._rosettes(hx, hy, hrx, hry, HAUNCH_ROSETTES)
        NSGraphicsContext.restoreGraphicsState()
        self.stroke(haunch, LINE, 1.0, 1.8)

    def _collar(self, rig, cat):
        x, y, w, h, tilt = rig["collar"]
        self.shape(oval(x, y, w, h, tilt), COLLAR, 1.0, 1.5)
        r = cat.charm_r * (1 + 0.12 * cat.charm_pulse + 0.15 * cat.flash)
        if r < 0.6:
            return
        cx, cy = rig["charm"]
        self.stroke(circle(cx, cy + r + 1.2, 1.6), LINE, 0.9, 1.0)   # the ring it hangs by
        bright = 1.0 + 0.9 * cat.flash + 0.35 * cat.charm_pulse
        mood = cat.mood
        if mood == "training":
            bright += 0.1 * math.sin(cat.t * 5.0)
        elif mood == "judging":
            bright += 0.3 * math.sin(cat.t * 2.4)
        alpha = 1.0
        if mood in ("failed", "fainted"):
            alpha = 0.45 + 0.35 * abs(math.sin(cat.t * 9.0) * math.sin(cat.t * 2.3))
        self._charm(cx, cy, r, cat.charm_cyan, alpha, bright)

    def _charm(self, x, y, r, cyan, alpha, bright):
        """The adapter: a glowing charm, gold for the headmaster, cyan for a skill."""
        tint = mix(GOLD_GLOW, CYAN_GLOW, cyan)
        glow = NSGradient.alloc().initWithColors_(
            [self.color(tint, min(0.8, 0.45 * bright) * alpha), self.color(tint, 0.0)])
        glow.drawFromCenter_radius_toCenter_radius_options_((x, y), r * 0.5, (x, y), r * 3.0, 0)
        inner = mix(mix(GOLD_IN, CYAN_IN, cyan), WHITE, max(0.0, bright - 1.0) * 0.5)
        outer = mix(GOLD_OUT, CYAN_OUT, cyan)
        orb = circle(x, y, r)
        gradient = NSGradient.alloc().initWithStartingColor_endingColor_(
            self.color(inner, 0.97 * alpha), self.color(outer, 0.95 * alpha))
        gradient.drawInBezierPath_relativeCenterPosition_(orb, (-0.3, 0.35))
        self.stroke(orb, LINE, 0.55 * alpha, 1.0)
        self.fill(circle(x - r * 0.35, y + r * 0.38, max(0.8, r * 0.24)), WHITE, 0.8 * alpha)

    # ── the head ─────────────────────────────────────────────────────

    def _head(self, rig):
        hx, hy, hrx, hry, tilt = rig["head"]
        NSGraphicsContext.saveGraphicsState()
        local = NSAffineTransform.transform()
        local.translateXBy_yBy_(hx, hy)
        if tilt:
            local.rotateByDegrees_(tilt)
        local.concat()
        for base1, tip, base2 in rig["ears"]:
            # Rounded, not pointed: the curve bends round the tip.
            ear = NSBezierPath.bezierPath()
            ear.moveToPoint_(base1)
            ear.curveToPoint_controlPoint1_controlPoint2_(
                base2,
                (tip[0] + (base1[0] - base2[0]) * 0.45, tip[1] + (base1[1] - base2[1]) * 0.45),
                (tip[0] + (base2[0] - base1[0]) * 0.45, tip[1] + (base2[1] - base1[1]) * 0.45))
            ear.closePath()
            self.shape(ear, FUR)
            mid = ((base1[0] + base2[0]) / 2, (base1[1] + base2[1]) / 2)
            self.fill(oval(mid[0] + (tip[0] - mid[0]) * 0.45, mid[1] + (tip[1] - mid[1]) * 0.45,
                           8.0, 8.5), EAR_INNER, 0.95)
        head = oval(0.0, 0.0, 2 * hrx, 2 * hry)
        self.fill(head, FUR)
        NSGraphicsContext.saveGraphicsState()
        head.addClip()
        face_x = 3.0
        # Tiger-cat streaks: two up the forehead, spots beside them, two
        # back across each cheek from the corner of the eye.
        streaks = NSBezierPath.bezierPath()
        spots = NSBezierPath.bezierPath()
        for side in (-1.0, 1.0):
            streaks.moveToPoint_((face_x + side * 2.2, hry * 0.38))
            streaks.lineToPoint_((face_x + side * 3.2, hry * 0.9))
            for (x0, y0), (x1, y1) in (((12.5, -0.5), (21.0, -3.0)), ((12.0, -4.0), (19.0, -8.5))):
                streaks.moveToPoint_((face_x + side * x0, y0))
                streaks.lineToPoint_((face_x + side * x1, y1))
            spots.appendBezierPath_(circle(face_x + side * 8.5, hry * 0.62, 1.5))
            spots.appendBezierPath_(circle(face_x + side * 12.0, hry * 0.45, 1.2))
        self.stroke(streaks, ROSETTE, 0.88, 1.7)
        self.fill(spots, ROSETTE, 0.85)
        NSGraphicsContext.restoreGraphicsState()
        self.stroke(head, LINE, 1.0, 1.8)
        # A small, scrunched muzzle, set low.
        self.fill(oval(face_x, -7.0, 15.0, 10.0), CREAM)
        self._face(rig["face"], face_x)
        NSGraphicsContext.restoreGraphicsState()

    def _face(self, face, fx):
        lx, ly = face["look"]
        for side in (-1.0, 1.0):
            x, y = fx + side * 9.0, 1.5
            if face["blush"]:
                self.fill(oval(fx + side * 13.5, -4.0, 9.0, 4.5), BLUSH, 0.42)
            # Pale rings round the eyes, as tiger cats have.
            self.fill(oval(x, y + 0.3, 11.5, 13.0), CREAM, 0.95)
            self._eye(face["eyes"], x + lx, y + ly, 8.0,
                      10.5 * max(0.12, 1.0 - face["blink"]), side, (lx * 1.2, ly * 1.2))
        # Long whiskers, both sides: the face is turned to the viewer.
        whiskers = NSBezierPath.bezierPath()
        for side in (-1.0, 1.0):
            for lift, reach, droop in ((1.0, 30.0, 3.5), (-0.5, 32.0, -1.0), (-2.0, 28.0, -6.0)):
                whiskers.moveToPoint_((fx + side * 6.5, -6.5 + lift))
                whiskers.lineToPoint_((fx + side * reach, -6.5 + lift + droop))
        self.stroke(whiskers, LINE, 0.45, 0.8)
        nose = line((fx - 2.4, -3.4), (fx + 2.4, -3.4), (fx, -5.9))
        nose.closePath()
        self.fill(nose, NOSE)
        self._mouth(face["mouth"], fx, -6.2)

    def _eye(self, kind, x, y, w, h, side, gaze=(0.0, 0.0)):
        if kind in ("open", "wide"):
            if kind == "wide":
                w, h = w * 1.15, 12.5
            if h < 2.4:
                self.stroke(line((x - w / 2, y), (x + w / 2, y)), INK, 1.0, 2.0)
                return
            iris = oval(x, y, w, h)
            self.fill(iris, IRIS)
            self.stroke(iris, INK, 0.9, 1.1)
            # A slit by day; wide and round when startled.
            pw, ph = (w * 0.62, h * 0.62) if kind == "wide" else (w * 0.4, h * 0.8)
            self.fill(oval(x + gaze[0], y + gaze[1], pw, ph), INK)
            self.fill(circle(x - w * 0.2, y + h * 0.22, 1.5), WHITE, 0.95)
            self.fill(circle(x + w * 0.22, y - h * 0.24, 0.7), WHITE, 0.75)
            return
        if kind == "half":
            self.fill(oval(x, y - 2.0, w, h * 0.42), INK)
            return
        path = NSBezierPath.bezierPath()
        if kind == "happy":      # ∩
            path.moveToPoint_((x - w * 0.6, y - 2.0))
            path.curveToPoint_controlPoint1_controlPoint2_(
                (x + w * 0.6, y - 2.0), (x - w * 0.4, y + 4.5), (x + w * 0.4, y + 4.5))
        elif kind == "closed":   # ∪
            path.moveToPoint_((x - w * 0.6, y + 1.0))
            path.curveToPoint_controlPoint1_controlPoint2_(
                (x + w * 0.6, y + 1.0), (x - w * 0.4, y - 4.0), (x + w * 0.4, y - 4.0))
        elif kind == "x":
            path.moveToPoint_((x - 3.2, y - 3.2))
            path.lineToPoint_((x + 3.2, y + 3.2))
            path.moveToPoint_((x - 3.2, y + 3.2))
            path.lineToPoint_((x + 3.2, y - 3.2))
        elif kind == "squeeze":  # > <
            tip = -side * 3.2
            path.moveToPoint_((x - tip, y + 3.5))
            path.lineToPoint_((x + tip, y))
            path.lineToPoint_((x - tip, y - 3.5))
        self.stroke(path, INK, 1.0, 2.2)

    def _mouth(self, kind, x, y):
        def omega(scale=1.0):
            path = NSBezierPath.bezierPath()
            path.moveToPoint_((x, y + 0.6))
            path.lineToPoint_((x, y - 0.6 * scale))
            for side in (-1.0, 1.0):
                path.moveToPoint_((x, y - 0.6 * scale))
                path.curveToPoint_controlPoint1_controlPoint2_(
                    (x + side * 4.6 * scale, y - 0.4 * scale),
                    (x + side * 0.6 * scale, y - 3.6 * scale),
                    (x + side * 4.2 * scale, y - 3.6 * scale))
            self.stroke(path, LINE, 1.0, 1.3)

        if kind in ("w", "sleep"):
            omega(1.0 if kind == "w" else 0.8)
        elif kind == "tongue":
            omega()
            self.fill(oval(x + 0.8, y - 3.6, 3.8, 3.4), TONGUE)
        elif kind in ("open", "yawn", "grin"):
            w, h = {"open": (5.5, 5.0), "yawn": (8.5, 9.5), "grin": (8.0, 6.0)}[kind]
            mouth = oval(x, y - h / 2 - 0.6, w, h)
            self.fill(mouth, MOUTH)
            NSGraphicsContext.saveGraphicsState()
            mouth.addClip()
            self.fill(oval(x, y - h - 0.2, w * 0.8, h * 0.55), TONGUE)
            NSGraphicsContext.restoreGraphicsState()
            omega(0.9)
        elif kind == "flat":
            self.stroke(line((x - 3.5, y - 1.5), (x + 3.5, y - 1.5)), LINE, 1.0, 1.4)
        elif kind == "wavy":
            path = NSBezierPath.bezierPath()
            for i in range(11):
                px = x - 5.0 + i
                py = y - 1.8 + 1.0 * math.sin(i * 1.3)
                if i == 0:
                    path.moveToPoint_((px, py))
                else:
                    path.lineToPoint_((px, py))
            self.stroke(path, LINE, 1.0, 1.3)

    # ── things in the air ────────────────────────────────────────────

    def _particles(self, cat):
        for p in cat.particles:
            a = p.fade
            if p.kind == "crumb":
                self.fill(circle(p.x, p.y, p.r), TREAT, 0.9 * a)
            elif p.kind == "dust":
                self.fill(circle(p.x, p.y, p.r), (0.86, 0.85, 0.82), 0.5 * a)
            elif p.kind == "sparkle":
                self._star(p.x, p.y, p.r * (0.6 + 0.4 * a), p.life * 2.0, a)
            elif p.kind == "heart":
                self._heart(p.x, p.y, p.r, a)
            elif p.kind == "z":
                self.text("z", p.x, p.y, self.z_font(p.r + 5.0), (0.55, 0.70, 0.95), a,
                          shadow=True)

    def _star(self, x, y, r, spin, alpha):
        path = NSBezierPath.bezierPath()
        for i in range(8):
            radius = r if i % 2 == 0 else r * 0.3
            angle = spin + i * math.pi / 4
            point = (x + radius * math.cos(angle), y + radius * math.sin(angle))
            if i == 0:
                path.moveToPoint_(point)
            else:
                path.lineToPoint_(point)
        path.closePath()
        self.fill(path, mix(GOLD_GLOW, WHITE, 0.35), alpha)

    def _heart(self, x, y, r, alpha):
        path = NSBezierPath.bezierPath()
        path.moveToPoint_((x, y - r))
        path.curveToPoint_controlPoint1_controlPoint2_(
            (x, y + r * 0.45), (x - r * 1.5, y - r * 0.1), (x - r * 0.8, y + r * 1.15))
        path.curveToPoint_controlPoint1_controlPoint2_(
            (x, y - r), (x + r * 0.8, y + r * 1.15), (x + r * 1.5, y - r * 0.1))
        path.closePath()
        self.fill(path, (1.0, 0.45, 0.56), 0.9 * alpha)

    def _treat(self, treat):
        """A fish-shaped treat, nose first along its flight."""
        heading = math.degrees(math.atan2(treat.vy, treat.vx))
        NSGraphicsContext.saveGraphicsState()
        local = NSAffineTransform.transform()
        local.translateXBy_yBy_(treat.x, treat.y)
        local.rotateByDegrees_(heading)
        local.concat()
        fade = min(1.0, treat.age / 0.1)
        tail = line((-3.0, 0.0), (-7.0, 2.8), (-7.0, -2.8))
        tail.closePath()
        self.shape(tail, TREAT, fade, 0.8)
        self.shape(oval(0.0, 0.0, 9.0, 5.2), TREAT, fade, 0.8)
        self.fill(circle(2.2, 0.6, 0.7), INK, fade)
        NSGraphicsContext.restoreGraphicsState()

    def _thought(self, cat):
        b = cat.thought_bubble()
        if not b:
            return
        a, x, y, side = b["alpha"], b["x"], b["y"], b["side"]
        edge = (0.55, 0.6, 0.62)
        for cx, cy, r in ((x, y, 2.6), (x + side * 7, y + 8, 4.2)):
            self.fill(circle(cx, cy, r), WHITE, 0.92 * a)
            self.stroke(circle(cx, cy, r), edge, 0.5 * a, 1.0)
        bx, by = x + side * 20, y + 24
        bubble = oval(bx, by, 32.0, 22.0)
        self.fill(bubble, WHITE, 0.94 * a)
        self.stroke(bubble, edge, 0.5 * a, 1.0)
        for i in range(3):
            lit = 0.35 + 0.65 * max(0.0, math.sin(b["t"] * 5.0 - i * 0.9))
            self.fill(circle(bx - 7 + i * 7, by, 2.0), INK, lit * a)

    def _panel(self, cat):
        p = cat.panel()
        if not p:
            return
        x, y, w, h = PANEL
        tint = CYAN_OUT if p["cyan"] else GOLD_OUT
        box = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(((x, y), (w, h)), 9.0, 9.0)
        self.fill(box, (0.07, 0.09, 0.11), 0.84)
        self.stroke(box, WHITE, 0.1, 1.0)
        # Top row: what is happening, and the loss now. Bottom row: how far.
        self.text(p["title"], x + 9, y + h - 17, self.title_font, WHITE, 0.95)
        if p["loss"] is not None:
            self.text(f"{p['loss']:.3f}", x + w - 9, y + h - 17, self.small_font,
                      tint, 1.0, right=True)
        detail_width = self.text(p["detail"], x + w - 9, y + 2, self.tiny_font,
                                 WHITE, 0.6, right=True)
        gx, gy, gw, gh = x + 9, y + 16, w - 18, h - 35
        losses = [v for _, v in p["train"]] + [v for _, v in p["val"]]
        if losses:
            low, high = min(losses), max(losses)
            span = high - low or 1.0
            last_step = max([st for st, _ in p["train"]] + [st for st, _ in p["val"]] + [1])
            end = max(last_step, (cat.snap.iters or 0))

            def at(step, loss):
                return (gx + gw * step / end, gy + gh * (loss - low) / span)

            if len(p["train"]) >= 2:
                self.stroke(line(*[at(step, loss) for step, loss in p["train"]]),
                            tint, 0.95, 1.4)
            for step, loss in p["val"]:
                px, py = at(step, loss)
                self.fill(circle(px, py, 1.6), WHITE, 0.85)
        if p["progress"] is not None:
            bar = max(20.0, gw - detail_width - 8)
            self.fill(NSBezierPath.bezierPathWithRect_(((gx, y + 7), (bar, 1.8))), WHITE, 0.12)
            self.fill(NSBezierPath.bezierPathWithRect_(((gx, y + 7), (bar * p["progress"], 1.8))),
                      tint, 0.9)


# ── rendering without a window ───────────────────────────────────────

def render_png(painter: Painter, cat: Cat, path: Path, scale: float = 2.0,
               background=None) -> None:
    """Draw one frame into a PNG. For checks and screenshots; needs no window."""
    NSApplication.sharedApplication()
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, int(WIDTH * scale), int(HEIGHT * scale), 8, 4, True, False,
        NSDeviceRGBColorSpace, 0, 0)
    rep.setSize_((WIDTH, HEIGHT))
    context = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(context)
    try:
        if background is not None:
            painter.color(background).setFill()
            NSBezierPath.fillRect_(NSMakeRect(0, 0, WIDTH, HEIGHT))
        painter.draw(cat)
        context.flushGraphics()
    finally:
        NSGraphicsContext.restoreGraphicsState()
    data = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
    data.writeToFile_atomically_(str(path), True)


# ── the live pet ─────────────────────────────────────────────────────

def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


class CatView(NSView):
    """Hands everything to the Pet; PyObjC turns trailing underscores into
    selectors, so the logic lives in a plain class."""

    def initWithFrame_(self, frame):
        self = objc.super(CatView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.pet = None
        return self

    def isOpaque(self):
        return False

    def acceptsFirstMouse_(self, event):
        return True

    def drawRect_(self, rect):
        if self.pet is not None:
            self.pet.guard("draw", self.pet.painter.draw, self.pet.cat)

    def mouseDown_(self, event):
        self.pet.guard("mouse", self.pet.mouse_down)

    def mouseDragged_(self, event):
        self.pet.guard("mouse", self.pet.mouse_dragged)

    def mouseUp_(self, event):
        self.pet.guard("mouse", self.pet.mouse_up, event.clickCount())

    def rightMouseDown_(self, event):
        self.pet.guard("menu", self.pet.context_menu, event)

    def tick_(self, timer):
        self.pet.guard("tick", self.pet.tick)

    def openChat_(self, sender):
        self.pet.guard("chat", self.pet.open_chat)

    def toggleGraph_(self, sender):
        self.pet.cat.show_graph = not self.pet.cat.show_graph

    def toggleRoam_(self, sender):
        self.pet.guard("roam", self.pet.toggle_roam)

    def quitPet_(self, sender):
        NSApplication.sharedApplication().terminate_(None)


class Pet:
    def __init__(self, feed, position_file: Path | None, log_dir: Path | None,
                 port: int = 8742):
        self.feed = feed
        self.position_file = position_file
        self.log_dir = log_dir
        self.port = port
        self.cat = Cat()
        self.roamer = Roamer()
        self.painter = Painter()
        self.errors: set[str] = set()
        self.lock = threading.Lock()
        self.pending = None
        self.tooltip = ""
        self.last = time.monotonic()
        self.press = None
        self.origin = None
        self.dragging = False
        self.window = None
        self.view = None
        self.timer = None
        self.fps = 0.0
        self._visible = None
        self._visible_at = 0.0

    def guard(self, where, fn, *args):
        """A pet that throws in a timer or in drawRect_ takes the app down
        with an uncaught Objective-C exception. Say it once, keep going."""
        try:
            return fn(*args)
        except Exception:
            self._report(where, traceback.format_exc())

    def _report(self, where, text):
        if text not in self.errors:
            self.errors.add(text)
            print(f"[symbio-pet] {where} failed:\n{text}", file=sys.stderr)

    # ── state from the feed, off the main thread ─────────────────────

    def _poll_forever(self):
        interval = getattr(self.feed, "interval", 1.0)
        while True:
            try:
                snap = self.feed.poll()
            except Exception:
                self._report("poll", traceback.format_exc())
                snap = None
            if snap is not None:
                with self.lock:
                    # A verdict is an event: never let a newer snapshot that
                    # arrives before the next frame swallow it.
                    if self.pending is not None and self.pending.verdict and not snap.verdict:
                        snap.verdict = self.pending.verdict
                        snap.verdict_skill = self.pending.verdict_skill
                    self.pending = snap
            time.sleep(interval)

    def tick(self):
        now = time.monotonic()
        dt, self.last = now - self.last, now
        with self.lock:
            snap, self.pending = self.pending, None
        self._track_pointer()
        self.cat.update(dt, snap)
        if not self.dragging:
            self._roam(min(dt, 0.1))
        if snap is not None and snap.status != self.tooltip:
            self.tooltip = snap.status
            self.view.setToolTip_(snap.status)
        self.view.setNeedsDisplay_(True)
        self._pace()

    def _pace(self):
        c = self.cat
        moving = (c.treats or c.ejecta or c.anim or c.held or c.hop_t is not None
                  or c.walking or c.walk_w > 0.01 or c.pose_t < 1.0 or self.roamer.falling
                  or c.mood == "training" or any(p.kind != "z" for p in c.particles))
        fps = (FPS_MOVING if moving else FPS_ASLEEP if c.mood == "asleep"
               else FPS_CALM)
        if fps == self.fps:
            return
        self.fps = fps
        if self.timer is not None:
            self.timer.invalidate()
        self.timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / fps, self.view, "tick:", None, True)
        # Slack lets macOS fold these wakeups in with others.
        self.timer.setTolerance_(0.2 / fps)
        # Common modes: keep animating while a drag or the menu is tracking.
        NSRunLoop.currentRunLoop().addTimer_forMode_(self.timer, NSRunLoopCommonModes)

    def _track_pointer(self):
        mouse = NSEvent.mouseLocation()
        origin = self.window.frame().origin
        x, y = mouse.x - origin.x, mouse.y - origin.y
        self.cat.pointer = (x, y)
        left, right = self.cat.bounds()
        _, top = self.cat.head_top()
        self.cat.hovered = left + 12 < x < right - 12 and GROUND - 4 < y < top

    # ── walking about ────────────────────────────────────────────────

    def _screen(self):
        frame = self.window.frame()
        x = frame.origin.x + WIDTH / 2
        y = frame.origin.y + GROUND + 20
        for screen in NSScreen.screens():
            f = screen.frame()
            if (f.origin.x <= x < f.origin.x + f.size.width
                    and f.origin.y <= y < f.origin.y + f.size.height):
                return screen
        return NSScreen.mainScreen()

    def _roam(self, dt):
        if not self.roamer.enabled and not self.roamer.falling:
            return
        now = time.monotonic()
        # The screen list is a round trip to the window server; it changes
        # when a display does, not thirty times a second.
        if self._visible is None or now - self._visible_at > 1.0:
            self._visible, self._visible_at = self._screen().visibleFrame(), now
        visible = self._visible
        floor = visible.origin.y - GROUND + 3.0
        # The cat is narrower than its window: let the window hang off the
        # screen's edge until the cat's own tail or nose reaches it.
        left = visible.origin.x - (CX - 60.0)
        right = visible.origin.x + visible.size.width - (CX + 60.0)
        origin = self.window.frame().origin
        x, y = self.roamer.step(self.cat, dt, origin.x, origin.y, floor, left, right)
        if abs(x - origin.x) > 0.01 or abs(y - origin.y) > 0.01:
            self.window.setFrameOrigin_((x, y))
        if self.roamer.arrived:
            self._save_position()

    def toggle_roam(self):
        self.roamer.enabled = not self.roamer.enabled
        if not self.roamer.enabled:
            self.cat.walking = False
        self._save_position()

    # ── handling ─────────────────────────────────────────────────────

    def mouse_down(self):
        self.press = NSEvent.mouseLocation()
        self.origin = self.window.frame().origin
        self.dragging = False

    def mouse_dragged(self):
        if self.press is None:
            return
        mouse = NSEvent.mouseLocation()
        dx, dy = mouse.x - self.press.x, mouse.y - self.press.y
        if not self.dragging and math.hypot(dx, dy) > 4:
            self.dragging = True
            self.cat.grab()
        if self.dragging:
            self.window.setFrameOrigin_((self.origin.x + dx, self.origin.y + dy))

    def mouse_up(self, clicks: int = 1):
        if self.dragging:
            self.dragging = False
            self.cat.drop()
            self._save_position()
        elif self.press is not None:
            self.cat.poke()
            # A double click, not a single one: people pet a pet. Opening the
            # chat on every touch started a server and a browser tab each time.
            if clicks >= 2:
                self.open_chat()
        self.press = None

    def context_menu(self, event):
        menu = NSMenu.alloc().initWithTitle_("Symbio")
        for title, action, state in (
                ("Open chat", "openChat:", None),
                ("Wander around", "toggleRoam:", self.roamer.enabled),
                ("Always show the loss graph", "toggleGraph:", self.cat.show_graph),
                (None, None, None),
                ("Quit Symbio Pet", "quitPet:", None)):
            if title is None:
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
            item.setTarget_(self.view)
            if state is not None:
                item.setState_(1 if state else 0)
            menu.addItem_(item)
        NSMenu.popUpContextMenu_withEvent_forView_(menu, event, self.view)

    def open_chat(self):
        """The desktop chat window: reused when it is already serving,
        started (detached, so it outlives the pet) when it is not. It opens
        the browser itself once it is listening.

        `open`, not webbrowser.open: on macOS that drives the browser over
        AppleScript, and a browser that is not answering blocked this thread
        (the one that draws) for the two minutes an AppleEvent takes to
        time out.
        """
        url = f"http://127.0.0.1:{self.port}"
        if _listening(self.port):
            subprocess.Popen(["open", url], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        log = subprocess.DEVNULL
        if self.log_dir is not None:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                log = open(self.log_dir / "desktop.log", "ab")
            except OSError:
                log = subprocess.DEVNULL
        try:
            subprocess.Popen(
                [sys.executable, "-m", "symbio_desktop.cli", "--port", str(self.port)],
                cwd=str(Path(__file__).resolve().parent.parent),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        finally:
            if log is not subprocess.DEVNULL:
                log.close()

    # ── where it sits ────────────────────────────────────────────────

    def _default_origin(self):
        frame = NSScreen.mainScreen().visibleFrame()
        return (frame.origin.x + frame.size.width - WIDTH - 24,
                frame.origin.y - GROUND + 3.0)

    def _load_place(self):
        """Where it was left, and whether it may wander; None when unknown."""
        if self.position_file is None:
            return None
        try:
            saved = json.loads(self.position_file.read_text(encoding="utf-8"))
            x, y = float(saved["x"]), float(saved["y"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        self.roamer.enabled = bool(saved.get("roam", True))
        # Only if some screen still shows the cat there: a display that was
        # unplugged since would leave it somewhere nobody can reach.
        for screen in NSScreen.screens():
            f = screen.frame()
            if (f.origin.x - WIDTH / 2 < x < f.origin.x + f.size.width - WIDTH / 2
                    and f.origin.y - HEIGHT / 2 < y < f.origin.y + f.size.height - HEIGHT / 2):
                return (x, y)
        return None

    def _save_position(self):
        if self.position_file is None or self.window is None:
            return
        origin = self.window.frame().origin
        try:
            self.position_file.parent.mkdir(parents=True, exist_ok=True)
            self.position_file.write_text(json.dumps(
                {"x": origin.x, "y": origin.y, "roam": self.roamer.enabled}),
                encoding="utf-8")
        except OSError:
            pass

    # ── running ──────────────────────────────────────────────────────

    def run(self) -> int:
        app = NSApplication.sharedApplication()
        # No Dock icon and no menu bar: the cat is the whole interface.
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        x, y = self._load_place() or self._default_origin()
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, WIDTH, HEIGHT),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(False)
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setFloatingPanel_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle
            | NSWindowCollectionBehaviorFullScreenAuxiliary)

        view = CatView.alloc().initWithFrame_(NSMakeRect(0, 0, WIDTH, HEIGHT))
        view.pet = self
        panel.setContentView_(view)
        self.window, self.view = panel, view
        panel.orderFrontRegardless()

        threading.Thread(target=self._poll_forever, name="pet-feed", daemon=True).start()
        self._pace()

        def stop(*_):
            self._save_position()
            app.terminate_(None)

        # The handler runs at the next timer tick, at most 1/6 s away.
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        app.run()
        return 0
