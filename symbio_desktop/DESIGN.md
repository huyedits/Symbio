---
version: alpha
name: Symbio Desktop
description: The chat window of a personal agent that lives on one Mac, learns from its own mistakes, and asks before it acts; it should feel like a companion's window, not another company's chat app.
colors:
  ground: "#f2f5f3"        # light; dark #0f1e20
  surface: "#ffffff"       # light; dark #162a2d
  ink: "#2a211b"           # light; dark #efe7da
  muted: "#675b50"         # light; dark #a39684
  signal: "#bf7c16"        # the charm; dark #f0b44c
  pond: "#1f7a6f"          # focus, links, resting; dark #58b8aa
  danger: "#b3361f"        # dark #f28a72
typography:
  display:
    fontFamily: ui-rounded, 'SF Pro Rounded', system-ui
    fontSize: 16.5px-30px
    fontWeight: 600-650
  body:
    fontFamily: -apple-system, system-ui
    fontSize: 15.5px
    lineHeight: 1.65
  data:
    fontFamily: ui-monospace, 'SF Mono', Menlo
rounded:
  sm: 6px
  md: 10px
  lg: 16px
  xl: 22px
components:
  button-primary:
    backgroundColor: "{colors.signal}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
---

## Overview

Symbio is one person's agent on one Mac: it chats, drives the browser and the
desktop, and fine-tunes itself overnight on what it got wrong. It has a body —
tilcayo, the tabby cat on the Dock and in `static/icon.png`, whose collar charm
glows as its adapter grows. The window takes its look from that icon and
nothing else. Until 2026-09-27 it wore Claude's (cream, `#d97757`, a serif for
replies, "Good afternoon, Huy"), which made it a copy of another product.

## Colors

- **Pond (ground):** the teal the icon sits on — a misty pond by day
  (`#f2f5f3`), deep water at night (`#0f1e20`). The dominant surface.
- **Tabby ink:** the coat's stripe brown for text by day (14:1 on ground),
  muzzle cream by night (14:1). Muted text is 6:1 and 5.9:1 — AA everywhere.
- **Charm (signal):** the amber bead. One job: what Symbio is about to do or
  is doing — the primary action (Send, Allow), the status dot while it works,
  the bead beside an approval's reason. Ink text on it (4.6:1 and 9:1).
- **Pond teal (secondary):** focus rings, links, the resting status dot.
- **Swat red (danger):** refusals, Never, a post whose words don't match.
- The adapter map uses the same five: charm, pond, coat, nose, stripe
  (`--accent` … `--accent-5`). Never a raw hex in a view.

## Typography

- **Display — SF Pro Rounded** (`ui-rounded`): Symbio's own voice. Its name on
  the empty window, section titles, and the headline of an approval card —
  the sentence where it says what it is about to do. In Chrome this falls
  back to SF Pro; the app window is WebKit and shows the rounded face.
- **Body — SF Pro:** everything else, replies included. Sentence case.
- **Data — SF Mono:** only for machine text a person reads as such: code, the
  tool log, and the exact command or post on an approval card.

## Layout

Chat first: sidebar, one reading column (720px), the composer card at the
bottom; the workspace is a drawer. The empty window is the icon and the
assistant's name over the composer.

## Elevation & Depth

Flat surfaces separated by the pond tints. Shadow only on the composer, the
settings panel and the back-to-latest button — the things that float. The
modal scrim is solid, never blurred.

## Shapes

6px controls, 10px blocks (code, activity, notices), 16px cards that hold a
decision (approvals, settings), 22px composer.

## Components

- **Approval card:** a charm bead + the reason naming the user's own setting;
  the headline (display face); a red warning when the words about to go out
  aren't the ones asked for; the exact payload in mono; Deny / Always allow /
  Allow. A warned card has no Always allow and a red-outlined Allow.
- **Guardrails panel:** one row per kind of action, four segments
  (Allow / If risky / Ask / Never), the floors as plain sentences, recent
  decisions with their answers.

## Do's and Don'ts

- Do add a token here before a view uses a colour. Read colours from the
  tokens in JS too (`getComputedStyle`), as the adapter map does.
- Do keep every label in sentence case, in the body face.
- Don't change: the icon, the pet's name, the assistant's name the user chose.
- Don't use: cream + terracotta or a serif for replies (SD1); tracked
  ALL-CAPS labels, `A · B` meta strings, mono for decoration (SD4);
  backdrop blur (K3); emoji as icons (I2); Tailwind's stock violet, sky,
  emerald, amber, pink.
