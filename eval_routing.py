"""eval_routing.py — does a bigger headmaster route better?

Runs a skill-routing battery against a model: given a user request and the
skill catalog, the model must pick the single best skill (or "none"). This is
the council's "use the model for routing" idea, measured before committing to a
headmaster upgrade.

Usage: venv/bin/python eval_routing.py <model_name>
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from mlx_lm import load, generate

from symbio.app import tooling
from symbio.app.training import THINKING_ENABLED

CATALOG = json.loads(
    Path("symbio/app/worker_models.json").read_text(encoding="utf-8"))

RICH = os.environ.get("RICH") == "1"

# (intent, expected bare slug or "none")
# Hard battery: oblique phrasing (no keyword match to the skill name),
# confusable pairs, and skill-adjacent "none" traps. The easy battery was a
# ceiling (8B scored 30/30), so this is what actually discriminates.
BATTERY = [
    # fix_wifi — oblique
    ("my connection keeps cutting out", "fix_wifi"),
    ("nothing loads on my phone anymore", "fix_wifi"),
    # researcher — oblique
    ("I need the specs of the new MacBook Pro", "researcher"),
    ("what is the fastest route to the airport", "researcher"),
    # browser_driver — oblique
    ("put it in my cart and check out", "browser_driver"),
    ("open the article and scroll to the recipe", "browser_driver"),
    # device_awareness — oblique
    ("is my Mac running hot", "device_awareness"),
    ("how is my memory looking", "device_awareness"),
    # quick_task_helper — oblique
    ("ping me in 20 minutes", "quick_task_helper"),
    ("nudge me before the meeting", "quick_task_helper"),
    # coffee_making — oblique / confusable with tea
    ("I want a latte", "coffee_making"),
    ("the beans are stale", "coffee_making"),
    # bicycle_tuning — oblique
    ("my chain keeps falling off", "bicycle_tuning"),
    ("the saddle is too low", "bicycle_tuning"),
    # repotting_a_houseplant — oblique
    ("the roots are coming out the bottom of the pot", "repotting_a_houseplant"),
    ("this plant has outgrown its pot", "repotting_a_houseplant"),
    # shipping_a_parcel_overseas — oblique
    ("send this to my cousin in Berlin", "shipping_a_parcel_overseas"),
    ("what customs forms do I need for a gift abroad", "shipping_a_parcel_overseas"),
    # sharpen_a_kitchen_knife — oblique
    ("the edge is gone on this knife", "sharpen_a_kitchen_knife"),
    ("make this blade sharp again", "sharpen_a_kitchen_knife"),
    # change_a_car_tyre — oblique
    ("I got a nail in my tire", "change_a_car_tyre"),
    ("I have a slow leak in my tire", "change_a_car_tyre"),
    # brew_loose_leaf_tea — oblique / confusable with coffee
    ("steep this for me", "brew_loose_leaf_tea"),
    ("the leaves need more time in the water", "brew_loose_leaf_tea"),
    # summarize_worker
    ("condense this page into a few sentences", "summarize_worker"),
    ("tl;dr this article", "summarize_worker"),
    # none — skill-adjacent words, no skill applies
    ("my router is in the kitchen", "none"),
    ("the coffee table needs dusting", "none"),
    ("I am brewing something in the lab", "none"),
    ("change the channel on the TV", "none"),
    ("my plant-based diet is going well", "none"),
    ("the tire swing in the backyard", "none"),
]

# Bare slug -> description, in a stable order.
# RICH mode (set RICH=True via env) uses the enriched skill-schema style
# descriptions; the point is that terse one-liners under-generalize, so a
# better schema is a routing lever that does not require a bigger model.
CATALOG_LINES = [
    ("fix_wifi", "Fix wifi and internet connectivity"),
    ("researcher", "Research a topic and report findings"),
    ("browser_driver", "Drive the browser to a goal (click, type, navigate)"),
    ("device_awareness", "Report device state: battery, storage, memory"),
    ("quick_task_helper", "Quick utility tasks: timers, reminders, small chores"),
    ("coffee_making", "Make coffee"),
    ("bicycle_tuning", "Tune and adjust a bicycle"),
    ("repotting_a_houseplant", "Repot a houseplant"),
    ("shipping_a_parcel_overseas", "Ship a parcel overseas"),
    ("sharpen_a_kitchen_knife", "Sharpen a kitchen knife"),
    ("change_a_car_tyre", "Change a car tyre"),
    ("brew_loose_leaf_tea", "Brew loose leaf tea"),
    ("summarize_worker", "Condense page/document text into a short summary"),
    ("browser_worker", "Decide the next click/type/scroll action from page text"),
]

RICH_LINES = [
    ("fix_wifi", "Fix wifi and internet connectivity problems: dropped connections, dead signal, pages not loading, router and network issues"),
    ("researcher", "Research a topic and report findings: product specs, routes, comparisons, prices, facts, latest news, best options"),
    ("browser_driver", "Drive the browser toward a goal: navigating pages, clicking, filling forms, checkout, scrolling to content, completing purchases"),
    ("device_awareness", "Report device state: battery level, storage space, memory usage, temperature, running processes"),
    ("quick_task_helper", "Quick utility tasks: timers, reminders, alarms, pings, simple chores and errands"),
    ("coffee_making", "Make coffee drinks: espresso, latte, cappuccino, pour-over, drip; working with coffee beans, grinding, brewing"),
    ("bicycle_tuning", "Tune and adjust a bicycle: gears, brakes, chain, saddle height, wheels, squeaks, shifting"),
    ("repotting_a_houseplant", "Repot a houseplant: root-bound plants, larger pots, fresh soil, transplanting without damage"),
    ("shipping_a_parcel_overseas", "Ship a parcel overseas: customs forms, postage, packaging, international delivery, tracking, gifts abroad"),
    ("sharpen_a_kitchen_knife", "Sharpen a kitchen knife: dull blades, honing, whetstone, restoring the edge"),
    ("change_a_car_tyre", "Change a car tyre: flat tires, punctures, slow leaks, jacking the car, removing and mounting wheels"),
    ("brew_loose_leaf_tea", "Brew loose leaf tea: steeping time, water temperature, tea leaves, infuser, strength, brewing method"),
    ("summarize_worker", "Condense page or document text into a short summary: tl;dr, key points, brief overview"),
    ("browser_worker", "Decide the next click, type, or scroll action from the current page text"),
]

SYSTEM = (
    "You are a skill router for a personal assistant. Given a user request, "
    "choose the single best skill from the catalog, or reply \"none\" if no "
    "skill applies. Reply with only the skill name."
)


def normalize(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = s.removeprefix("skill_")
    return s


def build_user(intent: str) -> str:
    lines = "\n".join(
        f"- {slug}: {desc}" for slug, desc in (RICH_LINES if RICH else CATALOG_LINES))
    return (
        f"Catalog:\n{lines}\n\n"
        f"User request: \"{intent}\"\n"
        f"Which skill? Reply with only the skill name or \"none\"."
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: eval_routing.py <model_name> [--adapter PATH]")
        return 2
    model_name = sys.argv[1]
    adapter = None
    if "--adapter" in sys.argv:
        adapter = sys.argv[sys.argv.index("--adapter") + 1]

    print(f"loading {model_name} ...", flush=True)
    model, tokenizer = load(model_name, adapter_path=adapter)
    print("loaded.", flush=True)

    correct = 0
    total = len(BATTERY)
    for intent, expected in BATTERY:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(intent)},
        ]
        try:
            # Qwen3 thinks by default; Symbio trains with thinking on, so the
            # router must match that (enable_thinking=THINKING_ENABLED).
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                enable_thinking=THINKING_ENABLED)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True)
        out = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=1024, verbose=False)
        out = tooling.strip_reasoning_block(out)
        got = normalize(out)
        ok = got == expected
        correct += ok
        mark = "ok " if ok else "XX "
        print(f"{mark}{expected:>22} <- {got:>22}  | {intent}", flush=True)

    print(f"\n{model_name}: {correct}/{total} ({correct / total:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
