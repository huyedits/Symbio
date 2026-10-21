# Charms — the Symbio pet

> Charms. A tabby with an amber bead on her collar — the one that glows when
> she learns. She roams, she loafs, she stalks the cursor when the trainer
> runs. Smug about held-out wins. Chews on rolled-back runs, then shakes it
> off and walks away pretending nothing happened. When the brain dictates a
> new chapter, she sits and listens, tail ticking.

*(inspo acknowledged: ThatPet proves a pet with personality and signature
reactions makes the work warmer — the character here is Symbio's own, drawn
from its design canon: pond teal, tabby ink, the collar charm that glows as
it learns.)*

## Signature reactions — the character sheet

| Life event | Charms' reaction | Mechanic (existing rig) |
|---|---|---|
| Training iters stream by | eats a droplet per step; body visibly gains weight; wobble follows the loss | `activity=training`, droplet queue |
| Loss wobbles upward | tail flicks, ear twitches | wobble-follows-loss |
| Held-out eval: wins on every task | **the charm flares** — celebration anim, tooltip reads the wins | `verdict="kept"` one-shot on new eval |
| Golden gate rolls a run back | droop, chew, shake-off, walk away | `rolled_back` anim |
| The brain dictates education | sits and listens; tooltip quotes the reasoning | `dictated: <action> — <reasoning>` |
| Owner away, nobody training | roams the desk, loafs, blinks, lives her little life | Roamer + idle kit |

## What she reads

* `logs/training_live.json` — the harness's fine-tune runs (`/train`, golden gate)
* `bench/arc/train_log.txt`, `eval_results.json` — the ARC apprenticeship organ
* `bench/arc/last_plan.json` — whatever the brain dictated
* `adapters/training_progress.json` — the headmaster adapter's total steps

She imports nothing from the agent. She watches from the outside, like a real
animal in a real house.