# Adapter marketplace — design doc

Status: **design only, not implemented.** This document specifies the shape of a future feature; nothing here should be assumed to exist in the codebase today except where explicitly noted.

## Why

Symbio's headmaster and each MoA worker (see the README's "Mixture of agents" section) can be fine-tuned locally, but every install starts that fine-tuning from zero. A marketplace lets people share and reuse trained adapters — a well-tuned `browser` worker, a headmaster adapter tuned for a particular workflow — without retraining from scratch, while keeping Symbio's core promise intact: **nothing leaves your machine, and nothing arrives on it, unless you explicitly ask.**

## Non-negotiable constraints

- **Opt-in, offline by default.** No config flag here defaults to `true`. No background checks, no telemetry, no "recommended adapters" prompts. A user has to run a command to talk to the network at all.
- **Never trust a downloaded adapter's own claims.** A manifest's self-reported golden-set score is a trust *signal*, not a verification. The consuming machine re-runs the golden set itself after installing, using the exact same infrastructure that already protects local training (`symbio/app/golden.py`, `ChatSession._guarded_train` / `dispatch.guarded_train_worker`).
- **Reuse existing infrastructure over building new hosting.** LoRA weight files are exactly what Hugging Face Hub already hosts well (safetensors + a config file); there's no reason to build or operate a custom backend for this.

## Package format

A shareable adapter package is the existing LoRA output — `adapter_config.json` + `adapters.safetensors` (the same files `training.run_training` / `dispatch.guarded_train_worker` already produce) — plus a new `manifest.json` committed alongside them:

```json
{
  "name": "browser-click-lite",
  "role": "browser",
  "base_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
  "description": "Faster, more reliable click/type/scroll decisions for e-commerce sites.",
  "author": "someone",
  "license": "MIT",
  "symbio_schema_version": 1,
  "golden_results": {
    "pass_count": 4,
    "total": 4,
    "case_ids": ["browser_emits_known_action"]
  },
  "created": "2026-07-21T00:00:00"
}
```

- `role: null` (or omitted) means it's a headmaster adapter, not a worker's — installing one of those replaces `adapters/`, the same thing `/train` already produces locally.
- `golden_results` is the publisher's own `golden.run_golden_set` output at publish time. It tells a consumer roughly what to expect; it does not substitute for re-running the check locally (see "Installing" below).

## Distribution: Hugging Face Hub, not custom hosting

Publish a package as a small HF Hub model repo — safetensors + a config file is exactly the shape HF Hub is built around, and Symbio already depends on `mlx-community/*` repos being pulled from there, so no new hosting infrastructure is needed. A `symbio-adapter` tag on the repo (plus `manifest.json` committed in the same repo) is enough for discovery; no separate index/registry service.

## CLI surface

`symb adapter <subcommand>`, added to `symbio/app/cli.py` alongside the existing `config`/`gateway`/`train` subcommands.

| Subcommand | Status | What it does |
|---|---|---|
| `symb adapter list` | **could ship now** | List locally installed adapters — headmaster's `adapters/` plus each `adapters/workers/<role>/`, trivially available from the filesystem, no network. |
| `symb adapter info <path-or-repo>` | **could ship now** | Parse and pretty-print a `manifest.json` without installing anything — no network required for a local path; a repo argument would need read-only HF Hub access. |
| `symb adapter search <query>` | not implemented | Search HF Hub repos tagged `symbio-adapter`. |
| `symb adapter install <repo>` | not implemented | Download a package, verify it against the local golden set, install (see below). |
| `symb adapter publish <role\|headmaster>` | not implemented | Package the current adapter + a generated manifest and push to an HF Hub repo the user owns. |

`list`/`info` are pure/local and low-risk enough to implement as a first, small follow-up; `search`/`install`/`publish` need real network-error handling, auth (HF tokens), and the install-time verification flow below, which is why they're scoped out of this pass.

## Installing: reuse the training guard rail, don't reinvent it

The key safety property: **installing a third-party adapter is treated as structurally the same operation as training one locally** — the same backup → apply → golden-check → rollback-on-regression sequence, just with "copy the downloaded files into place" substituted for "run `mlx_lm lora`":

1. `training.backup_adapter(role=...)` — snapshot whatever's currently installed for that role (or the headmaster, if `role` is `None`).
2. Run the golden set (role-scoped `WORKER_GOLDEN_CASES`, or the headmaster's `GOLDEN_CASES`) against the *current* adapter — the baseline.
3. Copy the downloaded `adapter_config.json` + `adapters.safetensors` into `constants.adapter_dir_for(role)`, replacing what was there.
4. Reload the model with the new adapter.
5. Run the golden set again. A case that passed before and fails now is a regression — same threshold/rollback config as training (`golden_rollback_on_regression` / `dispatch.worker_golden_rollback_on_regression`).
6. Regression with rollback enabled → `training.restore_adapter(backup_dir, role=...)`, back to the pre-install state automatically.

This is why the manifest's `golden_results` field matters even though it isn't trusted blindly: a consumer can compare "what the publisher claimed" against "what actually happened on my machine, with my base model, my hardware" and immediately see if something doesn't line up.

## Threat model notes

LoRA weights are data (safetensors), not executable code — there is no arbitrary-code-execution surface from the file format itself, unlike installing a package with a build script. The real risk is *behavioral*: a malicious or careless adapter could bias the model toward unsafe tool-calling (e.g. nudging it toward destructive `<cmd>` suggestions, or toward fabricating confidence instead of searching). The golden-set-guarded install flow above is the primary mitigation — it's a coarse behavioral smoke test, not a full safety audit, so:

- Anything downloaded should still go through the same sandbox/approval gates as normal use (`sandbox.py`'s command blocking, the Telegram confirm-before-dangerous-tools flow) — installing an adapter never bypasses those.
- A worse-but-still-passing adapter (subtly biased in ways the golden set doesn't probe) is a real residual risk. Expanding the golden set's coverage over time is the mitigation, not a one-time fix.

## Open questions (not resolved by this doc)

- Should `symb adapter install` require an explicit `--yes` even after a clean golden-check, or is a clean pass sufficient to install without a second confirmation?
- Versioning/updates: if a repo publishes a v2, does `install` overwrite silently, or always go through the same backup/verify/rollback flow as a fresh install (leaning toward: always, for consistency)?
- Should worker catalog entries (`symbio/app/worker_models.json`) be able to reference a marketplace adapter by repo id directly, so `dispatch.enabled` workers can bootstrap pretrained instead of starting from a base model with no adapter?
