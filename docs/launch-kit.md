# Launch kit

Ready-to-post copy for telling people about Symbio. Every number here is from
the README and is a single run on a 16 GB Mac; the posts say so, because the
audiences below (HN, r/LocalLLaMA) punish overclaiming faster than anything.

Landing page: `docs/index.html`. To serve it at
`https://huyedits.github.io/Symbio/`, turn on GitHub Pages under
**Settings → Pages → Deploy from a branch → `main` / `/docs`**.

---

## One-liners

- **Tagline:** Correct it once. It fine-tunes itself so it remembers.
- **GitHub "About":** Local agent for Apple Silicon that turns your corrections
  into LoRA fine-tunes, and rolls back a fine-tune that made it worse.
- **Suggested GitHub topics:** `llm` `agent` `lora` `fine-tuning` `mlx`
  `apple-silicon` `local-llm` `self-improving` `continual-learning` `macos`

---

## Show HN

**Title** (under 80 chars):

> Show HN: Symbio – a local Mac agent that fine-tunes itself from your corrections

**Body:**

> Most agents forget a correction as soon as the session ends. You can stuff
> fixes into the system prompt, but that makes every turn slower and teaches
> the model nothing.
>
> Symbio takes the other route. When you correct it, the correction is saved as
> a training example. Once enough pile up, it runs a short LoRA fine-tune on
> your Mac (MLX), reloads the adapter, and re-scores a golden regression set. If
> the new adapter made anything worse, it is rolled back.
>
> A few things I learned building it:
>
> - Not every mistake wants the same training. A wrong fact needs passes over
>   the corpus; a wrong *action shape* (a GNU flag on BSD, a tool that doesn't
>   exist) needs repetition of the exact form. Mistakes are classified at
>   capture time and trained differently.
> - On a 13-task held-out battery it hit 13/13 at iteration 100 of 1000. The
>   other 900 iterations changed nothing.
> - Pointed at a simulator it had never seen (50 verbs, deliberately
>   unfamiliar), with no human-written samples (it keeps only attempts that
>   changed the world the way the task asked), it went from 10/45 to 21/45 on a
>   clean held-out run. Single runs, so read the direction, not the decimals.
> - My first benchmark mimicked the AWS CLI and the model scored 13/13 before
>   any training, because it was reciting pretraining. The README keeps that
>   and a few other harness mistakes next to the results.
>
> It also drives the Mac through the accessibility API, automates a browser,
> and has a Telegram gateway. No cloud inference; telemetry is off by default.
> Apple Silicon only for now (MLX).
>
> `pip install "symbio-cli[mlx]" && symb setup && symb chat`
>
> Repo: https://github.com/huyedits/Symbio
> Browser demo (no fine-tuning, just the parsing/memory/RAG pieces):
> https://huggingface.co/spaces/HuyEdits/symbio-demo
>
> Happy to answer anything, especially "why not just RAG".

**Have an answer ready for "why not just RAG?"** Retrieval puts a fact back
into the context; it does not change what the model reaches for. Symbio does
both: notes and RAG for facts, adapter updates for behaviour (the "reflex"
mistakes), with the golden set guarding the weights.

---

## r/LocalLLaMA

**Title:**

> I built a local agent that LoRA-fine-tunes itself from my corrections on a 16 GB Mac, with automatic rollback when a fine-tune regresses

**Body:**

> Setup: Qwen3 8B/14B class models, MLX, Apple Silicon, 16 GB.
>
> Loop: correction detected → saved as a training sample (tagged "knowledge"
> vs "reflex") → threshold scales with corpus size (log, clamped 2–20) → short
> LoRA run → golden set re-scored → rollback on consistent regression (flaky
> regressions get a retry first).
>
> Numbers (single runs, README has the tables and the caveats):
>
> - 13-task held-out battery: 0/13 → 13/13 by iteration 100. Training
>   1.5 s/iter at 8.4 GB peak on a 16 GB Mac.
> - Self-taught on an unseen simulator with zero human-written samples:
>   10/45 → 21/45 on a clean held-out run; on the 10 reserved tasks, 2 → 4.
>
> Gotchas that cost me days, in case they save you some:
>
> - `mlx_lm lora` trains q/k/v/o plus the MLP projections. An eval that
>   attaches only q/v with `load_weights(strict=False)` silently drops the
>   rest and reads as "the model learned nothing".
> - Validating on 8 batches ended a two-hour run on a 0.005 difference.
>   Use the whole validation set.
>
> Repo: https://github.com/huyedits/Symbio (Apache 2.0)

---

## X / Bluesky thread

1. I got tired of correcting my AI agent and watching it make the same mistake
   the next day. So Symbio fine-tunes itself. On my Mac. From my corrections. 🧵
2. You say "no, it's X". That becomes a training example. After a few, a small
   LoRA run happens locally and the adapter reloads. No cloud, no subscription.
3. The scary part of self-training is getting worse without noticing. So every
   run is re-scored against a golden set, and a regression rolls it back.
4. Measured: 0/13 → 13/13 on a held-out battery by iteration 100 (of 1000).
   The other 900 iterations did nothing.
5. On a simulator it had never seen, with no human-written training data at
   all, it taught itself from 10/45 to 21/45. Single runs, caveats in the
   README.
6. Apple Silicon, Apache 2.0. `pip install "symbio-cli[mlx]"`
   https://github.com/huyedits/Symbio

---

## Where else

- **Hugging Face:** link the Space from the model cards of the presets it uses,
  and post in the MLX community discussions.
- **Awesome lists:** open PRs to `awesome-mlx`, `awesome-local-ai` and
  `awesome-llm-agents`-style lists with the GitHub "About" line above.
- **Timing:** Show HN does best on a weekday morning, US Pacific. Stay in the
  thread for the first two hours and answer every question.
