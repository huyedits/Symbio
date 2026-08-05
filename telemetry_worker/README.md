# Symbio Telemetry Worker

A Cloudflare Worker that receives **anonymous telemetry pings** and **`/feedback` messages** from opted-in Symbio installs. It appends `/feedback` to `feedback/feedback.txt` (human-readable) and telemetry to `telemetry/pings.ndjson`, both on a dedicated `telemetry` branch (keeps `main` clean). Review feedback by PR'ing the `telemetry` branch into `main`.

The GitHub PAT lives **only inside the Worker** — never in any user's config. Each Symbio install authenticates with a shared secret header (`X-Telemetry-Secret`).

The Worker uses **only the GitHub Contents API** — no Issues / Pull-requests permission — so a fine-grained PAT needs just **Contents: Read and write**.

## Abuse & DoS defense (all free, edge-side)
This is a public URL that writes into your repo, so it's spam/abuse-bait. Defenses layer:

1. **Cloudflare automatic DDoS** — always-on, free at the edge; absorbs volumetric floods before they reach the Worker.
2. **`X-Telemetry-Secret` gate** — constant-time compare; no secret → `401`, no write, no GitHub call.
3. **Per-IP rate limits** via the Cache API (edge-local, **no KV / no billing**):
   - coarse pre-auth: **120 / 10min** (blinds floods that lack the secret)
   - feedback post-auth: **5 / 1hr** (each appends to the repo = a commit)
   - telemetry post-auth: **30 / 10min**
4. **Strict payload validation + size caps** — even if the secret leaks, every field is type-checked and length-capped (body ≤32KB, feedback text ≤2000 chars, env ≤24 keys, each string ≤200 chars, control chars stripped, unknown telemetry fields rejected). A leaked secret can't dump blobs or junk.
5. **Kill switch** — rotate `TELEMETRY_SECRET` (`wrangler secret put TELEMETRY_SECRET`) to revoke every install at once. If `feedback.txt` or `pings.ndjson` ever fills with junk, just delete/reset it on the `telemetry` branch.

Rate limits are edge-local (per Cloudflare colo) and slightly leaky under concurrency — that's intentional: volumetric floods are Cloudflare's job; the limits suppress junk from a single misbehaving install and protect your repo.

## Privacy contract
Symbio only ever sends: Symbio version, OS, Python version, model_name, LoRA rank, enabled tool groups, speed_mode, session count, tool-error count, and (for `/feedback`) the user's typed text. **Never** user name, message text, note contents, prompts, or file paths (except the feedback text the user explicitly typed).

## Deploy (6 steps)

1. **Install wrangler** and log in:
   ```sh
   npm i -g wrangler
   wrangler login
   ```

2. **Create a target repo** (or use an existing one) and a `telemetry` branch in it. The Worker writes `feedback/feedback.txt` and `telemetry/pings.ndjson` onto that branch. A fine-grained PAT needs Contents access on the `telemetry` branch.

3. **Create a fine-grained PAT** (GitHub → Settings → Developer settings → Fine-grained tokens) scoped to that repo with:
   - **Contents: Read and Write**  ← the only permission needed

4. **Set the three secrets** (do not put these in `wrangler.toml`):
   ```sh
   wrangler secret put TELEMETRY_SECRET   # a long random string you invent (e.g. openssl rand -hex 32)
   wrangler secret put GITHUB_TOKEN       # the fine-grained PAT from step 3
   wrangler secret put GITHUB_REPO         # e.g. your-user/symbio-data
   ```
   Save the `TELEMETRY_SECRET` value — you'll paste it into Symbio next.

5. **Deploy**:
   ```sh
   wrangler deploy
   ```
   Note the URL it prints, e.g. `https://symbio-telemetry.<your-subdomain>.workers.dev`.

6. **Point Symbio at it**. In Symbio:
   ```
   /telemetry on            # re-asks consent with the full data set
   /config set telemetry.endpoint https://symbio-telemetry.<your-subdomain>.workers.dev/ingest
   /config set telemetry.shared_secret <the TELEMETRY_SECRET from step 4>
   /feedback hello world     # smoke test — should append to feedback/feedback.txt
   ```
   Or set those two keys directly in `config.json` under the `telemetry` section.

## Endpoints
- `GET /` → `{"ok":true,"service":"symbio-telemetry"}` health check.
- `POST /ingest` with header `X-Telemetry-Secret` and JSON body:
  - `{"type":"feedback","text":"...","env":{...},"session_count":N}` → appends a human-readable block to `feedback/feedback.txt` on the `telemetry` branch.
  - `{"type":"telemetry","env":{...},"session_count":N,"error_count":M}` → appends one JSON line to `telemetry/pings.ndjson` on the `telemetry` branch.
- `401` bad/missing secret · `413` body too large · `429` rate limited · `400` bad/invalid body · `502` GitHub error.

## Notes
- The `GITHUB_BRANCH` var (default `telemetry`) controls which branch feedback.txt / pings.ndjson append to. Review feedback by PR'ing that branch into `main`.
- Appends use the contents API with one retry on 409 (concurrent writers).
- Quick local smoke test without deploying:
  ```sh
  npx wrangler dev   # serves on http://localhost:8787
  curl http://localhost:8787/
  curl -X POST http://localhost:8787/ingest -H "X-Telemetry-Secret: test" \
       -H "Content-Type: application/json" -d '{"type":"feedback","text":"hi","env":{"os":"Darwin"}}'
  ```
  (Local dev secrets can be put in a `.dev.vars` file: `TELEMETRY_SECRET="test"`, `GITHUB_TOKEN="..."`, `GITHUB_REPO="..."`.)