/**
 * Symbio telemetry + /feedback receiver (hardened).
 *
 * Receives anonymous telemetry pings and /feedback messages from opted-in
 * Symbio installs, authenticated by a shared secret header. Appends /feedback
 * to feedback/feedback.txt (human-readable) and telemetry to
 * telemetry/pings.ndjson, both on a dedicated branch (keeps main clean).
 * Review feedback by PR'ing that branch into main.
 *
 * Uses ONLY the GitHub Contents API — no Issues / Pull-requests permission —
 * so a fine-grained PAT needs just "Contents: Read and write".
 *
 * Abuse defenses (all free, edge-side):
 *  - Cloudflare automatic DDoS absorbs volumetric floods before the Worker.
 *  - X-Telemetry-Secret gate (constant-time compare); no secret -> 401.
 *  - Per-IP rate limits via the Cache API (edge-local, no KV/billing):
 *      coarse pre-auth  120 / 10min   (blinds floods that lack the secret)
 *      feedback post-auth  5 / 1hr    (each makes a GitHub issue = expensive)
 *      telemetry post-auth 30 / 10min
 *  - Strict payload validation + size caps: a leaked secret still can't dump
 *    large or malformed data — every field is type-checked and length-capped.
 *  - Kill switch: rotate TELEMETRY_SECRET (wrangler secret put) to revoke all
 *    installs at once.
 *
 * Secrets (set with `wrangler secret put`):
 *   TELEMETRY_SECRET  — shared bearer key (matches Symbio's telemetry.shared_secret)
 *   GITHUB_TOKEN      — fine-grained PAT with Contents: Read and write (only)
 *   GITHUB_REPO       — "owner/repo" to append feedback.txt / pings.ndjson into
 * Vars:
 *   GITHUB_BRANCH      — branch to append pings.ndjson to (default "telemetry")
 */

const GITHUB_API = "https://api.github.com";

// --- size / shape limits -------------------------------------------------
const MAX_BODY = 32 * 1024;     // hard cap on any single ingest body
const MAX_TEXT = 2000;         // /feedback text length
const MAX_ENV_KEYS = 24;       // keys in the env object
const MAX_ENV_KEY = 64;        // length of an env key
const MAX_ENV_STR = 200;       // length of any string env value
const MAX_ENV_ARR = 20;        // length of an array env value
const MAX_ENV_ARR_ITEM = 64;   // length of an array env string item

// --- small helpers -------------------------------------------------------

/** Constant-time string compare so timing attacks can't leak the secret. */
function safeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function json(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

function isPlainObject(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function authed(request, env) {
  const got = request.headers.get("X-Telemetry-Secret") || "";
  const want = env.TELEMETRY_SECRET || "";
  if (!want) return false; // misconfigured: refuse all ingestion
  return safeEqual(got, want);
}

async function ghFetch(env, path, init = {}) {
  const headers = Object.assign(
    {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "symbio-telemetry-worker",
    },
    init.headers || {},
  );
  return fetch(`${GITHUB_API}${path}`, Object.assign({}, init, { headers }));
}

// --- per-IP rate limiting (edge-local Cache API, no billing) -------------

/**
 * Count this IP within a time bucket. Edge-local (per colo) and slightly
 * leaky under concurrency — that's fine: volumetric floods are Cloudflare's
 * job; this just suppresses junk from a single misbehaving install.
 */
async function rateCount(request, bucket, windowSec) {
  const ip =
    request.headers.get("CF-Connecting-IP") ||
    request.headers.get("X-Real-IP") ||
    "anon";
  const slot = Math.floor(Date.now() / 1000 / windowSec);
  const key = new Request(`https://symbio-rl/${bucket}/${ip}/${slot}`, {
    method: "GET",
  });
  const cache = caches.default;
  let count = 0;
  try {
    const existing = await cache.match(key);
    if (existing) count = parseInt(await existing.text(), 10) || 0;
  } catch (e) {
    /* cache miss = first request in window */
  }
  count += 1;
  try {
    await cache.put(
      key,
      new Response(String(count), {
        headers: { "Cache-Control": `max-age=${windowSec}` },
      }),
    );
  } catch (e) {
    /* non-fatal: worst case we under-count */
  }
  return count;
}

const COARSE_MAX = 120;       // pre-auth: /10min/IP
const FEEDBACK_MAX = 5;      // post-auth: /1hr/IP
const TELEMETRY_MAX = 30;    // post-auth: /10min/IP

// --- payload validation --------------------------------------------------

/** Normalize+cap the anonymous env block. Rejects oversized/odd values so a
 * leaked secret still can't store arbitrary junk in the repo. */
function clampEnv(env) {
  if (env == null) return { ok: true, env: null };
  if (!isPlainObject(env)) return { ok: false, error: "env must be an object" };
  const keys = Object.keys(env);
  if (keys.length > MAX_ENV_KEYS) return { ok: false, error: "env too large" };
  const out = {};
  for (const k of keys) {
    if (typeof k !== "string" || k.length > MAX_ENV_KEY)
      return { ok: false, error: "env key too long" };
    const v = env[k];
    if (v == null) {
      out[k] = v;
      continue;
    }
    const tv = typeof v;
    if (tv === "string") {
      if (v.length > MAX_ENV_STR) return { ok: false, error: "env value too long" };
      out[k] = v;
    } else if (tv === "number" || tv === "boolean") {
      out[k] = v;
    } else if (
      Array.isArray(v) &&
      v.every((x) => typeof x === "string" && x.length <= MAX_ENV_ARR_ITEM)
    ) {
      out[k] = v.slice(0, MAX_ENV_ARR);
    } else {
      return { ok: false, error: "env value type not allowed" };
    }
  }
  return { ok: true, env: out };
}

function validateFeedback(payload) {
  if (!isPlainObject(payload)) return { ok: false, error: "payload must be an object" };
  const text = payload.text;
  if (typeof text !== "string" || text.length < 1 || text.length > MAX_TEXT)
    return { ok: false, error: "text must be 1..2000 chars" };
  // Strip control chars except \n and \t (keeps feedback readable, kills binary junk).
  const clean = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  const envRes = clampEnv(payload.env);
  if (!envRes.ok) return envRes;
  const out = { type: "feedback", text: clean, env: envRes.env };
  if (typeof payload.session_count === "number")
    out.session_count = Math.floor(payload.session_count);
  return { ok: true, payload: out };
}

function validateTelemetry(payload) {
  if (!isPlainObject(payload)) return { ok: false, error: "payload must be an object" };
  const allowed = new Set(["type", "env", "session_count", "error_count"]);
  for (const k of Object.keys(payload)) {
    if (!allowed.has(k)) return { ok: false, error: `unexpected field: ${k}` };
  }
  const envRes = clampEnv(payload.env);
  if (!envRes.ok) return envRes;
  const out = { type: "telemetry", env: envRes.env };
  if (typeof payload.session_count === "number")
    out.session_count = Math.floor(payload.session_count);
  if (typeof payload.error_count === "number")
    out.error_count = Math.floor(payload.error_count);
  return { ok: true, payload: out };
}

// --- handlers ------------------------------------------------------------

/**
 * Append `addText` to `path` on GITHUB_BRANCH via the contents API:
 * GET current sha + content → PUT updated content. One retry on 409 (sha race
 * with a concurrent writer). Shared by feedback (.txt) and telemetry (.ndjson).
 *
 * This deliberately uses the Contents API only — no Issues / Pull requests
 * permission — so a fine-grained PAT needs just "Contents: Read and write".
 * Feedback lands in feedback/feedback.txt; you review it by PR'ing the
 * telemetry branch into main.
 */
async function appendFile(env, path, addText) {
  const branch = env.GITHUB_BRANCH || "telemetry";
  const owner_repo = env.GITHUB_REPO;

  const getResp = await ghFetch(env, `/repos/${owner_repo}/contents/${path}?ref=${branch}`);
  let sha = null;
  let existing = "";
  if (getResp.ok) {
    const data = await getResp.json();
    sha = data.sha || null;
    try {
      existing = atob(data.content.replace(/\n/g, ""));
    } catch (e) {
      existing = "";
    }
  } else if (getResp.status !== 404) {
    const detail = await getResp.text();
    return json(502, { error: "github_get_failed", status: getResp.status, detail });
  }

  let content = existing + addText;
  const putBody = { message: `append ${path}`, content: btoa(content), branch };
  if (sha) putBody.sha = sha;

  let putResp = await ghFetch(env, `/repos/${owner_repo}/contents/${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(putBody),
  });
  // Retry once on 409 (sha race with a concurrent writer).
  if (putResp.status === 409) {
    const retryGet = await ghFetch(env, `/repos/${owner_repo}/contents/${path}?ref=${branch}`);
    if (retryGet.ok) {
      const rdata = await retryGet.json();
      putBody.sha = rdata.sha || null;
      try {
        content = atob(rdata.content.replace(/\n/g, "")) + addText;
        putBody.content = btoa(content);
      } catch (e) {
        /* fall back to our own content */
      }
      putResp = await ghFetch(env, `/repos/${owner_repo}/contents/${path}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(putBody),
      });
    }
  }
  if (!putResp.ok) {
    const detail = await putResp.text();
    return json(502, { error: "github_put_failed", status: putResp.status, detail });
  }
  return json(200, { ok: true });
}

/** Human-readable feedback block appended to feedback/feedback.txt. */
function formatFeedbackBlock(payload) {
  const ts = new Date().toISOString();
  const parts = [ts];
  if (payload.session_count != null) parts.push(`session_count=${payload.session_count}`);
  if (payload.env && typeof payload.env === "object") {
    for (const [k, v] of Object.entries(payload.env)) {
      parts.push(`${k}=${Array.isArray(v) ? v.join(",") : v}`);
    }
  }
  return `=== ${parts.join(" | ")} ===\n${payload.text || ""}\n---\n`;
}

async function handleFeedback(env, payload) {
  return appendFile(env, "feedback/feedback.txt", formatFeedbackBlock(payload));
}

/** Append one JSON line to telemetry/pings.ndjson. */
async function handleTelemetry(env, payload) {
  return appendFile(env, "telemetry/pings.ndjson", JSON.stringify(payload) + "\n");
}

// --- entry ---------------------------------------------------------------

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/health")) {
      return json(200, { ok: true, service: "symbio-telemetry" });
    }
    if (request.method !== "POST" || url.pathname !== "/ingest") {
      return json(404, { error: "not_found", hint: "POST /ingest with X-Telemetry-Secret" });
    }

    // 1) Coarse pre-auth flood limit — blunts blind floods lacking the secret.
    try {
      if ((await rateCount(request, "coarse", 600)) > COARSE_MAX)
        return json(429, { error: "rate_limited" });
    } catch (e) {
      /* Cache API hiccup: don't block legit traffic */
    }

    // 2) Secret gate.
    if (!authed(request, env)) return json(401, { error: "unauthorized" });

    // 3) Body size cap (header first to avoid reading giant bodies).
    const cl = parseInt(request.headers.get("Content-Length") || "0", 10);
    if (cl > MAX_BODY) return json(413, { error: "body_too_large" });

    let raw;
    try {
      const text = await request.text();
      if (text.length > MAX_BODY) return json(413, { error: "body_too_large" });
      raw = JSON.parse(text);
    } catch (e) {
      return json(400, { error: "bad_json" });
    }

    const type = raw && raw.type;

    // 4) Per-kind post-auth rate limit (feedback is expensive: makes an issue).
    if (type === "feedback") {
      try {
        if ((await rateCount(request, "fb", 3600)) > FEEDBACK_MAX)
          return json(429, { error: "rate_limited", kind: "feedback" });
      } catch (e) {}
      const v = validateFeedback(raw);
      if (!v.ok) return json(400, { error: v.error });
      return handleFeedback(env, v.payload);
    }
    if (type === "telemetry") {
      try {
        if ((await rateCount(request, "tl", 600)) > TELEMETRY_MAX)
          return json(429, { error: "rate_limited", kind: "telemetry" });
      } catch (e) {}
      const v = validateTelemetry(raw);
      if (!v.ok) return json(400, { error: v.error });
      return handleTelemetry(env, v.payload);
    }
    return json(400, { error: "unknown_type", type });
  },
};