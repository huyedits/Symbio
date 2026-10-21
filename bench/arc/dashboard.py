#!/usr/bin/env python3
"""Caine's Growth Lab — a live window on the fluid-learning experiment.

Serves one dark page (Symbio's own tokens: pond teal, tabby ink, charm gold)
that polls a JSON snapshot every 2 seconds. Everything is shown:
the apprenticeship loss curve with every iteration line, the paged-bank
telemetry, the brain's dictated plan verbatim, the self-model it read, the
paired held-out scoreboard when it lands, per-task rows, and the raw feed.

stdlib only. Run:  python3 dashboard.py [--port 8765]
"""
import argparse
import json
import re
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ARC_DIR = Path("/Users/huygpt/Downloads/agi/bench/arc")
PAGED = Path("/Users/huygpt/Downloads/agi/bench/paged_moe")
TRAIN_LOG = ARC_DIR / "train_log.txt"
LAST_PLAN = ARC_DIR / "last_plan.json"
EVAL_RESULTS = ARC_DIR / "eval_results.json"
FRONTIER = ARC_DIR / "frontier_results.json"
ADAPTER_CFG = PAGED / "adapters_lfm_arc" / "adapter_config.json"

_ITER_RE = re.compile(
    r"Iter (\d+): Train loss ([-+\d.eE]+)\s*\|\s*(?:peak (\d+)MB)?\s*\|\s*"
    r"res=(\d+)MB hot=(\d+) pin(?:ned)?=(\d+) h=(\d+) m=(\d+) e=(\d+)"
    r"\s*\|\s*([\d.]+)s")


def parse_train_log():
    iters, raw = [], []
    if TRAIN_LOG.exists():
        for ln in TRAIN_LOG.read_text(errors="replace").splitlines():
            if "Train loss" not in ln:
                continue
            raw.append(ln)
            m = _ITER_RE.search(ln)
            if m:
                it, loss, peak = int(m[1]), float(m[2]), int(m[3] or 0)
                iters.append({
                    "iter": it, "loss": loss, "peak_mb": peak,
                    "res_mb": int(m[4]), "hot": int(m[5]),
                    "pinned": int(m[6]), "hits": int(m[7]),
                    "misses": int(m[8]), "evicts": int(m[9]),
                    "t": float(m[10]),
                })
    return iters, raw


def as_json(path, label=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def snapshot():
    iters, raw = parse_train_log()
    cfg = as_json(ADAPTER_CFG)
    plan = as_json(LAST_PLAN)
    ev = as_json(EVAL_RESULTS)
    fr = as_json(FRONTIER)
    import subprocess
    try:
        running = bool(subprocess.run(["pgrep", "-f", "train_lfm.py"],
                                      capture_output=True, timeout=10).stdout)
    except Exception:
        running = False

    files = {}
    for p in (TRAIN_LOG, LAST_PLAN, EVAL_RESULTS, FRONTIER, ADAPTER_CFG,
              ARC_DIR / "train.jsonl", ARC_DIR / "eval.jsonl"):
        if p.exists():
            files[str(p.name)] = {
                "bytes": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime(
                    "%H:%M:%S"),
            }

    # a sample training task so a human can see WHAT it learns
    sample = None
    try:
        first = json.loads((ARC_DIR / "train.jsonl").read_text().splitlines()[0])
        sample = {"task_id": first.get("task_id"),
                  "user_excerpt": first["messages"][-2]["content"][:1400],
                  "assistant_target": first["messages"][-1]["content"][:600]}
    except Exception:
        pass

    return {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_running": running,
        "curve": iters,
        "raw_line_count": len(raw),
        "spawn_config": cfg,
        "decision": plan and {
            "plan": plan.get("plan"),
            "state_seen": plan.get("state"),
        },
        "held_out_eval": ev and {
            "summary": ev.get("summary"),
            "per_task": ev.get("per_task"),
        },
        "frontier_withdrawn": (fr | {"note": "invalid run: response field "
                                     "misparsed; superseded by published "
                                     "yardsticks"} if fr else None),
        "sample_task": sample,
        "files": files,
    }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Caine's Growth Lab</title>
<style>
  :root {
    --bg: #0e1412; --surface: #141c19; --elev: #1a2420;
    --border: rgba(88,184,170,.16); --ink: #efe7da; --dim: #a39684;
    --pond: #58b8aa; --charm: #bf7c16; --gold: #ffd700;
    --danger: #b3361f; --ok: #4aa06a;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--ink); margin: 0;
    font: 14px/1.5 ui-monospace, "SF Mono", Menlo, monospace; }
  header { padding: 18px 24px 10px; border-bottom: 1px solid var(--border);
    display: flex; align-items: baseline; gap: 14px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .mark { color: var(--gold); }
  .pill { margin-left: auto; padding: 3px 12px; border-radius: 20px;
    border: 1px solid var(--border); font-size: 12px; }
  .pill.on { color: var(--ok); border-color: var(--ok); }
  .pill.off { color: var(--dim); }
  main { padding: 18px 24px 40px; max-width: 1200px; }
  section { background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .1em;
    color: var(--pond); margin: 0 0 12px; font-weight: 600; }
  .kv { color: var(--dim); }
  .card { background: var(--elev); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 14px; margin-top: 10px; }
  .reason { border-left: 3px solid var(--charm); padding: 6px 12px;
    color: var(--ink); font-style: italic; white-space: pre-wrap; }
  pre { margin: 0; white-space: pre-wrap; word-break: break-word;
    color: var(--dim); font-size: 12px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  td, th { padding: 4px 8px; border-bottom: 1px solid var(--border);
    text-align: left; }
  th { color: var(--pond); font-weight: 500; }
  .num { text-align: right; }
  .up { color: var(--ok); } .down { color: var(--danger); }
  details summary { cursor: pointer; color: var(--dim); }
  #feed { max-height: 300px; overflow: auto; }
  .stamp { color: var(--dim); font-size: 11px; }
  .cast { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
  .girl { background: var(--elev); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px; text-align: center; position: relative;
    overflow: hidden; }
  .girl b { color: var(--ink); }
  .girl .sub { color: var(--dim); font-size: 11px; min-height: 34px; }
  .girl svg { width: 120px; height: 130px; }
  .drop { position: absolute; top: 6px; width: 6px; height: 10px;
    background: var(--pond); border-radius: 50% 50% 60% 60%;
    animation: fall linear infinite; opacity: .8; }
  @keyframes fall { from { transform: translateY(-10px); } to { transform: translateY(190px); opacity: .1; } }
  .hop { animation: hop 1s ease-in-out; }
  @keyframes hop { 0%,100% { transform: translateY(0);} 40% { transform: translateY(-8px);} }
  .flare circle.bell { animation: flaregold 1.2s ease-out; }
  @keyframes flaregold { 0% { r: 5; } 50% { r: 9; filter: drop-shadow(0 0 8px #ffd700);} 100% { r: 5; } }
  .zzz { color: var(--dim); font-size: 12px; letter-spacing: .2em; }
  svg { display: block; width: 100%; height: 120px; }
  .legend { color: var(--dim); font-size: 11px; margin-top: 4px; }
</style></head>
<body>
<header><span class="mark">☤</span><h1>Caine's Growth Lab</h1>
  <span class="kv">the fluid-learning experiment, nothing cut</span>
  <span id="status" class="pill off">…</span></header>
<main>
  <section><h2>Apprenticeship — every iteration, spawned expert on ARC-AGI-2</h2>
    <svg id="curve" viewBox="0 0 1100 120" preserveAspectRatio="none"></svg>
    <div class="legend">gold: train loss · teal line: paged bank resident MB ·
      hover data in the raw feed below</div>
    <div id="curveStats" class="kv" style="margin-top:8px"></div></section>

  <section><h2>Caine's decision — dictated, verbatim</h2>
    <div id="decision">waiting for a decision…</div></section>

  <section><h2>Held-out scoreboard — 120 eval tasks it never trained on</h2>
    <div id="scoreboard" class="kv">pending — scores land when the
      apprenticeship finishes</div>
    <div id="perTask"></div></section>

  <section><h2>The self-model Caine read before deciding</h2>
    <details><summary>expand full JSON</summary><pre id="selfmodel">pending</pre>
    </details></section>

  <section><h2>What it is actually learning — a real training task</h2>
    <div id="taskSample" class="kv">…</div></section>

  <section><h2>The girls — live reactions</h2>
    <div class="cast">
      <div class="girl" id="gKomi">
        <svg viewBox="0 0 120 130">
          <g class="komibody">
            <path d="M35 40 Q30 10 60 14 Q90 10 85 40 L82 62 Q60 74 38 62 Z" fill="#58b8aa"/>
            <circle cx="60" cy="46" r="24" fill="#f7e6dd"/>
            <path d="M36 44 Q42 18 60 20 Q78 18 84 44 L78 30 Q60 24 42 30 Z" fill="#2a6b62"/>
            <path d="M30 44 L14 78 Q12 92 22 90 Z" fill="#58b8aa"/>
            <path d="M90 44 L106 78 Q108 92 98 90 Z" fill="#58b8aa"/>
            <circle cx="51" cy="48" r="3.4" fill="#2a211b"/>
            <circle cx="69" cy="48" r="3.4" fill="#2a211b"/>
            <circle cx="52" cy="53" r="3" fill="#f3b6a0" opacity=".7"/>
            <circle cx="68" cy="53" r="3" fill="#f3b6a0" opacity=".7"/>
            <path d="M57 56 Q60 59 63 56" stroke="#2a211b" stroke-width="1.2" fill="none"/>
            <circle class="bell" cx="60" cy="64" r="6" fill="#bf7c16" stroke="#ffd700" stroke-width="1.4"/>
            <rect x="42" y="78" width="36" height="34" rx="8" fill="#e6ecea"/>
          </g>
        </svg>
        <b>Kōmi</b> <span class="kv">· the Brain</span>
        <div class="sub" id="komiNote">droplets rain while she trains</div>
      </div>
      <div class="girl" id="gTata">
        <svg viewBox="0 0 120 130">
          <circle cx="34" cy="26" r="10" fill="#f3b8a2"/>
          <circle cx="86" cy="26" r="10" fill="#f3b8a2"/>
          <circle cx="60" cy="46" r="24" fill="#f7e6dd"/>
          <path d="M38 36 Q60 22 82 36 Q80 28 60 26 Q40 28 38 36Z" fill="#8a5a3c"/>
          <circle cx="51" cy="48" r="3.4" fill="#2a211b"/>
          <circle cx="69" cy="48" r="3.4" fill="#2a211b"/>
          <path d="M54 56 Q60 62 66 56" stroke="#2a211b" stroke-width="1.4" fill="none"/>
          <rect x="40" y="70" width="40" height="42" rx="10" fill="#3c3c46"/>
          <rect x="34" y="70" width="10" height="30" rx="6" fill="#3c3c46"/>
          <rect x="76" y="70" width="10" height="30" rx="6" fill="#3c3c46"/>
        </svg>
        <b>Tata</b> <span class="kv">· the Apprentice</span>
        <div class="sub" id="tataNote">watches the golden gate</div>
      </div>
      <div class="girl" id="gSumi">
        <svg viewBox="0 0 120 130">
          <path d="M34 52 Q28 12 60 16 Q92 12 86 52 L82 40 Q60 30 38 40 Z" fill="#c9ccd4"/>
          <circle cx="60" cy="46" r="24" fill="#f7e6dd"/>
          <circle cx="51" cy="49" r="6" fill="none" stroke="#51453b" stroke-width="1.4"/>
          <circle cx="69" cy="49" r="6" fill="none" stroke="#51453b" stroke-width="1.4"/>
          <line x1="57" y1="49" x2="63" y2="48.6" stroke="#51453b" stroke-width="1.2"/>
          <circle cx="51" cy="49" r="2.6" fill="#2a211b"/>
          <circle cx="69" cy="49" r="2.6" fill="#2a211b"/>
          <path d="M55 58 L65 58" stroke="#b3361f" stroke-width="1" opacity=".25"/>
          <rect x="40" y="72" width="40" height="40" rx="9" fill="#2b3a4a"/>
        </svg>
        <b>Sumi</b> <span class="kv">· the Governor</span>
        <div class="sub" id="sumiNote">decides how and when she learns</div>
        <pre id="sumiQuote" style="font-size:10px; text-align:left; max-height:44px; overflow:auto"></pre>
      </div>
      <div class="girl" id="gMofu">
        <svg viewBox="0 0 120 130">
          <circle cx="60" cy="46" r="24" fill="#f7e6dd"/>
          <path d="M36 42 Q32 14 60 16 Q88 14 84 42 Q74 26 60 28 Q46 26 36 42Z" fill="#6e625a"/>
          <path d="M55 48 Q60 52 65 48" stroke="#2a211b" stroke-width="1.4" fill="none" transform="translate(0,2)"/>
          <rect x="38" y="70" width="44" height="42" rx="12" fill="#4d3b2e"/>
          <text id="mofuZ" x="88" y="26" class="zzz">z z z</text>
        </svg>
        <b>Mofu</b> <span class="kv">· the Elder</span>
        <div class="sub" id="mofuNote">purrs per milestone</div>
      </div>
    </div>
  </section>

  <section><h2>Raw feed — every line, nothing cut</h2>
    <details open><summary id="feedSummary">0 lines</summary>
      <pre id="feed"></pre></details></section>

  <section><h2>Artifacts on disk</h2>
    <div id="files" class="kv"></div>
    <div class="stamp" id="stamp"></div></section>
</main>
<script>
const $ = id => document.getElementById(id);
function poly(pts, w, h, key, color, yMin, yMax) {
  if (pts.length < 2) return "";
  const px = i => 40 + (w - 60) * i / (pts.length - 1);
  const py = v => h - 14 - (h - 30) * (v - yMin) / ((yMax - yMin) || 1);
  return `<polyline fill="none" stroke="${color}" stroke-width="1.6" points="${
    pts.map((p, i) => px(i).toFixed(1) + "," + py(p[key]).toFixed(1)).join(" ")}"/>`;
}
async function tick() {
  const s = await (await fetch("/api/state")).json();
  const pill = $("status");
  pill.textContent = s.training_running ? "TRAINING — the expert is forming"
                                        : "idle";
  pill.className = "pill " + (s.training_running ? "on" : "off");

  const c = s.curve || [];
  if (c.length >= 2) {
    const losses = c.map(p => p.loss);
    const lo = Math.min(...losses), hi = Math.max(...losses);
    const resM = c.map(p => p.res_mb);
    const rlo = Math.min(...resM), rhi = Math.max(...resM);
    $("curve").innerHTML =
      poly(c, 1100, 120, "loss", "#ffd700", lo, hi) +
      poly(c, 1100, 120, "res_mb", "#58b8aa", rlo, rhi);
    $("curveStats").textContent =
      `iters ${c.length} · loss ${losses[0]?.toFixed(3)} → ${losses.at(-1)?.toFixed(3)}` +
      ` · last: iter ${c.at(-1).iter}, ${c.at(-1).t}s cumulative` +
      ` · hot ${c.at(-1).hot} tensors, resident ${c.at(-1).res_mb} MB, evictions ${c.at(-1).evicts}`;
  } else { $("curveStats").textContent = "no iterations yet…"; }

  const d = s.decision;
  if (d && d.plan) {
    const p = d.plan;
    $("decision").innerHTML =
      `<div class="card"><b>action:</b> ${p.action} · tasks: ${p.tasks ?? "—"} · ` +
      `spawn slots: ${p.spawn ?? "—"} · iters: ${p.iters ?? "—"} · decider: ${p._decider ?? "?"}` +
      `<div class="reason">${p.reasoning || ""}</div></div>`;
    $("selfmodel").textContent = JSON.stringify(d.state_seen, null, 1);
  }

  const ev = s.held_out_eval;
  if (ev && ev.summary) {
    const s2 = ev.summary;
    const dUp = s2.cell_delta >= 0;
    $("scoreboard").innerHTML =
      `<table><tr><th></th><th class="num">loss</th><th class="num">cell acc</th><th class="num">exact grids</th></tr>` +
      `<tr><td>base brain</td><td class="num">${s2.base_loss?.toFixed(3)}</td>` +
      `<td class="num">${(s2.base_cell_acc * 100).toFixed(1)}%</td>` +
      `<td class="num">${s2.base_exact_count}</td></tr>` +
      `<tr><td>grown brain</td><td class="num">${s2.adapted_loss?.toFixed(3)}</td>` +
      `<td class="num" class="${dUp ? "up" : "down"}">${(s2.adapted_cell_acc * 100).toFixed(1)}%</td>` +
      `<td class="num">${s2.adapted_exact_count}</td></tr></table>` +
      `<div class="kv" style="margin-top:8px">paired: ${s2.wins} tasks improved, ` +
      `${s2.losses} worsened, ${s2.ties} tied · cell delta ` +
      `<span class="${dUp ? "up" : "down"}">${(dUp ? "+" : "")}${(s2.cell_delta * 100).toFixed(1)}%</span>` +
      ` · n=${s2.n} held-out tasks</div>`;
    const pt = (ev.per_task || []).slice(0, 400);
    $("perTask").innerHTML = pt.length ?
      `<details><summary>all ${pt.length} per-task rows, cut nothing</summary>` +
      `<table><tr><th>task</th><th class="num">Δ loss</th><th class="num">Δ cell</th><th>exact→</th></tr>` +
      pt.map(r => `<tr><td>${r.task_id}</td><td class="num ${r.loss_delta < 0 ? "up" : ""}">${r.loss_delta.toFixed(3)}</td>` +
        `<td class="num">${(r.cell_delta * 100).toFixed(1)}%</td>` +
        `<td>${r.base_exact ? "✓→" : "·→"}${r.adapted_exact ? "✓" : "·"}</td></tr>`).join("") +
      `</table></details>` : "";
  }

  // ---- the girls ----
  const komi = $("gKomi");
  if (komi) {
    komi.querySelectorAll(".drop").forEach(e => e.remove());
    if (s.training_running) {
      for (let i = 0; i < 4; i++) {
        const d = document.createElement("div"); d.className = "drop";
        d.style.left = (18 + Math.random() * 84) + "px";
        d.style.animationDuration = (0.9 + Math.random() * 0.9) + "s";
        komi.appendChild(d);
      }
      $("komiNote").textContent = "eating droplets — one per iteration ♪";
    } else {
      $("komiNote").textContent = "resting between chapters";
    }
    const ev = s.held_out_eval;
    if (ev && ev.summary && ev.summary.wins > 0) {
      const bell = komi.querySelector(".bell");
      bell.classList.remove("flare"); void bell.getBoundingClientRect();
      if (!bell.dataset.flared) { bell.classList.add("flare");
        bell.dataset.flared = "1"; }
      $("komiNote").textContent =
        `bell flare — ${ev.summary.wins}/${ev.summary.n} held-out wins`;
    }
  }
  const d2 = s.decision;
  if (d2 && d2.plan && d2.plan.reasoning) {
    $("sumiQuote").textContent = "“" + d2.plan.reasoning.slice(0, 220) + "”";
    $("sumiNote").textContent = "dictated: " + d2.plan.action;
  }
  if (s.training_running) {
    $("tataNote").textContent = "wobbling with every spike";
    $("mofuZ").textContent = "";
  } else {
    $("tataNote").textContent = "watching the golden gate";
    $("mofuZ").textContent = "z z z";
  }

  const t = s.sample_task;
  $("taskSample").innerHTML = t ?
    `<div class="card"><b>${t.task_id}</b><pre>${t.user_excerpt.replace(/</g, "&lt;")}</pre>` +
    `<pre style="color:var(--pond)">TARGET ↴ ${t.assistant_target.replace(/</g, "&lt;")}</pre></div>`
    : "…";

  $("feed").textContent = (s.rawFeed || []).join("\\n");
  $("feedSummary").textContent = `${s.raw_line_count || 0} lines`;

  $("files").innerHTML = Object.entries(s.files || {}).map(([k, v]) =>
    `${k} — ${v.bytes} bytes, ${v.mtime}`).join("<br>");
  $("stamp").textContent = "refreshed " + s.now;
}
tick(); setInterval(tick, 2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            try:
                import io  # raw feed lines pass through untouched
                iters, raw = parse_train_log()
                snap = snapshot()
                snap["rawFeed"] = raw[-400:]
            except Exception as e:
                snap = {"error": str(e), "rawFeed": [], "curve": []}
            body = json.dumps(snap, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            body = PAGE.encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"growth lab on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()