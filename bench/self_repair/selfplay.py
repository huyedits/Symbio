"""Controlled self-play on the model's own code: it writes its own bugs, fixes
them, learns from the fixes the tests confirm, and keeps an update only when
it does not lose ground on bugs it never trained on.

One round:
  propose   the current model breaks one line in a function its tests cover
            (propose.py); a proposal counts only if the tests catch it
  solve     the current model tries every task ATTEMPTS times (drive.py),
            the first greedy, the rest sampled
  learn     a task solved some but not all of the time sits at the edge of
            what it can do: its passing fixes become solver data and its
            proposal becomes proposer data. Always-solved and never-solved
            tasks teach nothing and are dropped (the Absolute Zero rule)
  train     a fresh LoRA on everything earned so far
  gate      the candidate repairs the held-out bugs (modules never proposed
            from). Kept if it fixes at least as many as the best so far,
            otherwise rejected and archived; the next round plays with the
            kept model

Every decision and number goes to OUT/curve.jsonl; rejected adapters stay
on disk. Each step is skipped when its output exists, so a crash resumes.

    MODEL=... CLONE=<repo copy> OUT=<dir> python bench/self_repair/selfplay.py

Knobs: ROUNDS (4), PROPOSALS (16), ATTEMPTS (4), HELD_N (all; an evenly
spaced subset of the held-out bugs, to make each gate faster), plus rsi.py's RSI_LR,
RSI_LAYERS and genlib's GEN/MAXTOK. Only one model is in memory at a time:
every step is its own process.
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsi  # noqa: E402  (reads MODEL, CLONE, OUT, GEN from the environment)

ROUNDS = int(os.environ.get("ROUNDS", 4))
PROPOSALS = int(os.environ.get("PROPOSALS", 16))
ATTEMPTS = int(os.environ.get("ATTEMPTS", 4))
HELD_N = int(os.environ.get("HELD_N", 0))
OUT, CLONE, PY, log = rsi.OUT, rsi.CLONE, rsi.PY, rsi.log


def held_bugs(n):
    """The held-out list, or N of it spread evenly so every module stays represented."""
    bugs = json.loads(rsi.HELD.read_text())
    if not n or n >= len(bugs):
        return rsi.HELD
    pick = [bugs[i * len(bugs) // n] for i in range(n)]
    path = OUT / f"held_{n}.json"
    path.write_text(json.dumps(pick, indent=1))
    return path


def step(tag, script, args, adapter=None, **env):
    """Run one model-loading script in the clone; skipped once TAG.done exists."""
    if (OUT / f"{tag}.done").exists():
        return
    e = dict(os.environ, SYMBIO_HOME=str(CLONE), **{k: str(v) for k, v in env.items()})
    e.pop("ADAPTER", None)
    if adapter:
        e["ADAPTER"] = str(adapter)
    log(f"{tag}: {script} adapter={adapter}")
    with open(OUT / f"{tag}.log", "w") as f:
        subprocess.run([PY, str(HERE / script), rsi.MODEL, tag, *map(str, args)],
                       cwd=CLONE, env=e, stdout=f, stderr=subprocess.STDOUT, check=True)
    (OUT / f"{tag}.done").write_text("ok")


def held_score(tag, adapter):
    # one greedy try per bug: the gate measures pass@1, whatever ATTEMPTS the solve step uses
    step(tag, "drive.py", [held_bugs(HELD_N), OUT], adapter, ATTEMPTS=1)
    d = json.loads((OUT / f"{tag}.repair.json").read_text())
    return sum(r["fixed"] for r in d), len(d)


def edge_data(tasks, attempts, rnd):
    """Solver and proposer samples from tasks solved some, but not all, of the time."""
    by_task = {}
    for r in attempts:
        by_task.setdefault(r["i"], []).append(r)
    solver, proposer, rates = [], [], []
    for i, rs in sorted(by_task.items()):
        rate = sum(r["fixed"] for r in rs) / len(rs)
        rates.append(rate)
        if not 0 < rate < 1:
            continue
        t = tasks[i]
        proposer.append({"key": f"p{rnd}:{i}", "messages": [
            {"role": "user", "content": t["prompt"]}, {"role": "assistant", "content": t["reply"]}]})
        for r in rs:
            if r["fixed"] and r.get("prompts") and r.get("replies"):
                solver.append({"key": f"s{rnd}:{i}:{r['attempt']}", "messages": [
                    {"role": "user", "content": r["prompts"][-1]},
                    {"role": "assistant", "content": r["replies"][-1]}]})
    return solver, proposer, rates


def record(row):
    with open(OUT / "curve.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    log("CURVE " + json.dumps(row))


def main():
    state_file = OUT / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if "best" not in state:
        fixed, total = held_score("sp_base", None)
        state = {"best": fixed, "total": total, "adapter": None, "data": [], "round": 0}
        state_file.write_text(json.dumps(state))
        record({"round": 0, "held": f"{fixed}/{total}", "decision": "baseline"})
    for rnd in range(state["round"] + 1, ROUNDS + 1):
        cur = state["adapter"]
        step(f"sp_r{rnd}_propose", "propose.py", [rsi.TRAIN, OUT, PROPOSALS], cur)
        tasks_file = OUT / f"sp_r{rnd}_propose.tasks.json"
        tasks = json.loads(tasks_file.read_text()) if tasks_file.exists() else []
        if not tasks:
            record({"round": rnd, "valid": 0, "decision": "no valid proposals"})
            state["round"] = rnd
            state_file.write_text(json.dumps(state))
            continue
        step(f"sp_r{rnd}_solve", "drive.py", [tasks_file, OUT], cur, ATTEMPTS=ATTEMPTS)
        attempts = json.loads((OUT / f"sp_r{rnd}_solve.repair.json").read_text())
        solver, proposer, rates = edge_data(tasks, attempts, rnd)
        seen = {s["key"] for s in state["data"]}
        state["data"] += [s for s in solver + proposer if s["key"] not in seen]
        row = {"round": rnd, "valid": len(tasks), "proposals": PROPOSALS,
               "solve_rates": [round(x, 2) for x in rates],
               "edge_tasks": len(proposer), "new_solver": len(solver), "data": len(state["data"])}
        if not solver:
            row["decision"] = "nothing at the edge; no training"
        else:
            cand = rsi.train(f"sp_r{rnd}", state["data"])
            fixed, total = held_score(f"sp_r{rnd}_held", cand)
            row["held"] = f"{fixed}/{total}"
            if fixed >= state["best"]:
                row["decision"] = f"KEEP ({fixed} >= best {state['best']})"
                state["best"], state["adapter"] = fixed, str(cand)
            else:
                row["decision"] = f"REJECT ({fixed} < best {state['best']}); archived, keeping {cur}"
                (Path(cand) / "REJECTED").write_text(row["decision"] + "\n")
        record(row)
        state["round"] = rnd
        state_file.write_text(json.dumps(state))
    log(f"DONE best {state['best']}/{state['total']} adapter {state['adapter']}")


if __name__ == "__main__":
    main()
