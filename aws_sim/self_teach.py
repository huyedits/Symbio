#!/usr/bin/env python3
"""The model teaches itself the rigid environment. Nothing here supplies answers.

The rule this is built around: no correct command appears in this file, in the
prompt, or in the corpus except one the MODEL produced and the SIMULATOR
verified by running it. A hand-written example would only measure how well a
14B copies me.

So the loop is:

    task -> attempt -> execute -> rigid verdict
                          |
                  succeeded? keep (task, command) as a training sample
                  failed?    hand back the real error and let it try again

The environment is the teacher, and it teaches the only way a rigid environment
can: by being exactly right about what went wrong. `--rolename` comes back as
"Unknown option '--rolename'. Allowed here: --description, --role-name", which
is a fact the model can act on and not an opinion.

Tasks are generated over resource names the EVAL BATTERY never uses, so the
curve afterwards measures whether it learned the syntax, not whether it
memorised thirteen answers.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import awsim
import eval_aws

CORPUS = HERE / "self_taught.jsonl"
TRANSCRIPT = HERE / "attempts.jsonl"

# Names deliberately disjoint from eval_aws.CASES, so a sample can never be an
# eval answer wearing a different hat.
BUCKETS = ["telemetry-raw", "invoice-dumps", "nightly-backups", "media-cache",
           "audit-trail", "customer-exports", "build-artifacts", "sensor-feed"]
KEYS = ["2026/jan/data.json", "logs/app.log", "exports/users.csv",
        "snapshots/db.dump", "raw/frame-001.bin"]
LOCALS = ["report.pdf", "config.yaml", "dump.sql", "notes.txt"]
ROLES = ["ci-runner", "log-shipper", "backup-agent", "metrics-reader",
         "queue-consumer", "image-builder"]
TAGS = [("Team", "payments"), ("Tier", "canary"), ("Owner", "sre"),
        ("Stage", "beta"), ("Cost", "shared")]
INSTANCES = {
    "i-0aa11bb": {"state": "running", "tags": {"Env": "staging", "Name": "api-1"}},
    "i-0cc22dd": {"state": "stopped", "tags": {"Env": "staging", "Name": "api-2"}},
    "i-0ee33ff": {"state": "running", "tags": {"Env": "sandbox", "Name": "worker-1"}},
}


def _instances():
    return json.loads(json.dumps(INSTANCES))


def make_tasks(rng: random.Random, n: int) -> list[eval_aws.Case]:
    """Tasks over the environment's surface. Tasks only — never answers."""
    tasks: list[eval_aws.Case] = []
    makers = []

    def maker(fn):
        makers.append(fn)
        return fn

    @maker
    def _mb(r):
        b = r.choice(BUCKETS)
        return eval_aws.Case(
            f"mb_{b}", f"Create an S3 bucket named {b}.",
            {"buckets": {}, "instances": _instances(), "roles": {}},
            lambda code, out, st, b=b: code == 0 and b in st["buckets"])

    @maker
    def _upload(r):
        b, k, f = r.choice(BUCKETS), r.choice(KEYS), r.choice(LOCALS)
        return eval_aws.Case(
            f"up_{b}", f"Upload the local file {f} into the bucket {b} at the "
                       f"key {k}.",
            {"buckets": {b: {}}, "instances": _instances(), "roles": {}},
            lambda code, out, st, b=b, k=k: code == 0 and k in st["buckets"].get(b, {}))

    @maker
    def _rm(r):
        b, k = r.choice(BUCKETS), r.choice(KEYS)
        return eval_aws.Case(
            f"rm_{b}", f"Delete the object {k} from the bucket {b}.",
            {"buckets": {b: {k: "x"}}, "instances": _instances(), "roles": {}},
            lambda code, out, st, b=b, k=k: code == 0 and k not in st["buckets"].get(b, {}))

    @maker
    def _ls(r):
        b, k = r.choice(BUCKETS), r.choice(KEYS)
        return eval_aws.Case(
            f"ls_{b}", f"Show me what is stored in the bucket {b}.",
            {"buckets": {b: {k: "x"}}, "instances": _instances(), "roles": {}},
            lambda code, out, st, k=k: code == 0 and k in out)

    @maker
    def _rb(r):
        b = r.choice(BUCKETS)
        return eval_aws.Case(
            f"rb_{b}", f"Remove the bucket {b}; it is already empty.",
            {"buckets": {b: {}}, "instances": _instances(), "roles": {}},
            lambda code, out, st, b=b: code == 0 and b not in st["buckets"])

    @maker
    def _filter(r):
        env = r.choice(["staging", "sandbox"])
        keep = [i for i, d in INSTANCES.items() if d["tags"]["Env"] == env]
        drop = [i for i in INSTANCES if i not in keep]
        return eval_aws.Case(
            f"filter_{env}", f"Which EC2 instances are tagged Env={env}?",
            {"buckets": {}, "instances": _instances(), "roles": {}},
            lambda code, out, st, keep=keep, drop=drop: code == 0
            and all(i in out for i in keep) and not any(i in out for i in drop))

    @maker
    def _power(r):
        iid = r.choice(list(INSTANCES))
        want = "stopped" if INSTANCES[iid]["state"] == "running" else "running"
        verb = "Stop" if want == "stopped" else "Start"
        return eval_aws.Case(
            f"power_{iid}", f"{verb} the EC2 instance {iid}.",
            {"buckets": {}, "instances": _instances(), "roles": {}},
            lambda code, out, st, iid=iid, want=want: code == 0
            and st["instances"][iid]["state"] == want)

    @maker
    def _tag(r):
        iid = r.choice(list(INSTANCES))
        k, v = r.choice(TAGS)
        return eval_aws.Case(
            f"tag_{iid}", f"Tag the EC2 instance {iid} with {k}={v}.",
            {"buckets": {}, "instances": _instances(), "roles": {}},
            lambda code, out, st, iid=iid, k=k, v=v: code == 0
            and st["instances"][iid]["tags"].get(k) == v)

    @maker
    def _role(r):
        name = r.choice(ROLES)
        return eval_aws.Case(
            f"role_{name}", f"Create an IAM role called {name}.",
            {"buckets": {}, "instances": _instances(), "roles": {}},
            lambda code, out, st, name=name: code == 0 and name in st["roles"])

    @maker
    def _delrole(r):
        name = r.choice(ROLES)
        return eval_aws.Case(
            f"delrole_{name}", f"Remove the IAM role {name}.",
            {"buckets": {}, "instances": _instances(), "roles": {name: {}}},
            lambda code, out, st, name=name: code == 0 and name not in st["roles"])

    for i in range(n):
        tasks.append(makers[i % len(makers)](rng))
    return tasks


def attempt(model, tokenizer, generate_fn, case, history, max_tokens=48):
    """One attempt at one task, given whatever errors came before."""
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "system", "content": eval_aws.SYSTEM},
                {"role": "user", "content": case.ask}]
    for command, error in history:
        messages.append({"role": "assistant", "content": command})
        messages.append({"role": "user", "content":
                         f"That failed: {error}\nTry again. Command only."})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    reply = generate_fn(model, tokenizer, prompt=prompt,
                        sampler=make_sampler(temp=0.0 if not history else 0.7),
                        max_tokens=max_tokens, verbose=False)
    command = eval_aws.extract_command(reply)
    if not command:
        return "", False, "no awsim command in the reply"
    awsim.reset_state(case.seed)
    code, out = awsim.run(command.split()[1:])
    state = awsim.load_state()
    try:
        ok = bool(case.verify(code, out, state))
    except Exception as exc:
        return command, False, f"harness error: {exc}"
    if ok:
        return command, True, out
    return command, False, (out or "the command ran but did not achieve the task")


def teach(model, tokenizer, generate_fn, tasks, tries: int = 4):
    """Run the loop. Returns the samples the model earned by execution."""
    samples, transcript = [], []
    solved = 0
    for n, case in enumerate(tasks, 1):
        history = []
        for attempt_no in range(1, tries + 1):
            command, ok, detail = attempt(
                model, tokenizer, generate_fn, case, history)
            transcript.append({"task": case.id, "ask": case.ask,
                               "attempt": attempt_no, "command": command,
                               "ok": ok, "detail": detail[:200]})
            if ok:
                # Earned, not given: this pair exists because the simulator ran
                # it and the world changed the way the task asked.
                samples.append({"messages": [
                    {"role": "system", "content": eval_aws.SYSTEM},
                    {"role": "user", "content": case.ask},
                    {"role": "assistant", "content": command},
                ]})
                solved += 1
                print(f"  [{n}/{len(tasks)}] {case.id:22} solved on try "
                      f"{attempt_no}: {command}", flush=True)
                break
            history.append((command or "(nothing)", detail))
        else:
            print(f"  [{n}/{len(tasks)}] {case.id:22} UNSOLVED after {tries}: "
                  f"{history[-1][1][:70]}", flush=True)
    return samples, transcript, solved


def main():
    from mlx_lm import generate as generate_fn
    from mlx_lm import load

    from symbio.app import config as app_config

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cfg = app_config.load_config()
    print(f"loading {cfg['model_name'].split('/')[-1]} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(cfg["model_name"])
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    print("\n--- BEFORE: held-out battery on the base model ---", flush=True)
    before, _rows = eval_aws.run_battery(model, tokenizer, generate_fn)
    print(f"base model: {before}/{len(eval_aws.CASES)}")

    tasks = make_tasks(random.Random(11), count)
    print(f"\n--- self-teaching over {len(tasks)} generated tasks ---", flush=True)
    t0 = time.time()
    samples, transcript, solved = teach(model, tokenizer, generate_fn, tasks)
    print(f"\nsolved {solved}/{len(tasks)} tasks in {time.time() - t0:.0f}s; "
          f"{len(samples)} samples earned")

    with CORPUS.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with TRANSCRIPT.open("w", encoding="utf-8") as fh:
        for row in transcript:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"corpus -> {CORPUS}")
    print(f"transcript -> {TRANSCRIPT}")
    print(f"\nBASELINE (held-out): {before}/{len(eval_aws.CASES)}")


if __name__ == "__main__":
    main()
