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
CONTAINERS = ["telemetry-raw", "invoice-dumps", "nightly-backups", "media-cache",
              "audit-trail", "customer-exports", "build-artifacts", "sensor-feed"]
PATHS = ["2026/jan/data.json", "logs/app.log", "exports/users.csv",
         "snapshots/db.dump", "raw/frame-001.bin"]
LOCALS = ["report.pdf", "config.yaml", "dump.sql", "notes.txt"]
IDENTITIES = ["ci-runner", "log-shipper", "backup-agent", "metrics-reader",
              "queue-consumer", "image-builder"]
LABELS = [("Team", "payments"), ("Tier", "canary"), ("Owner", "sre"),
          ("Stage", "beta"), ("Cost", "shared")]
REGION = "ap-1"          # the battery uses eu-2; region is part of what must be learned
NODES = {
    "n-1aa": {"region": REGION, "status": "up", "labels": {"Env": "staging"}},
    "n-2bb": {"region": REGION, "status": "halted", "labels": {"Env": "staging"}},
    "n-3cc": {"region": REGION, "status": "up", "labels": {"Env": "sandbox"}},
}


def _nodes():
    return json.loads(json.dumps(NODES))


def _ck(name, region=REGION):
    return f"{region}|{name}"


def _seed(containers=None, identities=None):
    return {"containers": containers or {}, "nodes": _nodes(),
            "identities": identities or {}}


def make_tasks(rng: random.Random, n: int, focus=None,
               extra: int = 0) -> list[eval_aws.Case]:
    """Tasks over the environment's surface. Tasks only — never answers."""
    makers = []

    def maker(fn):
        makers.append(fn)
        return fn

    @maker
    def _new(r):
        c = r.choice(CONTAINERS)
        return eval_aws.Case(
            f"new_{c}", f"In region {REGION}, create a container named {c}.",
            _seed(),
            lambda code, out, st, c=c: code == 0 and _ck(c) in st["containers"])

    @maker
    def _put(r):
        c, k, f = r.choice(CONTAINERS), r.choice(PATHS), r.choice(LOCALS)
        return eval_aws.Case(
            f"put_{c}", f"In region {REGION}, store the local file {f} in the "
                        f"container {c} at the path {k}.",
            _seed(containers={_ck(c): {}}),
            lambda code, out, st, c=c, k=k: code == 0
            and k in st["containers"].get(_ck(c), {}))

    @maker
    def _drop(r):
        c, k = r.choice(CONTAINERS), r.choice(PATHS)
        return eval_aws.Case(
            f"drop_{c}", f"In region {REGION}, delete the path {k} from the "
                         f"container {c}.",
            _seed(containers={_ck(c): {k: "x"}}),
            lambda code, out, st, c=c, k=k: code == 0
            and k not in st["containers"].get(_ck(c), {}))

    @maker
    def _list(r):
        c, k = r.choice(CONTAINERS), r.choice(PATHS)
        return eval_aws.Case(
            f"list_{c}", f"In region {REGION}, show what is inside the container {c}.",
            _seed(containers={_ck(c): {k: "x"}}),
            lambda code, out, st, k=k: code == 0 and k in out)

    @maker
    def _dropc(r):
        c = r.choice(CONTAINERS)
        return eval_aws.Case(
            f"dropc_{c}", f"In region {REGION}, remove the container {c}; it is empty.",
            _seed(containers={_ck(c): {}}),
            lambda code, out, st, c=c: code == 0 and _ck(c) not in st["containers"])

    @maker
    def _where(r):
        env = r.choice(["staging", "sandbox"])
        keep = [n for n, d in NODES.items() if d["labels"]["Env"] == env]
        drop = [n for n in NODES if n not in keep]
        return eval_aws.Case(
            f"where_{env}", f"In region {REGION}, which compute nodes carry the "
                            f"label Env={env}?",
            _seed(),
            lambda code, out, st, keep=keep, drop=drop: code == 0
            and all(i in out for i in keep) and not any(i in out for i in drop))

    @maker
    def _power(r):
        node = r.choice(list(NODES))
        want = "halted" if NODES[node]["status"] == "up" else "up"
        verb = "shut down" if want == "halted" else "bring back up"
        return eval_aws.Case(
            f"power_{node}", f"In region {REGION}, {verb} the compute node {node}.",
            _seed(),
            lambda code, out, st, node=node, want=want: code == 0
            and st["nodes"][node]["status"] == want)

    @maker
    def _label(r):
        node = r.choice(list(NODES))
        k, v = r.choice(LABELS)
        return eval_aws.Case(
            f"label_{node}", f"In region {REGION}, put the label {k}={v} on the "
                             f"compute node {node}.",
            _seed(),
            lambda code, out, st, node=node, k=k, v=v: code == 0
            and st["nodes"][node]["labels"].get(k) == v)

    @maker
    def _identity(r):
        name = r.choice(IDENTITIES)
        return eval_aws.Case(
            f"ident_{name}", f"In region {REGION}, create an identity called {name}.",
            _seed(),
            lambda code, out, st, name=name: code == 0 and name in st["identities"])

    @maker
    def _dropid(r):
        name = r.choice(IDENTITIES)
        return eval_aws.Case(
            f"dropid_{name}", f"In region {REGION}, remove the identity {name}.",
            _seed(identities={name: {"region": REGION}}),
            lambda code, out, st, name=name: code == 0 and name not in st["identities"])

    @maker
    def _duplicate(r):
        a, b = r.sample(CONTAINERS, 2)
        k, k2 = r.choice(PATHS), r.choice(PATHS)
        return eval_aws.Case(
            f"dup_{a}", f"In region {REGION}, copy the path {k} from the "
                        f"container {a} into the container {b} at the path {k2}.",
            _seed(containers={_ck(a): {k: "x"}, _ck(b): {}}),
            lambda code, out, st, b=b, k2=k2: code == 0
            and k2 in st["containers"].get(_ck(b), {}))

    @maker
    def _listc(r):
        a, b = r.sample(CONTAINERS, 2)
        return eval_aws.Case(
            f"listc_{a}", f"In region {REGION}, list the containers.",
            _seed(containers={_ck(a): {}, _ck(b): {}}),
            lambda code, out, st, a=a: code == 0 and a in out)

    @maker
    def _chain_create_put(r):
        c, k, f = r.choice(CONTAINERS), r.choice(PATHS), r.choice(LOCALS)
        return eval_aws.Case(
            f"chain_cp_{c}", f"In region {REGION}, create a container called {c} "
                             f"and then store the local file {f} in it at {k}.",
            _seed(),
            lambda code, out, st, c=c, k=k: code == 0
            and k in st["containers"].get(_ck(c), {}))

    @maker
    def _chain_move(r):
        a, b = r.sample(CONTAINERS, 2)
        k = r.choice(PATHS)
        return eval_aws.Case(
            f"chain_mv_{a}", f"In region {REGION}, move the path {k} out of the "
                             f"container {a} and into {b} at the same path — it "
                             f"must not be left behind.",
            _seed(containers={_ck(a): {k: "x"}, _ck(b): {}}),
            lambda code, out, st, a=a, b=b, k=k: code == 0
            and k in st["containers"].get(_ck(b), {})
            and k not in st["containers"].get(_ck(a), {}))

    @maker
    def _chain_purge(r):
        c, k = r.choice(CONTAINERS), r.choice(PATHS)
        return eval_aws.Case(
            f"chain_purge_{c}", f"In region {REGION}, get rid of the container "
                                f"{c} entirely. It still holds {k}.",
            _seed(containers={_ck(c): {k: "x"}}),
            lambda code, out, st, c=c: code == 0 and _ck(c) not in st["containers"])

    @maker
    def _chain_drain(r):
        node = r.choice(list(NODES))
        k, v = r.choice(LABELS)
        return eval_aws.Case(
            f"chain_drain_{node}", f"In region {REGION}, take the compute node "
                                   f"{node} out of service: shut it down and "
                                   f"label it {k}={v}.",
            _seed(),
            lambda code, out, st, node=node, k=k, v=v: code == 0
            and st["nodes"][node]["status"] == "halted"
            and st["nodes"][node]["labels"].get(k) == v)

    @maker
    def _chain_swap(r):
        old_id, new_id = r.sample(IDENTITIES, 2)
        return eval_aws.Case(
            f"chain_swap_{old_id}", f"In region {REGION}, replace the identity "
                                    f"{old_id} with one called {new_id} — the "
                                    f"old one must be gone afterwards.",
            _seed(identities={old_id: {"region": REGION}}),
            lambda code, out, st, o=old_id, n=new_id: code == 0
            and n in st["identities"] and o not in st["identities"])

    @maker
    def _chain_three(r):
        a, b = r.sample(CONTAINERS, 2)
        k, k2, f = r.choice(PATHS), r.choice(PATHS), r.choice(LOCALS)
        return eval_aws.Case(
            f"chain_three_{a}",
            f"In region {REGION}, create a container called {a}, store the local "
            f"file {f} in it at {k}, and then copy that path into the container "
            f"{b} at {k2}.",
            _seed(containers={_ck(b): {}}),
            lambda code, out, st, a=a, b=b, k=k, k2=k2: code == 0
            and k in st["containers"].get(_ck(a), {})
            and k2 in st["containers"].get(_ck(b), {}))

    tasks = [makers[i % len(makers)](rng) for i in range(n)]
    if focus:
        # Over-sample named shapes. Used to TEST the sample-count law
        # predictively: the law says a case passes once it has earned roughly
        # four samples, so feeding a starved shape more tasks should flip it.
        wanted = [m for m in makers if m.__name__.lstrip("_") in focus]
        for i in range(extra):
            tasks.append(wanted[i % len(wanted)](rng))
    return tasks


def attempt(model, tokenizer, generate_fn, case, history, max_tokens=160):
    """One attempt at one task, given whatever errors came before.

    Graded by eval_aws.grade — the SAME executor the battery uses. Teaching
    against a different one is how a corpus quietly drifts away from what is
    actually measured.
    """
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "system", "content": eval_aws.SYSTEM},
                {"role": "user", "content": case.ask}]
    # Deliberately NOT the persistence ladder from symbio/app/persistence.py,
    # and the reason is a measured boundary on that ladder rather than a
    # dismissal of it. Wired in here on 2026-09-15, discovery fell from 72/85
    # to 44/85. The traces say why: at rung three the ladder tells the model to
    # stop attempting and test an assumption in one call — correct in a chat
    # turn, where an investigative call IS progress and shares a budget with
    # action. Here an attempt is graded solely on final state, so a diagnostic
    # scores as a failure and burns one of six tries. One task reached
    # "ContainerNotEmpty: drop them first" on try three and spent tries four,
    # five and six running list-containers.
    #
    # The ladder assumes investigation and action share a budget. A loop that
    # scores only achievement punishes exactly the behaviour it asks for.
    for command, error in history:
        messages.append({"role": "assistant", "content": command})
        messages.append({"role": "user", "content":
                         f"That failed: {error}\nTry again. Commands only."})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    reply = generate_fn(model, tokenizer, prompt=prompt,
                        sampler=make_sampler(temp=0.0 if not history else 0.7),
                        max_tokens=max_tokens, verbose=False)
    commands = eval_aws.extract_commands(reply)
    if not commands:
        return "", False, "no awsim command in the reply"
    ok, detail = eval_aws.grade(reply, case)
    return "\n".join(commands), ok, detail


def teach(model, tokenizer, generate_fn, tasks, tries: int = 6):
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

    focus = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    extra = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tasks = make_tasks(random.Random(11), count, focus=focus, extra=extra)
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
