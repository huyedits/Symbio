#!/usr/bin/env python3
"""Grade a model on the rigid AWS environment by RUNNING what it emits.

Not by keyword. A reply containing "s3 mb" scores nothing here; a reply whose
command, executed against a freshly seeded simulator, leaves the bucket
actually created scores one. That distinction is the whole point — this project
has already learned once that a keyword rubric cannot grade code, and a command
line is code.

Each case carries its own seed state, so cases cannot contaminate each other
and a checkpoint that scores better really is better rather than luckier.

Deliberately cheap: one short greedy completion per case, capped at 48 tokens,
because this battery has to run against fifty checkpoints. The headmaster's own
golden battery costs ten minutes a pass, which would be eight hours across a
1000-iteration run and is why this one is built separately.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import awsim

NODES = {
    "n-4kq": {"region": "eu-2", "status": "up", "labels": {"Env": "prod", "Tier": "web"}},
    "n-7wz": {"region": "eu-2", "status": "halted", "labels": {"Env": "dev"}},
    "n-9tm": {"region": "eu-2", "status": "up", "labels": {"Env": "prod", "Tier": "db"}},
}


def _seed(containers=None, identities=None):
    import json as _json
    return {"containers": containers or {},
            "nodes": _json.loads(_json.dumps(NODES)),
            "identities": identities or {}}


class Case:
    def __init__(self, cid, ask, seed, verify):
        self.id, self.ask, self.seed, self.verify = cid, ask, seed, verify


def _ck(name, region="eu-2"):
    return f"{region}|{name}"


CASES: list[Case] = [
    Case("new_container",
         "In region eu-2, create a container called quarterly-reports.",
         _seed(),
         lambda code, out, st: code == 0 and _ck("quarterly-reports") in st["containers"]),
    Case("put_object",
         "In region eu-2, store the local file summary.csv into the container "
         "quarterly-reports at the path q3/summary.csv.",
         _seed(containers={_ck("quarterly-reports"): {}}),
         lambda code, out, st: code == 0
         and "q3/summary.csv" in st["containers"].get(_ck("quarterly-reports"), {})),
    Case("list_container",
         "In region eu-2, show me what is inside the container quarterly-reports.",
         _seed(containers={_ck("quarterly-reports"): {"q3/summary.csv": "x"}}),
         lambda code, out, st: code == 0 and "q3/summary.csv" in out),
    Case("drop_path",
         "In region eu-2, delete q3/summary.csv from the container quarterly-reports.",
         _seed(containers={_ck("quarterly-reports"): {"q3/summary.csv": "x"}}),
         lambda code, out, st: code == 0
         and "q3/summary.csv" not in st["containers"].get(_ck("quarterly-reports"), {})),
    Case("drop_container",
         "In region eu-2, remove the empty container old-logs.",
         _seed(containers={_ck("old-logs"): {}}),
         lambda code, out, st: code == 0 and _ck("old-logs") not in st["containers"]),
    Case("duplicate",
         "In region eu-2, copy the path q3/summary.csv from the container "
         "quarterly-reports into the container archive at the path 2026/summary.csv.",
         _seed(containers={_ck("quarterly-reports"): {"q3/summary.csv": "x"},
                           _ck("archive"): {}}),
         lambda code, out, st: code == 0
         and "2026/summary.csv" in st["containers"].get(_ck("archive"), {})),
    Case("list_containers",
         "In region eu-2, list the containers.",
         _seed(containers={_ck("quarterly-reports"): {}, _ck("archive"): {}}),
         lambda code, out, st: code == 0 and "archive" in out),
    Case("filter_nodes",
         "In region eu-2, which compute nodes carry the label Env=prod?",
         _seed(),
         lambda code, out, st: code == 0 and "n-4kq" in out and "n-7wz" not in out),
    Case("halt_node",
         "In region eu-2, shut down the compute node n-4kq.",
         _seed(),
         lambda code, out, st: code == 0 and st["nodes"]["n-4kq"]["status"] == "halted"),
    Case("resume_node",
         "In region eu-2, bring the compute node n-7wz back up.",
         _seed(),
         lambda code, out, st: code == 0 and st["nodes"]["n-7wz"]["status"] == "up"),
    Case("label_node",
         "In region eu-2, put the label Owner=platform on the compute node n-7wz.",
         _seed(),
         lambda code, out, st: code == 0
         and st["nodes"]["n-7wz"]["labels"].get("Owner") == "platform"),
    Case("new_identity",
         "In region eu-2, create an identity named deployment-bot.",
         _seed(),
         lambda code, out, st: code == 0 and "deployment-bot" in st["identities"]),
    Case("drop_identity",
         "In region eu-2, remove the identity stale-role.",
         _seed(identities={"stale-role": {"region": "eu-2"}}),
         lambda code, out, st: code == 0 and "stale-role" not in st["identities"]),
]

# Chained cases: the answer is a SEQUENCE, and step two depends on step one
# having happened. These are the cases a single-command benchmark cannot ask —
# ordering, carried state, and a precondition the model has to create itself.
# Marked by id prefix so single-step and chained scores can be reported apart.
CHAIN_CASES: list[Case] = [
    Case("chain_create_then_put",
         "In region eu-2, create a container called release-notes and then "
         "store the local file v2.md in it at the path drafts/v2.md.",
         _seed(),
         lambda code, out, st: code == 0
         and "drafts/v2.md" in st["containers"].get(_ck("release-notes"), {})),
    Case("chain_move",
         "In region eu-2, move the path q3/summary.csv out of the container "
         "quarterly-reports and into the container archive at the same path — "
         "it must not be left behind.",
         _seed(containers={_ck("quarterly-reports"): {"q3/summary.csv": "x"},
                           _ck("archive"): {}}),
         lambda code, out, st: code == 0
         and "q3/summary.csv" in st["containers"].get(_ck("archive"), {})
         and "q3/summary.csv" not in st["containers"].get(_ck("quarterly-reports"), {})),
    Case("chain_empty_then_drop",
         "In region eu-2, get rid of the container scratch-space entirely. It "
         "still has the path tmp/a.bin in it.",
         _seed(containers={_ck("scratch-space"): {"tmp/a.bin": "x"}}),
         lambda code, out, st: code == 0
         and _ck("scratch-space") not in st["containers"]),
    Case("chain_halt_then_label",
         "In region eu-2, take the compute node n-4kq out of service: shut it "
         "down and label it Status:drained.",
         _seed(),
         lambda code, out, st: code == 0
         and st["nodes"]["n-4kq"]["status"] == "halted"
         and st["nodes"]["n-4kq"]["labels"].get("Status") == "drained"),
    Case("chain_replace_identity",
         "In region eu-2, replace the identity old-runner with one called "
         "new-runner — the old one must be gone afterwards.",
         _seed(identities={"old-runner": {"region": "eu-2"}}),
         lambda code, out, st: code == 0
         and "new-runner" in st["identities"]
         and "old-runner" not in st["identities"]),
    Case("chain_three_step",
         "In region eu-2, create a container called nightly, store the local "
         "file run.log in it at logs/run.log, and then copy that path into the "
         "container archive at nightly/run.log.",
         _seed(containers={_ck("archive"): {}}),
         lambda code, out, st: code == 0
         and "logs/run.log" in st["containers"].get(_ck("nightly"), {})
         and "nightly/run.log" in st["containers"].get(_ck("archive"), {})),
]

ALL_CASES = CASES + CHAIN_CASES

# Deliberately says nothing about the syntax. Listing the verbs and options
# here would turn the battery into a copying exercise, which is exactly what
# the first version of this environment accidentally measured — the model
# scored 13/13 before any training because it was reciting the real AWS CLI.
# What the model knows about awsim has to come from its own failed attempts.
SYSTEM = (
    "You drive a command-line tool called awsim. Reply with the command or "
    "commands to run, ONE PER LINE, in the order they should run, and nothing "
    "else — no explanation, no code fence."
)


# The command out of whatever wrapping the model put round it — a fenced block,
# a <cmd> tag, a tool call, or bare prose. Anchored on the tool's own name so a
# sentence about awsim does not read as a command.
_COMMAND_RE = re.compile(r"(?:^|[\s`\"'>\](}])(awsim\s+[^\n`\"']+)")


def extract_commands(reply: str) -> list[str]:
    """Every awsim command a reply proposes, in order.

    A chained task needs all of them: "create the container, then put the file
    in it" is two commands whose ORDER is the answer. Grading only the first
    would mark a correct two-step plan as a failure, and grading them as a set
    would let a model that got the order backwards pass.
    """
    text = (reply or "").replace("\\\n", " ")
    out = []
    for match in _COMMAND_RE.finditer(text):
        command = match.group(1).strip().rstrip("`\"';.").split("\n")[0].strip()
        if command:
            out.append(command)
    return out


def extract_command(reply: str) -> str:
    """The first command a reply proposes, or "". Kept for single-step callers."""
    found = extract_commands(reply)
    return found[0] if found else ""


def grade(reply: str, case: Case) -> tuple[bool, str]:
    """Run everything the reply proposes, in order, against ONE seeded state.

    The state is seeded once and then carried across the commands, because a
    chain is only a chain if step two sees what step one did. Execution stops
    at the first failure — a plan whose second command is wrong has not
    achieved the task, whatever the third would have done.
    """
    commands = extract_commands(reply)
    if not commands:
        return False, "(no awsim command in the reply)"
    awsim.reset_state(case.seed)
    trail, code, out = [], 0, ""
    try:
        for command in commands:
            code, out = awsim.run(command.split()[1:])
            trail.append(f"[{code}] {command}")
            if code != 0:
                break
        state = awsim.load_state()
        ok = bool(case.verify(code, out, state))
    except Exception as exc:
        return False, f"{' ; '.join(trail)}  -> harness error: {exc}"
    detail = " ; ".join(trail)
    if not ok and out:
        detail += f"  -> {out.splitlines()[0]}"
    return ok, detail


def run_battery(model, tokenizer, generate_fn, max_tokens: int = 160,
                verbose: bool = True, cases=None) -> tuple[int, list[tuple[str, bool, str]]]:
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    rows = []
    for case in (cases if cases is not None else ALL_CASES):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": case.ask}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        try:
            reply = generate_fn(model, tokenizer, prompt=prompt, sampler=sampler,
                                max_tokens=max_tokens, verbose=False)
        except Exception as exc:
            rows.append((case.id, False, f"generation error: {exc}"))
            continue
        ok, detail = grade(reply, case)
        rows.append((case.id, ok, detail))
        if verbose:
            print(f"    {'PASS' if ok else 'FAIL'} {case.id:20} {detail[:88]}",
                  flush=True)
    return sum(1 for _, ok, _ in rows if ok), rows


if __name__ == "__main__":
    # Self-check: the reference answers must all pass, or the battery is
    # grading something other than what it claims to.
    REFERENCE = {
        "new_container": "awsim store new-container --name quarterly-reports --region eu-2",
        "put_object": "awsim store put --container quarterly-reports --path q3/summary.csv --from summary.csv --region eu-2",
        "list_container": "awsim store list --container quarterly-reports --region eu-2",
        "drop_path": "awsim store drop --container quarterly-reports --path q3/summary.csv --region eu-2",
        "drop_container": "awsim store drop-container --name old-logs --region eu-2",
        "duplicate": "awsim store duplicate --from-container quarterly-reports --from-path q3/summary.csv --to-container archive --to-path 2026/summary.csv --region eu-2",
        "list_containers": "awsim store list-containers --region eu-2",
        "filter_nodes": "awsim compute list-nodes --where tag/Env:prod --region eu-2",
        "halt_node": "awsim compute halt --node n-4kq --region eu-2",
        "resume_node": "awsim compute resume --node n-7wz --region eu-2",
        "label_node": "awsim compute label --node n-7wz --label Owner:platform --region eu-2",
        "new_identity": "awsim access new-identity --identity deployment-bot --region eu-2",
        "drop_identity": "awsim access drop-identity --identity stale-role --region eu-2",
    }
    R = "--region eu-2"
    REFERENCE.update({
        "chain_create_then_put":
            f"awsim store new-container --name release-notes {R}\n"
            f"awsim store put --container release-notes --path drafts/v2.md "
            f"--from v2.md {R}",
        "chain_move":
            f"awsim store duplicate --from-container quarterly-reports "
            f"--from-path q3/summary.csv --to-container archive "
            f"--to-path q3/summary.csv {R}\n"
            f"awsim store drop --container quarterly-reports "
            f"--path q3/summary.csv {R}",
        "chain_empty_then_drop":
            f"awsim store drop --container scratch-space --path tmp/a.bin {R}\n"
            f"awsim store drop-container --name scratch-space {R}",
        "chain_halt_then_label":
            f"awsim compute halt --node n-4kq {R}\n"
            f"awsim compute label --node n-4kq --label Status:drained {R}",
        "chain_replace_identity":
            f"awsim access new-identity --identity new-runner {R}\n"
            f"awsim access drop-identity --identity old-runner {R}",
        "chain_three_step":
            f"awsim store new-container --name nightly {R}\n"
            f"awsim store put --container nightly --path logs/run.log "
            f"--from run.log {R}\n"
            f"awsim store duplicate --from-container nightly "
            f"--from-path logs/run.log --to-container archive "
            f"--to-path nightly/run.log {R}",
    })
    bad = 0
    for case in ALL_CASES:
        ok, detail = grade(REFERENCE[case.id], case)
        print(f"  {'ok  ' if ok else 'BAD '} {case.id:20} {detail[:92]}")
        bad += not ok
    print(f"\n{len(ALL_CASES) - bad}/{len(ALL_CASES)} reference answers pass "
          f"({len(CASES)} single-step, {len(CHAIN_CASES)} chained)")
    raise SystemExit(1 if bad else 0)
