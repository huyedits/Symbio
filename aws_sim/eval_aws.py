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

# Deliberately says nothing about the syntax. Listing the verbs and options
# here would turn the battery into a copying exercise, which is exactly what
# the first version of this environment accidentally measured — the model
# scored 13/13 before any training because it was reciting the real AWS CLI.
# What the model knows about awsim has to come from its own failed attempts.
SYSTEM = (
    "You drive a command-line tool called awsim. Reply with the single "
    "command to run and nothing else — no explanation, no code fence."
)


# The command out of whatever wrapping the model put round it — a fenced block,
# a <cmd> tag, a tool call, or bare prose. Anchored on the tool's own name so a
# sentence about awsim does not read as a command.
_COMMAND_RE = re.compile(r"(?:^|[\s`\"'>\](}])(awsim\s+[^\n`\"']+)")


def extract_command(reply: str) -> str:
    """The awsim command a reply is proposing, or ""."""
    text = (reply or "").replace("\\\n", " ")
    match = _COMMAND_RE.search(text)
    if not match:
        return ""
    command = match.group(1).strip().rstrip("`\"';.")
    # A model that emitted several lines gets its first command graded.
    return command.split("\n")[0].strip()


def grade(reply: str, case: Case) -> tuple[bool, str]:
    """Run what the reply proposes against a freshly seeded simulator."""
    command = extract_command(reply)
    if not command:
        return False, "(no awsim command in the reply)"
    awsim.reset_state(case.seed)
    try:
        args = command.split()[1:]
        code, out = awsim.run(args)
        state = awsim.load_state()
        ok = bool(case.verify(code, out, state))
    except Exception as exc:
        return False, f"{command}  -> harness error: {exc}"
    return ok, f"{command}  -> [{code}] {out.splitlines()[0] if out else ''}"


def run_battery(model, tokenizer, generate_fn, max_tokens: int = 64,
                verbose: bool = True) -> tuple[int, list[tuple[str, bool, str]]]:
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    rows = []
    for case in CASES:
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
    bad = 0
    for case in CASES:
        ok, detail = grade(REFERENCE[case.id], case)
        print(f"  {'ok  ' if ok else 'BAD '} {case.id:20} {detail[:92]}")
        bad += not ok
    print(f"\n{len(CASES) - bad}/{len(CASES)} reference answers pass")
    raise SystemExit(1 if bad else 0)
