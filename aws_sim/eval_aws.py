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

INSTANCES = {
    "i-0abc123": {"state": "running", "tags": {"Env": "prod", "Name": "web-1"}},
    "i-0def456": {"state": "stopped", "tags": {"Env": "dev", "Name": "batch-1"}},
    "i-0ghi789": {"state": "running", "tags": {"Env": "prod", "Name": "web-2"}},
}


def _seed(buckets=None, instances=True, roles=None):
    return {
        "buckets": buckets or {},
        "instances": dict(INSTANCES) if instances else {},
        "roles": roles or {},
    }


class Case:
    def __init__(self, cid, ask, seed, verify):
        self.id, self.ask, self.seed, self.verify = cid, ask, seed, verify


CASES: list[Case] = [
    Case("s3_make_bucket",
         "Create an S3 bucket called quarterly-reports.",
         _seed(),
         lambda code, out, st: code == 0 and "quarterly-reports" in st["buckets"]),
    Case("s3_upload",
         "Upload the local file summary.csv to the bucket quarterly-reports "
         "under the key q3/summary.csv.",
         _seed(buckets={"quarterly-reports": {}}),
         lambda code, out, st: code == 0
         and "q3/summary.csv" in st["buckets"].get("quarterly-reports", {})),
    Case("s3_list_bucket",
         "List everything stored in the bucket quarterly-reports.",
         _seed(buckets={"quarterly-reports": {"q3/summary.csv": "x"}}),
         lambda code, out, st: code == 0 and "q3/summary.csv" in out),
    Case("s3_remove_object",
         "Delete the object q3/summary.csv from the bucket quarterly-reports.",
         _seed(buckets={"quarterly-reports": {"q3/summary.csv": "x"}}),
         lambda code, out, st: code == 0
         and "q3/summary.csv" not in st["buckets"].get("quarterly-reports", {})),
    Case("s3_remove_bucket",
         "Remove the empty bucket old-logs.",
         _seed(buckets={"old-logs": {}}),
         lambda code, out, st: code == 0 and "old-logs" not in st["buckets"]),
    Case("s3_copy_between",
         "Copy s3://quarterly-reports/q3/summary.csv to "
         "s3://archive/2026/summary.csv.",
         _seed(buckets={"quarterly-reports": {"q3/summary.csv": "x"},
                        "archive": {}}),
         lambda code, out, st: code == 0
         and "2026/summary.csv" in st["buckets"].get("archive", {})),
    Case("ec2_filter_by_tag",
         "Show me the EC2 instances tagged Env=prod.",
         _seed(),
         lambda code, out, st: code == 0 and "i-0abc123" in out
         and "i-0def456" not in out),
    Case("ec2_stop_instance",
         "Stop the EC2 instance i-0abc123.",
         _seed(),
         lambda code, out, st: code == 0
         and st["instances"]["i-0abc123"]["state"] == "stopped"),
    Case("ec2_start_instance",
         "Start the EC2 instance i-0def456.",
         _seed(),
         lambda code, out, st: code == 0
         and st["instances"]["i-0def456"]["state"] == "running"),
    Case("ec2_tag_instance",
         "Tag the EC2 instance i-0def456 with Owner=platform.",
         _seed(),
         lambda code, out, st: code == 0
         and st["instances"]["i-0def456"]["tags"].get("Owner") == "platform"),
    Case("iam_create_role",
         "Create an IAM role named deployment-bot.",
         _seed(),
         lambda code, out, st: code == 0 and "deployment-bot" in st["roles"]),
    Case("iam_list_roles",
         "List the IAM roles.",
         _seed(roles={"deployment-bot": {"description": ""}}),
         lambda code, out, st: code == 0 and "deployment-bot" in out),
    Case("iam_delete_role",
         "Delete the IAM role stale-role.",
         _seed(roles={"stale-role": {"description": ""}}),
         lambda code, out, st: code == 0 and "stale-role" not in st["roles"]),
]

SYSTEM = (
    "You drive a command-line tool called awsim, which mimics the AWS CLI and "
    "is STRICT: option names, S3 URIs and filter syntax must be exact.\n"
    "Reply with the single command and nothing else. Do not explain it.\n"
    "Services and operations available:\n"
    "  awsim s3 ls [s3://bucket[/prefix]]\n"
    "  awsim s3 mb s3://bucket\n"
    "  awsim s3 rb s3://bucket\n"
    "  awsim s3 cp <source> <destination>\n"
    "  awsim s3 rm s3://bucket/key\n"
    "  awsim ec2 describe-instances [--filters Name=tag:<Tag>,Values=<v>] "
    "[--instance-ids <id>[,<id>]]\n"
    "  awsim ec2 start-instances --instance-ids <id>[,<id>]\n"
    "  awsim ec2 stop-instances --instance-ids <id>[,<id>]\n"
    "  awsim ec2 create-tags --resources <id> --tags Key=<K>,Value=<V>\n"
    "  awsim iam create-role --role-name <name> [--description <text>]\n"
    "  awsim iam list-roles\n"
    "  awsim iam delete-role --role-name <name>\n"
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


def run_battery(model, tokenizer, generate_fn, max_tokens: int = 48,
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
        "s3_make_bucket": "awsim s3 mb s3://quarterly-reports",
        "s3_upload": "awsim s3 cp summary.csv s3://quarterly-reports/q3/summary.csv",
        "s3_list_bucket": "awsim s3 ls s3://quarterly-reports",
        "s3_remove_object": "awsim s3 rm s3://quarterly-reports/q3/summary.csv",
        "s3_remove_bucket": "awsim s3 rb s3://old-logs",
        "s3_copy_between": "awsim s3 cp s3://quarterly-reports/q3/summary.csv s3://archive/2026/summary.csv",
        "ec2_filter_by_tag": "awsim ec2 describe-instances --filters Name=tag:Env,Values=prod",
        "ec2_stop_instance": "awsim ec2 stop-instances --instance-ids i-0abc123",
        "ec2_start_instance": "awsim ec2 start-instances --instance-ids i-0def456",
        "ec2_tag_instance": "awsim ec2 create-tags --resources i-0def456 --tags Key=Owner,Value=platform",
        "iam_create_role": "awsim iam create-role --role-name deployment-bot",
        "iam_list_roles": "awsim iam list-roles",
        "iam_delete_role": "awsim iam delete-role --role-name stale-role",
    }
    bad = 0
    for case in CASES:
        ok, detail = grade(REFERENCE[case.id], case)
        print(f"  {'ok  ' if ok else 'BAD '} {case.id:20} {detail[:92]}")
        bad += not ok
    print(f"\n{len(CASES) - bad}/{len(CASES)} reference answers pass")
    raise SystemExit(1 if bad else 0)
