"""The workers a fresh install starts with.

They used to live only in symbio/app/worker_models.json, which was committed —
and that file is also where every skill the user saves is written. The shipped
copy had grown 21 of one person's skills in it: Coffee Making, Bicycle Tuning,
Repotting a Houseplant, Pitch My Tent In Wind, one of them carrying an absolute
/Users/... path. Anyone installing Symbio got a worker roster advertising
specialists that do not exist on their machine, and the roster goes into the
system prompt, so the model was told it had them.

Same rule as tools/ and commands/, and for the same reason: the defaults live
in code, the file on disk is this install's, and it is seeded on first use and
never committed.
"""

from __future__ import annotations

import json
from typing import Any

from symbio import constants

BUILTIN_WORKERS: dict[str, dict[str, Any]] = {   'summarize_worker': {   'model_name': 'mlx-community/Qwen3.5-4B-MLX-4bit',
                            'role': 'summarize',
                            'description': 'Condense page/document text handed off by '
                                           'the headmaster into a short summary.',
                            'adapter_compatible': True,
                            'memory_note': '~2.3 GB on disk, ~2.5-3 GB RAM at runtime'},
    'browser_worker': {   'model_name': 'mlx-community/Qwen3.5-4B-MLX-4bit',
                          'role': 'browser',
                          'description': 'Decide the next click/type/scroll action '
                                         'from the current page text.',
                          'adapter_compatible': True,
                          'memory_note': '~2.3 GB on disk, ~2.5-3 GB RAM at runtime'},
    'second_opinion_worker': {   'model_name': 'mlx-community/phi-4-4bit',
                                 'role': 'second_opinion',
                                 'description': 'Re-work a question independently on a '
                                                'different model family and say '
                                                "whether the headmaster's answer "
                                                'holds.',
                                 'adapter_compatible': False,
                                 'memory_note': '~7.7 GB on disk, ~9 GB RAM; needs '
                                                'headmaster_deep_sleep_while_workers',
                                 'system_prompt': 'You are a reviewer running on a '
                                                  'different model from the one that '
                                                  'produced the answer you are given.\n'
                                                  '\n'
                                                  'You will receive a QUESTION and a '
                                                  'PROPOSED ANSWER. Work the question '
                                                  'out yourself from scratch. Do not '
                                                  'assume the proposed answer is '
                                                  'right, and do not assume it is '
                                                  'wrong.\n'
                                                  '\n'
                                                  'Reply in exactly two lines:\n'
                                                  'VERDICT: correct | incorrect\n'
                                                  'ANSWER: <your own answer, just the '
                                                  'value>',
                                 'advisory': True}}


def seed_worker_catalog() -> dict[str, dict[str, Any]]:
    """Write the built-in roster if there is no catalog yet, and return it.

    Never overwrites: a file that exists is this install's, skills and all.
    """
    path = constants.WORKER_MODELS_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(BUILTIN_WORKERS, indent=2) + "\n",
                        encoding="utf-8")
    except OSError:
        # A read-only install still gets the roster, just not a file.
        pass
    return dict(BUILTIN_WORKERS)
