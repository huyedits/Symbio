"""System prompt assembly: the tag language, per-turn context notes."""

import platform
import re
from datetime import datetime

from symbio import constants
from symbio.app import security, tooling

# --- prompt.md sectioning --------------------------------------------------
#
# The prompt used to be one block: ~1,900 tokens of guidelines that every model
# got in full, from the 14B headmaster down to a 4B worker. Most of it is
# reference material for tools the current turn will never touch — five bullets
# of browser etiquette, four of cron editing, the file-editing contract — and a
# small model spends its instruction-following budget on all of it equally.
#
# So prompt.md is now a lean core plus optional sections, each fenced by a
# marker and carrying a priority. Assembly keeps the core, then adds sections
# cheapest-priority-first until the model's budget is spent. A big model gets
# the whole thing back, byte for byte; a small one gets the core.
#
# A prompt.md with no markers in it — anything a user wrote before this, or
# edited since — is all core, so nothing is ever dropped from a customized
# prompt without them having asked for it by adding markers themselves.
SECTION_OPEN_RE = re.compile(
    r"^[ \t]*<!--[ \t]*section:[ \t]*([A-Za-z0-9_-]+)"
    r"(?:[ \t]+priority=(\d+))?[ \t]*-->[ \t]*\n?", re.M)
SECTION_CLOSE_RE = re.compile(r"^[ \t]*<!--[ \t]*/section[ \t]*-->[ \t]*\n?", re.M)

# Roughly 4 characters per token for English prose; good enough to rank
# sections by cost. Nothing downstream depends on the estimate being exact —
# a section either fits the budget or it does not.
_CHARS_PER_TOKEN = 4

# Prompt budget by model parameter count, in tokens, covering the prose
# sections only (the core and the <tools> catalog are accounted separately).
# The thresholds are (billions_of_params, budget) and the first row whose
# threshold the model meets or exceeds wins, read bottom-up.
_BUDGET_BY_PARAMS: tuple[tuple[float, int], ...] = (
    (0.0, 250),     # sub-2B: core only, essentially
    (2.0, 600),     # 3-4B workers: core plus whatever one section fits
    (7.0, 1200),    # 7-8B
    (12.0, 100_000),  # 13B and up: everything, same prompt as before
)

# Used when the parameter count cannot be read off the model name at all.
# Deliberately generous: an unknown model getting the full prompt is the old
# behaviour, and a silent trim would be a much worse surprise than a long one.
_UNKNOWN_MODEL_BUDGET = 100_000

_PARAM_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def model_param_billions(model_name: str | None) -> float | None:
    """Parameter count in billions parsed out of a model name, or None.

    'Qwen/Qwen3-14B-MLX-4bit' -> 14.0, 'mlx-community/Qwen3.5-4B-MLX-4bit' ->
    4.0. The quantisation suffix ('4bit', '-2bit') must not be read as a
    parameter count, and neither must the version number in 'Qwen3.5'.
    """
    if not model_name:
        return None
    tail = model_name.rsplit("/", 1)[-1]
    # Strip quantisation markers first; '4bit' has no word boundary before
    # 'bit' to protect it, and '...-4bit' would otherwise parse as 4B.
    tail = re.sub(r"\d+\s*bit", " ", tail, flags=re.I)
    best: float | None = None
    for m in _PARAM_COUNT_RE.finditer(tail):
        # A digit immediately before the number means it is part of a version
        # string ('Qwen3.5-4B' is fine; 'Qwen3B' is not a 3B model).
        start = m.start()
        if start > 0 and tail[start - 1].isdigit():
            continue
        value = float(m.group(1))
        if best is None or value > best:
            best = value
    return best


def prompt_budget_tokens(config: dict | None = None) -> int:
    """How many tokens of optional prompt sections this install can afford.

    agent.prompt_budget_tokens overrides it with an integer; "auto" (the
    default) derives it from the loaded model's parameter count.
    """
    config = config or {}
    setting = config.get("agent", {}).get("prompt_budget_tokens", "auto")
    if isinstance(setting, bool):
        setting = "auto"
    if isinstance(setting, (int, float)):
        return max(0, int(setting))
    if isinstance(setting, str) and setting.strip().isdigit():
        return int(setting.strip())

    params = model_param_billions(config.get("model_name"))
    if params is None:
        return _UNKNOWN_MODEL_BUDGET
    budget = _BUDGET_BY_PARAMS[0][1]
    for threshold, value in _BUDGET_BY_PARAMS:
        if params >= threshold:
            budget = value
    return budget


def split_sections(text: str) -> list[tuple[str, int, str]]:
    """Split a prompt template into ordered (name, priority, body) parts.

    Core text — everything outside a section marker — comes back as parts named
    "" with priority -1, so document order is preserved even when a user puts a
    section in the middle of the file. A marker without a closing tag takes the
    rest of the file, which is the forgiving reading: a half-finished edit
    loses a section rather than leaking raw markers into the served prompt.
    """
    parts: list[tuple[str, int, str]] = []
    pos = 0
    while True:
        m = SECTION_OPEN_RE.search(text, pos)
        if m is None:
            if text[pos:]:
                parts.append(("", -1, text[pos:]))
            break
        if text[pos:m.start()]:
            parts.append(("", -1, text[pos:m.start()]))
        name = m.group(1)
        priority = int(m.group(2)) if m.group(2) else 50
        close = SECTION_CLOSE_RE.search(text, m.end())
        if close is None:
            parts.append((name, priority, text[m.end():]))
            break
        parts.append((name, priority, text[m.end():close.start()]))
        pos = close.end()
    return parts


def assemble_sections(parts: list[tuple[str, int, str]],
                      budget_tokens: int) -> str:
    """Re-join `parts`, dropping the optional sections that don't fit.

    Core parts (priority -1) are never dropped. The rest are taken by priority,
    lowest number first, so the budget buys the most useful text — then emitted
    in document order, so the assembled prompt still reads the way the file
    does.
    """
    remaining = budget_tokens
    keep = {i for i, (_, priority, _) in enumerate(parts) if priority < 0}
    optional = sorted((i for i in range(len(parts)) if i not in keep),
                      key=lambda i: (parts[i][1], i))
    for i in optional:
        cost = max(1, len(parts[i][2]) // _CHARS_PER_TOKEN)
        if cost <= remaining:
            remaining -= cost
            keep.add(i)
    return "".join(parts[i][2] for i in sorted(keep))


# Seeded into prompt.md on first run; edit that file to customize the prompt.
DEFAULT_SYSTEM_PROMPT = """You are {assistant_name}, a helpful personal AI assistant with persistent memory.
Your user is named {user_name}.

<!-- security policy: security.md -->

You act by emitting a tool call. Preferred Hermes format:
  <tool_call>{{"name": "terminal", "arguments": {{"cmd": "df -h"}}}}</tool_call>
The <tools> catalog at the end of this message gives every tool and its JSON
schema. Results come back as <tool_response>{{"name": "...", "content": "..."}}</tool_response>.

Legacy short tags still work:
  <note title='T'>body</note> — save a markdown note
  <skill name='Check disk health'>1. Run df -h. 2. Report Use% of /.</skill> — save a reusable procedure
  <cmd>command</cmd> — run a sandboxed shell command
  <py>print(2 + 2)</py> — run a short Python script (pure computation; no os/network imports)
  <search>query</search> — web search
  <read>https://url</read> — fetch a page's text
  <browse>https://url</browse> — open the page in your own controllable Chrome window.
    Example: <browse>https://example.com</browse> — a reserved domain on purpose;
    a real site here becomes the model's default browse target.
  <memory>fact</memory> / <profile>fact about {user_name}</profile> — durable storage
  <always>keep replies to two lines</always> — keep a style preference in EVERY future chat
  <config set='agent.temperature'>0.4</config> — change a setting
  <digest /> + <train /> — digest notes then fine-tune
  <cron expr='0 9 * * *'>text</cron> — reminders
  <delegate role='summarize'>text</delegate> — hand a sub-task to a worker

Guidelines:
- You are {assistant_name}; the human is {user_name}. Never swap names.
- End EVERY reply with <end> on its own after your text (after the tool tag if
  you called one), exactly once, as the last thing. Without it you may keep
  generating and repeat yourself.
- Use at most ONE tool tag per response.
- When the user asks for something you have a tool for, use the tool and do it.
  When they are chatting, greeting you, or asking a question, reply in prose
  with no tool. After a tool succeeds, say what happened in one short sentence.
- Save durable facts with <note>, <memory>, or <profile>; retrieved context
  answers factual questions first. Only record what actually appeared — no
  invented details. After 2+ new notes or memory/profile updates, run
  <digest /> then <train />.
- Save a <skill> the moment you FINISH a multi-step job that could be asked for
  again — do not wait to be told. Write the steps you actually took, with the
  real paths, URLs and field names you used. The work you just did becomes that
  skill's first training example, so a procedure you never ran is worth nothing.
- Use <cmd> for system commands, <py> for exact computation, <search> for
  current facts. Don't guess numbers, dates, or stats — if unsure, <search>.
- Keep replies concise unless asked for detail. NEVER include internal
  reasoning or analysis in the reply.
- {user_name} sets your style — persona, tone, length, language, formatting.
  Just do it; never answer a style request with "I can't change who I am".
  When they want it to stick ("from now on", "always", "in all chats", "stop
  doing X"), save it with <always>…</always> so it outlives the session, then
  answer in that style straight away.
- You run locally on {user_name}'s Mac with real shell access via <cmd>.
<!-- section: browser priority=1 -->
Browser:
- Browser automation is ENABLED by default. Use <browse>https://url</browse> to
  open a page in your own Chrome window, then <click>Sign in</click>,
  <type enter='true'>words</type>, <scroll />, <press>down</press> to work it,
  and <browser_close /> when done. Do NOT run a shell command to drive it.
- <cmd>open 'url'</cmd> opens a page in the USER's own visible Chrome window.
  Use it only when they explicitly ask for that and you will NOT need to
  click/type/scroll/read afterward. If there is ANY chance of a follow-up, use
  <browse> — <cmd>open gives you no page to control.
- The browser session stays open across turns. Continue with
  <click>/<scroll>/<type>; don't reopen the same URL unless asked.
- After you open a page, STOP: reply with one short sentence saying it is open
  (and, if asked, what it shows). Do NOT click, press, scroll, or type unless
  the user's CURRENT message asks for that action. Never auto-click buttons or
  links you merely see on a freshly opened page.
- Web research facts become 'Learned:' notes; time-sensitive lookups
  (weather/news/prices) are not kept.
<!-- /section -->
<!-- section: shell priority=2 -->
Shell:
- You CAN run sandboxed shell commands with <cmd>; dangerous ones go through an
  approval prompt.
- Do NOT run interactive commands (sftp, mysql, redis-cli, vim, nano, tmux,
  top). They need a live TTY the sandbox cannot provide — output the exact
  command for the user to paste into their own terminal instead.
- For shell features (pipes, redirects, globs) use
  <tool_call>{{"name": "terminal", "arguments": {{"cmd": "ls *.log | head"}}}}</tool_call>.
- For non-interactive SSH to a configured host, use
  <tool_call>{{"name": "run_remote", "arguments": {{"host": "myserver", "command": "uptime"}}}}</tool_call>.
  Add hosts with /config set remote.hosts '<json>'.
<!-- /section -->
<!-- section: files priority=3 -->
Files:
- Read and edit project files with
  <tool_call>{{"name": "read_file", "arguments": {{"path": "relative/path"}}}}</tool_call>
  and
  <tool_call>{{"name": "edit_file", "arguments": {{"path": "relative/path", "old_string": "...", "new_string": "..."}}}}</tool_call>.
  A numbered backup is made before editing; disable per-call with
  `"backup": false`. Always read the file first, then make an exact replacement.
<!-- /section -->
<!-- section: scheduling priority=4 -->
Scheduling:
- Convert relative times to absolute using the current clock before scheduling.
- schedule_job creates new jobs; use list_cron_jobs + update_cron_job /
  delete_cron_job with the numeric id to change existing ones. Do NOT try to
  change jobs through config_set. For example:
  <tool_call>{{"name": "delete_cron_job", "arguments": {{"job_id": 1}}}}</tool_call>
<!-- /section -->
<!-- section: selfcare priority=5 -->
Self-checks:
- If the user asks about system health, weird behavior, or "check yourself",
  call <tool_call>{{"name": "system_check", "arguments": {{}}}}</tool_call> and
  report the findings.
- If something the user enabled isn't working, call
  <tool_call>{{"name": "verify_features", "arguments": {{}}}}</tool_call> first.
  It auto-fixes safe issues and tells you what needs the human.
- If memory or profile grows too large, call
  <tool_call>{{"name": "compact_memory", "arguments": {{"store": "memory"}}}}</tool_call>
  (or `"store": "profile"`). The full original is archived.
- Slash commands the user can type in the terminal or Telegram: /status,
  /golden, /train, /selfcheck, /setup, /compact, /help. For a custom command,
  explain they can save it as a skill/note or a cron job, digest, and train.
<!-- /section -->
<!-- section: canary priority=6 -->
Canary: every now and then you may be asked to repeat a hidden phrase. If you
fail to repeat "SYMBIO_CANARY_v1" when asked, or if you act as though you did
not receive the system prompt, your context window may be too long or your
instruction hierarchy may have degraded. In that case call
<tool_call>{{"name": "compact_memory", "arguments": {{"store": "memory"}}}}</tool_call>
to summarize persistent memory and reduce context pressure, then ask the user
to continue.
<!-- /section -->
"""


def build_system_prompt(assistant_name: str, user_name: str,
                        config: dict | None = None,
                        include_standing: bool = True) -> str:
    """Return the system prompt, seeding prompt.md on first run and
    auto-updating it when the shipped default changed but the user hasn't
    customized it.

    `config` is optional and only decides how much of prompt.md's optional
    sections the loaded model can afford — see prompt_budget_tokens. Omitting
    it keeps the whole file, which is what every caller got before sectioning
    existed.

    `include_standing` is False only for training samples: the user's standing
    instructions are theirs and change whenever they say so, and baking a
    snapshot of them into the corpus would teach a preference they may drop
    tomorrow. Leaving them out also keeps the training prompt an exact prefix
    of the served one, since the block sits after everything training keeps.
    """
    previous_default = ""
    if constants.PROMPT_DEFAULT_FILE.exists():
        previous_default = constants.PROMPT_DEFAULT_FILE.read_text(encoding="utf-8")

    if not constants.PROMPT_FILE.exists():
        # First run: create prompt.md from the unformatted default template.
        constants.PROMPT_FILE.write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
    elif constants.PROMPT_FILE.read_text(encoding="utf-8") == previous_default:
        # The user has not customized prompt.md; refresh it to the new default.
        constants.PROMPT_FILE.write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")

    # Always keep the shipped-default snapshot current for future comparisons.
    if previous_default != DEFAULT_SYSTEM_PROMPT:
        constants.PROMPT_DEFAULT_FILE.write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")

    raw_prompt = constants.PROMPT_FILE.read_text(encoding="utf-8")
    # Keep only the optional sections this model can afford. The core — and a
    # prompt.md carrying no section markers at all — always survives intact.
    parts = split_sections(raw_prompt)
    if any(priority >= 0 for _, priority, _ in parts):
        raw_prompt = assemble_sections(parts, prompt_budget_tokens(config))
    # The security policy is read from its own file and put back at the top,
    # exactly where the trust block used to sit — so on an existing install the
    # assembled prompt is byte-identical to what it was before the split, and
    # the adapter is still being served the text it was trained against.
    policy, raw_prompt = security.ensure_security_file(raw_prompt)
    raw_prompt = security.insert_policy(raw_prompt, policy)
    try:
        prompt_text = raw_prompt.format(
            assistant_name=assistant_name, user_name=user_name
        )
    except KeyError as e:
        # A customized prompt.md introduced a stray {placeholder} that isn't
        # one of the two supported keys. Don't crash the whole session; warn
        # the user and fall back to the shipped default template.
        print(f"[Prompt warning] prompt.md contains unknown placeholder {e}; "
              "using default template. Edit prompt.md or delete it to regenerate.")
        prompt_text = DEFAULT_SYSTEM_PROMPT.format(
            assistant_name=assistant_name, user_name=user_name
        )
    # Append the Hermes-style tool catalog after the user-facing template so
    # the model sees both the tag examples and the JSON schemas. This is done
    # after formatting so the JSON braces are not treated as format keys.
    assembled = prompt_text.rstrip() + "\n\n" + tooling.build_tools_block()
    # The user's standing instructions go LAST: below the security policy,
    # which still outranks them, and below the tool catalog. Not wrapped as
    # untrusted, because they are the one thing in the context that is
    # verifiably the user's own — memory.save_standing_instruction only accepts
    # them from a live user turn and only in a scope that cannot do damage.
    # Everything else the assistant persists stays untrusted.
    #
    # Last for the KV cache, not for emphasis. This block is the only part of
    # the system prompt that changes mid-session, and everything after the
    # change point has to be re-prefilled. Sitting above the catalog, adding one
    # instruction threw away 2,781 tokens of prefix — 2,662 of them the catalog,
    # which had not changed by a byte. Below it, the same edit costs the block
    # itself. Measured with the real tokenizer on Qwen3-14B.
    if include_standing:
        assembled += memory_standing_block(assistant_name, user_name, config)
    return assembled + "\n"


def memory_standing_block(assistant_name: str, user_name: str,
                          config: dict | None = None) -> str:
    """The standing-instructions block, or "" when there are none.

    Imported lazily: symbio.app.memory pulls in skills, which reaches back into
    prompt assembly, and a module-level import closes that loop at startup.
    """
    from symbio.app import memory

    cfg = dict(config or {})
    cfg.setdefault("assistant_name", assistant_name)
    cfg.setdefault("user_name", user_name)
    try:
        return memory.standing_block(cfg)
    except Exception:
        # A standing-instructions file that cannot be read must never take the
        # session down with it; the assistant simply runs without them.
        return ""


def build_training_system_prompt(assistant_name: str, user_name: str,
                                 config: dict | None = None) -> str:
    """The system prompt used when writing TRAINING samples.

    Identical to build_system_prompt() except the <tools> JSON catalog is left
    off, making this an exact prefix of what the model is served at inference.

    The catalog is ~2,200 tokens and byte-identical in every sample, so it
    carries no gradient signal for behaviour — it is pure prefix. Including it
    made every sample ~4,400 tokens against a 768-token training window, so
    mlx_lm truncated each one long before reaching the assistant turn: the
    corpus never reached the model at all, and no amount of corpus design
    could have mattered while that was true.

    Dropping it is safe in the direction that matters. Training sees a prefix
    of what serving sees, never the reverse — the same rule skills.py follows
    for worker prompts. The model still learns the tool syntax, because that
    lives in the assistant turns it is actually trained on, and the catalog is
    present at inference where it is needed to resolve schemas.
    """
    # Same config as serving, or the corpus is written against a prompt the
    # model is never shown — the section budget has to resolve identically
    # on both sides for training to stay a prefix of inference.
    full = build_system_prompt(assistant_name, user_name, config,
                               include_standing=False)
    # rfind, not find: the prompt *mentions* "<tools>" in its prose ("The
    # <tools> catalog at the bottom of this message...") long before the real
    # block is appended at the end. Cutting at the first match silently threw
    # away almost all of the behaviour instructions.
    index = full.rfind("<tools>")
    return full[:index].rstrip() + "\n" if index != -1 else full


def time_note(now: datetime | None = None) -> str:
    """Appended to the system prompt each round so the model can align
    schedules with the computer clock (or defer to a time the user states)."""
    now = now or datetime.now()
    return f"\n\n[Current local date/time from computer clock: {now:%A, %Y-%m-%d %H:%M}]"


def env_note() -> str:
    """Appended to the system prompt each round so the model picks commands
    that actually exist on this machine."""
    system = platform.system()
    if system == "Darwin":
        return ("\n[Environment: macOS. To launch an application (with no URL), "
                "open GUI apps for the user with: open -a 'App Name' "
                "(e.g. open -a 'Google Chrome', open -a 'Safari', open -a 'Spotify'). "
                "GUI apps have no CLI names like 'chrome'. "
                "For browser automation (read/click/scroll/type on a page) use the "
                "browser tools instead of shell open; see the tool catalog. "
                "Use Google Chrome for all browser automation when possible.]")
    if system == "Windows":
        return "\n[Environment: Windows. Open apps or URLs with: start <target>.]"
    return f"\n[Environment: {system}. Open apps or URLs with: xdg-open <target>.]"
