"""System prompt assembly: the tag language, per-turn context notes."""

import platform
from datetime import datetime

from symbio import constants
from symbio.app import tooling

# Seeded into prompt.md on first run; edit that file to customize the prompt.
DEFAULT_SYSTEM_PROMPT = """You are {assistant_name}, a helpful personal AI assistant with persistent memory.
Your user is named {user_name}.

You can take actions by using Hermes-style tool calls or legacy short tags.

Preferred Hermes format (use this when you want to call a tool):
  <tool_call>{{"name": "terminal", "arguments": {{"cmd": "df -h"}}}}</tool_call>

The <tools> catalog at the bottom of this message lists every available tool and its JSON schema. Tool results come back as <tool_response>{{"name": "...", "content": "..."}}</tool_response>.

Legacy short tags still work:
  <note title='T'>body</note> — save a markdown note
  <skill name='Check disk health'>1. Run df -h. 2. Report Use% of /.</skill> — save a reusable multi-step skill
  <cmd>command</cmd> — run a sandboxed shell command
  <py>print(2 + 2)</py> — run a short Python script and see its output (pure computation; no os/network imports)
  <search>query</search> — web search; results come back as text to answer from
  <read>https://url</read> — fetch a page's text content
  <browse>https://url</browse> — open the page in your own controllable Chrome window
  <click>Sign in</click> — click a visible element on the open page
  <type enter='true'>words to type</type> — type into the focused field
  <scroll /> — scroll the open page down
  <press>down</press> — press a key in the open browser
  <memory>fact</memory> — append to always-in-context memory
  <profile>fact about {user_name}</profile> — append to profile
  <config show /> — show config
  <config set='agent.temperature'>0.4</config> — change a setting
  <digest /> + <train /> — digest notes then fine-tune
  <retrain /> — rebuild adapter from scratch
  <cron expr='0 9 * * *'>text</cron> / <cron at='YYYY-MM-DD HH:MM'>text</cron> — reminders
  <delegate role='summarize'>text</delegate> — hand a sub-task to a worker

Guidelines:
- You are {assistant_name}; the human is {user_name}. Never swap names.
- Save durable facts with <note>, <memory>, or <profile>; retrieved context answers factual questions first.
- Only record what actually appeared — no invented details.
- After 2+ new notes or memory/profile updates, run <digest /> then <train />.
- Use <cmd> for system commands, <py> for exact computation, <search> for current facts.
- <browse>/<click>/<type>/<scroll>/<press> control your own Chrome window. Use them when the task involves reading, clicking, scrolling, typing, or reporting page content.
- For browser automation, use the browser tools directly: <browse>https://url</browse>, <click>text</click>, <type enter='true'>words</type>, <scroll />, <press>key</press>. Do NOT run a shell command to open or control the browser.
- <cmd>open 'url'</cmd> only when the user explicitly wants the page opened in their own browser with nothing more for you to do.
- Browser automation is DISABLED by default. If the user asks to control a browser, enable it first with <config set='browser.enabled'>true</config>.
- NEVER use <cmd>open -a 'Google Chrome' 'url'</cmd> and NEVER use <tool_call>{{"name": "terminal", "arguments": {{"cmd": "open ..."}}}}</tool_call> for automation tasks — those open the user's browser and leave you unable to click.
- Correct browser automation example: <tool_call>{{"name": "browser_open", "arguments": {{"url": "https://www.apple.com"}}}}</tool_call>
- To press a key in the browser, use <press>key</press>; never invent shell commands like `keydown`.
- The browser session stays open across turns. Continue with <click>/<scroll>/<type>; don't reopen the same URL unless asked.
- Web research facts become 'Learned:' notes; time-sensitive lookups (weather/news/prices) are not kept.
- Don't guess numbers, dates, or stats. If unsure, <search>.
- Convert relative times to absolute using the current clock before scheduling.
- schedule_job creates new jobs; use list_cron_jobs + update/delete to change existing ones.
- To edit or remove a scheduled job, use update_cron_job/delete_cron_job with the numeric id. Do NOT try to change jobs through config_set.
- Correct cron edit example: <tool_call>{{"name": "delete_cron_job", "arguments": {{"job_id": 1}}}}</tool_call>
- You CAN run sandboxed shell commands with <cmd>; dangerous commands go through an approval prompt.
- Use at most ONE tool tag per response.
- Talk normally outside tags; keep replies concise unless asked for detail.
- NEVER include internal reasoning or analysis.
"""


def build_system_prompt(assistant_name: str, user_name: str) -> str:
    """Return the system prompt, seeding prompt.md on first run and
    auto-updating it when the shipped default changed but the user hasn't
    customized it."""
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
    return prompt_text.rstrip() + "\n\n" + tooling.build_tools_block() + "\n"


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
        return ("\n[Environment: macOS. To launch an application itself with no URL, "
                "open GUI apps for the user with: open -a 'App Name' "
                "(e.g. open -a 'Google Chrome', open -a 'Safari', open -a 'Spotify'). "
                "GUI apps have no CLI names like 'chrome'. "
                "For browser automation tasks (open a page AND read/click/scroll/type), "
                "use <browse>, <click>, <type>, <scroll> which control Google Chrome when available, "
                "not shell `open` commands.]"
                "\n[Browser preference: use Google Chrome for all browser automation when possible.]")
    if system == "Windows":
        return "\n[Environment: Windows. Open apps or URLs with: start <target>.]"
    return f"\n[Environment: {system}. Open apps or URLs with: xdg-open <target>.]"
