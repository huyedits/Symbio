"""The desktop pet: a cat that draws the fine-tune as it happens.

Two processes that never import each other meet in one JSON file, so the file
is the contract worth testing from both ends: training writes it (from the
trainer's own lines, and from the golden gate's two exits), and the pet reads
it without importing `symbio` at all. The cat and its strolls are plain
numbers stepped in time, so they are tested without a window; one test draws
every demo frame offscreen where AppKit is available.
"""
import ast
import json
import os
import pathlib
import subprocess
import time
import types

import pytest

from symbio import constants
from symbio.app import training, training_live
from symbio_pet import feed as pet_feed
from symbio_pet.cat import SWAT_AT, Cat, charm_radius
from symbio_pet.demo import LOOP_S, DemoFeed
from symbio_pet.feed import Feed, Snapshot, local_chat_pid, parse_cputime
from symbio_pet.roam import Roamer

REPO = pathlib.Path(__file__).resolve().parent.parent
TRAIN_LINE = ("Iter {}: Train loss {}, Learning Rate 1.000e-05, It/sec 1.000, "
              "Tokens/sec 99.000, Trained Tokens 900, Peak mem 9.000 GB\n")


@pytest.fixture
def live(tmp_path, monkeypatch):
    """training_live writing into a scratch LOG_DIR, with no run in flight."""
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(training_live, "_state", None)
    monkeypatch.setattr(training_live, "_gate_armed", {})
    return tmp_path


def _dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


# ---- the budget -------------------------------------------------------------

def test_the_pet_never_imports_the_agent_package():
    """`symbio` is ~105 MB of agent stack; constants.py is loaded by path."""
    for path in sorted((REPO / "symbio_pet").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            offenders = [n for n in names if n == "symbio" or n.startswith("symbio.")]
            assert not offenders, f"{path.name} imports {offenders}"


# ---- the writer: training_live ---------------------------------------------

def test_a_run_keeps_the_losses_the_trainer_prints(live):
    training_live.begin(None, 20)
    for text in ("Loading pretrained model\n",
                 "Iter 1: Val loss 2.500, Val took 1.000s\n",
                 TRAIN_LINE.format(10, "2.100"),
                 # symbio/cuda_lora.py spells it in lower case.
                 "Iter 20: train loss 1.500, lr 1e-05\n"):
        training_live.line(text)
    state = training_live.read()
    assert state["phase"] == "training"
    assert state["train"] == [[10, 2.1], [20, 1.5]]
    assert state["val"] == [[1, 2.5]]
    assert state["iter"] == 20
    assert state["pid"] == os.getpid()


def test_a_nan_is_reported_not_plotted(live):
    """JSON has no nan, and a curve with one in it cannot be drawn."""
    training_live.begin(None, 20)
    training_live.line("Iter 30: Val loss nan, Val took 1.0s\n")
    state = json.loads(training_live.live_file().read_text(encoding="utf-8"))
    assert state["val"] == []
    assert "nan" in state["reason"]


def test_nothing_is_written_without_a_run(live):
    training_live.line(TRAIN_LINE.format(10, "1.0"))
    assert not training_live.live_file().exists()


def test_a_run_nobody_judges_is_kept_when_it_ends(live):
    training_live.begin(None, 20)
    training_live.end("trained", iter=20, total_iters=20)
    state = training_live.read()
    assert (state["phase"], state["verdict"], state["total_iters"]) == ("trained", "kept", 20)


def test_a_gated_run_waits_for_the_gate(live):
    training_live.arm_gate(None)
    training_live.begin(None, 20)
    training_live.end("trained", iter=20, total_iters=20)
    assert training_live.read()["verdict"] is None
    training_live.mark_kept()
    assert training_live.read()["verdict"] == "kept"


def test_the_cleanup_never_overturns_a_rollback(live):
    """discard_adapter_backup runs in the `finally` of every gated flow."""
    training_live.arm_gate(None)
    training_live.begin(None, 20)
    training_live.end("trained", iter=20, total_iters=20)
    training_live.mark_rolled_back(None)
    training_live.mark_kept()
    assert training_live.read()["verdict"] == "rolled_back"


def test_a_failed_run_is_never_marked_kept(live):
    training_live.arm_gate(None)
    training_live.begin(None, 20)
    training_live.end("failed", reason="boom")
    training_live.mark_kept()
    assert training_live.read()["verdict"] is None


def test_the_first_ending_wins(live):
    training_live.begin(None, 20)
    training_live.end("stopped")
    training_live.end("failed")
    assert training_live.read()["phase"] == "stopped"


# ---- the writer, wired into training ---------------------------------------

class _Trainer:
    """A trainer child that prints `lines` and exits with `code`."""

    def __init__(self, lines, code=0, on_exit=None):
        self.stdout = iter(lines)
        self.returncode = code
        self.on_exit = on_exit

    def wait(self, timeout=None):
        if self.on_exit:
            self.on_exit()
        return self.returncode


def test_the_plain_trainer_streams_its_lines(live, tmp_path, monkeypatch):
    """It was subprocess.run with the child on this process's stdout: the
    losses reached a terminal and nothing else."""
    lines = [TRAIN_LINE.format(10, "2.000"), TRAIN_LINE.format(20, "1.000")]
    monkeypatch.setattr(training.subprocess, "Popen", lambda *a, **k: _Trainer(lines))
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    heard = []
    training.set_log_sink(heard.append)
    try:
        training_live.begin(None, 20)
        assert training._run_trainer(["trainer"], str(config)) is True
    finally:
        training.set_log_sink(None)
    assert heard == [line.rstrip("\n") for line in lines]
    assert training_live.read()["train"] == [[10, 2.0], [20, 1.0]]
    assert not config.exists()


def test_a_trainer_that_exits_nonzero_failed(live, tmp_path, monkeypatch):
    monkeypatch.setattr(training.subprocess, "Popen", lambda *a, **k: _Trainer([], code=1))
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    assert training._run_trainer(["trainer"], str(config)) is False


def test_early_stop_ends_the_run_on_a_garbage_loss(live, tmp_path, monkeypatch):
    process = _Trainer(["Iter 1: Val loss 2.795, Val took 8s\n",
                        "Iter 30: Val loss nan, Val took 8s\n"])
    process.send_signal = lambda sig: None
    monkeypatch.setattr(training.subprocess, "Popen", lambda *a, **k: process)
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    training_live.begin(None, 60)
    assert training._run_training_with_early_stop(
        ["echo"], {"early_stop_patience": 2, "save_every": 100},
        tmp_path, str(config)) is False
    state = training_live.read()
    assert state["phase"] == "failed"
    assert "implausible" in state["reason"]
    assert state["val"] == [[1, 2.795]]


def test_run_training_opens_and_closes_the_run(live, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(constants, "TRAIN_FILE", data / "train.jsonl")
    monkeypatch.setattr(constants, "VALID_FILE", data / "valid.jsonl")
    (data / "train.jsonl").write_text(json.dumps({"text": "a sample to train on"}) + "\n",
                                      encoding="utf-8")
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    monkeypatch.setattr(constants, "ADAPTER_DIR", adapters)
    for name, stub in (("_memory_shortfall", lambda *a, **k: None),
                       ("drop_foreign_template_samples", lambda *a, **k: {}),
                       ("upgrade_corpus_to_messages", lambda *a, **k: {"upgraded": 0, "left": 0}),
                       ("_supports_prompt_masking", lambda **k: False),
                       ("drop_degenerate_samples", lambda *a, **k: {}),
                       ("model_block_count", lambda name: 36),
                       # It rebuilds the adapter root in PROJECT_DIR.
                       ("reseal_adapter", lambda adapter_dir: True)):
        monkeypatch.setattr(training, name, stub)

    def write_adapter():
        (adapters / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapters / "adapters.safetensors").write_bytes(b"weights")

    lines = [TRAIN_LINE.format(1, "2.000"), TRAIN_LINE.format(2, "1.000")]
    monkeypatch.setattr(training.subprocess, "Popen",
                        lambda *a, **k: _Trainer(lines, on_exit=write_adapter))
    lora = {"rank": 8, "dropout": 0.0, "scale": 20.0, "num_layers": 8,
            "batch_size": 1, "learning_rate": 1e-4, "iters": 2, "epochs": 2,
            "max_iters": 10, "max_seq_length": 512, "steps_per_eval": 10,
            "save_every": 10, "early_stop_enabled": False}
    # A path, not a repo id: the tokenizer lookup fails at once, offline.
    config = {"model_name": str(tmp_path / "no-model"), "gpu": {}, "lora": lora}

    assert training.run_training(config) is True
    state = training_live.read()
    assert (state["phase"], state["verdict"]) == ("trained", "kept")
    assert state["train"] == [[1, 2.0], [2, 1.0]]
    assert state["total_iters"] == state["iters"]


def test_the_golden_gate_exits_record_the_verdict(live, tmp_path, monkeypatch):
    """backup_adapter arms the gate, restore_adapter is a rollback, and the
    discard that follows in every `finally` leaves it one."""
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"old")
    monkeypatch.setattr(constants, "ADAPTER_DIR", adapters)
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR", adapters / "workers")

    backup = training.backup_adapter()
    training_live.begin(None, 10)
    (adapters / "adapters.safetensors").write_bytes(b"new")
    training_live.end("trained", iter=10, total_iters=10)
    assert training_live.read()["verdict"] is None

    training.restore_adapter(backup)
    assert (adapters / "adapters.safetensors").read_bytes() == b"old"
    assert training_live.read()["verdict"] == "rolled_back"
    training.discard_adapter_backup(backup)
    assert training_live.read()["verdict"] == "rolled_back"


# ---- the reader: symbio_pet.feed -------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    """The paths the pet reads, and no `ps` or `sysctl` behind them."""
    c = types.SimpleNamespace(
        LOG_DIR=tmp_path / "logs", DAEMON_PID_FILE=tmp_path / "daemon.pid",
        DAEMON_SOCKET=tmp_path / "daemon.sock", ADAPTER_DIR=tmp_path / "adapters",
        TRAINING_LIVE_NAME="training_live.json")
    c.LOG_DIR.mkdir()
    c.ADAPTER_DIR.mkdir()
    monkeypatch.setattr(pet_feed, "_run", lambda argv: "")
    return c


def _write_live(c, **fields):
    state = {"run": "r1", "role": None, "phase": "training", "gated": False,
             "pid": os.getpid(), "iters": 100, "prior_iters": 0, "iter": 30,
             "train": [[10, 2.0], [20, 1.5], [30, 1.2]], "val": [],
             "started_at": time.time(), "updated_at": time.time(), "ended_at": None,
             "total_iters": None, "reason": "", "verdict": None, "verdict_at": None,
             "verdict_role": None}
    state.update(fields)
    (c.LOG_DIR / c.TRAINING_LIVE_NAME).write_text(json.dumps(state), encoding="utf-8")


def test_with_no_model_loaded_it_sleeps(home):
    snap = Feed(home).poll()
    assert (snap.presence, snap.activity) == ("asleep", "idle")
    assert snap.status.startswith("Asleep")


def test_a_daemon_wakes_it_once_the_socket_exists(home):
    home.DAEMON_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    feed = Feed(home)
    assert feed.poll().presence == "waking"
    home.DAEMON_SOCKET.write_text("", encoding="utf-8")
    assert feed.poll().presence == "awake"


def test_busy_is_the_model_process_using_cpu(home, monkeypatch):
    home.DAEMON_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    home.DAEMON_SOCKET.write_text("", encoding="utf-8")
    readings = iter([10.0, 10.0, 10.0, 12.0])
    monkeypatch.setattr(pet_feed, "cpu_seconds", lambda pid: next(readings))
    feed = Feed(home)
    feed.poll()
    assert feed.poll().busy is False
    feed.poll()
    assert feed.poll().busy is True


def test_a_run_in_progress_is_training(home):
    _write_live(home)
    snap = Feed(home).poll()
    assert snap.activity == "training"
    assert (snap.iter, snap.iters) == (30, 100)
    assert "step 30/100" in snap.status


def test_a_run_whose_process_died_fainted_and_is_then_forgotten(home):
    _write_live(home, pid=_dead_pid())
    assert Feed(home).poll().activity == "fainted"
    _write_live(home, pid=_dead_pid(), updated_at=time.time() - pet_feed.FAINT_S - 5)
    assert Feed(home).poll().activity == "idle"


def test_a_pid_from_before_the_last_boot_is_not_trusted(home):
    """After a reboot the dead trainer's pid is soon somebody else's."""
    _write_live(home, started_at=time.time() - 3600)
    feed = Feed(home)
    feed.boot = time.time() - 60
    assert feed.poll().activity != "training"


def test_a_gated_run_is_judged_until_it_has_a_verdict(home):
    _write_live(home, phase="trained", gated=True, ended_at=time.time(), total_iters=100)
    assert Feed(home).poll().activity == "judging"


def test_a_verdict_is_an_event_seen_once(home):
    _write_live(home, phase="trained", ended_at=time.time(), verdict="kept",
                verdict_at=time.time(), total_iters=100)
    feed = Feed(home)
    assert feed.poll().verdict == "kept"
    assert feed.poll().verdict is None


def test_an_old_verdict_is_history(home):
    _write_live(home, phase="trained", ended_at=time.time() - 120, verdict="rolled_back",
                verdict_at=time.time() - 120)
    assert Feed(home).poll().verdict is None


def test_the_adapter_on_disk_sizes_the_idle_charm(home):
    (home.ADAPTER_DIR / "adapters.safetensors").write_bytes(b"w")
    (home.ADAPTER_DIR / "training_progress.json").write_text(
        json.dumps({"total_iters": 600}), encoding="utf-8")
    snap = Feed(home).poll()
    assert (snap.has_adapter, snap.adapter_iters) == (True, 600)


@pytest.mark.parametrize("text,seconds", [
    ("0:01.52", 1.52), ("12:34.56", 754.56), ("1:02:03.45", 3723.45),
    ("2-01:02:03.45", 2 * 86400 + 3723.45), ("", None), ("junk", None)])
def test_ps_cputime_is_read_in_every_shape(text, seconds):
    if seconds is None:
        assert parse_cputime(text) is None
    else:
        assert parse_cputime(text) == pytest.approx(seconds)


def test_a_local_chat_is_found_by_its_argv():
    ps = ("  101 /opt/venv/bin/python3.14 /opt/venv/bin/symb daemon run\n"
          "  102 /bin/zsh -c symb chat\n"
          "  103 /usr/bin/python3 -m symbio.app.cli chat --no-attach\n")
    assert local_chat_pid(ps) == 103
    assert local_chat_pid("  104 /opt/venv/bin/python3 /opt/venv/bin/symb\n") == 104
    assert local_chat_pid("  105 /opt/venv/bin/python3 /opt/venv/bin/symb train\n") is None


# ---- the cat ----------------------------------------------------------------

def _step(cat, seconds, snap=None, dt=1 / 30):
    for i in range(int(round(seconds / dt))):
        cat.update(dt, snap if i == 0 else None)


def _training(iter_, loss_ratio=1.0, run="r1"):
    return Snapshot(presence="awake", activity="training", run=run, iter=iter_, iters=200,
                    train=[[10, 2.0], [iter_ or 10, 2.0 * loss_ratio]])


def test_the_charm_grows_with_steps_and_is_nothing_without_an_adapter():
    sizes = [charm_radius(s) for s in (0, 20, 200, 2000, 20000)]
    assert sizes[0] == 0.0
    assert sizes == sorted(sizes)
    assert sizes[-1] == sizes[-2]         # capped


def test_it_catches_a_treat_for_every_couple_of_steps():
    cat = Cat(seed=1)
    eaten = []
    cat._eat = eaten.append
    _step(cat, 0.5, _training(0))
    _step(cat, 0.5, _training(12))
    _step(cat, 4.0)
    assert len(eaten) == 6


def test_a_long_gap_between_reports_is_not_a_flood():
    """At most eight treats a report, however many steps it covers."""
    cat = Cat(seed=1)
    eaten = []
    cat._eat = eaten.append
    _step(cat, 0.5, _training(0))
    _step(cat, 0.5, _training(200))
    _step(cat, 6.0)
    assert len(eaten) == 8


def test_it_owes_nothing_for_steps_taken_before_it_was_watching():
    cat = Cat(seed=1)
    _step(cat, 0.5, _training(300))
    assert cat.queue == 0 and not cat.treats


def test_the_tail_lashes_with_the_loss():
    high, low = Cat(seed=1), Cat(seed=1)
    _step(high, 3.0, _training(40, loss_ratio=1.0))
    _step(low, 3.0, _training(40, loss_ratio=0.2))
    assert high.tail_amp > low.tail_amp + 0.2


def test_a_rollback_swats_the_charm_off():
    cat = Cat(seed=1)
    _step(cat, 3.0, _training(100))
    assert cat.charm_r > 3.0
    verdict = Snapshot(presence="awake", run="r1", iter=100, iters=200, total_iters=100,
                       verdict="rolled_back")
    _step(cat, SWAT_AT - 0.1, verdict)
    assert cat.ejecta is None
    _step(cat, 0.2)
    assert cat.ejecta is not None and cat.charm_r < 0.5


def test_a_kept_charm_stays_on():
    cat = Cat(seed=1)
    _step(cat, 3.0, _training(100))
    kept = Snapshot(presence="awake", run="r1", iter=100, iters=200, total_iters=100,
                    verdict="kept", has_adapter=True, adapter_iters=100)
    _step(cat, 4.0, kept)
    assert cat.ejecta is None
    assert cat.charm_r == pytest.approx(charm_radius(100), abs=0.2)


@pytest.mark.parametrize("snap,walking,held,pose", [
    (Snapshot(presence="asleep"), False, False, "loaf"),
    (Snapshot(presence="awake"), False, False, "sit"),
    (Snapshot(presence="awake"), True, False, "stand"),
    (Snapshot(presence="awake"), False, True, "held"),
])
def test_each_state_has_its_pose(snap, walking, held, pose):
    cat = Cat(seed=1)
    cat.held = held
    cat.walking = walking
    _step(cat, 1.0, snap)
    assert cat.pose == pose and cat.pose_t == 1.0


def test_turning_round_passes_through_zero():
    cat = Cat(seed=1)
    cat.facing_goal = -1.0
    _step(cat, 0.05)
    assert -1.0 < cat.facing < 1.0
    _step(cat, 1.0)
    assert cat.facing == pytest.approx(-1.0, abs=0.01)


# ---- the strolls ------------------------------------------------------------

def test_dropped_it_falls_to_the_floor_and_lands():
    cat, roam = Cat(seed=1), Roamer(seed=1)
    x, y = 500.0, 400.0
    for _ in range(90):
        cat.update(1 / 30)
        x, y = roam.step(cat, 1 / 30, x, y, floor=80.0, left=0.0, right=1600.0)
    assert (x, y) == (500.0, 80.0)
    assert not roam.falling


def test_it_strolls_when_idle_and_stays_on_screen():
    cat, roam = Cat(seed=1), Roamer(seed=2)
    awake = Snapshot(presence="awake")
    roam.next_stroll = 0.5
    x, y, xs = 1500.0, 80.0, []
    for i in range(30 * 20):
        cat.update(1 / 30, awake if i == 0 else None)
        x, y = roam.step(cat, 1 / 30, x, y, floor=80.0, left=0.0, right=1600.0)
        xs.append(x)
    assert max(xs) - min(xs) > 50, "it never went anywhere"
    assert 0.0 <= min(xs) and max(xs) <= 1600.0


@pytest.mark.parametrize("snap,hovered,enabled", [
    (_training(10), False, True),          # busy eating
    (Snapshot(presence="asleep"), False, True),
    (Snapshot(presence="awake"), True, True),   # someone has the pointer on it
    (Snapshot(presence="awake"), False, False),  # wandering switched off
])
def test_it_stays_put_when_it_should(snap, hovered, enabled):
    cat, roam = Cat(seed=1), Roamer(seed=2)
    roam.enabled = enabled
    roam.next_stroll = 0.0
    x, y = 700.0, 80.0
    for i in range(30 * 6):
        cat.update(1 / 30, snap if i == 0 else None)
        cat.hovered = hovered
        x, y = roam.step(cat, 1 / 30, x, y, floor=80.0, left=0.0, right=1600.0)
    assert x == 700.0


# ---- the demo, and drawing it -----------------------------------------------

def test_the_demo_passes_through_every_state():
    now = [0.0]
    feed = DemoFeed(clock=lambda: now[0])
    presences, activities, verdicts = set(), set(), []
    while now[0] < LOOP_S:
        snap = feed.poll()
        presences.add(snap.presence)
        activities.add(snap.activity)
        if snap.verdict:
            verdicts.append(snap.verdict)
        now[0] += 0.25
    assert presences == {"asleep", "waking", "awake"}
    assert activities == {"idle", "training", "judging", "failed"}
    assert verdicts == ["kept", "rolled_back"]


def test_every_demo_frame_draws(tmp_path):
    pytest.importorskip("AppKit")
    from symbio_pet.cli import FRAMES, snapshot

    written = snapshot(tmp_path)
    assert len(written) == len(FRAMES)
    assert all(path.stat().st_size > 2000 for path in written)
