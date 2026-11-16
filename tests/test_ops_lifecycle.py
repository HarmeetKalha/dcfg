"""Lifecycle tests for ops/run_queue.sh, watchdog.sh, status.sh (signals, pinger, alerts, CRLF)."""
import os
import signal
import subprocess
import time
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "ops"


def _env(tmp_path, queue_text, **extra):
    q = tmp_path / "queue.txt"
    q.write_bytes(queue_text.encode())
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    runs = tmp_path / "runs"
    env = dict(os.environ, HOME=str(home), DCFG_QUEUE_FILE=str(q), DCFG_RUNS_ROOT=str(runs),
               DCFG_NO_NOTIFY="1", RETRY_PAUSE="0")
    env.pop("HC_PING_URL", None)
    env.update(extra)
    return env, runs


def _wait_for(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def _alive(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _group_alive(pgid):
    r = subprocess.run(["ps", "-o", "stat=", "-g", str(pgid)], capture_output=True, text=True)
    return any(not s.startswith("Z") for s in r.stdout.split())


def _start_runner(env):
    return subprocess.Popen(["bash", str(OPS / "run_queue.sh")], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_sigterm_kills_job_group_and_pinger_without_marking(tmp_path):
    env, runs = _env(tmp_path, "long | sleep 300\n", DCFG_NO_NOTIFY="0",
                     HC_PING_URL="http://127.0.0.1:1/", PINGER_CHECK_S="1")
    p = _start_runner(env)
    try:
        assert _wait_for(lambda: (runs / "long" / "job.pid").exists()
                         and (runs / ".pinger.pid").exists())
        jpid, pgid = map(int, (runs / "long" / "job.pid").read_text().split())
        pinger = int((runs / ".pinger.pid").read_text())
        assert _group_alive(pgid) and _alive(pinger)
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=15)
        assert _wait_for(lambda: not _group_alive(pgid), 5)
        assert _wait_for(lambda: not _alive(pinger), 5)
        assert not (runs / "long" / "FAILED").exists()
        assert not (runs / "long" / "DONE").exists()
        assert not (runs / "_current_job").exists()
        assert not (runs / ".runner.pid").exists()
        # no retry consumed: only one attempt header in the log
        assert (runs / "long" / "log.txt").read_text().count("attempt") == 1
    finally:
        if p.poll() is None:
            p.kill()
        if (runs / "long" / "job.pid").exists():
            pg = (runs / "long" / "job.pid").read_text().split()[1]
            subprocess.run(["kill", "-KILL", "--", f"-{pg}"], capture_output=True)


def test_sigkill_runner_pinger_exits(tmp_path):
    env, runs = _env(tmp_path, "long | sleep 300\n", DCFG_NO_NOTIFY="0",
                     HC_PING_URL="http://127.0.0.1:1/", PINGER_CHECK_S="1")
    p = _start_runner(env)
    try:
        assert _wait_for(lambda: (runs / "long" / "job.pid").exists()
                         and (runs / ".pinger.pid").exists())
        pinger = int((runs / ".pinger.pid").read_text())
        assert _alive(pinger)
        p.kill()
        p.wait(timeout=5)
        assert _wait_for(lambda: not _alive(pinger), 10)
    finally:
        if p.poll() is None:
            p.kill()
        pg = (runs / "long" / "job.pid").read_text().split()[1]
        subprocess.run(["kill", "-KILL", "--", f"-{pg}"], capture_output=True)


def test_crlf_queue(tmp_path):
    env, runs = _env(tmp_path, "# c\r\nj1 | echo hi $DCFG_JOB_ID\r\nj2 | echo two #noheartbeat\r\n")
    r = subprocess.run(["bash", str(OPS / "run_queue.sh")], env=env, timeout=60)
    assert r.returncode == 0
    assert (runs / "j1" / "DONE").exists() and (runs / "j2" / "DONE").exists()
    log = (runs / "j2" / "log.txt").read_text()
    assert "two" in log and "#noheartbeat" not in log.split("attempt 1:")[1].splitlines()[1]
    assert (runs / "j2" / ".noheartbeat").exists()
    s = subprocess.run(["bash", str(OPS / "status.sh")], env=env, capture_output=True, text=True)
    assert "done=2" in s.stdout and "total=2" in s.stdout


def _fake_live_runner(runs):
    """A process whose cmdline contains run_queue.sh, registered as the runner."""
    runs.mkdir(exist_ok=True)
    fake = subprocess.Popen(["bash", "-c", "sleep 60; true", "run_queue.sh"])
    (runs / ".runner.pid").write_text(str(fake.pid))
    return fake


def test_watchdog_restarts_dead_runner(tmp_path):
    marker = tmp_path / "restarted"
    env, runs = _env(tmp_path, "j | true\n", DCFG_RESTART_CMD=f"touch {marker}")
    runs.mkdir()
    (runs / ".runner.pid").write_text("999999")  # stale pidfile
    subprocess.run(["bash", str(OPS / "watchdog.sh")], env=env, timeout=30)
    assert marker.exists()


def test_watchdog_live_runner_no_restart(tmp_path):
    marker = tmp_path / "restarted"
    env, runs = _env(tmp_path, "j | true\n", DCFG_RESTART_CMD=f"touch {marker}")
    fake = _fake_live_runner(runs)
    try:
        subprocess.run(["bash", str(OPS / "watchdog.sh")], env=env, timeout=30)
        assert not marker.exists()
        s = subprocess.run(["bash", str(OPS / "status.sh")], env=env, capture_output=True, text=True)
        assert "runner alive: yes" in s.stdout
    finally:
        fake.kill()
        fake.wait()


def _hb_setup(tmp_path, queue_line):
    nlog = tmp_path / "notify.log"
    env, runs = _env(tmp_path, queue_line, DCFG_NOTIFY_LOG=str(nlog))
    fake = _fake_live_runner(runs)
    (runs / "j").mkdir()
    (runs / "_current_job").write_text("j\n")
    (runs / "j" / "attempt_start").write_text(str(int(time.time()) - 3600))
    return env, runs, nlog, fake


def _wd(env):
    subprocess.run(["bash", str(OPS / "watchdog.sh")], env=env, timeout=30)


def _hb_alerts(nlog):
    return [ln for ln in (nlog.read_text().splitlines() if nlog.exists() else [])
            if "eartbeat" in ln]


def test_heartbeat_alert_and_cooldown(tmp_path):
    env, runs, nlog, fake = _hb_setup(tmp_path, "j | sleep 1\n")
    try:
        _wd(env)
        assert len(_hb_alerts(nlog)) == 1
        _wd(env)
        assert len(_hb_alerts(nlog)) == 1  # cooldown
    finally:
        fake.kill()
        fake.wait()


def test_old_heartbeat_ignored_and_per_job_threshold(tmp_path):
    env, runs, nlog, fake = _hb_setup(tmp_path, "j | sleep 1 #hb=120\n")
    try:
        hb = runs / "j" / "heartbeat"
        hb.write_text("{}")
        os.utime(hb, (time.time() - 7200, time.time() - 7200))  # before attempt_start
        _wd(env)
        assert _hb_alerts(nlog) == []  # old hb ignored; 60 min since start < 120 min threshold
    finally:
        fake.kill()
        fake.wait()


def test_noheartbeat_tag_no_alert(tmp_path):
    env, runs, nlog, fake = _hb_setup(tmp_path, "j | sleep 1 #noheartbeat\n")
    try:
        _wd(env)
        assert _hb_alerts(nlog) == []
    finally:
        fake.kill()
        fake.wait()


def test_noheartbeat_marker_no_alert(tmp_path):
    env, runs, nlog, fake = _hb_setup(tmp_path, "j | sleep 1\n")
    (runs / "j" / ".noheartbeat").touch()
    try:
        _wd(env)
        assert _hb_alerts(nlog) == []
    finally:
        fake.kill()
        fake.wait()


def test_status_lists_failed(tmp_path):
    env, runs = _env(tmp_path, "a | true\nb | true\n")
    (runs / "b").mkdir(parents=True)
    (runs / "b" / "FAILED").write_text("x")
    s = subprocess.run(["bash", str(OPS / "status.sh")], env=env, capture_output=True, text=True)
    assert "failed jobs: b" in s.stdout
