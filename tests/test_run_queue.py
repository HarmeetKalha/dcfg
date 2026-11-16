import os
import subprocess
import time
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "ops"


def _env(tmp_path, queue_text):
    q = tmp_path / "queue.txt"
    q.write_text(queue_text)
    runs = tmp_path / "runs"
    env = dict(os.environ, DCFG_QUEUE_FILE=str(q), DCFG_RUNS_ROOT=str(runs),
               DCFG_NO_NOTIFY="1", RETRY_PAUSE="0")
    return env, runs


def _run(env, script="run_queue.sh"):
    return subprocess.run(["bash", str(OPS / script)], env=env, timeout=60,
                          capture_output=True, text=True)


def test_success_and_skips(tmp_path):
    env, runs = _env(tmp_path, "# comment\n\n  ok1 | echo hi $DCFG_JOB_ID\n# x | false\n")
    assert _run(env).returncode == 0
    assert (runs / "ok1" / "DONE").exists()
    assert "hi ok1" in (runs / "ok1" / "log.txt").read_text()
    assert not (runs / "x").exists()
    assert not (runs / "_current_job").exists()


def test_failure_three_attempts(tmp_path):
    cnt = tmp_path / "count"
    env, runs = _env(tmp_path, f"bad | echo x >> {cnt}; exit 3\n")
    assert _run(env).returncode == 0
    assert (runs / "bad" / "FAILED").exists() and not (runs / "bad" / "DONE").exists()
    assert len(cnt.read_text().split()) == 3
    assert "exit=3" in (runs / "bad" / "FAILED").read_text()


def test_done_not_rerun(tmp_path):
    cnt = tmp_path / "count"
    env, runs = _env(tmp_path, f"j | echo x >> {cnt}\n")
    _run(env)
    _run(env)
    assert len(cnt.read_text().split()) == 1


def test_second_runner_exits_immediately(tmp_path):
    env, runs = _env(tmp_path, "slow | sleep 4\n")
    p = subprocess.Popen(["bash", str(OPS / "run_queue.sh")], env=env)
    try:
        for _ in range(50):
            if (runs / "_current_job").exists():
                break
            time.sleep(0.1)
        t = time.time()
        r = _run(env)
        assert r.returncode == 0 and time.time() - t < 2
    finally:
        p.wait(timeout=30)
    assert (runs / "slow" / "DONE").exists()


def test_status_empty_runs(tmp_path):
    env, runs = _env(tmp_path, "# nothing\n")
    r = _run(env, "status.sh")
    assert r.returncode == 0
    assert "pending=0" in r.stdout and len(r.stdout.splitlines()) <= 20
