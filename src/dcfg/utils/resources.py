"""Cheap resource logging for training runs: background sampler + per-step timing window.

* :class:`StepTimer` accumulates ``data_wait_s`` (blocked in ``next(loader_iter)``) and ``compute_s``
  (H2D + forward + backward + optimizer step) per step; the sampler drains it every interval.
* :class:`ResourceSampler` is a daemon thread that appends one row per ``interval_s`` to
  ``<run_dir>/resources.csv``: timestamp, step, phase (train/val), GPU util/mem/temp, CPU %, RAM, and the
  timing-window averages (mean data wait, mean compute, wait share, steps/s).
* :func:`summarize_resources` turns the csv into ``resources_summary.json`` (training rows only for
  the utilisation statistics).

GPU: ``pynvml`` if importable, else ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``.
CPU/RAM: ``psutil`` if importable, else ``/proc/stat`` deltas and ``/proc/meminfo``.
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

FIELDS = ["timestamp", "step", "phase", "gpu_util_pct", "gpu_mem_mib", "gpu_temp_c", "cpu_pct", "ram_used_mib",
          "n_steps", "data_wait_s", "compute_s", "data_wait_share", "steps_per_s"]


class StepTimer:
    """Thread-safe accumulator of per-step (data_wait, compute) seconds, drained per window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset(time.perf_counter())

    def _reset(self, now: float) -> None:
        self.n, self.wait, self.comp, self.t0 = 0, 0.0, 0.0, now

    def add(self, wait_s: float, compute_s: float) -> None:
        with self._lock:
            self.n += 1
            self.wait += wait_s
            self.comp += compute_s

    def drain(self) -> dict[str, float | int | None]:
        """Window statistics since the last drain (None values when no step ran), then reset."""
        with self._lock:
            now = time.perf_counter()
            dt, n, w, c = now - self.t0, self.n, self.wait, self.comp
            self._reset(now)
        if n == 0:
            return {"n_steps": 0, "data_wait_s": None, "compute_s": None, "data_wait_share": None,
                    "steps_per_s": None}
        return {"n_steps": n, "data_wait_s": w / n, "compute_s": c / n,
                "data_wait_share": w / (w + c) if (w + c) > 0 else None, "steps_per_s": n / max(dt, 1e-9)}


# ----------------------------------------------------------------------------- readers
def read_gpu_nvidia_smi() -> tuple[float, float, float]:
    """(util %, mem used MiB, temp C) of GPU 0 from nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=10, stderr=subprocess.DEVNULL)
    u, m, t = (float(x) for x in out.strip().splitlines()[0].split(","))
    return u, m, t


def make_gpu_reader() -> Callable[[], tuple[float, float, float] | None]:
    """pynvml reader when available, else nvidia-smi; returns None per call if no GPU can be read."""
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def nvml() -> tuple[float, float, float] | None:
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                return float(u.gpu), m.used / 2**20, float(pynvml.nvmlDeviceGetTemperature(h, 0))
            except Exception:  # noqa: BLE001
                return None
        return nvml
    except Exception:  # noqa: BLE001
        pass

    def smi() -> tuple[float, float, float] | None:
        try:
            return read_gpu_nvidia_smi()
        except Exception:  # noqa: BLE001
            return None
    return smi


def _proc_stat() -> tuple[float, float]:
    """(busy, total) jiffies from the aggregate cpu line of /proc/stat."""
    with open("/proc/stat") as f:
        vals = [float(x) for x in f.readline().split()[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
    return sum(vals) - idle, sum(vals)


def _meminfo_used_mib() -> float:
    d: dict[str, float] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            d[k] = float(v.split()[0])
    return (d["MemTotal"] - d["MemAvailable"]) / 1024.0


class CpuRamReader:
    """CPU % (psutil, else /proc/stat delta since the previous call) and RAM used MiB."""

    def __init__(self) -> None:
        try:
            import psutil  # type: ignore[import-not-found]
            self._ps = psutil
            psutil.cpu_percent(None)
        except Exception:  # noqa: BLE001
            self._ps = None
        self._prev = _proc_stat()

    def __call__(self) -> tuple[float, float]:
        if self._ps is not None:
            return float(self._ps.cpu_percent(None)), self._ps.virtual_memory().used / 2**20
        busy, total = _proc_stat()
        pb, pt = self._prev
        self._prev = (busy, total)
        cpu = 100.0 * (busy - pb) / (total - pt) if total > pt else 0.0
        return cpu, _meminfo_used_mib()


# ----------------------------------------------------------------------------- sampler
class ResourceSampler:
    """Daemon thread writing one csv row (and one ``on_row`` callback) per ``interval_s``.

    ``step_fn`` / ``phase_fn`` report the trainer's current step and phase ("train" | "val").
    Readers are injectable for tests."""

    def __init__(self, path: Path, interval_s: float, timer: StepTimer, step_fn: Callable[[], int],
                 phase_fn: Callable[[], str], on_row: Callable[[dict[str, Any]], None] | None = None,
                 gpu_reader: Callable[[], tuple[float, float, float] | None] | None = None,
                 cpu_reader: Callable[[], tuple[float, float]] | None = None) -> None:
        self.path, self.interval_s, self.timer = Path(path), float(interval_s), timer
        self.step_fn, self.phase_fn, self.on_row = step_fn, phase_fn, on_row
        self.gpu_reader = gpu_reader or make_gpu_reader()
        self.cpu_reader = cpu_reader or CpuRamReader()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)

    def sample(self) -> dict[str, Any]:
        """One row now (drains the timer window). Phase is the one at sampling time."""
        gpu = self.gpu_reader()
        cpu, ram = self.cpu_reader()
        row: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "step": int(self.step_fn()),
                               "phase": self.phase_fn(),
                               "gpu_util_pct": gpu[0] if gpu else None, "gpu_mem_mib": gpu[1] if gpu else None,
                               "gpu_temp_c": gpu[2] if gpu else None, "cpu_pct": cpu, "ram_used_mib": ram,
                               **self.timer.drain()}
        return row

    def _write(self, row: dict[str, Any]) -> None:
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: ("" if v is None else (f"{v:.4f}" if isinstance(v, float) else v))
                        for k, v in row.items()})

    def _emit(self) -> None:
        try:
            row = self.sample()
            self._write(row)
            if self.on_row:
                self.on_row(row)
        except Exception as e:  # noqa: BLE001  never let monitoring kill training
            print(f"WARNING: resource sampler error: {e}", flush=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._emit()

    def start(self) -> "ResourceSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the thread and write a final (partial-window) row."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_s + 15)
        self._emit()


# ----------------------------------------------------------------------------- summary
def _num(x: str) -> float | None:
    return float(x) if x not in ("", None) else None


def summarize_resources(csv_path: Path, train_s: float, val_s: float) -> dict[str, Any]:
    """Aggregate resources.csv. GPU/CPU statistics use phase == "train" rows only (validation excluded);
    data-wait share is the step-weighted ratio sum(wait)/sum(wait+compute) over training rows."""
    rows = list(csv.DictReader(open(csv_path))) if Path(csv_path).exists() else []
    tr = [r for r in rows if r["phase"] == "train" and (_num(r["n_steps"]) or 0) > 0]  # windows with training steps only

    def col(rs: list[dict[str, str]], k: str) -> list[float]:
        return [v for v in (_num(r[k]) for r in rs) if v is not None]

    gu, cu = col(tr, "gpu_util_pct"), col(tr, "cpu_pct")
    w = c = 0.0
    for r in tr:
        n, dw, dc = _num(r["n_steps"]), _num(r["data_wait_s"]), _num(r["compute_s"])
        if n and dw is not None and dc is not None:
            w += n * dw
            c += n * dc
    mem, temp = col(rows, "gpu_mem_mib"), col(rows, "gpu_temp_c")
    return {"n_rows": len(rows), "n_train_rows": len(tr),
            "gpu_util_mean_train": statistics.fmean(gu) if gu else None,
            "gpu_util_median_train": statistics.median(gu) if gu else None,
            "data_wait_share_mean": w / (w + c) if (w + c) > 0 else None,
            "cpu_util_mean_train": statistics.fmean(cu) if cu else None,
            "gpu_mem_peak_mib": max(mem) if mem else None, "gpu_temp_max_c": max(temp) if temp else None,
            "wall_train_s": train_s, "wall_val_s": val_s, "wall_total_s": train_s + val_s}


def write_summary(run_dir: Path, train_s: float, val_s: float) -> dict[str, Any]:
    s = summarize_resources(Path(run_dir) / "resources.csv", train_s, val_s)
    (Path(run_dir) / "resources_summary.json").write_text(json.dumps(s, indent=2) + "\n")
    return s
