import json

from dcfg.utils.heartbeat import Heartbeat, read_heartbeat


def test_write_and_read(tmp_path):
    hb = Heartbeat(tmp_path, interval_s=60)
    assert hb.beat(5, val_auc=0.9)
    d = read_heartbeat(tmp_path / "heartbeat")
    assert d["step"] == 5 and d["val_auc"] == 0.9 and "time_iso" in d and "unix" in d
    assert len((tmp_path / "heartbeat").read_text().strip().splitlines()) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_throttle_and_force(tmp_path):
    hb = Heartbeat(tmp_path, interval_s=3600)
    assert hb.beat(1)
    assert not hb.beat(2)
    assert json.loads((tmp_path / "heartbeat").read_text())["step"] == 1
    assert hb.beat(3, force=True)
    assert read_heartbeat(tmp_path / "heartbeat")["step"] == 3


def test_read_missing(tmp_path):
    assert read_heartbeat(tmp_path / "nope") is None
