# Basic smoke test of the action registry

from core.actions import actions, write_file, read_file

def test_write_and_read(tmp_path):
    p = tmp_path / "hello.txt"
    res = write_file(str(p), "ok")
    assert res.get("ok")
    res2 = read_file(str(p))
    assert res2.get("text") == "ok"
