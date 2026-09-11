from vfs_shell.vfs import MemoryStore, TempDirStore


def test_memory_mkdir_write_move():
    fs = MemoryStore()
    fs.mkdir("/notes/archive", parents=True)
    fs.write("/notes/todo.txt", "buy milk\n")
    assert "milk" in fs.read("/notes/todo.txt")
    fs.move("/notes/todo.txt", "/notes/archive/todo.txt")
    assert not fs.exists("/notes/todo.txt")
    assert fs.read("/notes/archive/todo.txt") == "buy milk\n"


def test_normalize_blocks_escape():
    fs = MemoryStore()
    fs.mkdir("/notes", parents=True)
    fs.chdir("/notes")
    fs.mkdir("/notes/keep", parents=True)
    fs.chdir("..")
    assert fs.cwd() == "/"
    fs.chdir("/notes/keep")
    fs.chdir("../../..")
    assert fs.cwd() == "/"


def test_tempdir_roundtrip(tmp_path):
    fs = TempDirStore(root=tmp_path / "sandbox")
    fs.mkdir("/src/bin", parents=True)
    fs.write("/src/bin/readme.txt", "hello edge\n")
    fs.mkdir("/dst", parents=True)
    fs.copy("/src/bin/readme.txt", "/dst/readme.txt")
    fs.remove("/src/bin/readme.txt")
    assert fs.read("/dst/readme.txt") == "hello edge\n"
    assert not fs.exists("/src/bin/readme.txt")
