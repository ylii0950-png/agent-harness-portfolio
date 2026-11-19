import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab.workspace import FileErrorCode as Code
from agent_harness_lab.workspace import FileToolError, WorkspaceFiles


def test_utf8_create_read_and_explicit_overwrite(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        workspace.write_text("notes/example.txt", "你好 🌱\r\n")
        assert workspace.read_text("notes/example.txt") == "你好 🌱\r\n"
        with pytest.raises(FileToolError) as error:
            workspace.write_text("notes/example.txt", "refused")
        assert error.value.code == Code.ALREADY_EXISTS
        assert workspace.read_text("notes/example.txt") == "你好 🌱\r\n"
        workspace.write_text("notes/example.txt", "replacement", overwrite=True)
        assert workspace.read_text("notes/example.txt") == "replacement"
    assert sorted(p.name for p in (tmp_path / "notes").iterdir()) == ["example.txt"]
    assert stat.S_IMODE((tmp_path / "notes/example.txt").stat().st_mode) == 0o600


def test_empty_text_is_valid(tmp_path: Path) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        workspace.write_text("empty.txt", "")
        assert workspace.read_text("empty.txt") == ""


def test_default_write_permission_denies_without_mutation(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("original")
    with WorkspaceFiles(tmp_path) as workspace:
        assert workspace.read_text("existing.txt") == "original"
        for path in ["new.txt", "existing.txt"]:
            with pytest.raises(FileToolError) as error:
                workspace.write_text(path, "denied", overwrite=True)
            assert error.value.code == Code.PERMISSION_DENIED
    assert existing.read_text() == "original"
    assert not (tmp_path / "new.txt").exists()


@pytest.mark.parametrize("path", ["../outside.txt", "a/../../outside.txt", "/tmp/file", "a/../b"])
def test_outside_paths_are_denied_for_read_and_write(tmp_path: Path, path: str) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as read_error:
            workspace.read_text(path)
        with pytest.raises(FileToolError) as write_error:
            workspace.write_text(path, "denied")
    assert read_error.value.code == write_error.value.code == Code.PATH_DENIED
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "path", [None, 3, Path("a"), "", ".", "a//b", "a/./b", "a/", "a\x00b", "a\\b"]
)
def test_invalid_paths(tmp_path: Path, path: Any) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as read_error:
            workspace.read_text(path)
        with pytest.raises(FileToolError) as write_error:
            workspace.write_text(path, "denied")
    assert read_error.value.code == write_error.value.code == Code.INVALID_INPUT


@pytest.mark.parametrize(
    "path", [".git/config", ".GIT/config", ".venv/file", ".env", ".ENV.local", "sub/.env/a"]
)
def test_protected_paths(tmp_path: Path, path: str) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as read_error:
            workspace.read_text(path)
        with pytest.raises(FileToolError) as write_error:
            workspace.write_text(path, "denied")
    assert read_error.value.code == write_error.value.code == Code.PATH_DENIED


@pytest.mark.parametrize("dangling", [False, True])
def test_leaf_symlink_escape_is_denied(tmp_path: Path, dangling: bool) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    if not dangling:
        outside.write_text("untouched")
    (root / "link").symlink_to(outside)
    with WorkspaceFiles(root, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as read_error:
            workspace.read_text("link")
        with pytest.raises(FileToolError) as write_error:
            workspace.write_text("link", "attack", overwrite=True)
    assert read_error.value.code == write_error.value.code == Code.PATH_DENIED
    assert outside.exists() is not dangling
    if not dangling:
        assert outside.read_text() == "untouched"


def test_parent_symlink_escape_is_denied(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("untouched")
    (root / "alias").symlink_to(outside, target_is_directory=True)
    with WorkspaceFiles(root, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as read_error:
            workspace.read_text("alias/secret.txt")
        with pytest.raises(FileToolError) as write_error:
            workspace.write_text("alias/new.txt", "attack")
    assert read_error.value.code == write_error.value.code == Code.PATH_DENIED
    assert (outside / "secret.txt").read_text() == "untouched"
    assert not (outside / "new.txt").exists()


def test_internal_symlink_is_also_denied(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("original")
    (tmp_path / "alias").symlink_to("file.txt")
    with WorkspaceFiles(tmp_path) as workspace:
        with pytest.raises(FileToolError) as error:
            workspace.read_text("alias")
    assert error.value.code == Code.PATH_DENIED


def test_hardlink_is_denied(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched")
    os.link(outside, root / "alias")
    with WorkspaceFiles(root, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as read_error:
            workspace.read_text("alias")
        with pytest.raises(FileToolError) as write_error:
            workspace.write_text("alias", "attack", overwrite=True)
    assert read_error.value.code == write_error.value.code == Code.PATH_DENIED
    assert outside.read_text() == "untouched"


@pytest.mark.parametrize("path", ["missing.txt", "missing/file.txt"])
def test_missing_file_or_parent(tmp_path: Path, path: str) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as error:
            workspace.read_text(path)
        assert error.value.code == Code.NOT_FOUND
        if "/" in path:
            with pytest.raises(FileToolError) as error:
                workspace.write_text(path, "text")
            assert error.value.code == Code.NOT_FOUND
            assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("kind", ["directory", "fifo", "file_parent"])
def test_non_regular_targets(tmp_path: Path, kind: str) -> None:
    path = "target"
    if kind == "directory":
        (tmp_path / path).mkdir()
    elif kind == "fifo":
        os.mkfifo(tmp_path / path)
    else:
        (tmp_path / path).write_text("parent is a file")
        path += "/child"
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as read_error:
            workspace.read_text(path)
        with pytest.raises(FileToolError) as write_error:
            workspace.write_text(path, "denied", overwrite=True)
    assert read_error.value.code == write_error.value.code == Code.NOT_FILE


def test_limits_count_utf8_bytes_and_preserve_existing_file(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("old")
    (tmp_path / "large.txt").write_bytes(b"12345")
    with WorkspaceFiles(tmp_path, allow_write=True, max_bytes=4) as workspace:
        workspace.write_text("exact.txt", "éé")
        assert workspace.read_text("exact.txt") == "éé"
        with pytest.raises(FileToolError) as error:
            workspace.write_text("existing.txt", "你好", overwrite=True)
        assert error.value.code == Code.TOO_LARGE
        with pytest.raises(FileToolError) as error:
            workspace.read_text("large.txt")
        assert error.value.code == Code.TOO_LARGE
    assert (tmp_path / "existing.txt").read_text() == "old"


def test_invalid_utf8_read_and_write(tmp_path: Path) -> None:
    (tmp_path / "binary").write_bytes(b"\xff\xfe")
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as read_error:
            workspace.read_text("binary")
        with pytest.raises(FileToolError) as write_error:
            workspace.write_text("new.txt", "\ud800")
    assert read_error.value.code == write_error.value.code == Code.INVALID_TEXT
    assert not (tmp_path / "new.txt").exists()


@pytest.mark.parametrize(
    "content,overwrite", [(None, False), (7, False), (b"bytes", False), ("text", 1)]
)
def test_invalid_write_arguments(tmp_path: Path, content: Any, overwrite: Any) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        with pytest.raises(FileToolError) as error:
            workspace.write_text("new.txt", content, overwrite=overwrite)
    assert error.value.code == Code.INVALID_INPUT
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "options",
    [
        {"allow_write": 1},
        {"max_bytes": True},
        {"max_bytes": 0},
        {"max_bytes": -1},
        {"max_bytes": "4"},
    ],
)
def test_invalid_workspace_options(tmp_path: Path, options: Any) -> None:
    with pytest.raises(FileToolError) as error:
        WorkspaceFiles(tmp_path, **options)
    assert error.value.code == Code.INVALID_INPUT


@pytest.mark.parametrize("root", [None, 3, "", "bad\x00path"])
def test_invalid_root(root: Any) -> None:
    with pytest.raises(FileToolError) as error:
        WorkspaceFiles(root)
    assert error.value.code == Code.INVALID_INPUT


def test_missing_or_non_directory_root(tmp_path: Path) -> None:
    with pytest.raises(FileToolError) as error:
        WorkspaceFiles(tmp_path / "absent")
    assert error.value.code == Code.NOT_FOUND
    file = tmp_path / "file"
    file.write_text("text")
    with pytest.raises(FileToolError) as error:
        WorkspaceFiles(file)
    assert error.value.code == Code.NOT_FILE


def test_context_closes_on_exception_and_close_is_idempotent(tmp_path: Path) -> None:
    workspace = WorkspaceFiles(tmp_path, allow_write=True)
    with pytest.raises(RuntimeError, match="caller failed"), workspace:
        raise RuntimeError("caller failed")
    workspace.close()
    operations: list[Callable[[], object]] = [
        lambda: workspace.read_text("a"),
        lambda: workspace.write_text("a", "b"),
    ]
    for operation in operations:
        with pytest.raises(FileToolError) as error:
            operation()
        assert error.value.code == Code.CLOSED
    with pytest.raises(FileToolError) as error, workspace:
        pass
    assert error.value.code == Code.CLOSED


def test_looping_root_has_explicit_error(tmp_path: Path) -> None:
    (tmp_path / "loop").symlink_to("loop")
    with pytest.raises(FileToolError) as error:
        WorkspaceFiles(tmp_path / "loop")
    assert error.value.code == Code.PATH_DENIED


@pytest.mark.parametrize("operation", ["read", "write"])
def test_real_os_permission_denial(tmp_path: Path, operation: str) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses POSIX mode-bit denial")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "file").write_text("original")
    blocked.chmod(0)
    try:
        with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
            with pytest.raises(FileToolError) as error:
                if operation == "read":
                    workspace.read_text("blocked/file")
                else:
                    workspace.write_text("blocked/new", "denied")
        assert error.value.code == Code.PERMISSION_DENIED
        assert str(tmp_path) not in str(error.value)
    finally:
        blocked.chmod(0o700)
    assert (blocked / "file").read_text() == "original"
    assert not (blocked / "new").exists()


def test_parent_symlink_swap_after_check_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    parent = root / "parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("untouched")
    original_open = os.open

    def swapped_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == "parent":
            parent.rename(root / "saved")
            parent.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    with WorkspaceFiles(root, allow_write=True) as workspace:
        monkeypatch.setattr(os, "open", swapped_open)
        with pytest.raises(FileToolError):
            workspace.write_text("parent/secret", "attack", overwrite=True)
    assert (outside / "secret").read_text() == "untouched"


def test_leaf_symlink_swap_after_check_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "file"
    target.write_text("inside")
    outside = tmp_path / "outside"
    outside.write_text("secret")
    original_open = os.open

    def swapped_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == "file":
            target.unlink()
            target.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    with WorkspaceFiles(root) as workspace:
        monkeypatch.setattr(os, "open", swapped_open)
        with pytest.raises(FileToolError) as error:
            workspace.read_text("file")
    assert error.value.code == Code.PATH_DENIED
    assert outside.read_text() == "secret"


@pytest.mark.parametrize("overwrite", [False, True])
def test_failed_publish_preserves_destination_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool
) -> None:
    if overwrite:
        (tmp_path / "file").write_text("original")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.ENOSPC, "disk full")

    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        monkeypatch.setattr(os, "replace" if overwrite else "link", fail)
        with pytest.raises(FileToolError) as error:
            workspace.write_text("file", "new", overwrite=overwrite)
    assert error.value.code == Code.IO_ERROR
    assert sorted(p.name for p in tmp_path.iterdir()) == (["file"] if overwrite else [])
    if overwrite:
        assert (tmp_path / "file").read_text() == "original"


def test_concurrent_create_does_not_overwrite_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = os.link

    def raced_link(src: Any, dst: Any, **kwargs: Any) -> None:
        (tmp_path / "file").write_text("other writer won")
        original_link(src, dst, **kwargs)

    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        monkeypatch.setattr(os, "link", raced_link)
        with pytest.raises(FileToolError) as error:
            workspace.write_text("file", "must not overwrite")
    assert error.value.code == Code.ALREADY_EXISTS
    assert (tmp_path / "file").read_text() == "other writer won"
    assert [p.name for p in tmp_path.iterdir()] == ["file"]
