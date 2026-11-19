import os
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab.workspace import FileToolError, WorkspaceFiles

HEADER = b"---\nname: alpha\ndescription: Short description.\n---\n"


def test_frontmatter_read_stops_before_body_without_read_ahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "SKILL.md").write_bytes(HEADER + b"\xffprivate body must not be read")
    positions: list[int] = []
    original = os.fdopen

    class ObservedFile:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __enter__(self) -> Any:
            return self.stream

        def __exit__(self, *arguments: object) -> None:
            positions.append(self.stream.tell())
            self.stream.close()

    def observed(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["buffering"] == 0
        return ObservedFile(original(*args, **kwargs))

    with WorkspaceFiles(tmp_path) as workspace:
        monkeypatch.setattr(os, "fdopen", observed)
        assert workspace.read_frontmatter("SKILL.md") == HEADER.decode()
    assert positions == [len(HEADER)]


@pytest.mark.parametrize("path", ["../outside", "/etc", "a/../b", ".env", ".git", "a//b", ".", ""])
def test_listing_rejects_illegal_or_protected_directories(tmp_path: Path, path: str) -> None:
    with WorkspaceFiles(tmp_path) as workspace, pytest.raises(FileToolError):
        workspace.list_directories(path)


def test_directory_listing_is_bounded_sorted_and_never_follows_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    (root / "zeta").mkdir()
    (root / "alpha").mkdir()
    (root / "README.md").write_text("ignored regular root file")
    with WorkspaceFiles(tmp_path) as workspace:
        assert workspace.list_directories("skills") == ("alpha", "zeta")
        with pytest.raises(FileToolError, match="entry limit"):
            workspace.list_directories("skills", max_entries=2)
        (root / "linked").symlink_to(tmp_path)
        with pytest.raises(FileToolError, match="symbolic"):
            workspace.list_directories("skills")


@pytest.mark.parametrize(
    "attack", ["file_symlink", "parent_symlink", "hardlink", "fifo", "directory"]
)
def test_header_read_preserves_existing_no_follow_regular_file_boundary(
    tmp_path: Path, attack: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.md").write_bytes(HEADER + b"body")
    root = tmp_path / "workspace"
    root.mkdir()
    path = "SKILL.md"
    if attack == "file_symlink":
        (root / path).symlink_to(outside / "real.md")
    elif attack == "parent_symlink":
        (root / "link").symlink_to(outside)
        path = "link/real.md"
    elif attack == "hardlink":
        os.link(outside / "real.md", root / path)
    elif attack == "fifo":
        os.mkfifo(root / path)
    else:
        (root / path).mkdir()
    with WorkspaceFiles(root) as workspace, pytest.raises(FileToolError):
        workspace.read_frontmatter(path)


@pytest.mark.parametrize(
    "document", [b"name: no delimiter", b"---\nname: no closing", b"---\n\xff\n---\n"]
)
def test_missing_invalid_or_damaged_header_is_explicit_error(
    tmp_path: Path, document: bytes
) -> None:
    (tmp_path / "SKILL.md").write_bytes(document)
    with WorkspaceFiles(tmp_path) as workspace, pytest.raises(FileToolError):
        workspace.read_frontmatter("SKILL.md")


def test_header_and_full_read_caps_cannot_expand_workspace_limit(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_bytes(HEADER + b"x" * 100)
    with WorkspaceFiles(tmp_path, max_bytes=len(HEADER)) as workspace:
        assert workspace.read_frontmatter("SKILL.md", max_bytes=1000) == HEADER.decode()
        with pytest.raises(FileToolError, match="byte limit"):
            workspace.read_text("SKILL.md", max_bytes=1000)
        with pytest.raises(FileToolError, match="header exceeds"):
            workspace.read_frontmatter("SKILL.md", max_bytes=len(HEADER) - 1)
    with WorkspaceFiles(tmp_path) as workspace:
        with pytest.raises(FileToolError, match="byte limit"):
            workspace.read_text("SKILL.md", max_bytes=10)
        assert workspace.read_text("SKILL.md").endswith("x" * 100)


@pytest.mark.parametrize("limit", [0, -1, True, "9"])
def test_additive_file_api_options_fail_explicitly(tmp_path: Path, limit: Any) -> None:
    with WorkspaceFiles(tmp_path) as workspace:
        with pytest.raises(FileToolError):
            workspace.list_directories("skills", max_entries=limit)
        with pytest.raises(FileToolError):
            workspace.read_frontmatter("SKILL.md", max_bytes=limit)
        with pytest.raises(FileToolError):
            workspace.read_text("SKILL.md", max_bytes=limit)
