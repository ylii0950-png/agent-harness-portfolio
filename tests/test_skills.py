from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab.skills import SkillError, SkillRegistry
from agent_harness_lab.workspace import FileToolError, WorkspaceFiles


def seed(
    root: Path,
    folder: str = "alpha",
    *,
    name: str = "alpha",
    description: str = "Short summary.",
    body: str = "private alpha body",
) -> Path:
    path = root / "skills" / folder / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n", encoding="utf-8"
    )
    return path


def test_discover_metadata_only_stable_snapshot_and_lazy_requested_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(tmp_path)
    seed(tmp_path, "beta", name="beta", body="private beta body")
    reads: list[str] = []
    original = WorkspaceFiles.read_text

    def spy(self: WorkspaceFiles, path: str, *, max_bytes: int | None = None) -> str:
        reads.append(path)
        return original(self, path, max_bytes=max_bytes)

    monkeypatch.setattr(WorkspaceFiles, "read_text", spy)
    with WorkspaceFiles(tmp_path) as workspace:
        skills = SkillRegistry(workspace, ("skills",))
        assert reads == []
        catalog = skills.discover()
        assert [(info.name, info.description) for info in catalog] == [
            ("alpha", "Short summary."),
            ("beta", "Short summary."),
        ]
        assert catalog[0].entry == "skills/alpha/SKILL.md" and reads == []
        assert skills.discover() is catalog
        assert "private" not in repr(catalog)
        with pytest.raises(FrozenInstanceError):
            cast(Any, catalog[0]).name = "other"
        assert skills.load("alpha") == "private alpha body"
    assert reads == ["skills/alpha/SKILL.md"]


@pytest.mark.parametrize(
    "metadata",
    [
        "name: alpha",
        "description: only",
        "name: alpha\nname: alpha\ndescription: x",
        "name: alpha\ndescription: x\ncommand: run",
        "name: ../secret\ndescription: x",
        "name: Alpha\ndescription: x",
        "name: []\ndescription: x",
        "name: null\ndescription: x",
        "name: alpha\ndescription: []",
        "name: alpha\ndescription: false",
        "name: alpha\ndescription: 42",
        'name: alpha\ndescription: "unterminated',
        "name: alpha\ndescription: >",
        "name: alpha\ndescription: !include secret",
        "name: alpha\ndescription: &alias value",
        'name: alpha\ndescription: "line\\nline"',
        'name: alpha\ndescription: "\\ud800"',
        "name: alpha\ndescription: ",
        "name: alpha\ndescription: " + "x" * 257,
    ],
)
def test_invalid_frontmatter_rejected_without_any_body_loading(
    tmp_path: Path, metadata: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = seed(tmp_path)
    path.write_text("---\n" + metadata + "\n---\nprivate body")

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("discovery must never read body")

    monkeypatch.setattr(WorkspaceFiles, "read_text", forbidden)
    with WorkspaceFiles(tmp_path) as workspace, pytest.raises(SkillError, match="invalid_metadata"):
        SkillRegistry(workspace, ("skills",)).discover()


@pytest.mark.parametrize(
    "description,expected",
    [
        ('"Quoted: description."', "Quoted: description."),
        ("'Single ''quote'' text'", "Single 'quote' text"),
        ("Simple: punctuation", "Simple: punctuation"),
    ],
)
def test_supported_string_frontmatter_variants(
    tmp_path: Path, description: str, expected: str
) -> None:
    seed(tmp_path, description=description)
    with WorkspaceFiles(tmp_path) as workspace:
        assert SkillRegistry(workspace, ("skills",)).discover()[0].description == expected


def test_duplicate_names_across_folders_fail_atomically_and_can_be_repaired(tmp_path: Path) -> None:
    seed(tmp_path)
    other = seed(tmp_path, "other", name="alpha")
    with WorkspaceFiles(tmp_path) as workspace:
        skills = SkillRegistry(workspace, ("skills",))
        with pytest.raises(SkillError, match="duplicate_name"):
            skills.discover()
        other.write_text("---\nname: other\ndescription: Other.\n---\nother body")
        assert [x.name for x in skills.discover()] == ["alpha", "other"]


def test_unknown_and_unauthorized_skills_cannot_interpret_names_as_paths(tmp_path: Path) -> None:
    seed(tmp_path)
    seed(tmp_path, "beta", name="beta")
    with WorkspaceFiles(tmp_path) as workspace:
        skills = SkillRegistry(workspace, ("skills",), allowed_names=("alpha",))
        assert [x.name for x in skills.discover()] == ["alpha"]
        with pytest.raises(SkillError, match="not_authorized"):
            skills.load("beta")
        for name in ("../alpha", "/etc/passwd", "alpha/SKILL.md", "a\\b", "", ".."):
            with pytest.raises(SkillError, match="invalid_name"):
                skills.load(name)
        with pytest.raises(SkillError, match="not_found"):
            SkillRegistry(workspace, ("skills",)).load("missing")


@pytest.mark.parametrize(
    "directory", ["../skills", "/etc", "skills/../skills", ".git", "skills//alpha"]
)
def test_configured_directories_cannot_escape_workspace(tmp_path: Path, directory: str) -> None:
    with WorkspaceFiles(tmp_path) as workspace, pytest.raises(FileToolError):
        SkillRegistry(workspace, (directory,)).discover()


@pytest.mark.parametrize(
    "case",
    [
        "missing_root",
        "missing_document",
        "broken_header",
        "huge_header",
        "directory_link",
        "document_link",
    ],
)
def test_missing_damaged_and_escaped_documents_fail_closed(tmp_path: Path, case: str) -> None:
    path = seed(tmp_path)
    if case == "missing_root":
        roots = ("absent",)
    else:
        roots = ("skills",)
    if case == "missing_document":
        path.unlink()
    elif case == "broken_header":
        path.write_bytes(b"\xffmalformed")
    elif case == "huge_header":
        path.write_text("---\nname: alpha\ndescription: " + "x" * 4096 + "\n---\nbody")
    elif case == "directory_link":
        (tmp_path / "skills-link").symlink_to(tmp_path / "skills")
        roots = ("skills-link",)
    elif case == "document_link":
        path.unlink()
        path.symlink_to(tmp_path / "outside.txt")
        (tmp_path / "outside.txt").write_text("private data")
    with WorkspaceFiles(tmp_path) as workspace, pytest.raises((SkillError, FileToolError)):
        SkillRegistry(workspace, roots).discover()


@pytest.mark.parametrize("body", [b"\xffinvalid UTF-8", b"x" * 65536, b" \n"])
def test_body_is_unread_until_load_then_damaged_empty_or_oversized_fails(
    tmp_path: Path, body: bytes
) -> None:
    path = seed(tmp_path)
    header = b"---\nname: alpha\ndescription: Short summary.\n---\n"
    path.write_bytes(header + body)
    with WorkspaceFiles(tmp_path) as workspace:
        skills = SkillRegistry(workspace, ("skills",))
        assert skills.discover()[0].name == "alpha"
        with pytest.raises((SkillError, FileToolError)):
            skills.load("alpha")


def test_metadata_change_or_symlink_swap_after_discovery_is_rejected(tmp_path: Path) -> None:
    path = seed(tmp_path)
    with WorkspaceFiles(tmp_path) as workspace:
        skills = SkillRegistry(workspace, ("skills",))
        skills.discover()
        path.write_text("---\nname: alpha\ndescription: CHANGED.\n---\nbody")
        with pytest.raises(SkillError, match="changed"):
            skills.load("alpha")
        path.unlink()
        path.symlink_to(tmp_path / "outside.txt")
        (tmp_path / "outside.txt").write_text("secret outside configured skill dirs")
        with pytest.raises(FileToolError):
            skills.load("alpha")


@pytest.mark.parametrize(
    "roots", [[], ["skills", "skills"], "skills", [None], [[]], None, False, 3]
)
def test_invalid_directory_configuration_is_explicit_error(tmp_path: Path, roots: Any) -> None:
    with WorkspaceFiles(tmp_path) as workspace, pytest.raises(SkillError, match="invalid_config"):
        SkillRegistry(workspace, roots)


def test_catalog_size_is_bounded(tmp_path: Path) -> None:
    for i in range(65):
        seed(tmp_path, f"skill_{i}", name=f"skill_{i}")
    with WorkspaceFiles(tmp_path) as workspace, pytest.raises(SkillError, match="too_many"):
        SkillRegistry(workspace, ("skills",)).discover()


def test_exact_document_limit_is_accepted_and_one_more_byte_rejected(tmp_path: Path) -> None:
    path = seed(tmp_path)
    header = b"---\nname: alpha\ndescription: Short summary.\n---\n"
    path.write_bytes(header + b"x" * (65536 - len(header)))
    with WorkspaceFiles(tmp_path) as workspace:
        skills = SkillRegistry(workspace, ("skills",))
        assert len(skills.load("alpha").encode()) == 65536 - len(header)
        path.write_bytes(path.read_bytes() + b"x")
        with pytest.raises(FileToolError, match="byte limit"):
            skills.load("alpha")


def test_empty_allowlist_exposes_no_metadata_or_body(tmp_path: Path) -> None:
    seed(tmp_path)
    with WorkspaceFiles(tmp_path) as workspace:
        skills = SkillRegistry(workspace, ("skills",), allowed_names=())
        assert skills.discover() == ()
        with pytest.raises(SkillError, match="not_authorized"):
            skills.load("alpha")
