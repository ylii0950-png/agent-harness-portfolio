"""Trusted repository identity and a shared one-shot A_S journal across worktrees."""

import configparser
from pathlib import Path


def stabilization_journal(root: Path) -> Path:
    """Resolve Git metadata only; never execute Git, model text or repository scripts.

    This is a guard against accidental retries/alternate worktrees, not a defense
    against an owner deliberately deleting metadata or cloning/spoofing a repo.
    """
    root = root.resolve(strict=True)
    marker = root / ".git"
    if marker.is_symlink() or not (root / "pyproject.toml").is_file():
        raise ValueError("A_S requires the trusted project checkout")
    try:
        if marker.is_dir():
            common = marker
        elif marker.is_file():
            prefix, separator, location = marker.read_text().strip().partition(": ")
            if prefix != "gitdir" or not separator:
                raise ValueError("invalid checkout metadata")
            gitdir = (root / location).resolve(strict=True)
            common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve(strict=True)
            if gitdir.parent != common / "worktrees":
                raise ValueError("invalid linked worktree metadata")
            backlink = Path((gitdir / "gitdir").read_text().strip()).resolve(strict=True)
            if backlink != marker:
                raise ValueError("invalid worktree backlink")
        else:
            raise ValueError("missing checkout metadata")
        if not (common / "objects").is_dir() or not (common / "HEAD").is_file():
            raise ValueError("missing repository metadata")
        config = configparser.RawConfigParser()
        config.read_string((common / "config").read_text())
        origin = config.get('remote "origin"', "url")
        if origin not in {
            "https://github.com/liyongyong455-cell/agent-harness-lab.git",
            "https://github.com/liyongyong455-cell/agent-harness-lab",
            "git@github.com:liyongyong455-cell/agent-harness-lab.git",
        }:
            raise ValueError("unexpected repository identity")
        return common / "agent-harness-lab" / "phase26s-total-requests.jsonl"
    except (OSError, configparser.Error) as exc:
        raise ValueError("trusted repository metadata unavailable") from exc
