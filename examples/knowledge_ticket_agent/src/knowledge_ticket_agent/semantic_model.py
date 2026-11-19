"""Optional pinned CPU encoder. Inference is offline; preparation is explicit."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import urllib.request
from collections import OrderedDict
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

from knowledge_ticket_agent.strict import require

SPEC_SHA256 = "d8bdbe373d42357171d209ab381b15287a4d78e66a61fd098857f43e669a2f93"


class VectorEncoder(Protocol):
    identity: str

    def embed(
        self, texts: tuple[str, ...], *, query: bool = False
    ) -> tuple[tuple[float, ...], ...]: ...


def _windows(full: list[int], head: list[int]) -> tuple[tuple[int, ...], ...]:
    """Repeat the model prefix, retain every token, never emit a redundant tail."""
    require(full[1 : 1 + len(head)] == head, "semantic_token_prefix")
    body = full[1 + len(head) : -1]
    room = 510 - len(head)
    require(room > 32 and len(full) >= 2, "semantic_token_header")
    result = []
    start = 0
    while True:
        end = min(start + room, len(body))
        result.append((full[0], *head, *body[start:end], full[-1]))
        require(len(result) <= 32, "semantic_window_limit")
        if end == len(body):
            break
        start = end - 32
    return tuple(result)


def model_spec() -> dict[str, Any]:
    raw = files("knowledge_ticket_agent").joinpath("semantic-model.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SPEC_SHA256, "model_spec_identity")
    return dict(json.loads(raw))


def model_identity() -> str:
    return hashlib.sha256(json.dumps(model_spec(), sort_keys=True).encode()).hexdigest()


def _root(path: Path) -> Path:
    selected = path.absolute()
    require(all(not p.is_symlink() for p in (selected, *selected.parents)), "model_symlink")
    return selected


def _bytes(path: Path, expected: dict[str, Any]) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        require(os.fstat(stream.fileno()).st_size == expected["bytes"], "semantic_model_size")
        raw = stream.read(expected["bytes"] + 1)
    require(len(raw) == expected["bytes"], "semantic_model_size")
    require(hashlib.sha256(raw).hexdigest() == expected["sha256"], "semantic_model_hash")
    return raw


def verify_model(path: Path) -> None:
    root = _root(path)
    require(root.is_dir(), "semantic_model_unavailable")
    for name, expected in model_spec()["files"].items():
        p = root / name
        require(p.is_file() and not p.is_symlink(), "semantic_model_unavailable")
        require(p.stat().st_size == expected["bytes"], "semantic_model_size")
        sha = hashlib.sha256()
        with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        require(sha.hexdigest() == expected["sha256"], "semantic_model_hash")


def prepare_model(path: Path) -> None:
    """Only explicit trusted CLI/API use downloads fixed public, non-executable assets."""
    require(
        not os.getenv("CI") and not os.getenv("GITHUB_ACTIONS"), "model_download_disabled_in_ci"
    )
    root = _root(path)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    spec = model_spec()
    for name, expected in spec["files"].items():
        destination = root / name
        if destination.exists() or destination.is_symlink():
            require(not destination.is_symlink(), "model_symlink")
            require(destination.is_file(), "model_file")
            require(destination.stat().st_size == expected["bytes"], "semantic_model_size")
            # Existing bytes are never replaced, including a corrupt partial deployment.
            _bytes(destination, expected)
            continue
        temporary = root / (name + ".part")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{expected['remote']}"
            sha, size = hashlib.sha256(), 0
            with (
                os.fdopen(descriptor, "wb") as target,
                urllib.request.urlopen(url, timeout=60) as response,
            ):
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    require(size <= expected["bytes"], "semantic_model_size")
                    sha.update(block)
                    target.write(block)
            require(
                size == expected["bytes"] and sha.hexdigest() == expected["sha256"],
                "semantic_model_hash",
            )
            os.link(temporary, destination)  # Atomic new name; never overwrite another writer.
        finally:
            temporary.unlink(missing_ok=True)
    verify_model(root)


class E5Encoder:
    """Single-thread CPU, no provider/network/custom code, bounded text/vector cache."""

    def __init__(self, path: Path) -> None:
        verify_model(path)
        spec = model_spec()
        try:
            require(
                all(importlib.metadata.version(n) == v for n, v in spec["runtime"].items()),
                "semantic_runtime_version",
            )
            ort = importlib.import_module("onnxruntime")
            tokenizers = importlib.import_module("tokenizers")
            self._np: Any = importlib.import_module("numpy")
        except (ModuleNotFoundError, importlib.metadata.PackageNotFoundError):
            raise ValueError("semantic_extra_required") from None
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        model = next(name for name in spec["files"] if name.endswith(".onnx"))
        # Hash the exact bytes loaded into the runtime, closing a path-replacement race.
        self._session = ort.InferenceSession(
            _bytes(path / model, spec["files"][model]), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {v.name for v in self._session.get_inputs()}
        require(
            self._inputs <= {"input_ids", "attention_mask", "token_type_ids"}, "semantic_inputs"
        )
        self._tokenizer = tokenizers.Tokenizer.from_str(
            _bytes(path / "tokenizer.json", spec["files"]["tokenizer.json"]).decode()
        )
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.identity = model_identity()
        self.windows = 0
        self._cache: OrderedDict[tuple[str, bool], tuple[float, ...]] = OrderedDict()

    def embed(
        self, texts: tuple[str, ...], *, query: bool = False
    ) -> tuple[tuple[float, ...], ...]:
        require(
            type(texts) is tuple and len(texts) <= 512 and type(query) is bool, "semantic_batch"
        )
        vectors = []
        for text in texts:
            require(
                type(text) is str and len(text) <= 16384 and "\x00" not in text, "semantic_text"
            )
            key = text, query
            if key not in self._cache:
                self._cache[key] = self._encode(text, query)
                if len(self._cache) > 512:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            vectors.append(self._cache[key])
        return tuple(vectors)

    def _encode(self, text: str, query: bool) -> tuple[float, ...]:
        np = self._np
        prefix = "query: " if query else "passage: "
        full = self._tokenizer.encode(prefix + text).ids
        # A trailing space encoded alone becomes a separate token with this tokenizer.
        # Only repeat the prefix tokens that actually occur in the complete input.
        head = self._tokenizer.encode(prefix.rstrip()).ids[1:-1]
        windows = _windows(full, head)
        self.windows += len(windows)
        pooled = []
        for start in range(0, len(windows), 4):
            group = windows[start : start + 4]
            size = max(map(len, group))
            ids = np.array([(*row, *([1] * (size - len(row)))) for row in group], dtype=np.int64)
            mask = np.array(
                [[1] * len(row) + [0] * (size - len(row)) for row in group], dtype=np.int64
            )
            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            }
            hidden = self._session.run(
                None, {k: v for k, v in inputs.items() if k in self._inputs}
            )[0]
            require(
                hidden.ndim == 3 and hidden.shape[:2] == ids.shape and hidden.shape[2] == 384,
                "semantic_output_shape",
            )
            pooled.extend(
                (hidden.astype(np.float64) * mask[:, :, None]).sum(axis=1)
                / mask.sum(axis=1)[:, None]
            )
        vector = np.mean(pooled, axis=0)
        norm = float(np.linalg.norm(vector))
        require(
            math.isfinite(norm) and norm > 0 and bool(np.isfinite(vector).all()),
            "semantic_output_finite",
        )
        return tuple(float(v) for v in vector / norm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicit preparation of checksum-pinned local E5 assets"
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        require(
            not os.getenv("CI") and not os.getenv("GITHUB_ACTIONS"), "model_download_disabled_in_ci"
        )
        prepare_model(args.directory)
    else:
        verify_model(args.directory)
    print(
        json.dumps(
            {
                "model_identity": model_identity(),
                "verified": True,
                "network_preparation": args.prepare,
            }
        )
    )


if __name__ == "__main__":
    main()
