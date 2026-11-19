"""Pinned offline bilingual pair relevance, not semantic entailment or authority."""

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

from knowledge_ticket_agent.semantic_model import _bytes, _root
from knowledge_ticket_agent.strict import require

SPEC_SHA256 = "8e06c3b917daf9e840832373f9a258fdaaa0c6bb86b8eb60367f717dee158db3"


class PairScorer(Protocol):
    identity: str

    def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float, ...]: ...


def pair_spec() -> dict[str, Any]:
    raw = files("knowledge_ticket_agent").joinpath("pair-model.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SPEC_SHA256, "pair_spec_identity")
    return dict(json.loads(raw))


def pair_identity() -> str:
    return hashlib.sha256(json.dumps(pair_spec(), sort_keys=True).encode()).hexdigest()


def verify_pair_model(path: Path) -> None:
    root = _root(path)
    require(root.is_dir(), "pair_model_unavailable")
    for name, expected in pair_spec()["files"].items():
        _bytes(root / name, expected)


def prepare_pair_model(path: Path) -> None:
    require(
        not os.getenv("CI") and not os.getenv("GITHUB_ACTIONS"), "model_download_disabled_in_ci"
    )
    root = _root(path)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in pair_spec()["files"].items():
        destination = root / name
        if destination.exists():
            _bytes(destination, expected)
            continue
        temporary = root / (name + ".part")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            size, sha = 0, hashlib.sha256()
            with (
                os.fdopen(descriptor, "wb") as out,
                urllib.request.urlopen(expected["url"], timeout=60) as response,
            ):
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    require(size <= expected["bytes"], "pair_model_size")
                    sha.update(block)
                    out.write(block)
            require(
                size == expected["bytes"] and sha.hexdigest() == expected["sha256"],
                "pair_model_hash",
            )
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    verify_pair_model(root)


def token_windows(tokens: list[int], size: int) -> tuple[tuple[int, ...], ...]:
    require(size >= 64, "pair_window_size")
    rows = []
    start = 0
    while True:
        end = min(start + size, len(tokens))
        rows.append(tuple(tokens[start:end]))
        require(len(rows) <= 32, "pair_window_limit")
        if end == len(tokens):
            return tuple(rows)
        start = end - 32


class BgePairScorer:
    """CPU only, bounded cache/windows; never downloads or executes remote code."""

    def __init__(self, path: Path) -> None:
        verify_pair_model(path)
        spec = pair_spec()
        try:
            require(
                all(importlib.metadata.version(k) == v for k, v in spec["runtime"].items()),
                "pair_runtime_version",
            )
            ort = importlib.import_module("onnxruntime")
            tokenizers = importlib.import_module("tokenizers")
            self._np: Any = importlib.import_module("numpy")
        except (ModuleNotFoundError, importlib.metadata.PackageNotFoundError):
            raise ValueError("semantic_extra_required") from None
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._session = ort.InferenceSession(
            _bytes(path / "model.onnx", spec["files"]["model.onnx"]),
            options,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = {v.name for v in self._session.get_inputs()}
        require(self._inputs == {"input_ids", "attention_mask"}, "pair_inputs")
        self._tokenizer = tokenizers.Tokenizer.from_str(
            _bytes(path / "tokenizer.json", spec["files"]["tokenizer.json"]).decode()
        )
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        self.identity = pair_identity()
        self.windows = 0
        self._cache: OrderedDict[tuple[str, str], float] = OrderedDict()

    def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float, ...]:
        require(type(pairs) is tuple and len(pairs) <= 512, "pair_batch")
        values = []
        for pair in pairs:
            require(
                type(pair) is tuple
                and len(pair) == 2
                and all(type(s) is str and len(s) <= 16384 and "\x00" not in s for s in pair),
                "pair_text",
            )
            if pair not in self._cache:
                self._cache[pair] = self._encode(*pair)
                if len(self._cache) > 2048:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(pair)
            values.append(self._cache[pair])
        return tuple(values)

    def _encode(self, query: str, passage: str) -> float:
        windows: tuple[tuple[int, ...], ...]
        encoding = self._tokenizer.encode(query, passage)
        if len(encoding.ids) <= 512:
            windows = (tuple(encoding.ids),)
        else:
            # Preserve every query and passage token. XLM-R pair layout is verified,
            # not reconstructed by truncating a string or dropping middle clauses.
            ids, seq = encoding.ids, encoding.sequence_ids
            q = [t for t, s in zip(ids, seq, strict=True) if s == 0]
            p = [t for t, s in zip(ids, seq, strict=True) if s == 1]
            require(ids == [0, *q, 2, 2, *p, 2], "pair_token_layout")
            windows = tuple(
                (0, *a, 2, 2, *b, 2) for a in token_windows(q, 256) for b in token_windows(p, 252)
            )
            require(len(windows) <= 64, "pair_window_product")
        self.windows += len(windows)
        np = self._np
        scores = []
        for row in windows:
            ids = np.array([row], dtype=np.int64)
            output = self._session.run(
                None, {"input_ids": ids, "attention_mask": np.ones_like(ids)}
            )[0]
            require(output.shape == (1, 1), "pair_output_shape")
            value = float(output[0, 0])
            require(math.isfinite(value), "pair_output_finite")
            scores.append(value)
        return max(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_pair_model(args.directory)
    else:
        verify_pair_model(args.directory)
    print(
        json.dumps(
            {
                "model_identity": pair_identity(),
                "verified": True,
                "network_preparation": args.prepare,
            }
        )
    )


if __name__ == "__main__":
    main()
