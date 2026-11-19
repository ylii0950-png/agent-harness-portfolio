"""Pure inventory/schema/reference check; no retrieval, candidate import, or scoring."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "public-relevance-challenge-1.0.0"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def keys(value: dict, expected: set[str], label: str) -> None:
    require(isinstance(value, dict), f"{label}: not an object")
    require(set(value) == expected, f"{label}: unexpected object schema")


def run() -> dict:
    loaded = {
        name: json.loads((ROOT / name).read_text())
        for name in (
            "corpus.json",
            "requests.json",
            "oracle.json",
            "author-provenance.json",
        )
    }
    corpus = loaded["corpus.json"]
    requests = loaded["requests.json"]
    oracle = loaded["oracle.json"]
    provenance = loaded["author-provenance.json"]
    keys(corpus, {"version", "synthetic", "data_origin", "topics"}, "corpus")
    keys(requests, {"version", "cases", "evidence_cases"}, "requests")
    keys(oracle, {"version", "cases", "evidence_cases"}, "oracle")
    require(all(data["version"] == VERSION for data in loaded.values()), "version")
    require(corpus["synthetic"] is True, "synthetic corpus flag")
    require(corpus["data_origin"] == "authored_fiction", "corpus origin")
    require(len(corpus["topics"]) == 8, "topic denominator")
    refs = set()
    section_ids = set()
    word_counts = []
    for topic_index, topic in enumerate(corpus["topics"]):
        keys(topic, {"title", "sections"}, "topic")
        require(isinstance(topic["title"], str) and bool(topic["title"]), "title")
        require(3 <= len(topic["sections"]) <= 5, "section count")
        for section in topic["sections"]:
            keys(section, {"id", "heading", "body"}, "section")
            require(all(isinstance(s, str) and s for s in section.values()), "section strings")
            require(
                re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", section["id"]) is not None,
                "snake case",
            )
            ref = (topic_index, section["id"])
            require(ref not in refs, "duplicate section reference")
            refs.add(ref)
            section_ids.add(section["id"])
            words = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", section["body"]))
            require(words < 160, "section exceeds English-word budget")
            word_counts.append(words)

    def reference(value: dict) -> tuple[int, str]:
        keys(value, {"topic", "section"}, "reference")
        require(type(value["topic"]) is int, "integer topic")
        ref = (value["topic"], value["section"])
        require(ref in refs, "unknown reference")
        return ref

    case_map = {}
    for case in requests["cases"]:
        keys(case, {"id", "query", "language", "kind", "snapshot", "category"}, "request")
        require(case["id"].startswith("RCE-C-"), "request ID")
        require(case["id"] not in case_map, "duplicate request ID")
        require(case["language"] in {"en", "zh", "mixed"}, "request language")
        require(case["kind"] in {"single", "multi", "negative"}, "request kind")
        require(
            case["snapshot"] == "normal" and case["category"] == "other", "fixed request metadata"
        )
        require(isinstance(case["query"], str) and bool(case["query"].strip()), "nonempty query")
        require(
            not any(s in case["query"] for s in section_ids), "oracle section ID leaked into query"
        )
        case_map[case["id"]] = case
    positive = [c for c in case_map.values() if c["kind"] != "negative"]
    negative = [c for c in case_map.values() if c["kind"] == "negative"]
    require(len(positive) == 48 and len(negative) == 24, "retrieval denominators")
    require(
        Counter(c["language"] for c in positive) == {"en": 16, "zh": 16, "mixed": 16},
        "positive languages",
    )
    require(
        Counter(c["language"] for c in negative) == {"en": 8, "zh": 8, "mixed": 8},
        "negative languages",
    )
    require(
        Counter(c["kind"] for c in positive) == {"single": 24, "multi": 24},
        "positive kind denominators",
    )
    require(
        Counter((c["language"], c["kind"]) for c in positive)
        == {(language, k): 8 for language in ("en", "zh", "mixed") for k in ("single", "multi")},
        "language-kind balance",
    )
    require(len({c["query"] for c in case_map.values()}) == len(case_map), "duplicate query")
    gold_map = {}
    topic_coverage = Counter()
    max_chunks = 0
    total_units = 0
    for case in oracle["cases"]:
        keys(case, {"id", "units", "expected", "group"}, "oracle case")
        require(case["id"] not in gold_map, "duplicate oracle ID")
        require(case["id"] in case_map, "unmatched oracle ID")
        request = case_map[case["id"]]
        require(isinstance(case["group"], str) and bool(case["group"]), "group")
        if request["kind"] == "negative":
            require(case["units"] == [], "negative units not empty")
            require(case["expected"] in {"insufficient", "not_assessable"}, "negative expected")
        else:
            require(case["expected"] == "answerable", "positive expected")
            require(len(case["units"]) >= 1, "positive unit count")
            if request["kind"] == "single":
                require(len(case["units"]) == 1, "single unit count")
            else:
                require(len(case["units"]) >= 2, "multi complementary unit count")
            seen_units = set()
            alternatives = []
            used_topics = set()
            for unit in case["units"]:
                keys(unit, {"id", "alternatives"}, "unit")
                require(isinstance(unit["id"], str) and bool(unit["id"]), "unit ID")
                require(unit["id"] not in seen_units, "duplicate unit ID")
                seen_units.add(unit["id"])
                require(
                    isinstance(unit["alternatives"], list) and bool(unit["alternatives"]),
                    "empty alternatives",
                )
                converted = []
                for bundle in unit["alternatives"]:
                    require(isinstance(bundle, list) and bool(bundle), "empty AND bundle")
                    br = {reference(item) for item in bundle}
                    converted.append(br)
                    used_topics.update(t for t, _ in br)
                alternatives.append(converted)
            minimum = min(len(set().union(*choice)) for choice in itertools.product(*alternatives))
            require(minimum <= 3, "positive cannot fit in top three chunks")
            max_chunks = max(max_chunks, minimum)
            topic_coverage.update(used_topics)
            total_units += len(case["units"])
        gold_map[case["id"]] = case
    require(set(gold_map) == set(case_map), "request/oracle ID mismatch")
    require(all(topic_coverage[t] >= 4 for t in range(8)), "positive topic coverage")
    require(
        Counter(gold_map[c["id"]]["group"] for c in negative)
        == {
            "negative_unrelated": 6,
            "negative_missing_detail": 6,
            "negative_ambiguous": 6,
            "negative_false_premise": 6,
        },
        "negative category balance",
    )
    condition_ids = provenance["positive_condition_exception_correction_case_ids"]
    require(len(set(condition_ids)) >= 12, "condition/exception author inventory")
    require(
        all(i in case_map and case_map[i]["kind"] != "negative" for i in condition_ids),
        "condition inventory references",
    )
    ev_map = {}
    for case in requests["evidence_cases"]:
        keys(case, {"id", "query", "language", "available", "kind"}, "evidence request")
        require(case["id"] not in ev_map and case["id"] not in case_map, "duplicate evidence ID")
        require(case["language"] in {"en", "zh", "mixed"}, "evidence language")
        require(
            case["kind"]
            in {"complete", "partial", "missing", "contradiction", "conditional_nonconflict"},
            "evidence kind",
        )
        require(isinstance(case["query"], str) and bool(case["query"]), "evidence query")
        require(
            not any(s in case["query"] for s in section_ids), "oracle section ID in evidence query"
        )
        require(
            isinstance(case["available"], list) and bool(case["available"]), "available sources"
        )
        available = [reference(r) for r in case["available"]]
        require(len(set(available)) == len(available), "duplicate supplied source")
        if case["kind"] == "contradiction":
            require(len(available) >= 2, "conflict source count")
        ev_map[case["id"]] = case
    require(len(ev_map) == 15, "evidence denominator")
    require(
        Counter(c["kind"] for c in ev_map.values())
        == {
            k: 3
            for k in ("complete", "partial", "missing", "contradiction", "conditional_nonconflict")
        },
        "evidence kind balance",
    )
    require(
        Counter(c["language"] for c in ev_map.values()) == {"en": 5, "zh": 5, "mixed": 5},
        "evidence language balance",
    )
    ev_gold = {}
    expected_by_kind = {
        "complete": "supported",
        "partial": "insufficient",
        "missing": "insufficient",
        "contradiction": "not_assessable",
        "conditional_nonconflict": "supported",
    }
    for case in oracle["evidence_cases"]:
        keys(case, {"id", "expected", "reason"}, "evidence oracle")
        require(case["id"] in ev_map and case["id"] not in ev_gold, "evidence oracle ID")
        require(
            case["expected"] == expected_by_kind[ev_map[case["id"]]["kind"]], "evidence expectation"
        )
        require(isinstance(case["reason"], str) and bool(case["reason"]), "evidence reason")
        ev_gold[case["id"]] = case
    require(set(ev_gold) == set(ev_map), "evidence ID mismatch")
    require(
        provenance["synthetic"] is True and provenance["data_origin"] == "authored_fiction",
        "provenance origin",
    )
    require(not any(provenance["access_history"].values()), "prohibited contact declaration")
    return {
        "status": "PASS",
        "scope": "schema/inventory/unit mapping only; no retrieval or scoring",
        "topics": len(corpus["topics"]),
        "sections": len(refs),
        "maximum_section_english_words": max(word_counts),
        "retrieval_positive": len(positive),
        "retrieval_negative": len(negative),
        "positive_languages": dict(Counter(c["language"] for c in positive)),
        "negative_languages": dict(Counter(c["language"] for c in negative)),
        "positive_kinds": dict(Counter(c["kind"] for c in positive)),
        "positive_units": total_units,
        "minimum_positive_chunks_maximum": max_chunks,
        "positive_topic_coverage": dict(sorted(topic_coverage.items())),
        "author_declared_condition_exception_cases": len(set(condition_ids)),
        "evidence_total": len(ev_map),
        "evidence_kinds": dict(Counter(c["kind"] for c in ev_map.values())),
        "sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (*loaded, "selfcheck.py")
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
