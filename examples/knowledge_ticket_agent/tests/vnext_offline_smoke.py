"""Explicit local-model integration checks; never auto-collected or downloaded in CI."""

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.agent_grading import grade_run
from knowledge_ticket_agent.business_demo import run_request, summarize
from knowledge_ticket_agent.evaluation_runtime import CaseInput, execute_public_task
from knowledge_ticket_agent.retrieval_vnext import VNextRetriever
from knowledge_ticket_agent.schemas import Dataset
from knowledge_ticket_agent.semantic_model import E5Encoder
from knowledge_ticket_agent.workflow_controller import WorkflowController

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.providers import MockProvider, ModelResponse


def check(directory: Path) -> dict[str, object]:
    encoder = E5Encoder(directory)
    # Compare short-input embeddings with the upstream attention-mask mean recipe
    # applied directly to the complete tokenizer input, bypassing our window builder.
    np = encoder._np
    errors = []
    for query in (True, False):
        prefix = "query: " if query else "passage: "
        text = "Explain the first purchase credit policy."
        ids = np.array([encoder._tokenizer.encode(prefix + text).ids], dtype=np.int64)
        inputs = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),
        }
        hidden = encoder._session.run(
            None, {k: v for k, v in inputs.items() if k in encoder._inputs}
        )[0]
        direct = hidden.astype(np.float64).mean(axis=1)[0]
        direct /= np.linalg.norm(direct)
        actual = np.array(encoder.embed((text,), query=query)[0])
        error = float(np.max(np.abs(actual - direct)))
        assert error < 1e-6
        errors.append(error)

    data = load_dataset()
    flows = []
    for text in (
        "Please handle this ticket under the first purchase credit policy.",
        "请按首次购买的购买额度调整政策处理这张模拟工单。",
        "请依据 first purchase credit 处理模拟工单。",
        "orbital banana municipal orchestra",
    ):
        for approval in ("approve", "refuse"):
            run = run_request(
                data,
                "KT-D01",
                text,
                approval=approval,
                retriever=VNextRetriever(data, encoder, variant="D"),
            )
            receipt = summarize(run, "PUBLIC_OFFLINE_MODEL_SMOKE")
            supported = receipt["request"]["status"] == "supported"
            assert receipt["commits"] == int(supported and approval == "approve")
            assert receipt["verification"] == "PASS"
            if supported:
                assert receipt["retrieval_method"].startswith("lqd-semantic-hybrid-1.0.1:D:")
            else:
                assert receipt["unchanged_state_verified"]
            flows.append(receipt)

    def factory(dataset: Dataset) -> VNextRetriever:
        return VNextRetriever(dataset, encoder, variant="D")

    compacted = []
    for approval in ("approve", "refuse"):
        task = CaseInput(
            "KT-D01",
            "first purchase credit",
            "normal",
            "SYN-STAFF-02",
            approval,
            compaction=True,
        )
        run = execute_public_task(data, task, retriever_factory=factory)
        grade = grade_run(run)
        assert not grade.issues and grade.compactions > 0
        assert grade.commits == int(approval == "approve")
        assert grade.readback_verified if approval == "approve" else not grade.commits
        compacted.append(asdict(grade))
    controllers = []
    for approval in ("approve", "refuse"):
        provider = MockProvider(
            (
                ModelResponse(
                    tool_calls=(
                        ToolCall("search", "policy_search", '{"query":"first purchase credit"}'),
                    )
                ),
                "DONE",
                ModelResponse(tool_calls=(ToolCall("intent", "request_transition", "{}"),)),
                "DONE",
            )
        )
        profile = VNextRetriever(data, encoder, variant="D")
        controller = WorkflowController(
            data,
            "KT-D01",
            provider,
            approval=approval,
            retriever=profile,
        )
        receipt = controller.run()
        assert receipt["workflow_acceptance"] == "PASS", receipt
        assert receipt["commits"] == int(approval == "approve")
        assert all(s.result.method == profile.method_identity for s in controller.searches)
        controllers.append(receipt)
    return {
        "scope": "PUBLIC_ENGINEERING_OFFLINE_NATIVE_MODEL",
        "python": platform.python_version(),
        "model_identity": encoder.identity,
        "short_input_direct_pooling_max_errors": errors,
        "flows": flows,
        "compacted": compacted,
        "controllers": controllers,
        "external_inference_requests": 0,
        "formal_executions": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.model_directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        "PASS: direct pooling, eight multilingual/unsupported flows, "
        "two compacted and two controller flows"
    )


if __name__ == "__main__":
    main()
