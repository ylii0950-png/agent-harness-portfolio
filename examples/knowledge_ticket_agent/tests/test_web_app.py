"""Installed HTTP adapter tests through the actual Mock, Controller and WRITE gate."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from knowledge_ticket_agent.bilingual import BilingualRetriever
from knowledge_ticket_agent.web_app import APPROVAL_SECONDS, SESSION_SECONDS, create_app

ORIGIN = "http://127.0.0.1:8000"
REQUEST = "Please process this first purchase credit policy request."
BINDING = ("call_id", "binding", "request_sha256", "expected_revision")


def browser(app: FastAPI, *, peer: str = "127.0.0.1") -> TestClient:
    return TestClient(app, base_url=ORIGIN, client=(peer, 54321))


def sign_in(client: TestClient) -> dict[str, str]:
    session = client.get("/api/session")
    assert session.status_code == 200
    assert "HttpOnly" in session.headers["set-cookie"]
    assert "SameSite=strict" in session.headers["set-cookie"]
    return {"Origin": ORIGIN, "X-LQD-CSRF": session.json()["csrf_token"]}


def submission(ticket_id: str = "KT-D01", text: str = REQUEST, **fields: Any) -> dict[str, Any]:
    return {
        "ticket_id": ticket_id,
        "request_text": text,
        "expected_revision": 0,
        "idempotency_key": str(uuid4()),
        **fields,
    }


def start(
    client: TestClient, headers: dict[str, str], body: dict[str, Any] | None = None
) -> dict[str, Any]:
    response = client.post("/api/runs", headers=headers, json=body or submission())
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def vote(run: dict[str, Any], choice: str = "approve") -> dict[str, Any]:
    return {key: run["pending_approval"][key] for key in BINDING} | {"decision": choice}


def url(run: dict[str, Any]) -> str:
    return f"/api/runs/{run['run_id']}"


def test_dashboard_api_refresh_and_exact_ask() -> None:
    with browser(create_app()) as client:
        page = client.get("/", headers={"Sec-Fetch-Site": "cross-site"})
        assert page.status_code == 200 and "AI Customer Support Copilot" in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/docs").status_code == 200
        paths = client.get("/openapi.json").json()["paths"]
        assert "post" in paths["/api/runs/{run_id}/decision"]
        assert client.get("/api/tickets").status_code == 401
        cross_site = {"Sec-Fetch-Site": "cross-site"}
        assert client.get("/api/session", headers=cross_site).status_code == 403
        headers = sign_in(client)
        assert len(client.get("/api/tickets").json()) == 12
        assert client.get("/api/tickets/KT-E01").status_code == 404
        pending = start(client, headers)
        assert pending["status"] == "awaiting_approval"
        assert pending["documentary_check"] == "SUFFICIENT_FOR_SPECIFIED_EVIDENCE"
        assert [call["tool"] for call in pending["agent_calls"]] == [
            "policy_search",
            "request_transition",
        ]
        assert all(call["ok"] for call in pending["agent_calls"])
        assert len(pending["citations"]) == 6
        assert all(cite["quote"] and cite["source_verified"] for cite in pending["citations"])
        assert pending["journal"] == [] and pending["handler_entries"] == 0
        assert pending["approval_observation"] is None
        assert client.get("/api/tickets/KT-D01").json()["status"] == "open"

        assert client.get("/api/session").json()["csrf_token"] == headers["X-LQD-CSRF"]
        assert client.get("/api/runs").json()[0]["run_id"] == pending["run_id"]
        restored = client.get(url(pending)).json()
        assert vote(restored) == vote(pending)
        result = client.post(f"{url(pending)}/decision", headers=headers, json=vote(restored))
        assert result.status_code == 200, result.text
        report = result.json()
        assert report["status"] == "completed" and report["readback_verified"]
        assert report["receipt_checks"]["APPROVAL_VALID"]
        assert report["approval_observation"] == {
            k: vote(pending)[k] for k in ("call_id", "binding", "request_sha256")
        } | {"approved": True}
        assert report["before"]["revision"] == 0 and report["after"]["revision"] == 1
        assert report["after"]["status"] == "resolved"
        assert report["journal"][0]["binding"] == vote(restored)["binding"]
        assert report["journal"] == client.get("/api/tickets/KT-D01").json()["journal"]

        duplicate = client.post(f"{url(pending)}/decision", headers=headers, json=vote(restored))
        assert duplicate.status_code == 409
        assert len(client.get(url(pending)).json()["journal"]) == 1


def test_reject_dispatches_ask_and_leaves_state_unchanged() -> None:
    with browser(create_app()) as client:
        headers = sign_in(client)
        run = start(
            client, headers, submission(text="请按首次购买的购买额度政策处理这张模拟工单。")
        )
        result = client.post(f"{url(run)}/decision", headers=headers, json=vote(run, "reject"))
        assert result.status_code == 200
        report = result.json()
        assert report["status"] == "blocked" and report["code"] == "permission_denied"
        assert report["before"] == report["after"]
        assert report["handler_entries"] == 0 and report["journal"] == []
        assert not report["readback_verified"]
        assert report["approval_observation"]["binding"] == vote(run)["binding"]
        assert report["approval_observation"]["approved"] is False
        assert any(event["code"] == "approval_denied" for event in report["audit"])
        assert report["receipt_checks"]["APPROVAL_VALID"]


def test_binding_and_session_csrf_boundaries() -> None:
    app = create_app()
    with browser(app) as client, browser(app) as outsider:
        headers = sign_in(client)
        outsider_headers = sign_in(outsider)
        body = submission()
        assert client.post("/api/runs", json=body).status_code == 403
        assert (
            client.post(
                "/api/runs", headers=headers | {"Origin": "http://evil"}, json=body
            ).status_code
            == 403
        )
        assert (
            client.post("/api/runs", headers=headers | {"Host": "evil:8000"}, json=body).status_code
            == 403
        )
        assert (
            client.post(
                "/api/runs", headers=headers | {"Sec-Fetch-Site": "cross-site"}, json=body
            ).status_code
            == 403
        )
        assert client.post("/api/runs", headers=outsider_headers, json=body).status_code == 403
        assert browser(app, peer="192.0.2.1").get("/api/session").status_code == 403
        assert (
            client.post(
                "/api/runs", headers=headers, json=body | {"approval": "approve"}
            ).status_code
            == 422
        )
        run = start(client, headers, body)
        assert outsider.get(url(run)).status_code == 404
        decision = f"{url(run)}/decision"
        assert outsider.post(decision, headers=outsider_headers, json=vote(run)).status_code == 404
        assert (
            client.post(decision, headers=headers, json={"decision": "approve"}).status_code == 422
        )
        for key, value in (
            ("binding", "a" * 64),
            ("call_id", "write_" + "b" * 32),
            ("request_sha256", "c" * 64),
            ("expected_revision", 1),
        ):
            response = client.post(decision, headers=headers, json=vote(run) | {key: value})
            assert response.status_code == 409
            assert response.json()["detail"] == "approval_binding_mismatch"
        assert client.get(url(run)).json()["handler_entries"] == 0
        assert client.get("/api/tickets/KT-D01").json()["revision"] == 0


def test_idempotent_creation_concurrent_decision_and_stale_run() -> None:
    with browser(create_app()) as client:
        headers = sign_in(client)
        body = submission()
        first = start(client, headers, body)
        assert start(client, headers, body)["run_id"] == first["run_id"]
        conflict = client.post(
            "/api/runs", headers=headers, json=body | {"request_text": "changed"}
        )
        assert (
            conflict.status_code == 409 and conflict.json()["detail"] == "idempotency_key_conflict"
        )
        stale = start(client, headers)
        different = start(client, headers, submission("KT-D08", "account access guidance"))
        with ThreadPoolExecutor(2) as executor:
            futures = [
                executor.submit(
                    client.post, f"{url(first)}/decision", headers=headers, json=vote(first)
                )
                for _ in range(2)
            ]
            assert sorted(f.result().status_code for f in futures) == [200, 409]
        invalid = client.post(f"{url(stale)}/decision", headers=headers, json=vote(stale))
        assert invalid.status_code == 409 and invalid.json()["detail"] == "stale_proposal"
        stale_report = client.get(url(stale)).json()
        assert stale_report["handler_entries"] == 0 and stale_report["journal"] == []
        assert not stale_report["receipt_checks"]["STATE_UNCHANGED_VERIFIED"]
        assert client.get("/api/tickets/KT-D01").json()["revision"] == 1
        assert start(client, headers, body)["status"] == "completed"
        assert client.post("/api/runs", headers=headers, json=submission()).status_code == 409
        other = client.post(f"{url(different)}/decision", headers=headers, json=vote(different))
        assert other.json()["status"] == "completed" and len(other.json()["journal"]) == 1


def test_expiration_is_fail_closed() -> None:
    now = 0.0
    with browser(create_app(clock=lambda: now)) as client:
        headers = sign_in(client)
        run = start(client, headers)
        now = float(APPROVAL_SECONDS)
        result = client.post(f"{url(run)}/decision", headers=headers, json=vote(run))
        assert result.status_code == 409 and result.json()["detail"] == "approval_expired"
        expired = client.get(url(run)).json()
        assert expired["pending_approval"] is None and expired["before"] == expired["after"]
        assert expired["journal"] == []
        now = float(SESSION_SECONDS)
        assert client.get(url(run)).status_code == 401
        assert sign_in(client) != headers
        assert client.get(url(run)).status_code == 404


def test_missing_evidence_blocks_before_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    search = BilingualRetriever.search

    def no_hits(self: BilingualRetriever, *args: Any, **kwargs: Any) -> Any:
        return replace(search(self, *args, **kwargs), hits=())

    monkeypatch.setattr(BilingualRetriever, "search", no_hits)
    with browser(create_app()) as client:
        run = start(client, sign_in(client))
        assert run["code"] == "evidence_blocked"
        assert run["searches"] and run["citations"] == []
        assert run["assessment"] is None and run["pending_approval"] is None
        assert run["before"] == run["after"] and run["journal"] == []
        assert not run["receipt_checks"]["EVIDENCE_COMPLETE"]


def test_conflicting_fixture_and_rule_denial() -> None:
    with browser(create_app()) as client:
        headers = sign_in(client)
        conflict = start(client, headers, submission("KT-D10", "export", snapshot="conflict"))
        assert conflict["code"] == "evidence_blocked" and conflict["pending_approval"] is None
        denied = start(client, headers, submission("KT-D04"))
        assert denied["assessment"]["decision"] == "ineligible"
        assert denied["code"] == "no_executable_proposal" and denied["pending_approval"] is None
        assert denied["before"] == denied["after"] and denied["journal"] == []


@pytest.mark.parametrize(("point", "revision"), [("before_publish", 0), ("readback", 1)])
def test_transaction_faults_report_actual_atomic_state(point: str, revision: int) -> None:
    def fault(stage: str) -> None:
        if stage == point:
            raise RuntimeError("test fault")

    with browser(create_app(fault=fault)) as client:
        headers = sign_in(client)
        run = start(client, headers)
        decision = f"{url(run)}/decision"
        response = client.post(decision, headers=headers, json=vote(run))
        assert response.status_code == 200
        report = response.json()
        assert report["status"] == "failed" and report["pending_approval"] is None
        assert report["after"]["revision"] == revision and len(report["journal"]) == revision
        assert not report["readback_verified"] and report["handler_entries"] == 1
        assert client.post(decision, headers=headers, json=vote(run)).status_code == 409
