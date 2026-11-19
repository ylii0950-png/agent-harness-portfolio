"""Loopback-only HTTP adapter for the fictional Offline Mock Controller."""

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from hmac import compare_digest
from pathlib import Path
from secrets import token_urlsafe
from threading import RLock
from time import monotonic
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.middleware.base import RequestResponseEndpoint

from agent_harness_lab.permissions import ApprovalRequest
from agent_harness_lab.providers import MockProvider
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.agent_mock import scripted_calls
from knowledge_ticket_agent.approvals import ScriptedApproval
from knowledge_ticket_agent.audit import AuditEvent
from knowledge_ticket_agent.bilingual import contextualize
from knowledge_ticket_agent.citations import Citation
from knowledge_ticket_agent.rules import RuleEvaluation
from knowledge_ticket_agent.schemas import Split, Status, Ticket
from knowledge_ticket_agent.strict import FixtureError
from knowledge_ticket_agent.ticket_store import Commit, FaultHook, TicketState, TicketStore
from knowledge_ticket_agent.workflow_controller import SearchObservation, WorkflowController

APPROVAL_SECONDS = 300
SESSION_SECONDS = 7200
MAX_SESSIONS = 16
MAX_RUNS = 64
STATIC = Path(__file__).with_name("web_static")
CsrfHeader = Annotated[str, Header(alias="X-LQD-CSRF", min_length=1, max_length=128)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StartRun(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ticket_id: Annotated[str, StringConstraints(pattern=r"^KT-D(?:0[1-9]|1[0-2])$")]
    request_text: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    expected_revision: Annotated[int, Field(ge=0)]
    idempotency_key: Annotated[UUID, Field(strict=False)]
    snapshot: Literal["normal", "conflict"] = "normal"


class Decide(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["approve", "reject"]
    call_id: Annotated[str, StringConstraints(pattern=r"^write_[0-9a-f]{32}$")]
    binding: Digest
    request_sha256: Digest
    expected_revision: Annotated[int, Field(ge=0)]


class ApprovalView(BaseModel):
    call_id: str
    tool: str
    permission: str
    request_sha256: Digest
    binding: Digest
    expires_in_seconds: int
    run_id: str
    proposal_id: str
    ticket_id: str
    tenant_id: str
    actor_id: str
    expected_revision: int
    action: str
    destination: Status
    rule_version: str
    snapshot: str
    decision: str
    context_identity: Digest
    evidence_identity: Digest
    required_permission: str
    idempotency_id: str


@dataclass(frozen=True)
class CitationView(Citation):
    title: str
    operation_ordinal: int
    source_verified: bool


class RunView(BaseModel):
    run_id: str
    status: Literal["awaiting_approval", "completed", "blocked", "failed"]
    code: str
    request_text: str
    provider: str
    retrieval_profile: str
    snapshot: str
    pending_approval: ApprovalView | None
    stages: list[str]
    operations: list[dict[str, str]]
    agent_calls: list[dict[str, str | bool | None]]
    searches: list[SearchObservation]
    citations: list[CitationView]
    assessment: RuleEvaluation | None
    documentary_check: str
    assessment_reason: str | None
    before: TicketState
    after: TicketState | None
    journal: list[Commit]
    audit: list[AuditEvent]
    handler_entries: int
    readback_verified: bool
    receipt_checks: dict[str, bool | None] | None


@dataclass(frozen=True)
class TicketView(TicketState):
    subject: str
    request_example: str
    requested_action: str | None


@dataclass(frozen=True)
class TicketDetail(TicketView):
    journal: list[Commit]


class SessionInfo(BaseModel):
    csrf_token: str
    mode: Literal["fictional_offline"]


class RunSummary(BaseModel):
    run_id: str
    ticket_id: str
    code: str


class APIError(BaseModel):
    detail: str


DECISION_ERRORS: dict[int | str, dict[str, Any]] = {
    409: {"model": APIError, "description": "The pending approval is unavailable or mismatched"},
    422: {"model": APIError, "description": "Invalid request"},
}
START_ERRORS: dict[int | str, dict[str, Any]] = {
    409: {"model": APIError, "description": "Stale ticket or reused UUID with a different body"},
    422: DECISION_ERRORS[422],
}


@dataclass
class WebRun:
    submission: StartRun
    before: TicketState
    run_id: str = field(default_factory=lambda: uuid4().hex)
    controller: WorkflowController | None = None
    ask: ApprovalRequest | None = None
    expires: float = 0
    report: dict[str, Any] | None = None
    after: TicketState | None = None
    code: str = "awaiting_approval"

    def finish(
        self,
        store: TicketStore,
        *,
        decision: Decide | None = None,
        cancel: str | None = None,
    ) -> None:
        """Consume the browser decision before dispatch, even if dispatch raises."""
        ask, self.ask = self.ask, None
        if self.controller is not None:
            approver = (
                ScriptedApproval(ask, decision.decision == "approve")
                if ask is not None and decision is not None
                else None
            )
            try:
                self.report = self.controller.finish(approver, cancel=cancel)
                self.code = self.report["code"]
            except Exception:
                self.code = "execution_error"
        self.after = store.read(self.before.ticket_id)

    def pending(self, now: float) -> dict[str, Any] | None:
        if self.ask is None or self.controller is None or self.controller.request is None:
            return None
        return {
            "call_id": self.ask.call_id,
            "tool": self.ask.tool_name,
            "permission": self.ask.permission.value,
            "request_sha256": self.controller.request.input_sha256,
            "expires_in_seconds": max(0, int(self.expires - now)),
            **dict(self.ask.arguments),
        }

    def detail(self, store: TicketStore, now: float) -> dict[str, Any]:
        c = self.controller
        preparation = c.preparation if c else None
        proposal = preparation.proposal if preparation else None
        commit = store.replay(proposal.idempotency_id, proposal.binding) if proposal else None
        titles = {d.metadata.ref: d.metadata.title for d in c.data.documents} if c else {}
        return {
            "run_id": self.run_id,
            "status": (
                "awaiting_approval"
                if self.ask is not None
                else "completed"
                if self.report and self.report["status"] == "COMPLETED"
                else "failed"
                if commit or self.code in {"execution_error", "workflow_error"}
                else "blocked"
            ),
            "code": self.code,
            "request_text": self.submission.request_text,
            "provider": "Offline Mock",
            "retrieval_profile": "A1",
            "snapshot": self.submission.snapshot,
            "pending_approval": self.pending(now),
            "stages": list(c.stages) if c else [],
            "operations": list(c.operations) if c else [],
            "agent_calls": [
                {
                    "stage": record.stage,
                    "run_id": record.run_id,
                    "call_id": record.call.call_id,
                    "tool": record.call.name,
                    "ok": record.result.ok,
                    "error": record.result.error.code if record.result.error else None,
                }
                for record in c.model_tools
            ]
            if c
            else [],
            "searches": [asdict(search) for search in c.searches] if c else [],
            "citations": [
                {
                    **asdict(source.citation),
                    "title": titles[source.citation.doc_ref],
                    "operation_ordinal": source.operation_ordinal,
                    "source_verified": True,
                }
                for source in c.sources
            ]
            if c
            else [],
            "assessment": asdict(preparation.evaluation) if preparation else None,
            "documentary_check": preparation.evidence_outcome
            if preparation
            else ("blocked" if self.code == "evidence_blocked" else "not_assessed"),
            "assessment_reason": preparation.reason if preparation else None,
            "before": asdict(self.before),
            "after": asdict(self.after) if self.after else None,
            "journal": [asdict(commit)] if commit else [],
            "audit": [asdict(event) for event in c.workflow.audit.events] if c else [],
            "handler_entries": c.workflow.handler_invocations if c else 0,
            "readback_verified": bool(c and c.execution and c.execution.readback_verified),
            "receipt_checks": self.report["checks"] if self.report else None,
        }


@dataclass
class Session:
    csrf: str
    expires: float
    store: TicketStore
    runs: dict[str, WebRun] = field(default_factory=dict)
    submissions: dict[UUID, WebRun] = field(default_factory=dict)


def ticket_info(ticket: Ticket, state: TicketState) -> dict[str, Any]:
    return {
        **asdict(state),
        "subject": ticket.subject,
        "request_example": ticket.body,
        "requested_action": ticket.facts.requested_action,
    }


def create_app(
    port: int = 8000,
    *,
    clock: Callable[[], float] = monotonic,
    fault: FaultHook | None = None,
) -> FastAPI:
    data = load_dataset()
    tickets = {t.ticket_id: t for t in data.tickets if t.split is Split.DEV}
    host = f"127.0.0.1:{port}"
    origin = f"http://{host}"
    cookie = f"lqd_session_{port}"
    app = FastAPI(
        title="Knowledge & Ticket Lab",
        version="0.1.0",
        description=(
            "Fictional, in-memory Offline Mock demo. Start with GET /api/session. "
            "Keep its cookie and send its X-LQD-CSRF token and the exact local Origin "
            "on POST requests. Decisions must echo the pending call binding."
        ),
        docs_url=None,
        redoc_url=None,
    )
    lock = RLock()
    sessions: dict[str, Session] = {}

    @app.middleware("http")
    async def local_boundary(request: Request, call_next: RequestResponseEndpoint) -> Response:
        public_document = request.method == "GET" and request.url.path in {
            "/",
            "/docs",
            "/openapi.json",
        }
        same_origin = (
            request.headers.get("sec-fetch-site", "none") in {"none", "same-origin"}
            and request.headers.get("origin", origin) == origin
        )
        if (
            request.client is None
            or request.client.host != "127.0.0.1"
            or request.headers.getlist("host") != [host]
            or (not public_document and not same_origin)
        ):
            return JSONResponse({"detail": "local_origin_required"}, status_code=403)
        if request.method == "POST":
            if request.headers.getlist("origin") != [origin]:
                return JSONResponse({"detail": "local_origin_required"}, status_code=403)
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return JSONResponse({"detail": "json_required"}, status_code=415)
            lengths = request.headers.getlist("content-length")
            if len(lengths) != 1 or "transfer-encoding" in request.headers:
                return JSONResponse({"detail": "bounded_body_required"}, status_code=411)
            length = lengths[0]
            if not length.isascii() or not length.isdecimal() or len(length) > 5:
                return JSONResponse({"detail": "bounded_body_required"}, status_code=411)
            if int(length) > 24_576:
                return JSONResponse({"detail": "body_too_large"}, status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, _: Exception) -> Response:
        return await http_exception_handler(
            request, HTTPException(status_code=422, detail="invalid_request")
        )

    def session_for(request: Request, csrf: str | None = None) -> Session:
        session = sessions.get(request.cookies.get(cookie, ""))
        if session is None or session.expires <= clock():
            raise HTTPException(401, "session_required")
        if csrf is not None and (not csrf.isascii() or not compare_digest(session.csrf, csrf)):
            raise HTTPException(403, "csrf_denied")
        return session

    def expire(session: Session, run: WebRun) -> None:
        if run.ask is None:
            return
        if run.expires <= clock():
            run.finish(session.store, cancel="approval_expired")
        elif session.store.read(run.before.ticket_id) != run.before:
            run.finish(session.store, cancel="stale_proposal")

    def owned_run(session: Session, run_id: str) -> WebRun:
        run = session.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "run_not_found")
        expire(session, run)
        return run

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/docs", include_in_schema=False)
    def api_docs() -> FileResponse:
        return FileResponse(STATIC / "docs.html")

    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

    @app.get(
        "/api/session",
        response_model=SessionInfo,
        summary="Open or recover the local browser session",
    )
    def open_session(request: Request, response: Response) -> dict[str, Any]:
        with lock:
            now = clock()
            for key in [key for key, s in sessions.items() if s.expires <= now]:
                del sessions[key]
            token = request.cookies.get(cookie, "")
            session = sessions.get(token)
            if session is None:
                if len(sessions) >= MAX_SESSIONS:
                    raise HTTPException(429, "session_capacity")
                token = token_urlsafe(32)
                session = Session(
                    token_urlsafe(32), now + SESSION_SECONDS, TicketStore(tuple(tickets.values()))
                )
                sessions[token] = session
                response.set_cookie(
                    cookie, token, httponly=True, samesite="strict", max_age=SESSION_SECONDS
                )
            return {"csrf_token": session.csrf, "mode": "fictional_offline"}

    @app.get(
        "/api/tickets",
        response_model=list[TicketView],
        summary="List fictional DEV tickets in this session",
    )
    def list_tickets(request: Request) -> list[dict[str, Any]]:
        with lock:
            session = session_for(request)
            return [ticket_info(t, session.store.read(t.ticket_id)) for t in tickets.values()]

    @app.get(
        "/api/tickets/{ticket_id}",
        response_model=TicketDetail,
        summary="Read a ticket and its atomic journal",
    )
    def read_ticket(ticket_id: str, request: Request) -> dict[str, Any]:
        with lock:
            session = session_for(request)
            ticket = tickets.get(ticket_id)
            if ticket is None:
                raise HTTPException(404, "ticket_not_found")
            snapshot = session.store.snapshot
            return {
                **ticket_info(ticket, session.store.read(ticket_id)),
                "journal": [asdict(c) for c in snapshot.commits if c.after.ticket_id == ticket_id],
            }

    @app.get(
        "/api/runs", response_model=list[RunSummary], summary="Recover run history after reload"
    )
    def list_runs(request: Request) -> list[dict[str, str]]:
        with lock:
            session = session_for(request)
            for run in session.runs.values():
                expire(session, run)
            return [
                {"run_id": r.run_id, "ticket_id": r.before.ticket_id, "code": r.code}
                for r in reversed(tuple(session.runs.values()))
            ]

    @app.get(
        "/api/runs/{run_id}",
        response_model=RunView,
        summary="Read actual trace, evidence and approval status",
    )
    def read_run(run_id: str, request: Request) -> dict[str, Any]:
        with lock:
            session = session_for(request)
            return owned_run(session, run_id).detail(session.store, clock())

    @app.post(
        "/api/runs",
        status_code=201,
        response_model=RunView,
        summary="Start Agent and pause before WRITE",
        responses=START_ERRORS,
    )
    def start_run(body: StartRun, request: Request, csrf: CsrfHeader) -> dict[str, Any]:
        with lock:
            session = session_for(request, csrf)
            existing = session.submissions.get(body.idempotency_key)
            if existing is not None:
                if existing.submission != body:
                    raise HTTPException(409, "idempotency_key_conflict")
                expire(session, existing)
                return existing.detail(session.store, clock())
            state = session.store.read(body.ticket_id)
            if state.revision != body.expected_revision or state.status is not Status.OPEN:
                raise HTTPException(409, "stale_ticket")
            if len(session.runs) >= MAX_RUNS:
                raise HTTPException(429, "run_capacity")
            run = WebRun(body, state)
            session.runs[run.run_id] = session.submissions[body.idempotency_key] = run
            try:
                _, context = contextualize(data, body.ticket_id, body.request_text, "auto")
                provider = MockProvider(
                    (
                        scripted_calls(
                            "web-intake", (("policy_search", {"query": context.query}),)
                        ),
                        "DONE",
                        scripted_calls("web-intent", (("request_transition", {}),)),
                        "DONE",
                    )
                )
                run.controller = WorkflowController(
                    data,
                    body.ticket_id,
                    provider,
                    request_text=body.request_text,
                    snapshot=body.snapshot,
                    store=session.store,
                    fault=fault,
                )
                run.ask = run.controller.begin(stop_without_proposal=True)
                if run.ask is not None:
                    run.expires = clock() + APPROVAL_SECONDS
                else:
                    run.finish(session.store)
            except FixtureError as exc:
                run.code = exc.code
                run.after = session.store.read(body.ticket_id)
            except Exception:
                run.code = "execution_error"
                run.after = session.store.read(body.ticket_id)
            return run.detail(session.store, clock())

    @app.post(
        "/api/runs/{run_id}/decision",
        response_model=RunView,
        summary="Consume one bound Approve or Reject",
        responses=DECISION_ERRORS,
    )
    def decide(run_id: str, body: Decide, request: Request, csrf: CsrfHeader) -> dict[str, Any]:
        with lock:
            session = session_for(request, csrf)
            run = owned_run(session, run_id)
            pending = run.pending(clock())
            if pending is None:
                raise HTTPException(
                    409,
                    run.code
                    if run.code in {"approval_expired", "stale_proposal"}
                    else "decision_unavailable",
                )
            if any(
                pending[key] != value
                for key, value in body.model_dump(exclude={"decision"}).items()
            ):
                raise HTTPException(409, "approval_binding_mismatch")
            run.finish(session.store, decision=body)
            return run.detail(session.store, clock())

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    uvicorn.run(
        create_app(args.port),
        host="127.0.0.1",
        port=args.port,
        workers=1,
        proxy_headers=False,
        access_log=False,
    )
